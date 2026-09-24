#include <brass/runtime/osr_coordinator.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include "../vm/fast_interpreter_impl.hpp"
#include <brass/mir/dominators.hpp>
#include <brass/mir/osr.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace brass::runtime {

namespace {

// Bridges an interpreter's heap to the native runtime for the duration of an
// OSR call. OSR code allocates through brass_gc_alloc and polls
// brass_gc_safepoint, which use the thread's active GC and stack maps.
// Pointing them at the interpreter's collector (the one its own allocations
// use) and at the OSR module's stack maps makes native allocations land in the
// same heap as the migrated gcrefs. A collection then roots both the native
// frames (stack-walked through the code stack-map registry, which also holds
// the maps of tier-1 and tier-2 code the OSR code calls) and the interpreter
// frames (the collector's root provider, installed by the interpreter). Both
// are restored on exit, so code that runs later on this thread finds the
// maps it found before.
class NativeGcBridge {
public:
    NativeGcBridge(MiniCheneyGC* gc, GenerationalGC* gen_gc, const ModuleStackMap* maps) noexcept
        : prev_gc_(brass_get_active_gc()),
          prev_gen_gc_(brass_get_active_generational_gc()),
          prev_maps_(brass_get_active_stack_maps()) {
        brass_set_active_gc(gc);
        brass_set_active_generational_gc(gen_gc);
        brass_set_active_stack_maps(maps);
    }
    ~NativeGcBridge() {
        brass_set_active_gc(prev_gc_);
        brass_set_active_generational_gc(prev_gen_gc_);
        brass_set_active_stack_maps(prev_maps_);
    }
    NativeGcBridge(const NativeGcBridge&) = delete;
    NativeGcBridge& operator=(const NativeGcBridge&) = delete;

private:
    MiniCheneyGC* prev_gc_;
    GenerationalGC* prev_gen_gc_;
    const ModuleStackMap* prev_maps_;
};

// A native deopt frame's state values as the interpreter holds them, gcref
// values tagged as such.
std::vector<RuntimeValue> guard_state_values(const Function& fn, const DeoptFrame& dframe) {
    std::vector<RuntimeValue> vals = dframe.to_runtime_values();
    if (const Instruction* g = fn.find_guard(dframe.resume_id)) {
        for (size_t i = 0; i < g->state_map().size() && i < vals.size(); ++i) {
            const Value* sv = g->state_map()[i];
            if (sv && sv->type().is_gcref() && !vals[i].is_gcref()) {
                vals[i] = RuntimeValue::from_gcref(vals[i].as_u64());
            }
        }
    }
    return vals;
}

// The function whose code a guard failure seen by an OSR call's deopt
// handler came from. The OSR module is the whole module compiled again, so
// the OSR code calls the module's own copies of its callees, which have no
// resumer: `osr_fn` for its own code, another function of its module for
// that function's copy. Anything else (no code entry, code of another
// module, such as a module the host compiled and calls through a pointer) is
// not this call's to resume: null, and the handler passes it on
// (decline_foreign_deopt).
const Function* deopt_owner(const CompiledModule& osr_mod, const Function& osr_fn, const DeoptFrame& dframe) {
    if (dframe.code_entry) {
        if (osr_mod.get_symbol_address(osr_fn.name()) == dframe.code_entry) return &osr_fn;
        if (const Module* mod = osr_fn.parent()) {
            for (const Function* f : mod->functions()) {
                if (f && !f->blocks().empty() && osr_mod.get_symbol_address(f->name()) == dframe.code_entry) {
                    return f;
                }
            }
        }
    }
    return nullptr;
}

// A guard failure of code outside the OSR module: the handler the OSR call
// replaced (an outer OSR call's) gets it, else the deopt entry does what it
// does with no handler (the code's own exit stub).
void* decline_foreign_deopt(const DeoptHandlerFn& prev, const DeoptFrame& dframe) {
    if (prev) return prev(dframe);
    deopt_handler_decline();
    return nullptr;
}

// Runs a callee's Tier-0 continuation for OSR code. A MIR exception it throws
// must reach the OSR code's landing pads as a native throw does (a C++
// exception unwinding into them is not one): the deopt entry raises it to
// the first pad above the callee's frame and below `stack_limit`, or rethrows
// it when there is none.
template <typename Run>
RuntimeValue run_callee_continuation(uintptr_t stack_limit, Run&& run) {
    try {
        return run();
    } catch (const InterpreterThrownException& ex) {
        deopt_handler_throw_native(ex.value().raw_bits(), stack_limit, std::current_exception());
    } catch (const BrassException& ex) {
        deopt_handler_throw_native(ex.value().raw(), stack_limit, std::current_exception());
    }
    return RuntimeValue();
}

// Whether a guard of `fn` failing in its OSR code may come from an inner
// activation of `fn` rather than the OSR'd one: `fn` has a guard and its OSR
// code may call `fn` again (a call path back to it, or a call it cannot see
// through). The deopt record names only the code that failed, not which
// activation, so such a function is not OSR'd: resuming the OSR'd frame for
// an inner call's failure would finish the wrong activation.
bool osr_guard_frame_ambiguous(const Function& fn) {
    const Module* mod = fn.parent();
    if (!mod) return true;
    bool has_guard = false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::guard) has_guard = true;
        }
    }
    if (!has_guard) return false;
    std::vector<const Function*> work{&fn};
    std::unordered_set<const Function*> seen{&fn};
    while (!work.empty()) {
        const Function* cur = work.back();
        work.pop_back();
        for (const BasicBlock* bb : cur->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (!inst || !inst->is_call()) continue;
                if (inst->opcode() != Opcode::call && inst->opcode() != Opcode::invoke) return true;
                const Function* callee = mod->get_function(inst->symbol());
                if (callee == &fn) return true;
                if (callee && !callee->blocks().empty() && seen.insert(callee).second) work.push_back(callee);
            }
        }
    }
    return false;
}

} // namespace

static thread_local brass::Interpreter* t_active_interpreter = nullptr;
static thread_local const brass::Function* t_active_fn = nullptr;
static thread_local brass::InterpreterFrame* t_active_frame = nullptr;

OsrCoordinator& OsrCoordinator::instance() {
    static OsrCoordinator s_instance;
    return s_instance;
}

OsrCoordinator::OsrCoordinator() = default;

OsrCoordinator::OsrCoordinator(TieringRegistry& registry) : registry_(&registry) {
    if (registry.is_default()) {
        throw std::logic_error("OsrCoordinator: the default program's coordinator is OsrCoordinator::instance()");
    }
}

TieringRegistry& OsrCoordinator::registry() const noexcept {
    return registry_ ? *registry_ : TieringRegistry::instance();
}

void OsrCoordinator::set_threshold(uint64_t threshold) noexcept {
    threshold_ = threshold;
    registry().default_config().backedge_osr_threshold = threshold;
}

void OsrCoordinator::set_active_interpreter(Interpreter* interp) noexcept {
    t_active_interpreter = interp;
}

Interpreter* OsrCoordinator::active_interpreter() const noexcept {
    return t_active_interpreter;
}

void OsrCoordinator::set_active_frame(InterpreterFrame* frame) noexcept {
    t_active_frame = frame;
}

InterpreterFrame* OsrCoordinator::active_frame() const noexcept {
    return t_active_frame;
}

void OsrCoordinator::clear_cache() {
    std::lock_guard<std::mutex> lock(osr_mutex_);
    osr_modules_.clear();
    osr_targets_.clear();
    loop_latches_.clear();
    backedge_pairs_.clear();
}

bool OsrCoordinator::is_loop_backedge(const Function& fn, const BasicBlock* from_bb, const BasicBlock* to_bb) {
    if (!from_bb || !to_bb) return false;

    std::string fn_name(fn.name());
    std::lock_guard<std::mutex> lock(osr_mutex_);
    auto it = backedge_pairs_.find(fn_name);
    if (it == backedge_pairs_.end()) {
        const_cast<Function&>(fn).rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        auto& pairs = backedge_pairs_[fn_name];
        auto& latches = loop_latches_[fn_name];
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb || !dom.is_reachable(bb)) continue;
            for (const BasicBlock* succ : bb->successors()) {
                if (!succ || !dom.is_reachable(succ)) continue;
                if (dom.dominates(succ, bb)) {
                    latches.insert(bb);
                    uint64_t edge_key = (static_cast<uint64_t>(bb->id()) << 32) | static_cast<uint64_t>(succ->id());
                    pairs.insert(edge_key);
                }
            }
        }
        it = backedge_pairs_.find(fn_name);
    }

    uint64_t edge_key = (static_cast<uint64_t>(from_bb->id()) << 32) | static_cast<uint64_t>(to_bb->id());
    return it != backedge_pairs_.end() && it->second.find(edge_key) != it->second.end();
}

bool OsrCoordinator::try_osr_migration(
    Interpreter& interp,
    const Function& fn,
    BasicBlock* loop_header,
    InterpreterFrame& frame,
    RuntimeValue& out_result
) {
    if (!enabled_ || !loop_header) return false;

    TieringFeedback& feedback = registry().get_or_create(fn.name());
    feedback.record_backedge();

    if (feedback.is_bailed_out() || !feedback.should_trigger_osr(threshold_)) {
        return false;
    }

    std::string cache_key = std::string(fn.name()) + "@" + std::to_string(loop_header->id());
    CompiledModule* comp_mod = nullptr;
    void* osr_entry_addr = nullptr;
    OsrTarget target;

    {
        std::lock_guard<std::mutex> lock(osr_mutex_);
        auto mod_it = osr_modules_.find(cache_key);
        if (mod_it != osr_modules_.end()) {
            comp_mod = mod_it->second.get();
        } else {
            if (!fn.parent()) return false;
            if (osr_guard_frame_ambiguous(fn)) {
                feedback.record_bailout("OSR refused: a guard failure could be an inner activation's");
                return false;
            }
            HostEngine engine;
            auto compiled = engine.compile_with_osr(*fn.parent(), fn.name(), loop_header->id());
            if (!compiled) {
                feedback.record_bailout("OSR compilation failed");
                return false;
            }
            comp_mod = compiled.get();
            osr_modules_[cache_key] = std::move(compiled);

            OsrTarget t = analyze_osr_target(const_cast<Function&>(fn), loop_header);
            osr_targets_[cache_key] = std::move(t);
        }

        if (!comp_mod) return false;

        target = osr_targets_[cache_key];
        osr_entry_addr = comp_mod->get_osr_entry_address(fn.name());
        if (!osr_entry_addr) {
            feedback.record_bailout("OSR entry point not found");
            return false;
        }
    }

    // Pack migration frame with live-in values from the interpreter frame
    OsrMigrationFrame mig_frame;
    mig_frame.loop_header_id = loop_header->id();
    for (size_t i = 0; i < target.live_ins.size() && i < OsrMigrationFrame::kMaxInlineSlots; ++i) {
        Value* v = target.live_ins[i];
        RuntimeValue val = frame.get_value(v);
        mig_frame.add_slot(static_cast<uint32_t>(i), val.as_u64());
    }

    // Set active execution context for potential native deoptimization on this thread
    Interpreter* prev_interp = t_active_interpreter;
    const Function* prev_fn = t_active_fn;
    InterpreterFrame* prev_frame = t_active_frame;

    t_active_interpreter = &interp;
    t_active_fn = &fn;
    t_active_frame = &frame;

    bool deopt_occurred = false;
    RuntimeValue deopt_res;

    // Every native frame of this OSR call lies below this frame's locals.
    const uintptr_t osr_stack_limit = reinterpret_cast<uintptr_t>(&mig_frame);
    auto prev_handler = get_deopt_handler();
    register_deopt_handler([&](const DeoptFrame& dframe) -> void* {
        const Function* owner_p = deopt_owner(*comp_mod, fn, dframe);
        if (!owner_p) return decline_foreign_deopt(prev_handler, dframe);
        const Function& owner = *owner_p;
        total_native_deopts_++;
        registry().get_or_create(owner.name()).record_deoptimization();
        if (&owner != &fn) {
            // A callee's copy: finish that call in Tier 0 and hand its
            // result back to the OSR code, which carries on.
            RuntimeValue r = run_callee_continuation(osr_stack_limit, [&] {
                return interp.resume_after_guard(owner, dframe.resume_id, guard_state_values(owner, dframe), nullptr);
            });
            return reinterpret_cast<void*>(r.as_u64());
        }
        deopt_occurred = true;
        deopt_res = interp.resume_after_guard(fn, dframe.resume_id, guard_state_values(fn, dframe), &frame);
        return reinterpret_cast<void*>(deopt_res.as_u64());
    });

    struct HandlerScopeGuard {
        DeoptHandlerFn prev;
        Interpreter* prev_interp;
        const Function* prev_fn;
        InterpreterFrame* prev_frame;
        ~HandlerScopeGuard() {
            register_deopt_handler(prev);
            t_active_interpreter = prev_interp;
            t_active_fn = prev_fn;
            t_active_frame = prev_frame;
        }
    } guard{prev_handler, prev_interp, prev_fn, prev_frame};

    // The reference interpreter allocates only from its semispace collector
    // (its generational GC, when set, is used for write barriers alone), so
    // native allocations must not go to a generational heap either.
    NativeGcBridge gc_bridge(&interp.gc(), nullptr, &comp_mod->stack_maps());

    // Invoke specialized OSR entry stub
    Type ret_t = fn.return_type();
    if (ret_t.is_void()) {
        using NativeOsrFn = void (*)(const OsrMigrationFrame*);
        reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        out_result = deopt_occurred ? deopt_res : RuntimeValue::from_void();
    } else if (ret_t.is_float()) {
        if (ret_t.kind() == TypeKind::F32) {
            using NativeOsrFn = float (*)(const OsrMigrationFrame*);
            float r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
            out_result = deopt_occurred ? deopt_res : RuntimeValue::from_f32(r);
        } else {
            using NativeOsrFn = double (*)(const OsrMigrationFrame*);
            double r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
            out_result = deopt_occurred ? deopt_res : RuntimeValue::from_f64(r);
        }
    } else if (ret_t.is_vector()) {
#if defined(__x86_64__) || defined(_M_X64)
        using NativeOsrFn = __m128 (*)(const OsrMigrationFrame*);
        __m128 r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        if (deopt_occurred) {
            out_result = deopt_res;
        } else {
            alignas(16) uint8_t b[16];
            std::memcpy(b, &r, 16);
            out_result = RuntimeValue::from_v128(ret_t, b);
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
        using NativeOsrFn = uint8x16_t (*)(const OsrMigrationFrame*);
        uint8x16_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        if (deopt_occurred) {
            out_result = deopt_res;
        } else {
            alignas(16) uint8_t b[16];
            vst1q_u8(b, r);
            out_result = RuntimeValue::from_v128(ret_t, b);
        }
#else
        (void)osr_entry_addr;
        alignas(16) uint8_t b[16] = {0};
        out_result = RuntimeValue::from_v128(ret_t, b);
#endif
    } else if (ret_t.is_i32()) {
        using NativeOsrFn = int32_t (*)(const OsrMigrationFrame*);
        int32_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        out_result = deopt_occurred ? deopt_res : RuntimeValue::from_i32(r);
    } else {
        using NativeOsrFn = int64_t (*)(const OsrMigrationFrame*);
        int64_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        out_result = deopt_occurred ? deopt_res : RuntimeValue::from_bits(ret_t, static_cast<uint64_t>(r));
    }

    total_osr_migrations_++;
    return true;
}

bool OsrCoordinator::try_osr_migration(
    FastInterpreter& interp,
    const Function& fn,
    BasicBlock* loop_header,
    FastFrame& frame,
    RuntimeValue& out_result
) {
    if (!enabled_ || !loop_header) return false;

    TieringFeedback& feedback = registry().get_or_create(fn.name());
    feedback.record_backedge();

    if (feedback.is_bailed_out() || !feedback.should_trigger_osr(threshold_)) {
        return false;
    }

    std::string cache_key = std::string(fn.name()) + "@" + std::to_string(loop_header->id());
    CompiledModule* comp_mod = nullptr;
    void* osr_entry_addr = nullptr;
    OsrTarget target;

    {
        std::lock_guard<std::mutex> lock(osr_mutex_);
        auto mod_it = osr_modules_.find(cache_key);
        if (mod_it != osr_modules_.end()) {
            comp_mod = mod_it->second.get();
        } else {
            if (!fn.parent()) return false;
            if (osr_guard_frame_ambiguous(fn)) {
                feedback.record_bailout("OSR refused: a guard failure could be an inner activation's");
                return false;
            }
            HostEngine engine;
            auto compiled = engine.compile_with_osr(*fn.parent(), fn.name(), loop_header->id());
            if (!compiled) {
                feedback.record_bailout("OSR compilation failed");
                return false;
            }
            comp_mod = compiled.get();
            osr_modules_[cache_key] = std::move(compiled);

            OsrTarget t = analyze_osr_target(const_cast<Function&>(fn), loop_header);
            osr_targets_[cache_key] = std::move(t);
        }

        if (!comp_mod) return false;

        target = osr_targets_[cache_key];
        osr_entry_addr = comp_mod->get_osr_entry_address(fn.name());
        if (!osr_entry_addr) {
            feedback.record_bailout("OSR entry point not found");
            return false;
        }
    }

    // Pack migration frame with live-in values from the FastFrame
    OsrMigrationFrame mig_frame;
    mig_frame.loop_header_id = loop_header->id();
    const BytecodeFunction* bfn = frame.bfn;
    for (size_t i = 0; i < target.live_ins.size() && i < OsrMigrationFrame::kMaxInlineSlots; ++i) {
        Value* v = target.live_ins[i];
        uint64_t val = 0;
        if (bfn) {
            auto it = bfn->ssa_to_reg.find(v->id());
            if (it != bfn->ssa_to_reg.end() && it->second < frame.num_registers) {
                val = frame.registers[it->second];
            }
        }
        mig_frame.add_slot(static_cast<uint32_t>(i), val);
    }

    bool deopt_occurred = false;
    RuntimeValue deopt_res;

    // Every native frame of this OSR call lies below this frame's locals.
    const uintptr_t osr_stack_limit = reinterpret_cast<uintptr_t>(&mig_frame);
    auto prev_handler = get_deopt_handler();
    register_deopt_handler([&](const DeoptFrame& dframe) -> void* {
        // A callee's copy in the OSR module finishes that call in Tier 0 and
        // hands its result back to the OSR code, which carries on; the OSR'd
        // function's own failure finishes the whole OSR call.
        const Function* owner_p = deopt_owner(*comp_mod, fn, dframe);
        if (!owner_p) return decline_foreign_deopt(prev_handler, dframe);
        const Function& owner = *owner_p;
        total_native_deopts_++;
        registry().get_or_create(owner.name()).record_deoptimization();
        // The exits the interpreter's guard takes, in its order.
        const Instruction* g_inst = owner.find_guard(dframe.resume_id);
        if (!g_inst) {
            throw InterpreterException("No guard with resume id " + std::to_string(dframe.resume_id) +
                                       " in function " + std::string(owner.name()));
        }
        std::vector<RuntimeValue> state_vals = guard_state_values(owner, dframe);
        auto finish = [&] {
            if (const Function* stub = owner.guard_exit_stub(*g_inst)) return interp.run(*stub, state_vals);
            return interp.resume(owner, dframe.resume_id, state_vals);
        };
        // The OSR'd function's own continuation ran its handlers in Tier 0:
        // what it throws leaves the OSR call.
        RuntimeValue r = &owner == &fn ? finish() : run_callee_continuation(osr_stack_limit, finish);
        if (&owner == &fn) {
            deopt_occurred = true;
            deopt_res = r;
        }
        return reinterpret_cast<void*>(r.as_u64());
    });

    struct HandlerScopeGuard {
        DeoptHandlerFn prev;
        ~HandlerScopeGuard() {
            register_deopt_handler(prev);
        }
    } guard{prev_handler};

    // Native allocations go where FastInterpreter::allocate_gc sends them: its
    // generational GC when set, else its semispace collector.
    NativeGcBridge gc_bridge(&interp.gc(), interp.generational_gc(), &comp_mod->stack_maps());

    // Invoke specialized OSR entry stub
    Type ret_t = fn.return_type();
    if (ret_t.is_void()) {
        using NativeOsrFn = void (*)(const OsrMigrationFrame*);
        reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        out_result = deopt_occurred ? deopt_res : RuntimeValue::from_void();
    } else if (ret_t.is_float()) {
        if (ret_t.kind() == TypeKind::F32) {
            using NativeOsrFn = float (*)(const OsrMigrationFrame*);
            float r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
            out_result = deopt_occurred ? deopt_res : RuntimeValue::from_f32(r);
        } else {
            using NativeOsrFn = double (*)(const OsrMigrationFrame*);
            double r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
            out_result = deopt_occurred ? deopt_res : RuntimeValue::from_f64(r);
        }
    } else if (ret_t.is_vector()) {
#if defined(__x86_64__) || defined(_M_X64)
        using NativeOsrFn = __m128 (*)(const OsrMigrationFrame*);
        __m128 r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        if (deopt_occurred) {
            out_result = deopt_res;
        } else {
            alignas(16) uint8_t b[16];
            std::memcpy(b, &r, 16);
            out_result = RuntimeValue::from_v128(ret_t, b);
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
        using NativeOsrFn = uint8x16_t (*)(const OsrMigrationFrame*);
        uint8x16_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        if (deopt_occurred) {
            out_result = deopt_res;
        } else {
            alignas(16) uint8_t b[16];
            vst1q_u8(b, r);
            out_result = RuntimeValue::from_v128(ret_t, b);
        }
#else
        (void)osr_entry_addr;
        alignas(16) uint8_t b[16] = {0};
        out_result = RuntimeValue::from_v128(ret_t, b);
#endif
    } else if (ret_t.is_i32()) {
        using NativeOsrFn = int32_t (*)(const OsrMigrationFrame*);
        int32_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        out_result = deopt_occurred ? deopt_res : RuntimeValue::from_i32(r);
    } else {
        using NativeOsrFn = uint64_t (*)(const OsrMigrationFrame*);
        uint64_t r = reinterpret_cast<NativeOsrFn>(osr_entry_addr)(&mig_frame);
        if (deopt_occurred) {
            out_result = deopt_res;
        } else if (ret_t.is_gcref()) {
            out_result = RuntimeValue::from_gcref(static_cast<uintptr_t>(r));
        } else {
            out_result = RuntimeValue::from_bits(ret_t, r);
        }
    }

    total_osr_migrations_++;
    return true;
}

void* OsrCoordinator::handle_native_deopt(const DeoptFrame& deopt_frame) {
    total_native_deopts_++;
    const Function* cur_fn = t_active_fn;
    Interpreter* cur_interp = t_active_interpreter;
    InterpreterFrame* cur_frame = t_active_frame;
    if (cur_fn) {
        TieringFeedback& fb = registry().get_or_create(cur_fn->name());
        fb.record_deoptimization();
    }
    if (cur_interp && cur_fn) {
        RuntimeValue res = cur_interp->resume_after_guard(*cur_fn, deopt_frame.resume_id,
                                                          guard_state_values(*cur_fn, deopt_frame), cur_frame);
        return reinterpret_cast<void*>(res.as_u64());
    }
    return nullptr;
}

} // namespace brass::runtime
