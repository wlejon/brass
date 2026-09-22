#include <brass/runtime/osr_coordinator.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include "../vm/fast_interpreter_impl.hpp"
#include <brass/mir/dominators.hpp>
#include <brass/mir/osr.hpp>
#include <brass/embedding/embedding.hpp>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace brass::runtime {

static thread_local brass::Interpreter* t_active_interpreter = nullptr;
static thread_local const brass::Function* t_active_fn = nullptr;
static thread_local brass::InterpreterFrame* t_active_frame = nullptr;

OsrCoordinator& OsrCoordinator::instance() {
    static OsrCoordinator s_instance;
    return s_instance;
}

OsrCoordinator::OsrCoordinator() = default;

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

    TieringFeedback& feedback = TieringRegistry::instance().get_or_create(fn.name());
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

    auto prev_handler = get_deopt_handler();
    register_deopt_handler([&](const DeoptFrame& dframe) -> void* {
        deopt_occurred = true;
        total_native_deopts_++;
        TieringFeedback& fb = TieringRegistry::instance().get_or_create(fn.name());
        fb.record_deoptimization();
        std::vector<RuntimeValue> state_vals = dframe.to_runtime_values();
        const Instruction* g_inst = nullptr;
        for (const auto* bb : fn.blocks()) {
            if (!bb) continue;
            for (const auto* inst : *bb) {
                if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == dframe.resume_id) {
                    g_inst = inst;
                    break;
                }
            }
            if (g_inst) break;
        }
        if (g_inst) {
            for (size_t i = 0; i < g_inst->state_map().size() && i < state_vals.size(); ++i) {
                const Value* sv = g_inst->state_map()[i];
                if (sv && sv->type().is_gcref() && !state_vals[i].is_gcref()) {
                    state_vals[i] = RuntimeValue::from_gcref(state_vals[i].as_u64());
                }
            }
        }
        deopt_res = interp.resume_with_frame(fn, dframe.resume_id, state_vals, frame);
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

    TieringFeedback& feedback = TieringRegistry::instance().get_or_create(fn.name());
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

    auto prev_handler = get_deopt_handler();
    register_deopt_handler([&](const DeoptFrame& dframe) -> void* {
        deopt_occurred = true;
        total_native_deopts_++;
        TieringFeedback& fb = TieringRegistry::instance().get_or_create(fn.name());
        fb.record_deoptimization();
        std::vector<RuntimeValue> state_vals = dframe.to_runtime_values();
        const Instruction* g_inst = nullptr;
        for (const auto* bb : fn.blocks()) {
            if (!bb) continue;
            for (const auto* inst : *bb) {
                if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == dframe.resume_id) {
                    g_inst = inst;
                    break;
                }
            }
            if (g_inst) break;
        }
        if (g_inst) {
            for (size_t i = 0; i < g_inst->state_map().size() && i < state_vals.size(); ++i) {
                const Value* sv = g_inst->state_map()[i];
                if (sv && sv->type().is_gcref() && !state_vals[i].is_gcref()) {
                    state_vals[i] = RuntimeValue::from_gcref(state_vals[i].as_u64());
                }
            }
        }
        deopt_res = interp.resume(fn, dframe.resume_id, state_vals);
        return reinterpret_cast<void*>(deopt_res.as_u64());
    });

    struct HandlerScopeGuard {
        DeoptHandlerFn prev;
        ~HandlerScopeGuard() {
            register_deopt_handler(prev);
        }
    } guard{prev_handler};

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
        TieringFeedback& fb = TieringRegistry::instance().get_or_create(cur_fn->name());
        fb.record_deoptimization();
    }
    if (cur_interp && cur_fn) {
        std::vector<RuntimeValue> state_vals = deopt_frame.to_runtime_values();
        const Instruction* g_inst = nullptr;
        for (const auto* bb : cur_fn->blocks()) {
            if (!bb) continue;
            for (const auto* inst : *bb) {
                if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == deopt_frame.resume_id) {
                    g_inst = inst;
                    break;
                }
            }
            if (g_inst) break;
        }
        if (g_inst) {
            for (size_t i = 0; i < g_inst->state_map().size() && i < state_vals.size(); ++i) {
                const Value* sv = g_inst->state_map()[i];
                if (sv && sv->type().is_gcref() && !state_vals[i].is_gcref()) {
                    state_vals[i] = RuntimeValue::from_gcref(state_vals[i].as_u64());
                }
            }
        }
        RuntimeValue res;
        if (cur_frame) {
            res = cur_interp->resume_with_frame(*cur_fn, deopt_frame.resume_id, state_vals, *cur_frame);
        } else {
            res = cur_interp->resume(*cur_fn, deopt_frame.resume_id, state_vals);
        }
        return reinterpret_cast<void*>(res.as_u64());
    }
    return nullptr;
}

} // namespace brass::runtime
