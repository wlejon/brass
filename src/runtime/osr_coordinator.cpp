#include <brass/runtime/osr_coordinator.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/osr.hpp>
#include <brass/embedding/embedding.hpp>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace brass::runtime {

OsrCoordinator& OsrCoordinator::instance() {
    static OsrCoordinator s_instance;
    return s_instance;
}

OsrCoordinator::OsrCoordinator() = default;

void OsrCoordinator::clear_cache() {
    osr_modules_.clear();
    osr_targets_.clear();
    loop_latches_.clear();
}

bool OsrCoordinator::is_loop_backedge(const Function& fn, const BasicBlock* from_bb, const BasicBlock* to_bb) {
    if (!from_bb || !to_bb) return false;

    std::string fn_name(fn.name());
    auto it = loop_latches_.find(fn_name);
    if (it == loop_latches_.end()) {
        const_cast<Function&>(fn).rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        auto& latches = loop_latches_[fn_name];
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb || !dom.is_reachable(bb)) continue;
            for (const BasicBlock* succ : bb->successors()) {
                if (!succ || !dom.is_reachable(succ)) continue;
                if (dom.dominates(succ, bb)) {
                    latches.insert(bb);
                }
            }
        }
        it = loop_latches_.find(fn_name);
    }

    if (it != loop_latches_.end() && it->second.find(from_bb) != it->second.end()) {
        DominatorTree dom(fn);
        return dom.dominates(to_bb, from_bb);
    }
    return false;
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

        OsrTarget target = analyze_osr_target(const_cast<Function&>(fn), loop_header);
        osr_targets_[cache_key] = std::move(target);
    }

    if (!comp_mod) return false;

    const OsrTarget& target = osr_targets_[cache_key];
    void* osr_entry_addr = comp_mod->get_osr_entry_address(fn.name());
    if (!osr_entry_addr) {
        feedback.record_bailout("OSR entry point not found");
        return false;
    }

    // Pack migration frame with live-in values from the interpreter frame
    OsrMigrationFrame mig_frame;
    mig_frame.loop_header_id = loop_header->id();
    for (size_t i = 0; i < target.live_ins.size() && i < OsrMigrationFrame::kMaxInlineSlots; ++i) {
        Value* v = target.live_ins[i];
        RuntimeValue val = frame.get_value(v);
        mig_frame.add_slot(static_cast<uint32_t>(i), val.as_u64());
    }

    // Set active execution context for potential native deoptimization
    active_interpreter_ = &interp;
    active_fn_ = &fn;
    active_frame_ = &frame;

    bool deopt_occurred = false;
    RuntimeValue deopt_res;

    auto prev_handler = get_deopt_handler();
    register_deopt_handler([&](const DeoptFrame& dframe) -> void* {
        deopt_occurred = true;
        total_native_deopts_++;
        TieringFeedback& fb = TieringRegistry::instance().get_or_create(fn.name());
        fb.record_deoptimization();
        std::vector<RuntimeValue> state_vals = dframe.to_runtime_values();
        deopt_res = interp.resume_with_frame(fn, dframe.resume_id, state_vals, frame);
        return reinterpret_cast<void*>(deopt_res.as_u64());
    });

    struct HandlerScopeGuard {
        DeoptHandlerFn prev;
        Interpreter*& cur_interp;
        const Function*& cur_fn;
        InterpreterFrame*& cur_frame;
        ~HandlerScopeGuard() {
            register_deopt_handler(prev);
            cur_interp = nullptr;
            cur_fn = nullptr;
            cur_frame = nullptr;
        }
    } guard{prev_handler, active_interpreter_, active_fn_, active_frame_};

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

void* OsrCoordinator::handle_native_deopt(const DeoptFrame& deopt_frame) {
    total_native_deopts_++;
    if (active_fn_) {
        TieringFeedback& fb = TieringRegistry::instance().get_or_create(active_fn_->name());
        fb.record_deoptimization();
    }
    if (active_interpreter_ && active_fn_) {
        std::vector<RuntimeValue> state_vals = deopt_frame.to_runtime_values();
        RuntimeValue res;
        if (active_frame_) {
            res = active_interpreter_->resume_with_frame(*active_fn_, deopt_frame.resume_id, state_vals, *active_frame_);
        } else {
            res = active_interpreter_->resume(*active_fn_, deopt_frame.resume_id, state_vals);
        }
        return reinterpret_cast<void*>(res.as_u64());
    }
    return nullptr;
}

} // namespace brass::runtime
