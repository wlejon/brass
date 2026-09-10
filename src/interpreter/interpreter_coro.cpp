#include "interpreter_coro.hpp"
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

namespace brass {

RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod) {
    std::string_view callee = inst.symbol();
    const Function* target_fn = mod ? mod->get_function(callee) : nullptr;

    uint32_t slot_count = 16;
    uint64_t ptr_mask = 0;
    uintptr_t frame_addr = brass_coro_create(
        reinterpret_cast<void*>(const_cast<Function*>(target_fn)),
        slot_count,
        ptr_mask
    );

    auto* frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    if (frame_ptr) {
        for (size_t i = 0; i < inst.operand_count() && i < 16; ++i) {
            RuntimeValue arg = frame.get_value(inst.operand(i));
            frame_ptr->slots[i] = static_cast<uint64_t>(arg.raw_bits());
        }
    }

    return RuntimeValue::from_ptr(frame_addr);
}

void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame, BasicBlock* cur_bb) {
    RuntimeValue yield_val = (inst.operand_count() > 0 && inst.operand(0))
        ? frame.get_value(inst.operand(0))
        : RuntimeValue::from_i64(0);
    throw InterpreterSuspendException(yield_val, inst.resume_id(), inst.next(), cur_bb);
}

RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const Module* mod,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn,
    const std::function<RuntimeValue(const Function&, BasicBlock*, const std::vector<RuntimeValue>&)>& exec_bb
) {
    (void)mod;
    RuntimeValue coro_val = frame.get_value(inst.operand(0));
    uintptr_t frame_addr = coro_val.as_ptr();
    auto* frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);

    if (!frame_ptr || frame_ptr->is_done) {
        return RuntimeValue::from_i64(frame_ptr ? static_cast<int64_t>(frame_ptr->yielded_val) : 0);
    }

    RuntimeValue input_val = (inst.operand_count() > 1 && inst.operand(1))
        ? frame.get_value(inst.operand(1))
        : RuntimeValue::from_i64(0);
    frame_ptr->resume_arg = static_cast<uint64_t>(input_val.raw_bits());

    const Function* target_fn = reinterpret_cast<const Function*>(frame_ptr->fn_ptr);
    RuntimeValue yielded_res = RuntimeValue::from_i64(0);

    if (target_fn) {
        try {
            if (frame_ptr->state_id == 0) {
                std::vector<RuntimeValue> fn_args;
                if (target_fn->param_count() == 1 && target_fn->param_type(0).is_gcref()) {
                    fn_args.push_back(RuntimeValue::from_ptr(frame_addr));
                } else {
                    for (size_t i = 0; i < target_fn->param_count(); ++i) {
                        fn_args.push_back(RuntimeValue::from_i64(static_cast<int64_t>(frame_ptr->slots[i])));
                    }
                }
                yielded_res = exec_fn(*target_fn, fn_args);
                frame_ptr->is_done = 1;
                frame_ptr->yielded_val = static_cast<uint64_t>(yielded_res.raw_bits());
            } else {
                BasicBlock* resume_bb = target_fn->get_resume_target(frame_ptr->state_id);
                if (resume_bb) {
                    std::vector<RuntimeValue> state_args;
                    if (resume_bb->param_count() > 0) {
                        state_args.push_back(input_val);
                    }
                    yielded_res = exec_bb(*target_fn, resume_bb, state_args);
                    frame_ptr->is_done = 1;
                    frame_ptr->yielded_val = static_cast<uint64_t>(yielded_res.raw_bits());
                } else if (target_fn->param_count() == 1 && target_fn->param_type(0).is_gcref()) {
                    yielded_res = exec_fn(*target_fn, {RuntimeValue::from_ptr(frame_addr)});
                    frame_ptr->yielded_val = static_cast<uint64_t>(yielded_res.raw_bits());
                }
            }
        } catch (const InterpreterSuspendException& e) {
            frame_ptr->state_id = e.state_id;
            frame_ptr->yielded_val = static_cast<uint64_t>(e.yielded_val.raw_bits());
            yielded_res = e.yielded_val;
        }
    }

    return yielded_res;
}

void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame) {
    RuntimeValue coro_val = frame.get_value(inst.operand(0));
    brass_coro_destroy(coro_val.as_ptr());
}

} // namespace brass
