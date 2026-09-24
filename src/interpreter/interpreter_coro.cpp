#include "interpreter_coro.hpp"
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <algorithm>

namespace brass {

namespace {

// Coroutine bodies run only in their lowered form (CoroTransformPass): one
// frame parameter, no coro_suspend. Anything else has no defined resume
// behavior, so it is an error rather than a guess.
const Function& lowered_coro_target(std::string_view callee, const Module* mod) {
    const Function* target_fn = mod ? mod->get_function(callee) : nullptr;
    if (!target_fn) {
        throw InterpreterException("coro_create of undefined function: " + std::string(callee));
    }
    if (!is_lowered_coro_body(*target_fn)) {
        throw InterpreterException("coroutine body " + std::string(callee) +
                                   " has not been lowered by CoroTransformPass");
    }
    return *target_fn;
}

} // namespace

RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod) {
    const Function& target_fn = lowered_coro_target(inst.symbol(), mod);
    CoroFrameLayout layout = compute_coro_frame_layout(target_fn);
    uint32_t slot_count = std::max(layout.slot_count, static_cast<uint32_t>(inst.operand_count()));

    // No generated frame: this interpreter's roots reach the heap through
    // its scope and provider.
    uintptr_t frame_addr = brass_coro_create_at(
        reinterpret_cast<void*>(const_cast<Function*>(&target_fn)),
        slot_count,
        layout.pointer_mask,
        0, 0
    );
    auto* frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    if (!frame_ptr) {
        throw InterpreterException("coro_create: frame allocation failed");
    }
    // Argument i lives in frame slot i (the lowered body loads it from there).
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        frame_ptr->slots[i] = static_cast<uint64_t>(frame.get_value(inst.operand(i)).raw_bits());
    }
    return RuntimeValue::from_ptr(frame_addr);
}

void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame) {
    (void)frame;
    const Function* fn = (inst.parent() ? inst.parent()->parent() : nullptr);
    throw InterpreterException("coro_suspend in " + (fn ? std::string(fn->name()) : std::string("<unknown>")) +
                               ": coroutine bodies must be lowered by CoroTransformPass before execution");
}

RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn
) {
    RuntimeValue coro_val = frame.get_value(inst.operand(0));
    uintptr_t frame_addr = coro_val.as_ptr();
    auto* frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    if (!frame_ptr) {
        throw InterpreterException("coro_resume of a null coroutine frame");
    }
    if (frame_ptr->is_done) {
        return RuntimeValue::from_i64(static_cast<int64_t>(frame_ptr->yielded_val));
    }

    RuntimeValue input_val = (inst.operand_count() > 1 && inst.operand(1))
        ? frame.get_value(inst.operand(1))
        : RuntimeValue::from_i64(0);
    frame_ptr->resume_arg = static_cast<uint64_t>(input_val.raw_bits());

    // The lowered body dispatches on state_id and maintains is_done itself,
    // exactly as brass_coro_resume runs it natively.
    const auto* target_fn = reinterpret_cast<const Function*>(frame_ptr->fn_ptr);
    RuntimeValue yielded_res = exec_fn(*target_fn, {RuntimeValue::from_ptr(frame_addr)});
    frame_ptr->yielded_val = static_cast<uint64_t>(yielded_res.raw_bits());
    return yielded_res;
}

void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame) {
    RuntimeValue coro_val = frame.get_value(inst.operand(0));
    brass_coro_destroy(coro_val.as_ptr());
}

} // namespace brass
