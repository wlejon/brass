#include "interpreter_coro.hpp"
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <algorithm>
#include <cstring>

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

// A frame's fn_ptr is the Function* of the lowered body when an interpreter
// created it, a native code address when generated code did. Only a body of
// the current module is recognized; anything else runs natively.
const Function* interpreted_coro_body(const void* fn_ptr, const Module* mod) {
    if (!fn_ptr || !mod) return nullptr;
    for (const Function* f : mod->functions()) {
        if (static_cast<const void*>(f) == fn_ptr) return is_lowered_coro_body(*f) ? f : nullptr;
    }
    return nullptr;
}

// The coroutine handle operand. The verifier also accepts ptr and i64
// handles; the frame is still a movable heap object, so the operand's value
// becomes a gcref in this frame, where the root walk reports and updates it.
uintptr_t coro_handle(const Instruction& inst, InterpreterFrame& frame, const char* op, bool allow_null) {
    RuntimeValue v = frame.get_value(inst.operand(0));
    if (v.is_vector()) {
        throw InterpreterException(std::string(op) + ": coroutine handle is a vector value");
    }
    const uintptr_t handle = static_cast<uintptr_t>(v.raw_bits());
    if (handle == 0) {
        if (allow_null) return 0;
        throw InterpreterException(std::string(op) + " of a null coroutine frame");
    }
    if (!v.is_gcref()) {
        frame.set_value(inst.operand(0), RuntimeValue::from_gcref(handle));
    }
    return handle;
}

} // namespace

RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod) {
    const Function& target_fn = lowered_coro_target(inst.symbol(), mod);
    CoroFrameLayout layout = compute_coro_frame_layout(target_fn);
    uint32_t slot_count = std::max(layout.slot_count, coro_create_slot_count(inst));

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
    // Arguments fill consecutive slots, a vector spanning coro_slot_count
    // of them (the lowered body loads each from there).
    uint32_t slot = 0;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        const Type t = inst.operand(i)->type();
        const RuntimeValue v = frame.get_value(inst.operand(i));
        if (t.is_vector()) {
            std::memcpy(&frame_ptr->slots[slot], v.vec_bytes(), t.size_in_bytes());
        } else {
            frame_ptr->slots[slot] = static_cast<uint64_t>(v.raw_bits());
        }
        slot += coro_slot_count(t);
    }
    // The frame is a heap object that a collection may move: a gcref, so
    // this frame's roots report it (the builder types the result gcref).
    return RuntimeValue::from_gcref(frame_addr);
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
    const Module* mod,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn
) {
    uintptr_t frame_addr = coro_handle(inst, frame, "coro_resume", /*allow_null=*/false);
    auto* frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    if (frame_ptr->is_done) {
        return RuntimeValue::from_bits(inst.type(), frame_ptr->yielded_val);
    }

    RuntimeValue input_val = (inst.operand_count() > 1 && inst.operand(1))
        ? frame.get_value(inst.operand(1))
        : RuntimeValue::from_i64(0);
    const uint64_t input_bits = static_cast<uint64_t>(input_val.raw_bits());

    const Function* body = interpreted_coro_body(frame_ptr->fn_ptr, mod);
    if (!body) {
        if (!frame_ptr->fn_ptr) {
            throw InterpreterException("coro_resume: coroutine frame has no body");
        }
        // Generated code's frame: run it natively. It roots the frame
        // itself; this frame's gcref operand is updated by the root walk.
        return RuntimeValue::from_bits(inst.type(), brass_coro_resume(frame_addr, input_bits));
    }
    frame_ptr->resume_arg = input_bits;

    // The lowered body dispatches on state_id and maintains is_done itself,
    // exactly as brass_coro_resume runs it natively. It may collect and move
    // the frame: the argument is a gcref (the body's frame parameter is
    // one), and the operand here is a root, so it is re-read afterwards.
    RuntimeValue yielded_res = exec_fn(*body, {RuntimeValue::from_gcref(frame_addr)});
    frame_addr = static_cast<uintptr_t>(frame.get_value(inst.operand(0)).raw_bits());
    frame_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    frame_ptr->yielded_val = static_cast<uint64_t>(yielded_res.raw_bits());
    if (frame_ptr->is_done) {
        runtime::unregister_active_coro_frame(frame_ptr);
    }
    return yielded_res;
}

void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame) {
    // Any frame, interpreted or generated: done, and no longer a root.
    brass_coro_destroy(coro_handle(inst, frame, "coro_destroy", /*allow_null=*/true));
}

} // namespace brass
