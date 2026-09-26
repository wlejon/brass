#include "interpreter_coro.hpp"
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/gc/native_frames.hpp>
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

RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod,
                                runtime::FunctionDispatchTable* table) {
    const Function& target_fn = lowered_coro_target(inst.symbol(), mod);
    const runtime::CoroBody& body = runtime::coro_body_of(target_fn, table);

    // No generated frame: this interpreter's roots reach the heap through
    // its scope and provider. Its body is the descriptor of `target_fn` in
    // this program, which every tier resumes in the body's best tier.
    uintptr_t frame_addr = runtime::create_coro_frame(body, coro_create_slot_count(inst));
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
    const uint32_t mode = (inst.operand_count() > 2 && inst.operand(2))
        ? static_cast<uint32_t>(frame.get_value(inst.operand(2)).raw_bits())
        : 0u;

    // A body with native code (generated code's frame, or a descriptor whose
    // program compiled the body) runs natively; it roots the frame itself,
    // and this frame's gcref operand is updated by the root walk. A Tier-0
    // body (any module) runs here.
    if (!frame_ptr->fn_ptr) {
        throw InterpreterException("coro_resume: coroutine frame has no body");
    }
    if (runtime::coro_body_native_entry(frame_ptr)) {
        return RuntimeValue::from_bits(inst.type(), brass_coro_resume_with(frame_addr, input_bits, mode));
    }
    const Function* body = runtime::mir_coro_body(frame_ptr);
    if (!body) {
        throw InterpreterException("coro_resume: the module of the frame's coroutine body was destroyed");
    }
    frame_ptr->resume_arg = input_bits;
    frame_ptr->resume_mode = mode;
    runtime::coro_resume_value_barrier(frame_addr, input_bits);

    // The lowered body dispatches on state_id and maintains is_done itself,
    // exactly as brass_coro_resume runs it natively. It may collect and move
    // the frame: the argument is a gcref (the body's frame parameter is
    // one), and the operand here is a root, so it is re-read afterwards.
    RuntimeValue yielded_res;
    uintptr_t running_root = frame_addr;
    try {
        ThreadRootsScope running_slot([](void* ctx, std::vector<uintptr_t*>& roots) {
            roots.push_back(static_cast<uintptr_t*>(ctx));
        }, &running_root);
        runtime::RunningCoroScope running(&running_root);
        yielded_res = exec_fn(*body, {RuntimeValue::from_gcref(frame_addr)});
    } catch (...) {
        // A body that throws is finished (brass_coro_resume does the same).
        runtime::finish_thrown_coro_frame(reinterpret_cast<runtime::BrassCoroFrame*>(
            static_cast<uintptr_t>(frame.get_value(inst.operand(0)).raw_bits())));
        throw;
    }
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
