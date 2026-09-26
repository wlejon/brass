#pragma once

#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/coroutine.hpp>
#include <functional>

namespace brass {

// Coroutine opcodes over lowered coroutine bodies (see CoroTransformPass):
// the body takes the frame, dispatches on its state_id and keeps is_done.
// Handles are gcrefs: the frame is a heap object a collection may move.
// The frame's body descriptor belongs to `table`, the interpreter's program.
RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod,
                                runtime::FunctionDispatchTable* table);
// Runs a frame's Tier-0 body (a descriptor's MIR, any module) through
// `exec_fn`, which runs it in its own module; a body with native code (a
// fixed-code frame, or a descriptor whose program has compiled it) runs
// through brass_coro_resume_with.
RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn
);
void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame);
// Always throws: a coro_suspend only exists in an unlowered body.
[[noreturn]] void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame);

} // namespace brass
