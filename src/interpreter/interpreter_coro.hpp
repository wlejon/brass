#pragma once

#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/coroutine.hpp>
#include <functional>

namespace brass {

// Coroutine opcodes over lowered coroutine bodies (see CoroTransformPass):
// the body takes the frame, dispatches on its state_id and keeps is_done.
// Handles are gcrefs: the frame is a heap object a collection may move.
RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod);
// Runs a lowered body of `mod` in this interpreter; a frame generated code
// created (its fn_ptr a native address) runs through brass_coro_resume.
RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const Module* mod,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn
);
void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame);
// Always throws: a coro_suspend only exists in an unlowered body.
[[noreturn]] void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame);

} // namespace brass
