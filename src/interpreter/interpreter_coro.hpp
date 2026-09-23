#pragma once

#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/coroutine.hpp>
#include <functional>

namespace brass {

// Coroutine opcodes over lowered coroutine bodies (see CoroTransformPass):
// the body takes the frame, dispatches on its state_id and keeps is_done.
RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod);
RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn
);
void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame);
// Always throws: a coro_suspend only exists in an unlowered body.
[[noreturn]] void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame);

} // namespace brass
