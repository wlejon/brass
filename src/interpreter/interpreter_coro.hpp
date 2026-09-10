#pragma once

#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/coroutine.hpp>
#include <exception>
#include <functional>

namespace brass {

struct InterpreterSuspendException : public std::exception {
    RuntimeValue yielded_val;
    uint32_t state_id = 0;
    Instruction* next_inst = nullptr;
    BasicBlock* block = nullptr;

    InterpreterSuspendException(RuntimeValue val, uint32_t sid, Instruction* next, BasicBlock* bb)
        : yielded_val(val), state_id(sid), next_inst(next), block(bb) {}

    const char* what() const noexcept override {
        return "Interpreter coroutine suspended";
    }
};

RuntimeValue interp_coro_create(const Instruction& inst, InterpreterFrame& frame, const Module* mod);
RuntimeValue interp_coro_resume(
    const Instruction& inst,
    InterpreterFrame& frame,
    const Module* mod,
    const std::function<RuntimeValue(const Function&, const std::vector<RuntimeValue>&)>& exec_fn,
    const std::function<RuntimeValue(const Function&, BasicBlock*, const std::vector<RuntimeValue>&)>& exec_bb
);
void interp_coro_destroy(const Instruction& inst, InterpreterFrame& frame);
void interp_coro_suspend(const Instruction& inst, InterpreterFrame& frame, BasicBlock* cur_bb);

} // namespace brass
