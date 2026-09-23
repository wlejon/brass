#include "fast_interpreter_impl.hpp"
#include <brass/runtime/exception.hpp>
#include <limits>

namespace brass {

namespace {

const ExceptionEntry* find_exception_entry(const BytecodeFunction& fn, uint32_t pc) {
    const ExceptionEntry* best = nullptr;
    uint32_t best_span = std::numeric_limits<uint32_t>::max();

    for (const auto& ee : fn.exception_table) {
        bool in_range = false;
        if (ee.start_pc == ee.end_pc) {
            in_range = (pc == ee.start_pc);
        } else {
            in_range = (pc >= ee.start_pc && pc <= ee.end_pc);
        }

        if (in_range) {
            uint32_t span = (ee.end_pc >= ee.start_pc) ? (ee.end_pc - ee.start_pc) : 0;
            if (span <= best_span) {
                best_span = span;
                best = &ee;
            }
        }
    }
    return best;
}

} // namespace

void FastInterpreter::handle_throw(FastFrame& frame, uint32_t reg, const BytecodeWord*& pc, const BytecodeWord* code_base) {
    const BytecodeFunction& fn = *frame.bfn;
    current_exception_ = fast_reg_value(frame, reg);

    uint32_t cur_pc = static_cast<uint32_t>(pc - code_base);
    const ExceptionEntry* ee = find_exception_entry(fn, cur_pc);
    if (ee) {
        pc = code_base + ee->handler_pc;
        return;
    }

    throw InterpreterThrownException(current_exception_);
}

void FastInterpreter::handle_invoke(FastFrame& frame, uint32_t cs_idx, const BytecodeWord*& pc, const BytecodeWord* code_base) {
    const BytecodeFunction& fn = *frame.bfn;
    bool threw = false;

    try {
        execute_call(frame, cs_idx);
    } catch (const InterpreterThrownException& ex) {
        threw = true;
        current_exception_ = ex.value();
    } catch (const runtime::BrassException& be) {
        threw = true;
        current_exception_ = RuntimeValue::from_bits(Type::i64(), be.value().raw());
    }

    uint32_t cur_pc = static_cast<uint32_t>(pc - code_base);
    if (threw) {
        const ExceptionEntry* ee = find_exception_entry(fn, cur_pc);
        if (ee) {
            pc = code_base + ee->handler_pc;
            return;
        }
        throw InterpreterThrownException(current_exception_);
    }

    pc++;
}

void FastInterpreter::handle_resume(FastFrame& frame, uint32_t reg) {
    if (reg != kNoReg && reg < frame.num_registers) {
        current_exception_ = fast_reg_value(frame, reg);
    }
    throw InterpreterThrownException(current_exception_);
}

} // namespace brass
