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

void FastInterpreter::handle_throw(FastFrame& frame, uint8_t reg, const uint32_t*& pc, const uint32_t* code_base) {
    const BytecodeFunction& fn = *frame.bfn;
    Type reg_type = (reg < fn.register_types.size()) ? fn.register_types[reg] : Type::i64();
    if (reg_type.is_gcref()) {
        current_exception_ = RuntimeValue::from_gcref(frame.registers[reg]);
    } else {
        current_exception_ = RuntimeValue::from_bits(reg_type, frame.registers[reg]);
    }

    uint32_t cur_pc = static_cast<uint32_t>(pc - code_base);
    const ExceptionEntry* ee = find_exception_entry(fn, cur_pc);
    if (ee) {
        pc = code_base + ee->handler_pc;
        return;
    }

    throw InterpreterThrownException(current_exception_);
}

void FastInterpreter::handle_invoke(FastFrame& frame, const CallSiteInfo& cs, const uint32_t*& pc, const uint32_t* code_base) {
    const BytecodeFunction& fn = *frame.bfn;
    bool threw = false;

    try {
        execute_call(frame, cs, BytecodeOp::call);
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

void FastInterpreter::handle_resume(FastFrame& frame, uint8_t reg) {
    if (reg != 255 && reg < frame.num_registers) {
        const BytecodeFunction& fn = *frame.bfn;
        Type reg_type = (reg < fn.register_types.size()) ? fn.register_types[reg] : Type::i64();
        if (reg_type.is_gcref()) {
            current_exception_ = RuntimeValue::from_gcref(frame.registers[reg]);
        } else {
            current_exception_ = RuntimeValue::from_bits(reg_type, frame.registers[reg]);
        }
    }
    throw InterpreterThrownException(current_exception_);
}

} // namespace brass
