#include "verifier_coro.hpp"
#include <brass/mir/opcodes.hpp>

namespace brass {

bool verify_coro_instruction(
    const Instruction* inst,
    const BasicBlock* bb,
    const std::string& inst_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    (void)bb;
    if (!inst) return false;

    switch (inst->opcode()) {
        case Opcode::coro_create: {
            if (inst->symbol().empty()) {
                report_error(inst_prefix + "coro_create instruction must specify a callee function symbol.");
            }
            if (!inst->produces_value() || !inst->result()) {
                report_error(inst_prefix + "coro_create instruction must produce a result value.");
            } else if (inst->type() != Type::gcref() && inst->type() != Type::ptr()) {
                report_error(inst_prefix + "coro_create instruction result type must be gcref or ptr.");
            }
            return true;
        }

        case Opcode::coro_suspend: {
            if (inst->operand_count() == 0) {
                report_error(inst_prefix + "coro_suspend instruction requires at least 1 operand (yielded value).");
            } else {
                const Value* op = inst->operand(0);
                if (!op) {
                    report_error(inst_prefix + "coro_suspend yielded operand is null.");
                } else if (op->type().is_void()) {
                    report_error(inst_prefix + "coro_suspend yielded operand cannot be void.");
                }
            }
            return true;
        }

        case Opcode::coro_resume: {
            if (inst->operand_count() == 0) {
                report_error(inst_prefix + "coro_resume instruction requires at least 1 operand (coro frame).");
            } else {
                const Value* op0 = inst->operand(0);
                if (!op0) {
                    report_error(inst_prefix + "coro_resume frame operand is null.");
                } else if (op0->type() != Type::gcref() && op0->type() != Type::ptr() && op0->type() != Type::i64()) {
                    report_error(inst_prefix + "coro_resume frame operand must be gcref, ptr, or dynamic.");
                }
                if (inst->operand_count() > 1) {
                    const Value* op1 = inst->operand(1);
                    if (op1 && op1->type().is_void()) {
                        report_error(inst_prefix + "coro_resume input argument cannot be void.");
                    }
                }
            }
            return true;
        }

        case Opcode::coro_destroy: {
            if (inst->operand_count() != 1) {
                report_error(inst_prefix + "coro_destroy instruction requires exactly 1 operand.");
            } else {
                const Value* op0 = inst->operand(0);
                if (!op0) {
                    report_error(inst_prefix + "coro_destroy frame operand is null.");
                } else if (op0->type() != Type::gcref() && op0->type() != Type::ptr() && op0->type() != Type::i64()) {
                    report_error(inst_prefix + "coro_destroy frame operand must be gcref, ptr, or dynamic.");
                }
            }
            if (inst->produces_value()) {
                report_error(inst_prefix + "coro_destroy instruction must not produce a value.");
            }
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass
