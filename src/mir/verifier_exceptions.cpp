#include "verifier_exceptions.hpp"
#include <brass/mir/opcodes.hpp>
#include <brass/mir/function.hpp>

namespace brass {

bool verify_exception_instruction(
    const Instruction* inst,
    const BasicBlock* bb,
    const std::string& inst_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    if (!inst) return false;

    switch (inst->opcode()) {
        case Opcode::throw_: {
            if (inst->operand_count() != 1) {
                report_error(inst_prefix + "throw instruction requires exactly 1 operand.");
            } else {
                const Value* op = inst->operand(0);
                if (!op) {
                    report_error(inst_prefix + "throw operand is null.");
                } else if (op->type().is_void()) {
                    report_error(inst_prefix + "throw operand cannot be of void type.");
                }
            }
            if (bb && inst != bb->terminator()) {
                report_error(inst_prefix + "throw instruction must be the terminator of the block.");
            }
            return true;
        }

        case Opcode::resume: {
            if (inst->operand_count() > 1) {
                report_error(inst_prefix + "resume instruction accepts at most 1 operand.");
            } else if (inst->operand_count() == 1) {
                const Value* op = inst->operand(0);
                if (op && op->type().is_void()) {
                    report_error(inst_prefix + "resume operand cannot be of void type.");
                }
            }
            if (bb && inst != bb->terminator()) {
                report_error(inst_prefix + "resume instruction must be the terminator of the block.");
            }
            return true;
        }

        case Opcode::landing_pad: {
            if (inst->operand_count() != 0) {
                report_error(inst_prefix + "landing_pad instruction cannot have operands.");
            }
            if (!inst->produces_value() || !inst->result()) {
                report_error(inst_prefix + "landing_pad instruction must produce a non-void result value.");
            } else if (inst->type().is_void()) {
                report_error(inst_prefix + "landing_pad instruction result type cannot be void.");
            }
            if (bb && bb->parent() && bb == bb->parent()->entry_block()) {
                report_error(inst_prefix + "landing_pad instruction cannot be placed in the entry block.");
            }
            if (bb && inst != bb->head()) {
                report_error(inst_prefix + "landing_pad instruction must be the first instruction in its block.");
            }
            return true;
        }

        case Opcode::invoke: {
            if (inst->symbol().empty()) {
                report_error(inst_prefix + "invoke instruction must specify a callee symbol.");
            }
            if (!inst->normal_target().block) {
                report_error(inst_prefix + "invoke instruction normal branch target block is null.");
            } else {
                const auto* n_blk = inst->normal_target().block;
                if (inst->normal_target().args.size() != n_blk->param_count()) {
                    report_error(inst_prefix + "invoke normal target arguments count (" +
                                 std::to_string(inst->normal_target().args.size()) +
                                 ") does not match target block parameter count (" +
                                 std::to_string(n_blk->param_count()) + ").");
                }
            }
            if (!inst->unwind_target().block) {
                report_error(inst_prefix + "invoke instruction unwind branch target block is null.");
            } else {
                const auto* u_blk = inst->unwind_target().block;
                if (inst->unwind_target().args.size() != u_blk->param_count()) {
                    report_error(inst_prefix + "invoke unwind target arguments count (" +
                                 std::to_string(inst->unwind_target().args.size()) +
                                 ") does not match target block parameter count (" +
                                 std::to_string(u_blk->param_count()) + ").");
                }
                const Instruction* first_inst = u_blk->head();
                if (!first_inst || first_inst->opcode() != Opcode::landing_pad) {
                    report_error(inst_prefix + "invoke unwind target block must begin with a landing_pad instruction.");
                }
            }
            if (bb && inst != bb->terminator()) {
                report_error(inst_prefix + "invoke instruction must be the terminator of the block.");
            }
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass
