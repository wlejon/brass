#include <brass/mir/verifier.hpp>
#include "verifier_vec.hpp"
#include "verifier_exceptions.hpp"
#include "verifier_coro.hpp"
#include "verifier_dom.hpp"
#include "verifier_gc.hpp"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <sstream>
#include <vector>

namespace brass {

namespace {

// Whether some guard of `fn` without an exit-stub function resumes in a
// resume_table block of its id.
bool guards_resume_in_table(const Function& fn) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::guard && !fn.guard_exit_stub(*inst) &&
                fn.get_resume_target(inst->resume_id())) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

void Verifier::report_error(std::string msg) {
    has_error_ = true;
    if (diag_) {
        diag_->error(std::move(msg));
    }
}

void Verifier::report_warning(std::string msg) {
    if (diag_) {
        diag_->warning(std::move(msg));
    }
}

bool Verifier::verify_module(const Module& mod) {
    has_error_ = false;

    std::unordered_set<std::string_view> fn_names;
    for (const Function* fn : mod.functions()) {
        if (!fn) {
            report_error("Module contains a null function pointer.");
            continue;
        }
        if (!fn_names.insert(fn->name()).second) {
            report_error("Duplicate function name in module: '" + std::string(fn->name()) + "'");
        }
        verify_function(*fn);
    }

    return !has_error_;
}

bool Verifier::verify_function(const Function& fn) {
    const std::string fn_prefix = "Function '" + std::string(fn.name()) + "': ";

    if (fn.blocks().empty()) {
        report_error(fn_prefix + "Function has no basic blocks.");
        return false;
    }

    const BasicBlock* entry = fn.entry_block();
    if (!entry) {
        report_error(fn_prefix + "Function has no entry basic block.");
        return false;
    }

    // Check entry block predecessors
    if (!entry->predecessors().empty()) {
        report_error(fn_prefix + "Entry block '" + std::string(entry->name()) +
                     "' must have no predecessors, but has " +
                     std::to_string(entry->predecessors().size()) + ".");
    }

    // Check entry block parameters match function signature
    if (entry->param_count() != fn.param_count()) {
        report_error(fn_prefix + "Entry block parameter count (" +
                     std::to_string(entry->param_count()) + ") does not match function parameter count (" +
                     std::to_string(fn.param_count()) + ").");
    } else {
        for (size_t i = 0; i < fn.param_count(); ++i) {
            const Value* param = entry->param(i);
            if (!param) {
                report_error(fn_prefix + "Entry block parameter " + std::to_string(i) + " is null.");
            } else if (param->type() != fn.param_type(i)) {
                report_error(fn_prefix + "Entry block parameter " + std::to_string(i) + " type (" +
                             std::string(param->type().name()) + ") does not match signature (" +
                             std::string(fn.param_type(i).name()) + ").");
            }
        }
    }

    // Index blocks and instruction positions
    std::unordered_set<const BasicBlock*> block_set;
    std::unordered_map<const Instruction*, size_t> inst_index;

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) {
            report_error(fn_prefix + "Contains null basic block pointer.");
            continue;
        }
        if (!block_set.insert(bb).second) {
            report_error(fn_prefix + "Duplicate basic block in block list.");
        }

        size_t idx = 0;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst) {
                inst_index[inst] = idx++;
            }
        }
    }

    // Compute Dominance
    DominanceCalculator dom(fn);

    // Verify each block and instruction
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        const std::string bb_prefix = fn_prefix + "Block '" + std::string(bb->name()) + "': ";

        if (bb->is_empty()) {
            report_error(bb_prefix + "Basic block is empty (missing terminator).");
            continue;
        }

        // Check terminator rules
        const Instruction* term = bb->terminator();
        if (!term) {
            report_error(bb_prefix + "Basic block does not end with a valid terminator instruction.");
        }

        // Ensure no non-tail instruction is a terminator
        for (const Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst != bb->tail() && inst->is_terminator()) {
                report_error(bb_prefix + "Terminator instruction '" +
                             std::string(opcode_name(inst->opcode())) + "' is not at the end of the block.");
            }
        }

        // Verify block parameters
        for (size_t i = 0; i < bb->param_count(); ++i) {
            const Value* param = bb->param(i);
            if (!param) {
                report_error(bb_prefix + "Block parameter " + std::to_string(i) + " is null.");
            } else if (param->type().is_void()) {
                report_error(bb_prefix + "Block parameter " + std::to_string(i) + " cannot be void.");
            } else if (!param->is_block_param() || param->defining_block() != bb || param->param_index() != i) {
                report_error(bb_prefix + "Block parameter " + std::to_string(i) +
                             " does not record this block and index as its definition.");
            }
        }

        // Verify each instruction in block
        for (const Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            const std::string inst_prefix = bb_prefix + "Instruction '" +
                                            std::string(opcode_name(inst->opcode())) + "': ";

            // Check operands & SSA Dominance
            auto check_value_dom = [&](const Value* val, const std::string& desc) {
                if (!val) {
                    report_error(inst_prefix + desc + " operand is null.");
                    return;
                }
                if (val->is_block_param()) {
                    const BasicBlock* def_bb = val->defining_block();
                    if (!def_bb) {
                        report_error(inst_prefix + desc + " block parameter has null defining block.");
                    } else if (val->param_index() >= def_bb->param_count() ||
                               def_bb->param(val->param_index()) != val) {
                        // A pass dropped the parameter from its block but
                        // left this use behind.
                        report_error(inst_prefix + desc + " uses a parameter no longer on block '" +
                                     std::string(def_bb->name()) + "'.");
                    } else if (dom.is_reachable(bb) && !dom.dominates(def_bb, bb)) {
                        report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                     " block parameter of block '" + std::string(def_bb->name()) +
                                     "' does not dominate use in block '" + std::string(bb->name()) + "'.");
                    }
                } else if (val->is_instruction()) {
                    const Instruction* def_inst = val->defining_instruction();
                    if (!def_inst) {
                        report_error(inst_prefix + desc + " instruction result has null defining instruction.");
                    } else {
                        const BasicBlock* def_bb = def_inst->parent();
                        if (!def_bb) {
                            report_error(inst_prefix + desc + " defining instruction has null parent block.");
                        } else if (def_bb == bb) {
                            auto it_def = inst_index.find(def_inst);
                            auto it_use = inst_index.find(inst);
                            if (it_def != inst_index.end() && it_use != inst_index.end()) {
                                if (it_def->second >= it_use->second) {
                                    report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                                 " is used before or at its definition in the same block (def_idx=" +
                                                 std::to_string(it_def->second) + " use_idx=" + std::to_string(it_use->second) +
                                                 " def_op=" + std::to_string(static_cast<int>(def_inst->opcode())) + ").");
                                }
                            }
                        } else if (dom.is_reachable(bb) && !dom.dominates(def_bb, bb)) {
                            report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                         " defined in block '" + std::string(def_bb->name()) +
                                         "' does not dominate use in block '" + std::string(bb->name()) + "'.");
                        }
                    }
                }
            };

            // 1. Check general operands dominance
            for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                check_value_dom(inst->operand(op_i), "operand " + std::to_string(op_i));
            }

            // 2. Check state map dominance
            for (size_t sm_i = 0; sm_i < inst->state_map().size(); ++sm_i) {
                check_value_dom(inst->state_map()[sm_i], "state map operand " + std::to_string(sm_i));
            }

            // 3. Opcode-specific type & semantic rules
            Opcode op = inst->opcode();
            switch (op) {
                case Opcode::iconst_i32:
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::iconst_i64:
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::fconst_f64:
                    if (inst->type() != Type::f64() && inst->type() != Type::f32()) {
                        report_error(inst_prefix + "Result type must be f64 or f32, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::patchable_const_i32:
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Patchable const requires non-empty symbol.");
                    }
                    break;
                case Opcode::patchable_const_i64:
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Patchable const requires non-empty symbol.");
                    }
                    break;

                case Opcode::sext_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "Requires 1 i32 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::zext_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) ||
                        (inst->operand(0)->type() != Type::i32() && inst->operand(0)->type() != Type::i8())) {
                        report_error(inst_prefix + "Requires 1 i32 or i8 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::trunc_i32:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i64()) {
                        report_error(inst_prefix + "Requires 1 i64 operand.");
                    }
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32.");
                    }
                    break;
                case Opcode::trunc_i8:
                    if (inst->operand_count() != 1 || !inst->operand(0) || !inst->operand(0)->type().is_integer()) {
                        report_error(inst_prefix + "Requires 1 integer operand.");
                    }
                    if (inst->type() != Type::i8()) {
                        report_error(inst_prefix + "Result type must be i8.");
                    }
                    break;
                case Opcode::fptosi_i32:
                case Opcode::fptosi_i64:
                case Opcode::fptosi_i32_f32:
                case Opcode::fptosi_i64_f32:
                case Opcode::sitofp_f64_i32:
                case Opcode::sitofp_f64_i64:
                case Opcode::sitofp_f32_i32:
                case Opcode::sitofp_f32_i64:
                case Opcode::fptrunc_f32_f64:
                case Opcode::fpext_f64_f32: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Conversion requires 1 operand.");
                        break;
                    }
                    Type in_t = inst->operand(0)->type();
                    Type out_t = inst->type();
                    bool ok = false;
                    switch (inst->opcode()) {
                        case Opcode::fptosi_i32: ok = (in_t == Type::f64() && out_t == Type::i32()); break;
                        case Opcode::fptosi_i64: ok = (in_t == Type::f64() && out_t == Type::i64()); break;
                        case Opcode::fptosi_i32_f32: ok = (in_t == Type::f32() && out_t == Type::i32()); break;
                        case Opcode::fptosi_i64_f32: ok = (in_t == Type::f32() && out_t == Type::i64()); break;
                        case Opcode::sitofp_f64_i32: ok = (in_t == Type::i32() && out_t == Type::f64()); break;
                        case Opcode::sitofp_f64_i64: ok = (in_t == Type::i64() && out_t == Type::f64()); break;
                        case Opcode::sitofp_f32_i32: ok = (in_t == Type::i32() && out_t == Type::f32()); break;
                        case Opcode::sitofp_f32_i64: ok = (in_t == Type::i64() && out_t == Type::f32()); break;
                        case Opcode::fptrunc_f32_f64: ok = (in_t == Type::f64() && out_t == Type::f32()); break;
                        case Opcode::fpext_f64_f32: ok = (in_t == Type::f32() && out_t == Type::f64()); break;
                        default: break;
                    }
                    if (!ok) report_error(inst_prefix + "Invalid operand or result type for conversion.");
                    break;
                }
                case Opcode::bitcast_i64_f64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::f64()) {
                        report_error(inst_prefix + "Requires 1 f64 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::bitcast_f64_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i64()) {
                        report_error(inst_prefix + "Requires 1 i64 operand.");
                    }
                    if (inst->type() != Type::f64()) {
                        report_error(inst_prefix + "Result type must be f64.");
                    }
                    break;

                case Opcode::add:
                case Opcode::sub: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0.is_pointer_or_gcref() && t1.is_integer()) {
                            if (inst->type() != t0) {
                                report_error(inst_prefix + "Pointer arithmetic result type must match base pointer type.");
                            }
                        } else if (op == Opcode::sub && t0.is_pointer_or_gcref() && t0 == t1) {
                            if (inst->type() != Type::i64()) {
                                report_error(inst_prefix + "Pointer difference result type must be i64.");
                            }
                        } else if (t0 != t1) {
                            report_error(inst_prefix + "Operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operands must be numeric or pointer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t0.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::mul:
                case Opcode::sdiv:
                case Opcode::udiv:
                case Opcode::smod:
                case Opcode::umod: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0 != t1) {
                            report_error(inst_prefix + "Operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operands must be numeric.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t0.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::fma_f32: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Requires 3 operands.");
                    } else if (inst->operand(0)->type() != Type::f32() ||
                               inst->operand(1)->type() != Type::f32() ||
                               inst->operand(2)->type() != Type::f32() ||
                               inst->type() != Type::f32()) {
                        report_error(inst_prefix + "fma_f32 operands and result must be f32.");
                    }
                    break;
                }

                case Opcode::fma_f64: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Requires 3 operands.");
                    } else if (inst->operand(0)->type() != Type::f64() ||
                               inst->operand(1)->type() != Type::f64() ||
                               inst->operand(2)->type() != Type::f64() ||
                               inst->type() != Type::f64()) {
                        report_error(inst_prefix + "fma_f64 operands and result must be f64.");
                    }
                    break;
                }

                case Opcode::sqrt_f32:
                case Opcode::floor_f32:
                case Opcode::ceil_f32:
                case Opcode::round_f32:
                case Opcode::fabs_f32:
                case Opcode::sqrt_f64:
                case Opcode::floor_f64:
                case Opcode::ceil_f64:
                case Opcode::round_f64:
                case Opcode::fabs_f64: {
                    Type expected = (inst->opcode() == Opcode::sqrt_f32 || inst->opcode() == Opcode::floor_f32 ||
                                     inst->opcode() == Opcode::ceil_f32 || inst->opcode() == Opcode::round_f32 ||
                                     inst->opcode() == Opcode::fabs_f32) ? Type::f32() : Type::f64();
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != expected || inst->type() != expected) {
                        report_error(inst_prefix + "Unary float op requires 1 matching float operand and result.");
                    }
                    break;
                }

                case Opcode::fmin_f32:
                case Opcode::fmax_f32:
                case Opcode::fmin_f64:
                case Opcode::fmax_f64: {
                    Type expected = (inst->opcode() == Opcode::fmin_f32 || inst->opcode() == Opcode::fmax_f32) ? Type::f32() : Type::f64();
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1) ||
                        inst->operand(0)->type() != expected || inst->operand(1)->type() != expected || inst->type() != expected) {
                        report_error(inst_prefix + "Binary float op requires 2 matching float operands and result.");
                    }
                    break;
                }

                case Opcode::neg: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Requires 1 operand.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operand must be numeric.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match operand type.");
                        }
                    }
                    break;
                }

                case Opcode::and_:
                case Opcode::or_:
                case Opcode::xor_:
                case Opcode::shl:
                case Opcode::lshr:
                case Opcode::ashr: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (!t0.is_integer() || !t1.is_integer()) {
                            report_error(inst_prefix + "Bitwise/shift operands must be integer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match first operand type.");
                        }
                    }
                    break;
                }

                case Opcode::not_:
                case Opcode::clz:
                case Opcode::ctz:
                case Opcode::popcnt: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Requires 1 operand.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        if (!t0.is_integer()) {
                            report_error(inst_prefix + "Operand must be integer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match operand type.");
                        }
                    }
                    break;
                }

                case Opcode::eq:
                case Opcode::ne:
                case Opcode::slt:
                case Opcode::ult:
                case Opcode::sle:
                case Opcode::ule:
                case Opcode::sgt:
                case Opcode::ugt:
                case Opcode::sge:
                case Opcode::uge: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0 != t1) {
                            report_error(inst_prefix + "Comparison operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (t0.is_void()) {
                            report_error(inst_prefix + "Cannot compare void types.");
                        }
                        if (inst->type() != Type::i32()) {
                            report_error(inst_prefix + "Comparison result type must be i32.");
                        }
                    }
                    break;
                }

                case Opcode::sadd_overflow:
                case Opcode::ssub_overflow:
                case Opcode::smul_overflow:
                case Opcode::uadd_overflow:
                case Opcode::usub_overflow:
                case Opcode::umul_overflow: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (!t0.is_integer() || !t1.is_integer()) {
                            report_error(inst_prefix + "Overflow arithmetic operands must be integer.");
                        } else if (t0 != t1) {
                            report_error(inst_prefix + "Overflow arithmetic operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        }
                        if (inst->type() != Type::i32()) {
                            report_error(inst_prefix + "Overflow arithmetic result type must be i32.");
                        }
                    }
                    break;
                }

                case Opcode::select: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Select requires 3 operands (cond, true_val, false_val).");
                    } else {
                        Type cond_t = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        Type t2 = inst->operand(2)->type();
                        if (cond_t != Type::i32()) {
                            report_error(inst_prefix + "Select condition type must be i32, got " + std::string(cond_t.name()) + ".");
                        }
                        if (t1.is_void() || t2.is_void()) {
                            report_error(inst_prefix + "Select operands cannot be void.");
                        } else if (t1 != t2) {
                            report_error(inst_prefix + "Select value types mismatch: " +
                                         std::string(t1.name()) + " vs " + std::string(t2.name()) + ".");
                        } else if (inst->type() != t1) {
                            report_error(inst_prefix + "Select result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t1.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::alloca_: {
                    if (inst->operand_count() != 0) {
                        report_error(inst_prefix + "Alloca requires 0 operands.");
                    }
                    if (!inst->type().is_pointer_or_gcref() && inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Alloca result type must be ptr or i64.");
                    }
                    verify_tagged_alloca(*inst, inst_prefix, [this](const std::string& msg) { report_error(msg); });
                    break;
                }

                case Opcode::pinned_tls_read:
                case Opcode::read_sp: {
                    if (inst->operand_count() != 0) {
                        report_error(inst_prefix + std::string(opcode_name(inst->opcode())) +
                                     " requires 0 operands.");
                    }
                    if (!inst->type().is_pointer_or_gcref() && inst->type() != Type::i64()) {
                        report_error(inst_prefix + std::string(opcode_name(inst->opcode())) +
                                     " result type must be ptr or i64.");
                    }
                    break;
                }

                case Opcode::pinned_tls_write: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "pinned_tls_write requires 1 operand.");
                    }
                    break;
                }

                case Opcode::load: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Load requires 1 base operand.");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                            report_error(inst_prefix + "Load base must be ptr or gcref, got " +
                                         std::string(base_t.name()) + ".");
                        }
                        if (inst->memory_type().is_void()) {
                            report_error(inst_prefix + "Load memory type cannot be void.");
                        } else if (inst->type() != inst->memory_type()) {
                            report_error(inst_prefix + "Load result type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::store: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Store requires 2 operands (base, val).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type val_t = inst->operand(1)->type();
                        if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                            report_error(inst_prefix + "Store base must be ptr or gcref, got " +
                                         std::string(base_t.name()) + ".");
                        }
                        if (inst->memory_type().is_void()) {
                            report_error(inst_prefix + "Store memory type cannot be void.");
                        } else if (val_t != inst->memory_type()) {
                            report_error(inst_prefix + "Store value type (" + std::string(val_t.name()) +
                                         ") must match memory type (" + std::string(inst->memory_type().name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::load_indexed: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Load indexed requires 2 operands (base, index).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type idx_t = inst->operand(1)->type();
                        if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                            report_error(inst_prefix + "Load indexed base must be ptr or gcref.");
                        }
                        if (!idx_t.is_integer()) {
                            report_error(inst_prefix + "Load indexed index must be integer.");
                        }
                        uint8_t sc = inst->scale();
                        if (sc != 1 && sc != 2 && sc != 4 && sc != 8) {
                            report_error(inst_prefix + "Scale must be 1, 2, 4, or 8, got " + std::to_string(sc) + ".");
                        }
                        if (inst->memory_type().is_void() || inst->type() != inst->memory_type()) {
                            report_error(inst_prefix + "Load indexed result type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::store_indexed: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Store indexed requires 3 operands (base, index, val).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type idx_t = inst->operand(1)->type();
                        Type val_t = inst->operand(2)->type();
                        if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                            report_error(inst_prefix + "Store indexed base must be ptr or gcref.");
                        }
                        if (!idx_t.is_integer()) {
                            report_error(inst_prefix + "Store indexed index must be integer.");
                        }
                        uint8_t sc = inst->scale();
                        if (sc != 1 && sc != 2 && sc != 4 && sc != 8) {
                            report_error(inst_prefix + "Scale must be 1, 2, 4, or 8, got " + std::to_string(sc) + ".");
                        }
                        if (inst->memory_type().is_void() || val_t != inst->memory_type()) {
                            report_error(inst_prefix + "Store indexed value type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::write_barrier: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Write barrier requires 2 operands (obj, val).");
                    } else {
                        Type obj_t = inst->operand(0)->type();
                        if (!obj_t.is_pointer_or_gcref() && !obj_t.is_integer()) {
                            report_error(inst_prefix + "Write barrier obj must be ptr, gcref, or int.");
                        }
                    }
                    break;
                }

                case Opcode::call: {
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Call requires a non-empty callee symbol.");
                    }
                    break;
                }

                case Opcode::call_indirect: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || !inst->operand(0)->type().is_pointer()) {
                        report_error(inst_prefix + "Call indirect requires first operand to be ptr.");
                    }
                    break;
                }

                case Opcode::patchable_call: {
                    if (inst->symbol().empty() || inst->extra_symbol().empty()) {
                        report_error(inst_prefix + "Patchable call requires site symbol and callee symbol.");
                    }
                    break;
                }

                case Opcode::safepoint:
                    break;

                case Opcode::func_addr: {
                    if (inst->operand_count() != 0) {
                        report_error(inst_prefix + "func_addr requires 0 operands.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "func_addr requires a symbol name.");
                    }
                    break;
                }

                case Opcode::guard: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "Guard requires i32 condition operand.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Guard requires non-empty exit label.");
                    }
                    // Exit stub: stub(state values...) -> the function's type.
                    // Resume target (used only without a stub): parameter i
                    // takes state value i.
                    std::string why;
                    if (const Function* stub = fn.guard_exit_stub(*inst)) {
                        if (!fn.guard_exit_stub_matches(*inst, *stub, why)) {
                            report_error(inst_prefix + "Guard " + why + ".");
                        }
                    } else if (const BasicBlock* rt = fn.get_resume_target(inst->resume_id())) {
                        const auto& state = inst->state_map();
                        if (rt->param_count() > state.size()) {
                            report_error(inst_prefix + "Guard resume target '" + std::string(rt->name()) + "' has " +
                                         std::to_string(rt->param_count()) + " parameters but the guard has " +
                                         std::to_string(state.size()) + " state values.");
                        } else {
                            for (size_t p = 0; p < rt->param_count(); ++p) {
                                if (state[p] && rt->param(p) && state[p]->type() != rt->param(p)->type()) {
                                    report_error(inst_prefix + "Guard resume target parameter " + std::to_string(p) +
                                                 " is " + std::string(rt->param(p)->type().name()) +
                                                 " but state value " + std::to_string(p) + " is " +
                                                 std::string(state[p]->type().name()) + ".");
                                }
                            }
                        }
                    } else if (guards_resume_in_table(fn)) {
                        // The function resumes its guards in resume_table
                        // blocks, but not this one: Tier 0 would have nowhere
                        // to go when it fails (text MIR written when every
                        // guard had id 0 and shared `entry 0`). A function
                        // none of whose guards has a resume target is
                        // optimizer-level IR that passes and tests build; it
                        // is left alone.
                        report_error(inst_prefix + "Guard (resume id " + std::to_string(inst->resume_id()) +
                                     ") has neither an exit stub (no function '@" + std::string(inst->symbol()) +
                                     "') nor a resume target (no resume_table entry " +
                                     std::to_string(inst->resume_id()) + "). Guards resuming at one block need " +
                                     "a resume_table entry per resume id.");
                    }
                    break;
                }

                case Opcode::resume_point:
                    break;

                case Opcode::br: {
                    const BranchTarget& target = inst->branch_target();
                    if (!target.block) {
                        report_error(inst_prefix + "Branch has null target block.");
                    } else {
                        if (block_set.find(target.block) == block_set.end()) {
                            report_error(inst_prefix + "Branch target block does not belong to function.");
                        }
                        if (target.args.size() != target.block->param_count()) {
                            report_error(inst_prefix + "Branch passes " + std::to_string(target.args.size()) +
                                         " arguments, but target block '" + std::string(target.block->name()) +
                                         "' expects " + std::to_string(target.block->param_count()) + ".");
                        } else {
                            for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                check_value_dom(target.args[a_i], "branch arg " + std::to_string(a_i));
                                if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                    report_error(inst_prefix + "Branch argument " + std::to_string(a_i) +
                                                 " type (" + std::string(target.args[a_i]->type().name()) +
                                                 ") does not match target parameter type (" +
                                                 std::string(target.block->param(a_i)->type().name()) + ").");
                                }
                            }
                        }
                    }
                    break;
                }

                case Opcode::br_if: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "br_if condition must be i32.");
                    }
                    auto check_target = [&](const BranchTarget& target, const std::string& label) {
                        if (!target.block) {
                            report_error(inst_prefix + label + " target block is null.");
                        } else {
                            if (block_set.find(target.block) == block_set.end()) {
                                report_error(inst_prefix + label + " target block does not belong to function.");
                            }
                            if (target.args.size() != target.block->param_count()) {
                                report_error(inst_prefix + label + " branch passes " + std::to_string(target.args.size()) +
                                             " arguments, but target expects " + std::to_string(target.block->param_count()) + ".");
                            } else {
                                for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                    check_value_dom(target.args[a_i], label + " arg " + std::to_string(a_i));
                                    if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                        report_error(inst_prefix + label + " argument " + std::to_string(a_i) +
                                                     " type (" + std::string(target.args[a_i]->type().name()) +
                                                     ") does not match target parameter (" +
                                                     std::string(target.block->param(a_i)->type().name()) + ").");
                                    }
                                }
                            }
                        }
                    };
                    check_target(inst->true_target(), "true");
                    check_target(inst->false_target(), "false");
                    break;
                }

                case Opcode::switch_: {
                    if (inst->operand_count() != 1 || !inst->operand(0) || !inst->operand(0)->type().is_integer()) {
                        report_error(inst_prefix + "switch requires 1 integer condition operand.");
                    }
                    auto check_target = [&](const BranchTarget& target, const std::string& label) {
                        if (!target.block) {
                            report_error(inst_prefix + label + " target block is null.");
                        } else {
                            if (block_set.find(target.block) == block_set.end()) {
                                report_error(inst_prefix + label + " target block does not belong to function.");
                            }
                            if (target.args.size() != target.block->param_count()) {
                                report_error(inst_prefix + label + " switch passes " + std::to_string(target.args.size()) +
                                             " arguments, but target expects " + std::to_string(target.block->param_count()) + ".");
                            } else {
                                for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                    check_value_dom(target.args[a_i], label + " arg " + std::to_string(a_i));
                                    if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                        report_error(inst_prefix + label + " argument " + std::to_string(a_i) +
                                                     " type (" + std::string(target.args[a_i]->type().name()) +
                                                     ") does not match target parameter (" +
                                                     std::string(target.block->param(a_i)->type().name()) + ").");
                                    }
                                }
                            }
                        }
                    };
                    check_target(inst->default_target(), "default");
                    const auto& cases = inst->switch_cases();
                    for (size_t c_i = 0; c_i < cases.size(); ++c_i) {
                        check_target(cases[c_i].target, "case " + std::to_string(cases[c_i].value));
                    }
                    // Every engine must agree on which case a value takes: a
                    // case outside the condition's signed range would match
                    // in the JITs (low bits only) but never in the
                    // interpreter, and a duplicate is ambiguous.
                    if (inst->operand_count() == 1 && inst->operand(0) && inst->operand(0)->type().is_integer()) {
                        const Type ct = inst->operand(0)->type();
                        const unsigned bits = ct == Type::i8() ? 8u : ct == Type::i16() ? 16u : ct == Type::i32() ? 32u : 64u;
                        const int64_t lo = bits >= 64 ? INT64_MIN : -(int64_t{1} << (bits - 1));
                        const int64_t hi = bits >= 64 ? INT64_MAX : (int64_t{1} << (bits - 1)) - 1;
                        std::unordered_set<int64_t> seen;
                        for (const SwitchCase& sc : cases) {
                            if (sc.value < lo || sc.value > hi) {
                                report_error(inst_prefix + "switch case value " + std::to_string(sc.value) +
                                             " does not fit the " + std::string(ct.name()) + " condition.");
                            }
                            if (!seen.insert(sc.value).second) {
                                report_error(inst_prefix + "switch has duplicate case value " +
                                             std::to_string(sc.value) + ".");
                            }
                        }
                    }
                    break;
                }

                case Opcode::ret: {
                    if (fn.return_type().is_void()) {
                        if (inst->operand_count() > 0 && inst->operand(0) != nullptr && !inst->operand(0)->type().is_void()) {
                            report_error(inst_prefix + "Void function return instruction must not have a return value.");
                        }
                    } else {
                        if (inst->operand_count() != 1 || !inst->operand(0)) {
                            report_error(inst_prefix + "Non-void function must return a value of type " +
                                         std::string(fn.return_type().name()) + ".");
                        } else if (inst->operand(0)->type() != fn.return_type()) {
                            report_error(inst_prefix + "Return value type (" +
                                         std::string(inst->operand(0)->type().name()) +
                                         ") does not match function return type (" +
                                         std::string(fn.return_type().name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::unreachable:
                    break;
                default:
                    if (is_vector_op(inst->opcode())) {
                        verify_vector_instruction(inst, inst_prefix, [this](const std::string& msg) { report_error(msg); });
                    } else if (is_coro_op(inst->opcode())) {
                        verify_coro_instruction(inst, bb, inst_prefix, [this](const std::string& msg) { report_error(msg); });
                    } else if (verify_tagged_bitcast(*inst, inst_prefix, [this](const std::string& msg) { report_error(msg); })) {
                    } else if (verify_keep_alive(*inst, inst_prefix, [this](const std::string& msg) { report_error(msg); })) {
                    } else if (!verify_exception_instruction(inst, bb, inst_prefix, [this](const std::string& msg) { report_error(msg); })) {
                        report_error(inst_prefix + "Unhandled or invalid instruction opcode: " + std::string(opcode_name(inst->opcode())));
                    }
                    break;
            }
        }
    }

    verify_derived_gcrefs(fn, fn_prefix, [this](const std::string& msg) { report_error(msg); });

    // A guard's resume id names the one Tier-0 guard a deopt resumes at.
    // Guards may share an id only as copies of one guard (code duplication
    // in an optimized clone): same exit label and same state value types.
    {
        std::unordered_map<uint32_t, const Instruction*> first_guard;
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (!inst || inst->opcode() != Opcode::guard) continue;
                auto [it, fresh] = first_guard.emplace(inst->resume_id(), inst);
                if (fresh) continue;
                const Instruction* g = it->second;
                bool same = g->symbol() == inst->symbol() && g->state_map().size() == inst->state_map().size();
                for (size_t i = 0; same && i < g->state_map().size(); ++i) {
                    const Value* a = g->state_map()[i];
                    const Value* c = inst->state_map()[i];
                    same = a && c && a->type() == c->type();
                }
                if (!same) {
                    report_error(fn_prefix + "Two different guards have resume id " +
                                 std::to_string(inst->resume_id()) + " (exit labels '" + std::string(g->symbol()) +
                                 "' and '" + std::string(inst->symbol()) + "'); each guard needs its own id.");
                }
            }
        }
    }

    // Verify resume points
    for (const auto& entry_pair : fn.resume_points()) {
        if (!entry_pair.second) {
            report_error(fn_prefix + "Resume point " + std::to_string(entry_pair.first) + " has null target block.");
        } else if (block_set.find(entry_pair.second) == block_set.end()) {
            report_error(fn_prefix + "Resume point " + std::to_string(entry_pair.first) +
                         " points to block not in function.");
        }
    }

    return !has_error_;
}

bool verify_module(const Module& mod, DiagnosticReporter* diag) {
    Verifier v(diag);
    return v.verify_module(mod);
}

bool verify_function(const Function& fn, DiagnosticReporter* diag) {
    Verifier v(diag);
    return v.verify_function(fn);
}

} // namespace brass
