#include <brass/mir/f64_demote.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/opcodes.hpp>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

namespace brass {

namespace {

static constexpr double kMaxSafeInt = 9007199254740992.0; // 2^53
static constexpr double kMinSafeInt = -9007199254740992.0;

bool is_safe_integer_f64(double v) noexcept {
    return std::isfinite(v) && v == std::trunc(v) && v >= kMinSafeInt && v <= kMaxSafeInt;
}

bool get_const_int_or_f64_int(const Value* val, int64_t& out) noexcept {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out = def->imm_i64();
        return true;
    }
    if (def->opcode() == Opcode::fconst_f64) {
        double f = def->imm_f64();
        if (is_safe_integer_f64(f)) {
            out = static_cast<int64_t>(f);
            return true;
        }
    }
    return false;
}

bool are_equivalent_values(const Value* a, const Value* b) noexcept {
    if (a == b) return true;
    if (!a || !b) return false;
    int64_t c0, c1;
    if (get_const_int_or_f64_int(a, c0) && get_const_int_or_f64_int(b, c1)) {
        return c0 == c1;
    }
    return false;
}

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) inst->set_operand(i, new_val);
            }
            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                if (inst->branch_target().args[i] == old_val) inst->branch_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                if (inst->true_target().args[i] == old_val) inst->true_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                if (inst->false_target().args[i] == old_val) inst->false_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                if (inst->default_target().args[i] == old_val) inst->default_target().args[i] = new_val;
            }
            for (auto& sc : inst->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    if (sc.target.args[i] == old_val) sc.target.args[i] = new_val;
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) inst->state_map()[i] = new_val;
            }
        }
    }
}

bool is_proven_exact_div(
    const Instruction* div_inst,
    const Value* num,
    const Value* denom,
    const DominatorTree& dom
) {
    if (!num || !denom) return false;
    int64_t c_num, c_denom;
    bool has_c_num = get_const_int_or_f64_int(num, c_num);
    bool has_c_denom = get_const_int_or_f64_int(denom, c_denom);

    if (has_c_denom && (c_denom == 1 || c_denom == -1)) return true;
    if (has_c_num && c_num == 0 && (!has_c_denom || c_denom != 0)) return true;
    if (has_c_num && has_c_denom && c_denom != 0 && (c_num % c_denom == 0)) return true;

    const BasicBlock* cur_bb = div_inst->parent();
    if (!cur_bb) return false;
    const Function* fn = cur_bb->parent();
    if (!fn) return false;

    for (const BasicBlock* dom_bb : fn->blocks()) {
        if (!dom_bb || !dom.dominates(dom_bb, cur_bb)) continue;
        const Instruction* term = dom_bb->terminator();
        if (!term || term->opcode() != Opcode::br_if) continue;

        const Value* cond = term->operand(0);
        if (!cond || !cond->is_instruction()) continue;
        const Instruction* cond_def = cond->defining_instruction();
        if (!cond_def) continue;

        bool on_true_branch = dom.dominates(term->true_target().block, cur_bb);
        bool on_false_branch = dom.dominates(term->false_target().block, cur_bb);
        if (!on_true_branch && !on_false_branch) continue;

        Opcode cmp_op = cond_def->opcode();
        if (cmp_op != Opcode::eq && cmp_op != Opcode::ne) continue;

        const Value* cmp_lhs = cond_def->operand(0);
        const Value* cmp_rhs = cond_def->operand(1);
        if (!cmp_lhs || !cmp_rhs) continue;

        const Value* mod_val = nullptr;
        const Value* zero_val = nullptr;

        int64_t z;
        if (get_const_int_or_f64_int(cmp_rhs, z) && z == 0) {
            mod_val = cmp_lhs;
            zero_val = cmp_rhs;
        } else if (get_const_int_or_f64_int(cmp_lhs, z) && z == 0) {
            mod_val = cmp_rhs;
            zero_val = cmp_lhs;
        }
        if (!mod_val || !zero_val) continue;

        if (!mod_val->is_instruction()) continue;
        const Instruction* mod_def = mod_val->defining_instruction();
        if (!mod_def) continue;

        const Value* m_num = nullptr;
        const Value* m_denom = nullptr;

        if (mod_def->opcode() == Opcode::smod || mod_def->opcode() == Opcode::umod) {
            m_num = mod_def->operand(0);
            m_denom = mod_def->operand(1);
        } else if (mod_def->opcode() == Opcode::call && mod_def->symbol() == "bronze_f64_mod") {
            m_num = mod_def->operand(0);
            m_denom = mod_def->operand(1);
        }

        if (m_num && m_denom) {
            if (are_equivalent_values(m_num, num) && are_equivalent_values(m_denom, denom)) {
                if ((cmp_op == Opcode::eq && on_true_branch) || (cmp_op == Opcode::ne && on_false_branch)) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool is_value_exact_int(
    const Value* val,
    const std::unordered_set<const Value*>& exact_ints,
    const std::unordered_set<const Value*>& entry_params
) {
    if (!val) return false;
    if (val->type().is_integer()) return true;
    if (exact_ints.count(val) != 0) return true;
    if (entry_params.count(val) != 0) return true;
    return false;
}

} // namespace

bool f64_demote_pass(Function& fn, const F64DemoteOptions& options) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);

    std::unordered_set<const Value*> exact_ints;
    std::unordered_set<const Value*> entry_params;

    // 1. Initial Seeding:
    // Constants, int-to-float conversions, and non-entry block parameters
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        if (bb == fn.entry_block()) {
            for (size_t i = 0; i < bb->param_count(); ++i) {
                const Value* param = bb->param(i);
                if (param && param->type() == Type::f64()) {
                    entry_params.insert(param);
                }
            }
        } else {
            for (size_t i = 0; i < bb->param_count(); ++i) {
                const Value* param = bb->param(i);
                if (param && param->type() == Type::f64()) {
                    exact_ints.insert(param);
                }
            }
        }

        for (const Instruction* inst : *bb) {
            if (!inst || !inst->produces_value()) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::fconst_f64) {
                if (is_safe_integer_f64(inst->imm_f64())) {
                    exact_ints.insert(inst->result());
                }
            } else if (op == Opcode::sitofp_f64_i32 || op == Opcode::sitofp_f64_i64) {
                exact_ints.insert(inst->result());
            } else if (inst->type() == Type::f64()) {
                if (op == Opcode::add || op == Opcode::sub || op == Opcode::mul ||
                    op == Opcode::neg || op == Opcode::smod || op == Opcode::select ||
                    (op == Opcode::call && inst->symbol() == "bronze_f64_mod") ||
                    (op == Opcode::sdiv && options.enable_exact_div)) {
                    exact_ints.insert(inst->result());
                }
            }
        }
    }

    // 2. Fixed-Point Dataflow Iteration
    bool changed = true;
    while (changed) {
        changed = false;

        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;

            // Check non-entry block parameters
            if (bb != fn.entry_block()) {
                for (size_t p_i = 0; p_i < bb->param_count(); ++p_i) {
                    const Value* param = bb->param(p_i);
                    if (!param || !exact_ints.count(param)) continue;

                    bool all_preds_exact = true;
                    if (bb->predecessors().empty()) {
                        all_preds_exact = false;
                    } else {
                        for (const BasicBlock* pred : bb->predecessors()) {
                            if (!pred) continue;
                            const Instruction* term = pred->terminator();
                            if (!term) { all_preds_exact = false; break; }

                            auto check_bt = [&](const BranchTarget& bt) -> bool {
                                if (bt.block != bb) return true;
                                if (p_i >= bt.args.size()) return false;
                                return is_value_exact_int(bt.args[p_i], exact_ints, entry_params);
                            };

                            if (term->opcode() == Opcode::br) {
                                if (!check_bt(term->branch_target())) { all_preds_exact = false; break; }
                            } else if (term->opcode() == Opcode::br_if) {
                                if (!check_bt(term->true_target()) || !check_bt(term->false_target())) {
                                    all_preds_exact = false; break;
                                }
                            } else if (term->opcode() == Opcode::switch_) {
                                if (!check_bt(term->default_target())) { all_preds_exact = false; break; }
                                for (const auto& sc : term->switch_cases()) {
                                    if (!check_bt(sc.target)) { all_preds_exact = false; break; }
                                }
                            }
                        }
                    }

                    if (!all_preds_exact) {
                        exact_ints.erase(param);
                        changed = true;
                    }
                }
            }

            // Check instructions
            for (const Instruction* inst : *bb) {
                if (!inst || !inst->produces_value()) continue;
                const Value* res = inst->result();
                if (!res || !exact_ints.count(res)) continue;

                Opcode op = inst->opcode();
                bool still_exact = true;

                if (op == Opcode::fconst_f64) {
                    still_exact = is_safe_integer_f64(inst->imm_f64());
                } else if (op == Opcode::sitofp_f64_i32 || op == Opcode::sitofp_f64_i64) {
                    still_exact = true;
                } else if (op == Opcode::add || op == Opcode::sub || op == Opcode::mul) {
                    still_exact = is_value_exact_int(inst->operand(0), exact_ints, entry_params) &&
                                  is_value_exact_int(inst->operand(1), exact_ints, entry_params);
                } else if (op == Opcode::neg) {
                    still_exact = is_value_exact_int(inst->operand(0), exact_ints, entry_params);
                } else if (op == Opcode::smod) {
                    still_exact = is_value_exact_int(inst->operand(0), exact_ints, entry_params) &&
                                  is_value_exact_int(inst->operand(1), exact_ints, entry_params);
                } else if (op == Opcode::call && inst->symbol() == "bronze_f64_mod") {
                    still_exact = is_value_exact_int(inst->operand(0), exact_ints, entry_params) &&
                                  is_value_exact_int(inst->operand(1), exact_ints, entry_params);
                } else if (op == Opcode::sdiv) {
                    still_exact = options.enable_exact_div &&
                                  is_value_exact_int(inst->operand(0), exact_ints, entry_params) &&
                                  is_value_exact_int(inst->operand(1), exact_ints, entry_params) &&
                                  is_proven_exact_div(inst, inst->operand(0), inst->operand(1), dom);
                } else if (op == Opcode::select) {
                    still_exact = is_value_exact_int(inst->operand(1), exact_ints, entry_params) &&
                                  is_value_exact_int(inst->operand(2), exact_ints, entry_params);
                } else {
                    still_exact = false;
                }

                if (!still_exact) {
                    exact_ints.erase(res);
                    changed = true;
                }
            }
        }
    }

    // 3. Relevance Filtering:
    // Only demote values that are connected to non-entry block parameters, modulo, or proven exact division
    std::unordered_set<const Value*> relevant_demote;

    // Seed roots: non-entry block parameters in exact_ints, bronze_f64_mod, proven sdiv
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        if (bb != fn.entry_block()) {
            for (size_t i = 0; i < bb->param_count(); ++i) {
                const Value* p = bb->param(i);
                if (p && exact_ints.count(p)) {
                    relevant_demote.insert(p);
                }
            }
        }
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::call && inst->symbol() == "bronze_f64_mod") {
                if (inst->result() && exact_ints.count(inst->result())) {
                    relevant_demote.insert(inst->result());
                }
            } else if (op == Opcode::sdiv && options.enable_exact_div) {
                if (inst->result() && exact_ints.count(inst->result())) {
                    relevant_demote.insert(inst->result());
                }
            }
        }
    }

    if (relevant_demote.empty()) return false;

    // Propagate backward and forward across dataflow
    bool rel_changed = true;
    while (rel_changed) {
        rel_changed = false;

        // Backward propagation: if an instruction's result is relevant, its exact operands are relevant
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                if (inst->produces_value() && relevant_demote.count(inst->result())) {
                    for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                        const Value* opd = inst->operand(op_i);
                        if (opd && exact_ints.count(opd) && !relevant_demote.count(opd)) {
                            relevant_demote.insert(opd);
                            rel_changed = true;
                        }
                    }
                }

                // If branch target parameter is relevant, incoming exact argument is relevant
                auto check_bt_rel = [&](const BranchTarget& bt) {
                    if (!bt.block) return;
                    for (size_t i = 0; i < bt.args.size() && i < bt.block->param_count(); ++i) {
                        const Value* p = bt.block->param(i);
                        const Value* a = bt.args[i];
                        if (p && relevant_demote.count(p) && a && exact_ints.count(a) && !relevant_demote.count(a)) {
                            relevant_demote.insert(a);
                            rel_changed = true;
                        }
                    }
                };

                if (inst->opcode() == Opcode::br) {
                    check_bt_rel(inst->branch_target());
                } else if (inst->opcode() == Opcode::br_if) {
                    check_bt_rel(inst->true_target());
                    check_bt_rel(inst->false_target());
                } else if (inst->opcode() == Opcode::switch_) {
                    check_bt_rel(inst->default_target());
                    for (const auto& sc : inst->switch_cases()) check_bt_rel(sc.target);
                }
            }
        }

        // Forward propagation: if all operands of an exact arithmetic op are relevant, the result is relevant
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst || !inst->produces_value()) continue;
                const Value* res = inst->result();
                if (!res || !exact_ints.count(res) || relevant_demote.count(res)) continue;

                Opcode op = inst->opcode();
                if (op == Opcode::add || op == Opcode::sub || op == Opcode::mul || op == Opcode::neg) {
                    bool all_ops_rel = true;
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        const Value* o = inst->operand(i);
                        if (!o || (!relevant_demote.count(o) && !o->type().is_integer() && !entry_params.count(o))) {
                            all_ops_rel = false;
                            break;
                        }
                    }
                    if (all_ops_rel) {
                        relevant_demote.insert(res);
                        rel_changed = true;
                    }
                }
            }
        }
    }

    // 4. Construct demote set for rewriting
    std::unordered_set<Value*> demote_set;
    for (const Value* v : relevant_demote) {
        if (v && v->type() == Type::f64()) {
            demote_set.insert(const_cast<Value*>(v));
        }
    }

    if (demote_set.empty()) return false;

    Builder b(*fn.parent());
    b.set_function(&fn);

    // 5. Rewrite non-entry block parameters to i64
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || bb == fn.entry_block()) continue;
        for (size_t p_i = 0; p_i < bb->param_count(); ++p_i) {
            Value* param = bb->param(p_i);
            if (param && demote_set.count(param)) {
                param->set_type(Type::i64());
            }
        }
    }

    // 6. Rewrite demoted instructions to i64
    std::vector<Instruction*> dead_instructions;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        Instruction* inst = bb->head();
        while (inst) {
            Instruction* next = inst->next();
            if (inst->produces_value() && demote_set.count(inst->result())) {
                inst->set_type(Type::i64());
                inst->result()->set_type(Type::i64());

                Opcode op = inst->opcode();
                if (op == Opcode::fconst_f64) {
                    inst->set_opcode(Opcode::iconst_i64);
                    inst->set_imm_i64(static_cast<int64_t>(inst->imm_f64()));
                } else if (op == Opcode::call && inst->symbol() == "bronze_f64_mod") {
                    inst->set_opcode(Opcode::smod);
                    inst->set_symbol("");
                } else if (op == Opcode::sitofp_f64_i32) {
                    inst->set_opcode(Opcode::sext_i64);
                } else if (op == Opcode::sitofp_f64_i64) {
                    replace_all_uses(fn, inst->result(), inst->operand(0));
                    dead_instructions.push_back(inst);
                } else if (op == Opcode::bitcast_i64_f64 || op == Opcode::bitcast_f64_i64) {
                    replace_all_uses(fn, inst->result(), inst->operand(0));
                    dead_instructions.push_back(inst);
                }
            }
            inst = next;
        }
    }

    for (Instruction* dead_inst : dead_instructions) {
        if (dead_inst && dead_inst->parent()) {
            dead_inst->parent()->remove_instruction(dead_inst);
        }
    }

    // 7. Insert boundary conversions and fixup operand types
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        Instruction* inst = bb->head();
        while (inst) {
            Instruction* next = inst->next();
            Opcode op = inst->opcode();

            // Return instruction
            if (op == Opcode::ret) {
                if (inst->operand_count() >= 1) {
                    Value* ret_val = inst->operand(0);
                    if (ret_val) {
                        if (fn.return_type() == Type::f64() && ret_val->type() == Type::i64()) {
                            b.position_before(inst);
                            Value* conv = b.build_sitofp_f64_i64(ret_val);
                            inst->set_operand(0, conv);
                        } else if (fn.return_type() == Type::i64() && ret_val->type() == Type::f64()) {
                            b.position_before(inst);
                            Value* conv = b.build_fptosi_i64(ret_val);
                            inst->set_operand(0, conv);
                        }
                    }
                }
            } else if (op == Opcode::call) {
                if (inst->symbol() == "bronze_print_f64") {
                    if (inst->operand_count() >= 1 && inst->operand(0) && inst->operand(0)->type() == Type::i64()) {
                        b.position_before(inst);
                        Value* conv = b.build_sitofp_f64_i64(inst->operand(0));
                        inst->set_operand(0, conv);
                    }
                } else if (fn.parent()) {
                    Function* callee = fn.parent()->get_function(inst->symbol());
                    if (callee) {
                        for (size_t i = 0; i < inst->operand_count() && i < callee->param_count(); ++i) {
                            Value* arg = inst->operand(i);
                            if (!arg) continue;
                            if (callee->param_type(i) == Type::f64() && arg->type() == Type::i64()) {
                                b.position_before(inst);
                                Value* conv = b.build_sitofp_f64_i64(arg);
                                inst->set_operand(i, conv);
                            } else if (callee->param_type(i) == Type::i64() && arg->type() == Type::f64()) {
                                b.position_before(inst);
                                Value* conv = b.build_fptosi_i64(arg);
                                inst->set_operand(i, conv);
                            }
                        }
                    }
                }
            } else if (is_comparison(op)) {
                Value* op0 = inst->operand(0);
                Value* op1 = inst->operand(1);
                if (op0 && op1) {
                    if (op0->type() == Type::i64() && op1->type() == Type::f64()) {
                        b.position_before(inst);
                        Value* conv = b.build_sitofp_f64_i64(op0);
                        inst->set_operand(0, conv);
                    } else if (op0->type() == Type::f64() && op1->type() == Type::i64()) {
                        b.position_before(inst);
                        Value* conv = b.build_sitofp_f64_i64(op1);
                        inst->set_operand(1, conv);
                    }
                }
            } else if (op == Opcode::select) {
                // Operand 0 is condition (i32/i64), Operands 1 and 2 are true_val and false_val
                if (inst->type() == Type::f64()) {
                    for (size_t op_i = 1; op_i <= 2 && op_i < inst->operand_count(); ++op_i) {
                        Value* o = inst->operand(op_i);
                        if (o && o->type() == Type::i64()) {
                            b.position_before(inst);
                            Value* conv = b.build_sitofp_f64_i64(o);
                            inst->set_operand(op_i, conv);
                        }
                    }
                } else if (inst->type() == Type::i64()) {
                    for (size_t op_i = 1; op_i <= 2 && op_i < inst->operand_count(); ++op_i) {
                        Value* o = inst->operand(op_i);
                        if (o && o->type() == Type::f64()) {
                            b.position_before(inst);
                            Value* conv = b.build_fptosi_i64(o);
                            inst->set_operand(op_i, conv);
                        }
                    }
                }
            } else if (op == Opcode::add || op == Opcode::sub || op == Opcode::mul ||
                       op == Opcode::sdiv || op == Opcode::udiv || op == Opcode::smod ||
                       op == Opcode::umod || op == Opcode::neg) {
                if (inst->type() == Type::f64()) {
                    for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                        Value* o = inst->operand(op_i);
                        if (o && o->type() == Type::i64()) {
                            b.position_before(inst);
                            Value* conv = b.build_sitofp_f64_i64(o);
                            inst->set_operand(op_i, conv);
                        }
                    }
                } else if (inst->type() == Type::i64()) {
                    for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                        Value* o = inst->operand(op_i);
                        if (o && o->type() == Type::f64()) {
                            b.position_before(inst);
                            Value* conv = b.build_fptosi_i64(o);
                            inst->set_operand(op_i, conv);
                        }
                    }
                }
            } else if (op == Opcode::fptosi_i32) {
                if (inst->operand(0) && inst->operand(0)->type() == Type::i64()) {
                    inst->set_opcode(Opcode::trunc_i32);
                }
            } else if (op == Opcode::trunc_i32) {
                if (inst->operand(0) && inst->operand(0)->type() == Type::f64()) {
                    inst->set_opcode(Opcode::fptosi_i32);
                }
            } else if (op == Opcode::bitcast_i64_f64) {
                if (inst->operand(0) && inst->operand(0)->type() == Type::i64()) {
                    b.position_before(inst);
                    Value* conv = b.build_sitofp_f64_i64(inst->operand(0));
                    inst->set_operand(0, conv);
                }
            }

            // Fix branch target arguments
            auto fix_target_args = [&](BranchTarget& target) {
                if (!target.block) return;
                for (size_t i = 0; i < target.args.size() && i < target.block->param_count(); ++i) {
                    Value* arg = target.args[i];
                    Value* param = target.block->param(i);
                    if (!arg || !param) continue;
                    if (arg->type() == Type::i64() && param->type() == Type::f64()) {
                        b.position_before(inst);
                        Value* conv = b.build_sitofp_f64_i64(arg);
                        target.args[i] = conv;
                    } else if (arg->type() == Type::f64() && param->type() == Type::i64()) {
                        b.position_before(inst);
                        Value* conv = b.build_fptosi_i64(arg);
                        target.args[i] = conv;
                    }
                }
            };

            if (op == Opcode::br) {
                fix_target_args(inst->branch_target());
            } else if (op == Opcode::br_if) {
                fix_target_args(inst->true_target());
                fix_target_args(inst->false_target());
            } else if (op == Opcode::switch_) {
                fix_target_args(inst->default_target());
                for (auto& sc : inst->switch_cases()) fix_target_args(sc.target);
            }

            inst = next;
        }
    }

    fn.rebuild_cfg_predecessors();
    return true;
}

bool f64_demote_module_pass(Module& mod, const F64DemoteOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= f64_demote_pass(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
