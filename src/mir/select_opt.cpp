#include <brass/mir/select_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace brass {

namespace {

bool is_safe_for_select_hoist(const Instruction* inst) {
    if (!inst) return false;
    if (inst->has_side_effects() || inst->is_terminator() || inst->is_call()) {
        return false;
    }
    Opcode op = inst->opcode();
    switch (op) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64:
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::select:
            return true;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod: {
            // Div/mod is only safe to hoist if divisor is non-zero constant
            if (inst->operand_count() >= 2 && inst->operand(1)) {
                const Value* denom = inst->operand(1);
                if (denom->is_instruction()) {
                    const Instruction* ddef = denom->defining_instruction();
                    if (ddef && (ddef->opcode() == Opcode::iconst_i32 || ddef->opcode() == Opcode::iconst_i64)) {
                        return ddef->imm_i64() != 0;
                    }
                }
            }
            return false;
        }
        default:
            return false;
    }
}

bool is_valid_branch_arm(
    BasicBlock* arm,
    BasicBlock* pred,
    BasicBlock* join,
    size_t max_insts
) {
    if (!arm || arm == pred || arm == join) return false;
    if (arm->predecessors().size() != 1 || arm->predecessors()[0] != pred) return false;
    if (arm->successors().size() != 1 || arm->successors()[0] != join) return false;

    Instruction* term = arm->terminator();
    if (!term || term->opcode() != Opcode::br) return false;
    if (term->branch_target().block != join) return false;
    if (term->branch_target().args.size() != join->param_count()) return false;

    size_t count = 0;
    for (Instruction* inst = arm->head(); inst != nullptr; inst = inst->next()) {
        if (inst == term) break;
        if (!is_safe_for_select_hoist(inst)) return false;
        if (++count > max_insts) return false;
    }
    return true;
}

void hoist_block_instructions(
    BasicBlock* from_bb,
    BasicBlock* to_bb,
    Instruction* insert_before,
    const std::vector<Value*>& branch_args,
    std::unordered_map<Value*, Value*>& val_map
) {
    // Map block parameters of from_bb to branch_args
    for (size_t i = 0; i < from_bb->param_count() && i < branch_args.size(); ++i) {
        val_map[from_bb->param(i)] = branch_args[i];
    }

    Instruction* cur = from_bb->head();
    Instruction* term = from_bb->terminator();

    while (cur && cur != term) {
        Instruction* next = cur->next();
        from_bb->remove_instruction(cur);

        // Remap operands
        for (size_t op_i = 0; op_i < cur->operand_count(); ++op_i) {
            Value* op = cur->operand(op_i);
            if (op) {
                auto it = val_map.find(op);
                if (it != val_map.end()) {
                    cur->set_operand(op_i, it->second);
                }
            }
        }

        to_bb->insert_before(cur, insert_before);
        cur = next;
    }
}

} // namespace

bool simplify_cfg_diamonds(Function& fn, size_t max_instructions_per_branch) {
    bool any_changed = false;
    bool progress = true;

    while (progress) {
        progress = false;
        fn.rebuild_cfg_predecessors();

        for (size_t b_idx = 0; b_idx < fn.blocks().size(); ++b_idx) {
            BasicBlock* pred_bb = fn.blocks()[b_idx];
            if (!pred_bb) continue;

            Instruction* term = pred_bb->terminator();
            if (!term || term->opcode() != Opcode::br_if) continue;

            Value* cond = term->operand(0);
            if (!cond) continue;

            BasicBlock* true_bb = term->true_target().block;
            BasicBlock* false_bb = term->false_target().block;
            if (!true_bb || !false_bb || true_bb == false_bb) continue;

            // Pattern 1: Full Diamond (pred -> true_bb/false_bb -> join_bb)
            Instruction* true_term = true_bb->terminator();
            Instruction* false_term = false_bb->terminator();

            if (true_term && false_term &&
                true_term->opcode() == Opcode::br && false_term->opcode() == Opcode::br) {

                BasicBlock* join_bb = true_term->branch_target().block;
                if (join_bb && join_bb == false_term->branch_target().block &&
                    join_bb != pred_bb && join_bb != true_bb && join_bb != false_bb) {

                    if (is_valid_branch_arm(true_bb, pred_bb, join_bb, max_instructions_per_branch) &&
                        is_valid_branch_arm(false_bb, pred_bb, join_bb, max_instructions_per_branch)) {

                        Builder b(*fn.parent());
                        b.set_function(&fn);

                        std::unordered_map<Value*, Value*> val_map;
                        hoist_block_instructions(true_bb, pred_bb, term, term->true_target().args, val_map);
                        hoist_block_instructions(false_bb, pred_bb, term, term->false_target().args, val_map);

                        b.position_before(term);

                        std::vector<Value*> join_args;
                        join_args.reserve(join_bb->param_count());

                        for (size_t p_i = 0; p_i < join_bb->param_count(); ++p_i) {
                            Value* v_true = true_term->branch_target().args[p_i];
                            Value* v_false = false_term->branch_target().args[p_i];

                            if (val_map.count(v_true)) v_true = val_map[v_true];
                            if (val_map.count(v_false)) v_false = val_map[v_false];

                            if (v_true == v_false) {
                                join_args.push_back(v_true);
                            } else {
                                Value* sel = b.build_select(cond, v_true, v_false);
                                join_args.push_back(sel);
                            }
                        }

                        // Replace br_if with unconditional br to join_bb
                        b.build_br(join_bb, join_args);
                        pred_bb->remove_instruction(term);

                        // Remove true_bb and false_bb from function
                        fn.remove_block(true_bb);
                        fn.remove_block(false_bb);

                        progress = true;
                        any_changed = true;
                        break;
                    }
                }
            }
        }
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
    }
    return any_changed;
}

} // namespace brass
