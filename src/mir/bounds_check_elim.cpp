#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

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

size_t count_uses(const Function& fn, const Value* val) {
    if (!val) return 0;
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == val) count++;
            }
            for (const Value* arg : inst->branch_target().args) {
                if (arg == val) count++;
            }
            for (const Value* arg : inst->true_target().args) {
                if (arg == val) count++;
            }
            for (const Value* arg : inst->false_target().args) {
                if (arg == val) count++;
            }
            for (const Value* arg : inst->default_target().args) {
                if (arg == val) count++;
            }
            for (const auto& sc : inst->switch_cases()) {
                for (const Value* arg : sc.target.args) {
                    if (arg == val) count++;
                }
            }
            for (const Value* sv : inst->state_map()) {
                if (sv == val) count++;
            }
        }
    }
    return count;
}

Opcode invert_comparison_opcode(Opcode op) noexcept {
    switch (op) {
        case Opcode::slt: return Opcode::sge;
        case Opcode::sle: return Opcode::sgt;
        case Opcode::sgt: return Opcode::sle;
        case Opcode::sge: return Opcode::slt;
        case Opcode::ult: return Opcode::uge;
        case Opcode::ule: return Opcode::ugt;
        case Opcode::ugt: return Opcode::ule;
        case Opcode::uge: return Opcode::ult;
        case Opcode::eq:  return Opcode::ne;
        case Opcode::ne:  return Opcode::eq;
        default: return op;
    }
}

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::patchable_const_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64 || def->opcode() == Opcode::patchable_const_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

Value* cast_to_i64(Builder& b, Value* v) {
    if (!v) return nullptr;
    if (v->type() == Type::i64()) return v;
    if (v->type() == Type::i32()) {
        int64_t c = 0;
        if (get_const_int(v, c)) {
            return b.build_iconst_i64(c);
        }
        return b.build_sext_i64(v);
    }
    return v;
}

Value* clone_invariant_def_to_preheader(
    Builder& b,
    Value* v,
    const LoopInfo& loop,
    BasicBlock* preheader
) {
    if (!v) return nullptr;
    if (loop.is_loop_invariant(v)) return v;

    if (v->is_instruction()) {
        Instruction* def = v->defining_instruction();
        if (!def) return v;

        if (def->opcode() == Opcode::zext_i64) {
            Value* in_val = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            return b.build_zext_i64(in_val);
        }
        if (def->opcode() == Opcode::sext_i64) {
            Value* in_val = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            return b.build_sext_i64(in_val);
        }
        if (def->opcode() == Opcode::load) {
            Value* base = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            return b.build_load(def->type(), base, def->offset());
        }
        if (def->opcode() == Opcode::and_) {
            Value* op0 = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            Value* op1 = clone_invariant_def_to_preheader(b, def->operand(1), loop, preheader);
            return b.build_and(op0, op1);
        }
        if (def->opcode() == Opcode::ne) {
            Value* op0 = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            Value* op1 = clone_invariant_def_to_preheader(b, def->operand(1), loop, preheader);
            return b.build_ne(op0, op1);
        }
        if (def->opcode() == Opcode::eq) {
            Value* op0 = clone_invariant_def_to_preheader(b, def->operand(0), loop, preheader);
            Value* op1 = clone_invariant_def_to_preheader(b, def->operand(1), loop, preheader);
            return b.build_eq(op0, op1);
        }
    }
    return v;
}

bool is_derived_from_iv(const Value* idx, const Value* iv) {
    if (!idx || !iv) return false;
    if (idx == iv) return true;
    if (idx->is_instruction()) {
        const Instruction* def = idx->defining_instruction();
        if (def && (def->opcode() == Opcode::sext_i64 || def->opcode() == Opcode::zext_i64)) {
            return is_derived_from_iv(def->operand(0), iv);
        }
        if (def && def->opcode() == Opcode::add) {
            int64_t c = 0;
            if (get_const_int(def->operand(1), c) && c >= 0) {
                return is_derived_from_iv(def->operand(0), iv);
            }
            if (get_const_int(def->operand(0), c) && c >= 0) {
                return is_derived_from_iv(def->operand(1), iv);
            }
        }
    }
    return false;
}

bool is_invariant_load_or_val(const Value* len, const LoopInfo& loop) {
    if (!len) return false;
    if (loop.is_loop_invariant(len)) return true;
    if (len->is_instruction()) {
        const Instruction* def = len->defining_instruction();
        if (!def) return false;
        if (def->opcode() == Opcode::zext_i64 || def->opcode() == Opcode::sext_i64) {
            return is_invariant_load_or_val(def->operand(0), loop);
        }
        if (def->opcode() == Opcode::load) {
            const Value* base = def->operand(0);
            return loop.is_loop_invariant(base);
        }
    }
    return false;
}

bool hoist_loop_bounds_checks(
    Function& fn,
    Module& mod,
    LoopAnalysis& loops,
    const DominatorTree& dom,
    const RangeAnalysisOptions& opts
) {
    (void)dom;
    bool changed = false;

    for (LoopInfo* loop : loops.post_order_loops()) {
        if (!loop || loop->latches().size() != 1) continue;
        BasicBlock* header = loop->header();
        BasicBlock* latch = loop->latches()[0];
        if (!header || !latch) continue;

        BasicBlock* preheader = loop->preheader();
        if (!preheader) {
            preheader = LoopAnalysis::ensure_preheader(fn, *loop);
        }
        if (!preheader) continue;

        Instruction* ph_term = preheader->terminator();
        Instruction* latch_term = latch->terminator();
        Instruction* hdr_term = header->terminator();
        if (!ph_term || !latch_term || !hdr_term || hdr_term->opcode() != Opcode::br_if) continue;

        BranchTarget* ph_bt = nullptr;
        if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
            ph_bt = &ph_term->branch_target();
        } else if (ph_term->opcode() == Opcode::br_if) {
            if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
            else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
        }
        if (!ph_bt || ph_bt->args.size() != header->param_count()) continue;

        BranchTarget* latch_bt = nullptr;
        if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
            latch_bt = &latch_term->branch_target();
        }
        if (!latch_bt || latch_bt->args.size() != header->param_count()) continue;

        Value* cond = hdr_term->operand(0);
        if (!cond || !cond->is_instruction()) continue;
        Instruction* cmp = cond->defining_instruction();
        if (!cmp || !is_comparison(cmp->opcode())) continue;

        bool body_is_true = loop->contains(hdr_term->true_target().block);
        bool body_is_false = loop->contains(hdr_term->false_target().block);
        if (body_is_true == body_is_false) continue;

        Opcode cmp_op = body_is_true ? cmp->opcode() : invert_comparison_opcode(cmp->opcode());
        Value* cmp_lhs = cmp->operand(0);
        Value* cmp_rhs = cmp->operand(1);

        // Find primary induction variable
        Value* primary_iv = nullptr;
        Value* init_val = nullptr;
        Value* limit_val = nullptr;

        for (size_t i = 0; i < header->param_count(); ++i) {
            Value* param = header->param(i);
            if (param == cmp_lhs && loop->is_loop_invariant(cmp_rhs)) {
                Value* step_v = latch_bt->args[i];
                if (step_v && step_v->is_instruction()) {
                    Instruction* sdef = step_v->defining_instruction();
                    if (sdef && sdef->opcode() == Opcode::add) {
                        Value* sop0 = sdef->operand(0);
                        Value* sop1 = sdef->operand(1);
                        Value* step_c_v = (sop0 == param) ? sop1 : ((sop1 == param) ? sop0 : nullptr);
                        int64_t step_c = 0;
                        if (step_c_v && get_const_int(step_c_v, step_c) && step_c > 0) {
                            primary_iv = param;
                            init_val = ph_bt->args[i];
                            limit_val = cmp_rhs;
                            break;
                        }
                    }
                }
            }
        }

        if (!primary_iv || !init_val || !limit_val) continue;

        // Search for hoistable bounds checks in the loop
        for (BasicBlock* bb : loop->blocks()) {
            if (!bb || bb == header) continue;

            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (cur->opcode() == Opcode::ult || cur->opcode() == Opcode::slt) {
                    Value* idx = cur->operand(0);
                    Value* len = cur->operand(1);

                    if (is_derived_from_iv(idx, primary_iv) && is_invariant_load_or_val(len, *loop)) {
                        // Found hoistable bounds check!
                        Builder b(mod);
                        b.position_before(ph_term);

                        Value* hoisted_len = clone_invariant_def_to_preheader(b, len, *loop, preheader);
                        Value* len_i64 = cast_to_i64(b, hoisted_len);
                        Value* N_i64 = cast_to_i64(b, limit_val);
                        Value* init_i64 = cast_to_i64(b, init_val);

                        Value* check_upper = nullptr;
                        if (cmp_op == Opcode::slt || cmp_op == Opcode::ult) {
                            check_upper = b.build_ule(N_i64, len_i64);
                        } else {
                            check_upper = b.build_ult(N_i64, len_i64);
                        }
                        Value* check_lower = b.build_sge(init_i64, b.build_iconst_i64(0));
                        Value* in_bounds = b.build_and(check_upper, check_lower);

                        // Look for related fastpath checks in bb (e.g. elements != 0, obj_valid)
                        Instruction* bb_term = bb->terminator();
                        if (bb_term && bb_term->opcode() == Opcode::br_if) {
                            Value* branch_cond = bb_term->operand(0);
                            if (branch_cond && branch_cond->is_instruction()) {
                                Instruction* bc_def = branch_cond->defining_instruction();
                                if (bc_def && bc_def->opcode() == Opcode::and_) {
                                    Value* other_check = (bc_def->operand(0) == cur->result()) ? bc_def->operand(1) :
                                                         ((bc_def->operand(1) == cur->result()) ? bc_def->operand(0) : nullptr);
                                    if (other_check && is_invariant_load_or_val(other_check, *loop)) {
                                        Value* hoisted_other = clone_invariant_def_to_preheader(b, other_check, *loop, preheader);
                                        in_bounds = b.build_and(in_bounds, hoisted_other);
                                    }
                                }
                            }
                        }

                        // Also check preceding block for initial_guard (is_valid_obj && is_valid_idx)
                        for (BasicBlock* pred : bb->predecessors()) {
                            if (pred && loop->contains(pred)) {
                                Instruction* pterm = pred->terminator();
                                if (pterm && pterm->opcode() == Opcode::br_if && pterm->true_target().block == bb) {
                                    Value* ig = pterm->operand(0);
                                    if (ig && is_invariant_load_or_val(ig, *loop)) {
                                        Value* hoisted_ig = clone_invariant_def_to_preheader(b, ig, *loop, preheader);
                                        in_bounds = b.build_and(in_bounds, hoisted_ig);
                                        // Hoist initial_guard branch into unconditional jump
                                        pterm->set_opcode(Opcode::br);
                                        pterm->set_branch_target(pterm->true_target());
                                        pterm->operands().clear();
                                        pterm->true_target() = BranchTarget();
                                        pterm->false_target() = BranchTarget();
                                    }
                                }
                            }
                        }

                        // Emit preheader guard
                        b.build_guard(in_bounds, "@exit_stub");

                        // Eliminate per-iteration bounds check inside loop
                        b.position_before(cur);
                        Value* true_v = b.build_iconst_i32(1);
                        replace_all_uses(fn, cur->result(), true_v);

                        if (bb_term && bb_term->opcode() == Opcode::br_if) {
                            bb_term->set_opcode(Opcode::br);
                            bb_term->set_branch_target(bb_term->true_target());
                            bb_term->operands().clear();
                            bb_term->true_target() = BranchTarget();
                            bb_term->false_target() = BranchTarget();
                        }

                        if (opts.stats) {
                            opts.stats->bounds_checks_hoisted++;
                        }
                        changed = true;
                    }
                }
                cur = next;
            }
        }
    }

    return changed;
}

bool eliminate_local_bounds_checks(
    Function& fn,
    Module& mod,
    const RangeAnalysis& ra,
    const RangeAnalysisOptions& opts
) {
    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();

            // Check 1: Redundant comparison (e.g. ult %idx, %len)
            if (cur->opcode() == Opcode::ult || cur->opcode() == Opcode::slt) {
                Value* idx = cur->operand(0);
                Value* len = cur->operand(1);
                ValueRange r_idx = ra.get_range_at(idx, bb);
                ValueRange r_len = ra.get_range_at(len, bb);

                if (r_idx.is_non_negative() && r_idx.max_val < r_len.min_val) {
                    // Statically proven true!
                    Builder b(mod);
                    b.position_before(cur);
                    Value* one = b.build_iconst_i32(1);
                    replace_all_uses(fn, cur->result(), one);

                    // If followed by br_if using this condition, simplify branch directly
                    Instruction* term = bb->terminator();
                    if (term && term->opcode() == Opcode::br_if && term->operand(0) == one) {
                        term->set_opcode(Opcode::br);
                        term->set_branch_target(term->true_target());
                        term->operands().clear();
                        term->true_target() = BranchTarget();
                        term->false_target() = BranchTarget();
                        if (opts.stats) opts.stats->branches_folded++;
                    }

                    if (opts.stats) opts.stats->bounds_checks_eliminated++;
                    changed = true;
                } else if (r_idx.max_val < 0 || (r_len.max_val < INT64_MAX && r_idx.min_val >= r_len.max_val)) {
                    // Statically proven false!
                    Builder b(mod);
                    b.position_before(cur);
                    Value* zero = b.build_iconst_i32(0);
                    replace_all_uses(fn, cur->result(), zero);

                    Instruction* term = bb->terminator();
                    if (term && term->opcode() == Opcode::br_if && term->operand(0) == zero) {
                        term->set_opcode(Opcode::br);
                        term->set_branch_target(term->false_target());
                        term->operands().clear();
                        term->true_target() = BranchTarget();
                        term->false_target() = BranchTarget();
                        if (opts.stats) opts.stats->branches_folded++;
                    }

                    if (opts.stats) opts.stats->bounds_checks_eliminated++;
                    changed = true;
                }
            }

            // Check 2: Redundant Guard
            else if (cur->opcode() == Opcode::guard) {
                ValueRange r_cond = ra.get_range_at(cur->operand(0), bb);
                if (r_cond.is_constant() && r_cond.min_val != 0) {
                    bb->remove_instruction(cur);
                    if (opts.stats) opts.stats->guards_eliminated++;
                    changed = true;
                }
            }

            // Check 3: Redundant br_if
            else if (cur->opcode() == Opcode::br_if) {
                ValueRange r_cond = ra.get_range_at(cur->operand(0), bb);
                if (r_cond.is_constant()) {
                    if (r_cond.min_val != 0) {
                        cur->set_opcode(Opcode::br);
                        cur->set_branch_target(cur->true_target());
                        cur->operands().clear();
                        cur->true_target() = BranchTarget();
                        cur->false_target() = BranchTarget();
                    } else {
                        cur->set_opcode(Opcode::br);
                        cur->set_branch_target(cur->false_target());
                        cur->operands().clear();
                        cur->true_target() = BranchTarget();
                        cur->false_target() = BranchTarget();
                    }
                    if (opts.stats) opts.stats->branches_folded++;
                    changed = true;
                }
            }

            cur = next;
        }
    }

    return changed;
}

} // namespace

bool run_bounds_check_elimination(Function& fn, Module& mod, const RangeAnalysisOptions& opts) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;
    bool any_changed = false;

    // Step 1: Loop Bounds Check Hoisting
    if (opts.enable_hoisting) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);
        bool hoisted = hoist_loop_bounds_checks(fn, mod, loops, dom, opts);
        if (hoisted) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
        }
    }

    // Step 2: Local / Dominator-Based Bounds Check Elimination
    if (opts.enable_bce) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);
        RangeAnalysis ra(fn, dom, loops);
        bool local_elim = eliminate_local_bounds_checks(fn, mod, ra, opts);
        if (local_elim) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
        }
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
        cfg_simplify_function(fn);
        fn.rebuild_cfg_predecessors();

        // Dead code elimination pass for unused constants/instructions
        bool dce_progress = true;
        while (dce_progress) {
            dce_progress = false;
            for (BasicBlock* bb : fn.blocks()) {
                if (!bb) continue;
                Instruction* cur = bb->head();
                while (cur) {
                    Instruction* next = cur->next();
                    if (!cur->has_side_effects() && cur->produces_value()) {
                        if (count_uses(fn, cur->result()) == 0) {
                            bb->remove_instruction(cur);
                            dce_progress = true;
                        }
                    }
                    cur = next;
                }
            }
        }
        fn.rebuild_cfg_predecessors();
    }

    return any_changed;
}

bool run_bounds_check_elimination(Module& mod, const RangeAnalysisOptions& opts) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= run_bounds_check_elimination(*fn, mod, opts);
        }
    }
    return changed;
}

} // namespace brass
