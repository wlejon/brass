#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/builder.hpp>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace brass {

namespace {

void for_each_target_pointing_to(Instruction* term, BasicBlock* target_bb, auto&& fn) {
    if (!term || !target_bb) return;
    if (term->opcode() == Opcode::br) {
        if (term->branch_target().block == target_bb) {
            fn(term->branch_target());
        }
    } else if (term->opcode() == Opcode::br_if) {
        if (term->true_target().block == target_bb) {
            fn(term->true_target());
        }
        if (term->false_target().block == target_bb) {
            fn(term->false_target());
        }
    } else if (term->opcode() == Opcode::switch_) {
        if (term->default_target().block == target_bb) {
            fn(term->default_target());
        }
        for (auto& sc : term->switch_cases()) {
            if (sc.target.block == target_bb) {
                fn(sc.target);
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
            if (inst->opcode() == Opcode::br) {
                for (Value* arg : inst->branch_target().args) {
                    if (arg == val) count++;
                }
            } else if (inst->opcode() == Opcode::br_if) {
                for (Value* arg : inst->true_target().args) {
                    if (arg == val) count++;
                }
                for (Value* arg : inst->false_target().args) {
                    if (arg == val) count++;
                }
            } else if (inst->opcode() == Opcode::switch_) {
                for (Value* arg : inst->default_target().args) {
                    if (arg == val) count++;
                }
                for (const auto& sc : inst->switch_cases()) {
                    for (Value* arg : sc.target.args) {
                        if (arg == val) count++;
                    }
                }
            }
            for (Value* sv : inst->state_map()) {
                if (sv == val) count++;
            }
        }
    }
    return count;
}

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) {
                    inst->set_operand(i, new_val);
                }
            }
            if (inst->opcode() == Opcode::br) {
                for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                    if (inst->branch_target().args[i] == old_val) {
                        inst->branch_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::br_if) {
                for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                    if (inst->true_target().args[i] == old_val) {
                        inst->true_target().args[i] = new_val;
                    }
                }
                for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                    if (inst->false_target().args[i] == old_val) {
                        inst->false_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::switch_) {
                for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                    if (inst->default_target().args[i] == old_val) {
                        inst->default_target().args[i] = new_val;
                    }
                }
                for (auto& sc : inst->switch_cases()) {
                    for (size_t i = 0; i < sc.target.args.size(); ++i) {
                        if (sc.target.args[i] == old_val) {
                            sc.target.args[i] = new_val;
                        }
                    }
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) {
                    inst->state_map()[i] = new_val;
                }
            }
        }
    }
}

class CfgSimplifier {
public:
    CfgSimplifier(Function& fn, const CfgSimplifyOptions& options)
        : fn_(fn), options_(options) {
        for (const auto& rp : fn_.resume_points()) {
            if (rp.second) {
                resume_targets_.insert(rp.second);
            }
        }
    }

    bool run() {
        if (fn_.blocks().empty() || !fn_.entry_block()) {
            return false;
        }

        bool overall_changed = false;
        size_t iter = 0;
        while (iter < options_.max_iterations) {
            bool iter_changed = false;

            if (options_.enable_branch_simplify) {
                iter_changed |= simplify_branches();
            }

            if (options_.enable_dead_block_removal) {
                iter_changed |= remove_unreachable_blocks();
            }

            if (options_.enable_param_elimination) {
                iter_changed |= eliminate_dead_block_params();
            }

            if (options_.enable_trampoline_elimination) {
                iter_changed |= eliminate_trampolines();
            }

            if (options_.enable_dead_block_removal) {
                iter_changed |= remove_unreachable_blocks();
            }

            if (options_.enable_block_merge) {
                iter_changed |= merge_linear_blocks();
            }

            fn_.rebuild_cfg_predecessors();

            if (!iter_changed) {
                break;
            }
            overall_changed = true;
            iter++;
        }

        return overall_changed;
    }

private:
    Function& fn_;
    const CfgSimplifyOptions& options_;
    std::unordered_set<BasicBlock*> resume_targets_;

    bool simplify_branches() {
        bool changed = false;

        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            Instruction* term = bb->terminator();
            if (!term) continue;

            if (term->opcode() == Opcode::br_if) {
                Value* cond = term->operand(0);
                Instruction* def = cond ? cond->defining_instruction() : nullptr;

                if (def && def->opcode() == Opcode::iconst_i32) {
                    int32_t val = def->imm_i32();
                    BranchTarget target = (val != 0) ? term->true_target() : term->false_target();
                    term->set_opcode(Opcode::br);
                    term->set_branch_target(std::move(target));
                    term->operands().clear();
                    if (options_.stats) options_.stats->branches_simplified++;
                    changed = true;
                } else if (term->true_target().block == term->false_target().block &&
                           term->true_target().args == term->false_target().args) {
                    BranchTarget target = term->true_target();
                    term->set_opcode(Opcode::br);
                    term->set_branch_target(std::move(target));
                    term->operands().clear();
                    if (options_.stats) options_.stats->branches_simplified++;
                    changed = true;
                }
            } else if (term->opcode() == Opcode::switch_) {
                Value* sel = term->operand(0);
                Instruction* def = sel ? sel->defining_instruction() : nullptr;

                if (def && (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::iconst_i64)) {
                    int64_t val = (def->opcode() == Opcode::iconst_i32) ? def->imm_i32() : def->imm_i64();
                    BranchTarget chosen = term->default_target();
                    for (const auto& sc : term->switch_cases()) {
                        if (sc.value == val) {
                            chosen = sc.target;
                            break;
                        }
                    }
                    term->set_opcode(Opcode::br);
                    term->set_branch_target(std::move(chosen));
                    term->switch_cases().clear();
                    term->operands().clear();
                    if (options_.stats) options_.stats->branches_simplified++;
                    changed = true;
                }
            }
        }

        return changed;
    }

    bool remove_unreachable_blocks() {
        bool changed = false;
        std::unordered_set<BasicBlock*> reachable;
        std::vector<BasicBlock*> worklist;

        BasicBlock* entry = fn_.entry_block();
        if (entry) {
            reachable.insert(entry);
            worklist.push_back(entry);
        }

        for (BasicBlock* r_target : resume_targets_) {
            if (r_target && reachable.insert(r_target).second) {
                worklist.push_back(r_target);
            }
        }

        while (!worklist.empty()) {
            BasicBlock* curr = worklist.back();
            worklist.pop_back();

            for (BasicBlock* succ : curr->successors()) {
                if (succ && reachable.insert(succ).second) {
                    worklist.push_back(succ);
                }
            }
        }

        std::vector<BasicBlock*> to_remove;
        for (BasicBlock* bb : fn_.blocks()) {
            if (bb && reachable.find(bb) == reachable.end()) {
                to_remove.push_back(bb);
            }
        }

        for (BasicBlock* bb : to_remove) {
            for (BasicBlock* succ : bb->successors()) {
                if (succ) {
                    succ->remove_predecessor(bb);
                }
            }
            fn_.remove_block(bb);
            if (options_.stats) options_.stats->blocks_removed++;
            changed = true;
        }

        return changed;
    }

    bool eliminate_dead_block_params() {
        bool changed = false;

        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || bb == fn_.entry_block() || resume_targets_.find(bb) != resume_targets_.end()) {
                continue;
            }
            if (bb->param_count() == 0) continue;

            // Case A: Block has a single predecessor
            if (bb->predecessors().size() == 1) {
                BasicBlock* pred = bb->predecessors().front();
                Instruction* term = pred ? pred->terminator() : nullptr;
                if (term) {
                    std::vector<BranchTarget*> targets_to_bb;
                    if (term->opcode() == Opcode::br && term->branch_target().block == bb) {
                        targets_to_bb.push_back(&term->branch_target());
                    } else if (term->opcode() == Opcode::br_if) {
                        if (term->true_target().block == bb) targets_to_bb.push_back(&term->true_target());
                        if (term->false_target().block == bb) targets_to_bb.push_back(&term->false_target());
                    } else if (term->opcode() == Opcode::switch_) {
                        if (term->default_target().block == bb) targets_to_bb.push_back(&term->default_target());
                        for (auto& sc : term->switch_cases()) {
                            if (sc.target.block == bb) targets_to_bb.push_back(&sc.target);
                        }
                    }

                    if (targets_to_bb.size() == 1) {
                        BranchTarget* bt = targets_to_bb.front();
                        if (bt->args.size() == bb->param_count()) {
                            for (size_t i = 0; i < bb->param_count(); ++i) {
                                replace_all_uses(fn_, bb->param(i), bt->args[i]);
                            }
                            bt->args.clear();
                            if (options_.stats) options_.stats->params_removed += bb->param_count();
                            bb->params().clear();
                            changed = true;
                            continue;
                        }
                    }
                }
            }

            // Case B: Filter unused parameters or parameters where all incoming edges pass the same value
            std::vector<bool> keep_param(bb->param_count(), true);
            bool any_param_removed = false;

            for (size_t i = 0; i < bb->param_count(); ++i) {
                Value* param = bb->param(i);
                size_t use_count = count_uses(fn_, param);
                if (use_count == 0) {
                    keep_param[i] = false;
                    any_param_removed = true;
                    continue;
                }

                Value* common_val = nullptr;
                bool all_same = !bb->predecessors().empty();
                for (BasicBlock* pred : bb->predecessors()) {
                    Instruction* term = pred ? pred->terminator() : nullptr;
                    if (!term) { all_same = false; break; }
                    for_each_target_pointing_to(term, bb, [&](BranchTarget& bt) {
                        if (i < bt.args.size()) {
                            if (!common_val) {
                                common_val = bt.args[i];
                            } else if (common_val != bt.args[i]) {
                                all_same = false;
                            }
                        } else {
                            all_same = false;
                        }
                    });
                    if (!all_same) break;
                }

                if (all_same && common_val && common_val != param) {
                    replace_all_uses(fn_, param, common_val);
                    keep_param[i] = false;
                    any_param_removed = true;
                }
            }

            if (any_param_removed) {
                std::vector<Value*> new_params;
                for (size_t i = 0; i < bb->param_count(); ++i) {
                    if (keep_param[i]) {
                        new_params.push_back(bb->param(i));
                    } else {
                        if (options_.stats) options_.stats->params_removed++;
                    }
                }
                bb->params() = std::move(new_params);

                for (BasicBlock* pred : bb->predecessors()) {
                    Instruction* term = pred ? pred->terminator() : nullptr;
                    if (!term) continue;
                    for_each_target_pointing_to(term, bb, [&](BranchTarget& bt) {
                        std::vector<Value*> new_args;
                        for (size_t i = 0; i < bt.args.size(); ++i) {
                            if (i < keep_param.size() && keep_param[i]) {
                                new_args.push_back(bt.args[i]);
                            }
                        }
                        bt.args = std::move(new_args);
                    });
                }
                changed = true;
            }
        }

        return changed;
    }

    bool merge_linear_blocks() {
        bool changed = false;

        for (size_t i = 0; i < fn_.block_count(); ++i) {
            BasicBlock* P = fn_.blocks()[i];
            if (!P) continue;
            Instruction* p_term = P->terminator();
            if (!p_term || p_term->opcode() != Opcode::br) continue;

            BasicBlock* S = p_term->branch_target().block;
            if (!S || S == P || S == fn_.entry_block() || resume_targets_.find(S) != resume_targets_.end()) {
                continue;
            }

            if (S->predecessors().size() == 1 && P->successors().size() == 1) {
                // Forward any block parameters of S
                const auto& branch_args = p_term->branch_target().args;
                for (size_t p_i = 0; p_i < S->param_count(); ++p_i) {
                    Value* arg = (p_i < branch_args.size()) ? branch_args[p_i] : nullptr;
                    if (arg) {
                        replace_all_uses(fn_, S->param(p_i), arg);
                    }
                }
                S->params().clear();

                // Remove P's terminator
                P->remove_instruction(p_term);

                // Move all instructions from S to P
                Instruction* cur = S->head();
                while (cur) {
                    Instruction* next = cur->next();
                    S->remove_instruction(cur);
                    cur->set_parent(P);
                    P->append_instruction(cur);
                    cur = next;
                }

                // Update predecessor references in S's successors
                for (BasicBlock* succ : S->successors()) {
                    if (succ) {
                        for (BasicBlock*& pred_entry : succ->predecessors()) {
                            if (pred_entry == S) {
                                pred_entry = P;
                            }
                        }
                    }
                }

                fn_.remove_block(S);
                if (options_.stats) options_.stats->blocks_merged++;
                changed = true;
                --i; // Re-check P with its new instructions/terminator
            }
        }

        return changed;
    }

    bool eliminate_trampolines() {
        bool changed = false;

        for (BasicBlock* T : fn_.blocks()) {
            if (!T || T == fn_.entry_block() || resume_targets_.find(T) != resume_targets_.end()) {
                continue;
            }
            if (!T->head() || T->head() != T->terminator()) {
                continue; // Must contain only the terminator
            }
            Instruction* term = T->terminator();
            if (term->opcode() != Opcode::br) {
                continue;
            }

            BasicBlock* Target = term->branch_target().block;
            if (!Target || Target == T) {
                continue;
            }

            // Identity forwarding or empty forwarding trampoline
            bool is_identity = (T->param_count() > 0 && term->branch_target().args == T->params());
            bool is_empty_forward = (T->param_count() == 0 && term->branch_target().args.empty());

            if (is_identity || is_empty_forward) {
                std::vector<BasicBlock*> preds = T->predecessors();
                for (BasicBlock* P : preds) {
                    if (P == T) continue;
                    Instruction* p_term = P ? P->terminator() : nullptr;
                    if (!p_term) continue;

                    for_each_target_pointing_to(p_term, T, [&](BranchTarget& bt) {
                        bt.block = Target;
                    });
                    Target->add_predecessor(P);
                }
                T->clear_predecessors();
                if (options_.stats) options_.stats->trampolines_eliminated++;
                changed = true;
            }
        }

        return changed;
    }
};

} // namespace

bool cfg_simplify_function(Function& fn) {
    CfgSimplifyOptions opts;
    return cfg_simplify_function(fn, opts);
}

bool cfg_simplify_function(Function& fn, const CfgSimplifyOptions& options) {
    CfgSimplifier simplifier(fn, options);
    return simplifier.run();
}

bool cfg_simplify_module(Module& mod) {
    CfgSimplifyOptions opts;
    return cfg_simplify_module(mod, opts);
}

bool cfg_simplify_module(Module& mod, const CfgSimplifyOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= cfg_simplify_function(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
