#include "loop_unswitch_internal.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/printer.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace brass {

bool transform_unswitch_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    BasicBlock* cond_bb,
    Instruction* br_if_inst,
    const LoopUnswitchOptions& opts
) {
    (void)opts;
    if (!cond_bb || !br_if_inst || br_if_inst->opcode() != Opcode::br_if) {
        return false;
    }

    Value* cond = br_if_inst->operand(0);
    if (!cond) return false;

    // Ensure dedicated preheader
    BasicBlock* preheader = loop.preheader();
    if (!preheader) {
        preheader = LoopAnalysis::ensure_preheader(fn, loop);
    }
    if (!preheader || !preheader->terminator()) {
        return false;
    }

    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() != Opcode::br || ph_term->branch_target().block != loop.header()) {
        return false;
    }

    // Condition must dominate preheader
    if (cond->is_instruction()) {
        BasicBlock* def_bb = cond->defining_instruction()->parent();
        if (def_bb != preheader && !dom.dominates(def_bb, preheader)) {
            return false;
        }
    } else if (cond->is_block_param()) {
        BasicBlock* def_bb = cond->defining_block();
        if (def_bb != preheader && !dom.dominates(def_bb, preheader)) {
            return false;
        }
    }

    std::vector<BasicBlock*> blocks_to_clone = loop.blocks();
    std::unordered_set<BasicBlock*> clone_set(blocks_to_clone.begin(), blocks_to_clone.end());

    // Collect all definitions in blocks_to_clone
    std::unordered_set<const Value*> cloned_defs;
    auto collect_defs = [&]() {
        cloned_defs.clear();
        for (BasicBlock* bb : blocks_to_clone) {
            if (!bb) continue;
            for (size_t i = 0; i < bb->param_count(); ++i) {
                cloned_defs.insert(bb->param(i));
            }
            for (Instruction* inst : *bb) {
                if (inst && inst->produces_value()) {
                    cloned_defs.insert(inst->result());
                }
            }
        }
    };
    collect_defs();

    // Transitive closure: if any block outside clone_set directly uses a value defined
    // in clone_set without block parameters, it must also be cloned into L_false so that
    // SSA dominance is strictly preserved in both branches.
    bool added_block = true;
    while (added_block) {
        added_block = false;
        for (BasicBlock* fn_bb : fn.blocks()) {
            if (!fn_bb || clone_set.count(fn_bb)) continue;

            bool uses_cloned_def = false;
            for (Instruction* inst : *fn_bb) {
                if (!inst) continue;
                for (Value* op : inst->operands()) {
                    if (op && cloned_defs.count(op)) { uses_cloned_def = true; break; }
                }
                if (uses_cloned_def) break;
                for (Value* sv : inst->state_map()) {
                    if (sv && cloned_defs.count(sv)) { uses_cloned_def = true; break; }
                }
                if (uses_cloned_def) break;

                // Check branch target args
                if (inst->opcode() == Opcode::br) {
                    for (Value* a : inst->branch_target().args) {
                        if (a && cloned_defs.count(a)) { uses_cloned_def = true; break; }
                    }
                } else if (inst->opcode() == Opcode::br_if) {
                    for (Value* a : inst->true_target().args) {
                        if (a && cloned_defs.count(a)) { uses_cloned_def = true; break; }
                    }
                    for (Value* a : inst->false_target().args) {
                        if (a && cloned_defs.count(a)) { uses_cloned_def = true; break; }
                    }
                } else if (inst->opcode() == Opcode::switch_) {
                    for (Value* a : inst->default_target().args) {
                        if (a && cloned_defs.count(a)) { uses_cloned_def = true; break; }
                    }
                    for (auto& sc : inst->switch_cases()) {
                        for (Value* a : sc.target.args) {
                            if (a && cloned_defs.count(a)) { uses_cloned_def = true; break; }
                        }
                    }
                }
                if (uses_cloned_def) break;
            }

            if (uses_cloned_def) {
                clone_set.insert(fn_bb);
                blocks_to_clone.push_back(fn_bb);
                collect_defs();
                added_block = true;
                break;
            }
        }
    }

    std::vector<Value*> init_args = ph_term->branch_target().args;

    // 1. Create duplicate blocks for L_false
    std::unordered_map<BasicBlock*, BasicBlock*> block_map;
    std::unordered_map<const Value*, Value*> val_map;

    for (BasicBlock* bb : blocks_to_clone) {
        if (!bb) continue;
        std::string new_name = std::string(bb->name()) + "_unsw_f";
        BasicBlock* cloned_bb = fn.parent()
            ? fn.parent()->arena().make<BasicBlock>(fn.next_block_id(), fn.parent()->string_pool().intern(new_name))
            : new BasicBlock(fn.next_block_id(), new_name);
        cloned_bb->set_parent(&fn);
        fn.append_block(cloned_bb);
        block_map[bb] = cloned_bb;

        // Clone block parameters
        for (size_t p_i = 0; p_i < bb->param_count(); ++p_i) {
            Value* orig_p = bb->param(p_i);
            Value* cloned_p = fn.parent()
                ? fn.parent()->arena().make<Value>(fn.next_value_id(), orig_p->type(), ValueKind::BlockParam)
                : new Value(fn.next_value_id(), orig_p->type(), ValueKind::BlockParam);
            cloned_bb->add_param(cloned_p);
            val_map[orig_p] = cloned_p;
        }
    }

    // 2a. Create duplicate instructions and map their result values
    for (BasicBlock* bb : blocks_to_clone) {
        if (!bb) continue;
        BasicBlock* dst_bb = block_map[bb];

        for (Instruction* inst : *bb) {
            if (!inst || inst->is_terminator()) break;

            Instruction* cloned = fn.parent()
                ? fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type())
                : new Instruction(inst->opcode(), inst->type());

            cloned->set_imm_i64(inst->imm_i64());
            cloned->set_imm_f64(inst->imm_f64());
            cloned->set_scale(inst->scale());
            cloned->set_offset(inst->offset());
            cloned->set_memory_type(inst->memory_type());
            if (!inst->symbol().empty() && fn.parent()) {
                cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));
            }
            if (!inst->extra_symbol().empty() && fn.parent()) {
                cloned->set_extra_symbol(fn.parent()->string_pool().intern(inst->extra_symbol()));
            }

            if (inst->produces_value()) {
                Value* res = fn.parent()
                    ? fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult)
                    : new Value(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
                res->set_defining_instruction(cloned);
                cloned->set_result(res);
                val_map[inst->result()] = res;
            }

            dst_bb->append_instruction(cloned);
        }
    }

    // 2b. Populate operands and state values for all cloned instructions
    for (BasicBlock* bb : blocks_to_clone) {
        if (!bb) continue;
        BasicBlock* dst_bb = block_map[bb];

        auto orig_it = bb->begin();
        auto cloned_it = dst_bb->begin();

        while (orig_it != bb->end() && cloned_it != dst_bb->end()) {
            Instruction* orig_inst = *orig_it;
            Instruction* cloned_inst = *cloned_it;
            if (orig_inst->is_terminator() || cloned_inst->is_terminator()) break;

            for (Value* op : orig_inst->operands()) {
                if (!op) continue;
                auto it = val_map.find(op);
                cloned_inst->add_operand(it != val_map.end() ? it->second : op);
            }

            for (Value* sv : orig_inst->state_map()) {
                if (!sv) continue;
                auto it = val_map.find(sv);
                cloned_inst->add_state_value(it != val_map.end() ? it->second : sv);
            }

            ++orig_it;
            ++cloned_it;
        }
    }

    // 3. Clone terminators for L_false
    auto remap_val = [&](Value* v) -> Value* {
        if (!v) return nullptr;
        auto it = val_map.find(v);
        return it != val_map.end() ? it->second : v;
    };

    auto remap_target = [&](const BranchTarget& tgt) -> BranchTarget {
        BasicBlock* blk = block_map.count(tgt.block) ? block_map[tgt.block] : tgt.block;
        std::vector<Value*> args;
        args.reserve(tgt.args.size());
        for (Value* a : tgt.args) {
            args.push_back(remap_val(a));
        }
        return BranchTarget(blk, std::move(args));
    };

    for (BasicBlock* bb : blocks_to_clone) {
        if (!bb) continue;
        BasicBlock* dst_bb = block_map[bb];
        Instruction* term = bb->terminator();
        if (!term) continue;

        if (bb == cond_bb) {
            // In false clone, take false branch unconditionally!
            BranchTarget f_target = remap_target(term->false_target());
            Instruction* br = fn.parent()
                ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
                : new Instruction(Opcode::br, Type::void_type());
            br->set_branch_target(std::move(f_target));
            dst_bb->append_instruction(br);
        } else {
            if (term->opcode() == Opcode::br) {
                Instruction* br = fn.parent()
                    ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
                    : new Instruction(Opcode::br, Type::void_type());
                br->set_branch_target(remap_target(term->branch_target()));
                dst_bb->append_instruction(br);
            } else if (term->opcode() == Opcode::br_if) {
                Instruction* br_if = fn.parent()
                    ? fn.parent()->arena().make<Instruction>(Opcode::br_if, Type::void_type())
                    : new Instruction(Opcode::br_if, Type::void_type());
                br_if->add_operand(remap_val(term->operand(0)));
                br_if->set_true_target(remap_target(term->true_target()));
                br_if->set_false_target(remap_target(term->false_target()));
                dst_bb->append_instruction(br_if);
            } else if (term->opcode() == Opcode::switch_) {
                Instruction* sw = fn.parent()
                    ? fn.parent()->arena().make<Instruction>(Opcode::switch_, Type::void_type())
                    : new Instruction(Opcode::switch_, Type::void_type());
                sw->add_operand(remap_val(term->operand(0)));
                sw->set_default_target(remap_target(term->default_target()));
                for (const auto& sc : term->switch_cases()) {
                    sw->add_switch_case(sc.value, remap_target(sc.target));
                }
                dst_bb->append_instruction(sw);
            } else if (term->opcode() == Opcode::ret) {
                Instruction* ret = fn.parent()
                    ? fn.parent()->arena().make<Instruction>(Opcode::ret, Type::void_type())
                    : new Instruction(Opcode::ret, Type::void_type());
                if (term->operand_count() > 0) {
                    ret->add_operand(remap_val(term->operand(0)));
                }
                dst_bb->append_instruction(ret);
            } else {
                Instruction* other = fn.parent()
                    ? fn.parent()->arena().make<Instruction>(term->opcode(), term->type())
                    : new Instruction(term->opcode(), term->type());
                dst_bb->append_instruction(other);
            }
        }
    }

    // 4. In L_true (cond_bb), replace invariant br_if with unconditional br to true target
    BranchTarget t_target = br_if_inst->true_target();
    Instruction* br_true = fn.parent()
        ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
        : new Instruction(Opcode::br, Type::void_type());
    br_true->set_branch_target(t_target);
    cond_bb->remove_instruction(br_if_inst);
    cond_bb->append_instruction(br_true);

    // 5. In preheader, replace br with br_if %cond to L_true.header or L_false.header
    Instruction* ph_br_if = fn.parent()
        ? fn.parent()->arena().make<Instruction>(Opcode::br_if, Type::void_type())
        : new Instruction(Opcode::br_if, Type::void_type());
    ph_br_if->add_operand(cond);
    ph_br_if->set_true_target(BranchTarget(loop.header(), init_args));
    ph_br_if->set_false_target(BranchTarget(block_map[loop.header()], init_args));

    preheader->remove_instruction(ph_term);
    preheader->append_instruction(ph_br_if);

    // 6. Rebuild CFG & run CFG simplification
    fn.rebuild_cfg_predecessors();
    CfgSimplifyOptions cfg_opts;
    cfg_opts.enable_param_elimination = false;
    cfg_simplify_function(fn, cfg_opts);

    return true;
}

} // namespace brass
