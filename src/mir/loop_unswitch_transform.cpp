#include "loop_unswitch_internal.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
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

    std::unordered_set<BasicBlock*> loop_block_set(loop.blocks().begin(), loop.blocks().end());

    // Collect all definitions inside the loop (block parameters and instruction results)
    std::vector<Value*> loop_defs;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        for (size_t p_i = 0; p_i < bb->param_count(); ++p_i) {
            loop_defs.push_back(bb->param(p_i));
        }
        for (Instruction* inst : *bb) {
            if (inst && inst->produces_value()) {
                loop_defs.push_back(inst->result());
            }
        }
    }

    // Identify all exit blocks of the loop
    std::vector<BasicBlock*> exit_blocks;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        Instruction* term = bb->terminator();
        if (!term) continue;
        auto add_exit = [&](BasicBlock* target_bb) {
            if (target_bb && !loop_block_set.count(target_bb)) {
                if (std::find(exit_blocks.begin(), exit_blocks.end(), target_bb) == exit_blocks.end()) {
                    exit_blocks.push_back(target_bb);
                }
            }
        };
        if (term->opcode() == Opcode::br) {
            add_exit(term->branch_target().block);
        } else if (term->opcode() == Opcode::br_if) {
            add_exit(term->true_target().block);
            add_exit(term->false_target().block);
        } else if (term->opcode() == Opcode::switch_) {
            add_exit(term->default_target().block);
            for (auto& sc : term->switch_cases()) add_exit(sc.target.block);
        }
    }

    // For any loop def used outside the loop, ensure LCSSA by passing it through exit blocks
    for (Value* val : loop_defs) {
        if (!val) continue;

        bool used_outside = false;
        for (BasicBlock* fn_bb : fn.blocks()) {
            if (!fn_bb || loop_block_set.count(fn_bb)) continue;
            for (Instruction* inst : *fn_bb) {
                if (!inst) continue;
                for (size_t i = 0; i < inst->operand_count(); ++i) {
                    if (inst->operand(i) == val) { used_outside = true; break; }
                }
                for (size_t i = 0; i < inst->state_map().size(); ++i) {
                    if (inst->state_map()[i] == val) { used_outside = true; break; }
                }
                if (used_outside) break;
            }
            if (used_outside) break;
        }

        if (!used_outside) continue;

        for (BasicBlock* exit_bb : exit_blocks) {
            Value* exit_param = fn.parent()
                ? fn.parent()->arena().make<Value>(fn.next_value_id(), val->type(), ValueKind::BlockParam)
                : new Value(fn.next_value_id(), val->type(), ValueKind::BlockParam);
            exit_bb->add_param(exit_param);

            for (BasicBlock* bb : loop.blocks()) {
                if (!bb) continue;
                Instruction* term = bb->terminator();
                if (!term) continue;
                auto add_arg = [&](BranchTarget& bt) {
                    if (bt.block == exit_bb) {
                        bt.args.push_back(val);
                    }
                };
                if (term->opcode() == Opcode::br) add_arg(term->branch_target());
                else if (term->opcode() == Opcode::br_if) {
                    add_arg(term->true_target());
                    add_arg(term->false_target());
                } else if (term->opcode() == Opcode::switch_) {
                    add_arg(term->default_target());
                    for (auto& sc : term->switch_cases()) add_arg(sc.target);
                }
            }

            for (BasicBlock* fn_bb : fn.blocks()) {
                if (!fn_bb || loop_block_set.count(fn_bb)) continue;
                if (fn_bb == exit_bb || dom.dominates(exit_bb, fn_bb)) {
                    for (Instruction* inst : *fn_bb) {
                        if (!inst) continue;
                        for (size_t i = 0; i < inst->operand_count(); ++i) {
                            if (inst->operand(i) == val) {
                                inst->set_operand(i, exit_param);
                            }
                        }
                        for (size_t i = 0; i < inst->state_map().size(); ++i) {
                            if (inst->state_map()[i] == val) {
                                inst->state_map()[i] = exit_param;
                            }
                        }
                        if (inst->opcode() == Opcode::br) {
                            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                                if (inst->branch_target().args[i] == val) inst->branch_target().args[i] = exit_param;
                            }
                        } else if (inst->opcode() == Opcode::br_if) {
                            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                                if (inst->true_target().args[i] == val) inst->true_target().args[i] = exit_param;
                            }
                            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                                if (inst->false_target().args[i] == val) inst->false_target().args[i] = exit_param;
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<Value*> init_args = ph_term->branch_target().args;

    // 1. Create duplicate blocks for L_false
    std::unordered_map<BasicBlock*, BasicBlock*> block_map;
    std::unordered_map<const Value*, Value*> val_map;

    for (BasicBlock* bb : loop.blocks()) {
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

    // 2. Clone instructions for L_false
    for (BasicBlock* bb : loop.blocks()) {
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

            for (Value* op : inst->operands()) {
                if (!op) continue;
                auto it = val_map.find(op);
                cloned->add_operand(it != val_map.end() ? it->second : op);
            }

            for (Value* sv : inst->state_map()) {
                if (!sv) continue;
                auto it = val_map.find(sv);
                cloned->add_state_value(it != val_map.end() ? it->second : sv);
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

    // 3. Clone terminators for L_false
    auto remap_val = [&](Value* v) -> Value* {
        if (!v) return nullptr;
        auto it = val_map.find(v);
        return it != val_map.end() ? it->second : v;
    };

    auto remap_target = [&](const BranchTarget& tgt) -> BranchTarget {
        BasicBlock* blk = loop_block_set.count(tgt.block) ? block_map[tgt.block] : tgt.block;
        std::vector<Value*> args;
        args.reserve(tgt.args.size());
        for (Value* a : tgt.args) {
            args.push_back(remap_val(a));
        }
        return BranchTarget(blk, std::move(args));
    };

    for (BasicBlock* bb : loop.blocks()) {
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
