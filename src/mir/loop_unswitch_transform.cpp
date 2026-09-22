#include "loop_unswitch_internal.hpp"
#include "ir_clone.hpp"
#include <brass/mir/cfg_simplify.hpp>
#include <string>
#include <vector>

namespace brass {

namespace {

Instruction* make_branch(Function& fn, Opcode op) {
    return fn.parent() ? fn.parent()->arena().make<Instruction>(op, Type::void_type())
                       : new Instruction(op, Type::void_type());
}

} // namespace

// The false copy of the loop must be a closed region: every block that can
// see a value defined in the loop has to be copied with it, or the copy's
// values would not dominate their uses. The blocks dominated by the header
// are exactly that set — every predecessor of such a block (other than the
// header's own entry edge) is itself dominated by the header, and a block
// outside it can only receive loop values through block parameters.
bool transform_unswitch_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    BasicBlock* cond_bb,
    Instruction* br_if_inst,
    const LoopUnswitchOptions& opts
) {
    (void)dom;
    if (!cond_bb || !br_if_inst || br_if_inst->opcode() != Opcode::br_if) return false;
    Value* cond = br_if_inst->operand(0);
    BasicBlock* header = loop.header();
    if (!cond || !header) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader || !preheader->terminator()) return false;
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;

    // ensure_preheader may have changed the CFG since `dom` was built.
    fn.rebuild_cfg_predecessors();
    DominatorTree cur_dom(fn);

    const BasicBlock* cond_def = cond->is_instruction() ? cond->defining_instruction()->parent()
                               : (cond->is_block_param() ? cond->defining_block() : nullptr);
    if (cond_def && !cur_dom.dominates(cond_def, preheader)) return false;

    std::vector<BasicBlock*> region;
    size_t region_instructions = 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || !cur_dom.dominates(header, bb)) continue;
        region.push_back(bb);
        for (Instruction* inst : *bb) {
            if (inst) ++region_instructions;
        }
    }
    // The loop itself was already bounded by max_loop_instructions; allow the
    // code after it to cost at most as much again.
    if (region_instructions > 2 * opts.max_loop_instructions) return false;

    const std::vector<Value*> init_args = ph_term->branch_target().args;

    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + "_unsw_f");
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }

    // Shells first so every result exists before any operand is remapped.
    std::vector<std::pair<Instruction*, Instruction*>> pairs;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = blocks[bb];
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst == br_if_inst) {
                Instruction* br = make_branch(fn, Opcode::br);
                copy->append_instruction(br);
                pairs.emplace_back(inst, br);
                continue;
            }
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            copy->append_instruction(shell);
            pairs.emplace_back(inst, shell);
        }
    }
    for (auto& [orig, copy] : pairs) {
        if (orig == br_if_inst) {
            // The false copy always takes the false edge.
            const BranchTarget& f = orig->false_target();
            BranchTarget t(f.block, f.args);
            if (auto it = blocks.find(t.block); it != blocks.end()) t.block = it->second;
            for (Value*& a : t.args) {
                if (auto it = values.find(a); it != values.end()) a = it->second;
            }
            copy->set_branch_target(std::move(t));
            continue;
        }
        ir::clone_uses(*orig, *copy, values, blocks);
    }

    // The original copy always takes the true edge.
    Instruction* br_true = make_branch(fn, Opcode::br);
    br_true->set_branch_target(br_if_inst->true_target());
    cond_bb->remove_instruction(br_if_inst);
    cond_bb->append_instruction(br_true);

    Instruction* ph_br_if = make_branch(fn, Opcode::br_if);
    ph_br_if->add_operand(cond);
    ph_br_if->set_true_target(BranchTarget(header, init_args));
    ph_br_if->set_false_target(BranchTarget(blocks[header], init_args));
    preheader->remove_instruction(ph_term);
    preheader->append_instruction(ph_br_if);

    fn.rebuild_cfg_predecessors();
    CfgSimplifyOptions cfg_opts;
    cfg_opts.enable_param_elimination = false;
    cfg_simplify_function(fn, cfg_opts);
    return true;
}

} // namespace brass
