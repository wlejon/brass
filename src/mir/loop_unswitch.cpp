#include <brass/mir/loop_unswitch.hpp>
#include "loop_unswitch_internal.hpp"
#include <brass/mir/cfg_simplify.hpp>

namespace brass {

bool unswitch_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopUnswitchOptions& opts
) {
    if (loop.depth() > opts.max_unswitch_depth) {
        return false;
    }

    // Count instructions across all blocks in the loop
    size_t total_instructions = 0;
    for (BasicBlock* bb : loop.blocks()) {
        if (bb) {
            total_instructions += bb->instruction_count();
        }
    }
    if (total_instructions > opts.max_loop_instructions) {
        return false;
    }

    // Find candidate loop-invariant conditional branch
    BasicBlock* cand_bb = nullptr;
    Instruction* cand_term = nullptr;

    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        Instruction* term = bb->terminator();
        if (!term || term->opcode() != Opcode::br_if) continue;

        Value* cond = term->operand(0);
        if (!cond || !loop.is_loop_invariant(cond)) continue;

        if (term->true_target().block == term->false_target().block) {
            continue;
        }

        cand_bb = bb;
        cand_term = term;
        break;
    }

    if (!cand_bb || !cand_term) {
        return false;
    }

    return transform_unswitch_loop(fn, loop, dom, cand_bb, cand_term, opts);
}

bool unswitch_loop(Function& fn, LoopInfo& loop, const DominatorTree& dom) {
    return unswitch_loop(fn, loop, dom, LoopUnswitchOptions());
}

bool unswitch_loops_in_function(Function& fn, const LoopUnswitchOptions& opts, LoopUnswitchStats* stats) {
    bool changed = false;
    constexpr size_t kMaxRounds = 16;

    for (size_t round = 0; round < kMaxRounds; ++round) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loop_analysis(fn, dom);

        std::vector<LoopInfo*> post_order = loop_analysis.post_order_loops();
        bool round_changed = false;

        for (LoopInfo* loop : post_order) {
            if (!loop) continue;
            if (unswitch_loop(fn, *loop, dom, opts)) {
                round_changed = true;
                changed = true;
                if (stats) {
                    stats->loops_unswitched++;
                }
                break; // Restart analysis after CFG modification
            }
        }

        if (!round_changed) {
            break;
        }
    }

    return changed;
}

bool unswitch_loops_in_function(Function& fn) {
    return unswitch_loops_in_function(fn, LoopUnswitchOptions(), nullptr);
}

bool unswitch_loops_in_module(Module& mod, const LoopUnswitchOptions& opts, LoopUnswitchStats* stats) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= unswitch_loops_in_function(*fn, opts, stats);
        }
    }
    return changed;
}

bool unswitch_loops_in_module(Module& mod) {
    return unswitch_loops_in_module(mod, LoopUnswitchOptions(), nullptr);
}

} // namespace brass
