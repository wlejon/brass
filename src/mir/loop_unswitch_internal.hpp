#pragma once

#include <brass/mir/loop_unswitch.hpp>

namespace brass {

bool transform_unswitch_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    BasicBlock* cond_bb,
    Instruction* br_if_inst,
    const LoopUnswitchOptions& opts
);

} // namespace brass
