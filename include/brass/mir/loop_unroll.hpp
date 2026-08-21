#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>

namespace brass {

struct LoopUnrollOptions {
    size_t unroll_factor = 4;
    bool enable_reduction_jam = true;       // Integer associative reduction jam (default ON)
    bool enable_fp_reduction_jam = false;   // Floating-point reduction jam (default OFF / IEEE-strict)
    bool enable_general_unroll = true;
};

// Try to unroll a specific counted loop
bool unroll_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopUnrollOptions& options = {}
);

// Pass over all loops in a function attempting unrolling
bool loop_unroll_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopUnrollOptions& options = {}
);

} // namespace brass
