#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>
#include <string>

namespace brass {

struct LoopDistributionStats {
    size_t loops_distributed = 0;
    size_t candidates_checked = 0;
    size_t rejected_pure_vectorizable = 0;
    size_t rejected_pure_scalar = 0;
    size_t rejected_cycles = 0;

    std::string format_report() const;
};

struct LoopDistributionOptions {
    LoopDistributionStats* stats = nullptr;
};

// Check if a loop can be legally distributed into vectorizable and unvectorizable loops
bool can_distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options = {}
);

// Distribute loop into two consecutive loops. Returns true on success.
bool distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options = {}
);

// Pass over all loops in a function attempting distribution
bool loop_distribution_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopDistributionOptions& options = {}
);

} // namespace brass
