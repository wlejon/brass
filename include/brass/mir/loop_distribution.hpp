#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>
#include <string>
#include <unordered_set>

namespace brass {

struct LoopDistributionStats {
    size_t loops_distributed = 0;
    size_t candidates_checked = 0;
    size_t rejected_pure_vectorizable = 0;
    size_t rejected_pure_scalar = 0;
    size_t rejected_cycles = 0;

    std::string format_report() const;
};

// The module's functions a loop may call and still be distributed: one block
// of computation from the arguments alone (pure_leaf_function). Recorded for
// a whole module at once, before its functions are optimized in parallel, so
// that a caller never reads a callee another thread is changing.
struct PureLeafFunctions {
    const Module* module = nullptr;
    std::unordered_set<const Function*> functions;
    void record(const Module& mod);
};

bool pure_leaf_function(const Function& fn);

struct LoopDistributionOptions {
    LoopDistributionStats* stats = nullptr;
    // When set and recorded for the function's module, what a call's callee
    // is taken to be; otherwise the callee's body is read as it is now.
    const PureLeafFunctions* pure_leaves = nullptr;
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
