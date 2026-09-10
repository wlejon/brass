#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>
#include <string>

namespace brass {

struct LoopFusionStats {
    size_t loops_fused = 0;
    size_t candidates_checked = 0;
    size_t rejected_non_adjacent = 0;
    size_t rejected_domain_mismatch = 0;
    size_t rejected_dependencies = 0;

    std::string format_report() const;
};

struct LoopFusionOptions {
    bool aggressive = false;
    LoopFusionStats* stats = nullptr;
};

// Check if two loops can be legally fused
bool can_fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options = {}
);

// Fuse loop2 into loop1. Returns true on success.
bool fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options = {}
);

// Pass over all loops in a function attempting fusion on adjacent candidate loops
bool loop_fusion_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopFusionOptions& options = {}
);

} // namespace brass
