#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <cstddef>

namespace brass {

struct CriticalEdgeStats {
    size_t critical_edges_split = 0;
};

// Check if edge (src -> dst) is a critical edge:
// src has multiple unique successors AND dst has multiple unique predecessors.
bool is_critical_edge(const BasicBlock* src, const BasicBlock* dst);

// Split a specific critical edge between src and dst.
// Inserts a synthetic forwarding block between src and dst,
// updates branch targets and block arguments, and rebuilds CFG predecessors.
// Returns the newly created split block, or nullptr if (src, dst) is not a critical edge or invalid.
BasicBlock* split_critical_edge(Function& fn, BasicBlock* src, BasicBlock* dst, CriticalEdgeStats* stats = nullptr);

// Detects and splits all critical edges in the function.
// Returns true if any critical edges were split.
bool split_critical_edges(Function& fn, CriticalEdgeStats* stats = nullptr);

} // namespace brass
