#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>
#include <string>

namespace brass {

struct ArrayContractionStats {
    size_t arrays_contracted = 0;
    size_t loads_eliminated = 0;
    size_t stores_eliminated = 0;
    size_t allocations_eliminated = 0;

    std::string format_report() const;
};

struct ArrayContractionOptions {
    ArrayContractionStats* stats = nullptr;
    // Fuse producer and consumer loops first, which puts each write next to
    // its read. Loop fusion is a transform of its own that the pipeline can
    // disable, so contraction only runs it when asked.
    bool fuse_loops_first = false;
};

// Contract arrays in a single loop (or fused loop).
bool contract_arrays_in_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const ArrayContractionOptions& options = {}
);

// Array contraction pass across all loops / candidate buffers in a function.
bool array_contraction_pass(
    Function& fn,
    const DominatorTree& dom,
    const ArrayContractionOptions& options = {}
);

} // namespace brass
