#pragma once

#include <brass/codegen/lir.hpp>

namespace brass {
class Function;
class BasicBlock;
namespace mir {
class BranchProbabilityInfo;
class BlockFrequencyInfo;
} // namespace mir
} // namespace brass

namespace brass::codegen {

struct BlockLayoutOptions {
    bool cold_block_at_end = true;
    const mir::BranchProbabilityInfo* branch_prob = nullptr;
    const mir::BlockFrequencyInfo* block_freq = nullptr;
    const Function* mir_function = nullptr;
};

// Reorders LirFunction::blocks before EmitContext::compile() to maximize straight-line execution
// and eliminate taken jumps via trace scheduling.
void optimize_block_layout(LirFunction& fn, const BlockLayoutOptions& opts);
void optimize_block_layout(LirFunction& fn);

} // namespace brass::codegen
