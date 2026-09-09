#pragma once

#include <brass/codegen/lir.hpp>

namespace brass::codegen {

struct BlockLayoutOptions {
    bool cold_block_at_end = true;
};

// Reorders LirFunction::blocks before EmitContext::compile() to maximize straight-line execution
// and eliminate taken jumps via trace scheduling.
void optimize_block_layout(LirFunction& fn, const BlockLayoutOptions& opts);
void optimize_block_layout(LirFunction& fn);

} // namespace brass::codegen
