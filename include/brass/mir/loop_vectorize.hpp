#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>

namespace brass {

struct LoopVectorizeOptions {
    bool enable_f32x4 = true;
    bool enable_i32x4 = true;
    bool enable_f64x2 = true;
    bool allow_fp_reassociation = false;
    uint32_t vector_width = 4;
};

// Vectorize a single loop
bool vectorize_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopVectorizeOptions& options
);
bool vectorize_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom
);

// Pass over all loops in a function
bool loop_vectorize_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopVectorizeOptions& options
);
bool loop_vectorize_pass(
    Function& fn,
    const DominatorTree& dom
);

} // namespace brass