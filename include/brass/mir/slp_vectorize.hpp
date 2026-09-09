#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>

namespace brass {

struct SlpOptions {
    bool enable_f32x4 = true;
    bool enable_i32x4 = true;
    bool enable_f64x2 = true;
    bool allow_fp_reassociation = false;
};

// Vectorize straight-line SLP opportunities in a single basic block
bool slp_vectorize_block(BasicBlock& bb, const SlpOptions& options);
bool slp_vectorize_block(BasicBlock& bb);

// Vectorize straight-line SLP opportunities across all blocks in a function
bool slp_vectorize_function(Function& fn, const SlpOptions& options);
bool slp_vectorize_function(Function& fn);

} // namespace brass