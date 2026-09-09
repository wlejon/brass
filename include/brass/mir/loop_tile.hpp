#pragma once

#include <brass/mir/loop_nest.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>

namespace brass {

struct LoopTileOptions {
    size_t tile_size_i = 16;
    size_t tile_size_j = 16;
    size_t tile_size_k = 16;
    bool enable_loop_interchange = true;
};

// Tile a specific loop nest (2D or 3D)
bool tile_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
);
bool tile_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom
);

// Pass over all tileable loop nests in a function
bool loop_tile_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopTileOptions& options
);
bool loop_tile_pass(
    Function& fn,
    const DominatorTree& dom
);

} // namespace brass
