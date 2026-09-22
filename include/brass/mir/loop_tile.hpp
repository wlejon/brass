#pragma once

#include <brass/mir/loop_nest.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>

namespace brass {

// Tiling covers the outer two loops of a nest; loops nested deeper run whole
// inside each (i, j) iteration, so iteration order within them is kept.
struct LoopTileOptions {
    size_t tile_size_i = 16;
    size_t tile_size_j = 16;
};

// Tile a specific loop nest (its outer two levels)
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
