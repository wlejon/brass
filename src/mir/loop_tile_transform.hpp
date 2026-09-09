#pragma once

#include <brass/mir/loop_tile.hpp>
#include <brass/mir/builder.hpp>

namespace brass {

bool transform_2d_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
);

bool transform_3d_matmul_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
);

bool transform_3d_generic_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
);

} // namespace brass
