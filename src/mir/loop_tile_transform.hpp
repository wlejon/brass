#pragma once

#include <brass/mir/loop_tile.hpp>
#include <brass/mir/builder.hpp>

namespace brass {

// Tiles `outer_loop` and its single sub-loop in place (see
// loop_tile_transform.cpp for the exact shape and legality rules). Returns
// false, changing nothing, when the nest is not provably safe to tile.
bool tile_2d_loop_nest(Function& fn, LoopInfo& outer_loop, const LoopTileOptions& options);

} // namespace brass
