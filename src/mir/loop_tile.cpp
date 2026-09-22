#include <brass/mir/loop_tile.hpp>
#include "loop_tile_transform.hpp"

namespace brass {

bool tile_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    (void)dom;
    if (nest.depth() < 2 || !nest.level(0).loop) return false;
    return tile_2d_loop_nest(fn, *nest.level(0).loop, options);
}

bool tile_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom
) {
    return tile_loop_nest(fn, nest, dom, LoopTileOptions());
}

bool loop_tile_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    fn.rebuild_cfg_predecessors();
    LoopAnalysis loops(fn, dom);
    for (const auto& top : loops.top_level_loops()) {
        // One nest per call: the transform changes the CFG under the analysis.
        if (top && tile_2d_loop_nest(fn, *top, options)) return true;
    }
    return false;
}

bool loop_tile_pass(
    Function& fn,
    const DominatorTree& dom
) {
    return loop_tile_pass(fn, dom, LoopTileOptions());
}

} // namespace brass
