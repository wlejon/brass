#include <brass/mir/loop_tile.hpp>
#include "loop_tile_transform.hpp"

namespace brass {

bool tile_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    if (!nest.is_tileable()) {
        return false;
    }

    if (nest.depth() == 2) {
        return transform_2d_loop_nest(fn, nest, dom, options);
    }

    if (nest.depth() == 3) {
        if (nest.is_matrix_multiply()) {
            return transform_3d_matmul_loop_nest(fn, nest, dom, options);
        }
        return transform_3d_generic_loop_nest(fn, nest, dom, options);
    }

    return false;
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
    LoopNestAnalysis nest_analysis(fn, dom);

    bool any_changed = false;
    for (const auto& nest : nest_analysis.nests()) {
        if (nest && tile_loop_nest(fn, *nest, dom, options)) {
            any_changed = true;
            break;
        }
    }

    return any_changed;
}

bool loop_tile_pass(
    Function& fn,
    const DominatorTree& dom
) {
    return loop_tile_pass(fn, dom, LoopTileOptions());
}

} // namespace brass
