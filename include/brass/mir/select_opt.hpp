#pragma once

#include <brass/mir/function.hpp>

namespace brass {

// Pass to optimize small control-flow diamonds and triangles into branchless select instructions
bool simplify_cfg_diamonds(Function& fn, size_t max_instructions_per_branch = 16);

} // namespace brass
