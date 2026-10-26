#pragma once

#include <brass/mir/function.hpp>

namespace brass {

// Splits a branch on the and (or) of two conditions into a branch on each:
// `br_if (and a b), T, F` becomes `br_if a, M, F` with M: `br_if b, T, F`
// (an or: `br_if a, T, M`, M: `br_if b, T, F`). The backends fuse a
// comparison into the branch reading it, so two compare-and-branches
// replace two flag materializations, an and and a test. Only conditions
// that are comparisons, or ands and ors of them, all read once and made in
// the branch's block, are split; the second one's instructions move into M.
// Runs last: analyses that read a loop's exit condition as one and-tree
// (range analysis) see it whole before.
bool split_branch_conditions(Function& fn);

} // namespace brass
