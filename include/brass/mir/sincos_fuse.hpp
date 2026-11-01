#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// Answers a sine and a cosine of one value with one call. For every pair of
// calls to the module's `sin_f64` and `cos_f64` symbols (runtime_symbols.hpp)
// on the same operand, one of which dominates the other, the dominating call
// becomes a call to its `sincos_f64` symbol writing both results to a frame
// slot, the two results are that slot's words, and the other call is gone.
// Runs late (after GVN has merged equal operands and loop-invariant code has
// left its loops): the fused call writes memory, so it is no longer a pure
// call the earlier passes may merge or hoist. A module that does not declare
// all three roles is left as it is. Returns how many pairs it fused.
size_t fuse_sincos(Function& fn);

} // namespace brass
