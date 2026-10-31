#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// A float division by a constant power of two becomes a multiplication by its
// reciprocal: `x / 2` is `x * 0.5`. The two are the same operation in IEEE
// arithmetic, not an approximation — x / 2^k and x * 2^-k are the same real
// number, each operation rounds that number once, so they agree for every x,
// NaN, the infinities and both zeros included — whenever 2^-k is itself a
// normal double (a subnormal one would read as 0 on a core flushing
// denormal inputs, where the division would not). What changes is the cost: a divide is 13-15 cycles of latency on
// current x64 cores and on Apple's, a multiply 3-4, and a divide that feeds a
// call's argument (Quaternion.setFromEuler's `x / 2` into its sines and
// cosines) stalls the call on it. V8 makes the same rewrite.
//
// Scalar f64 and f32 only. Returns how many divisions it rewrote.
// BRASS_NO_FDIV_POW2=1 turns it off (the A/B seam).
size_t rewrite_fdiv_pow2(Function& fn);

// Whether `c` is a nonzero power of two whose reciprocal is exactly a double
// (and, for `f32`, both are exactly floats); `recip` receives it.
bool exact_pow2_reciprocal(double c, bool f32, double& recip);

} // namespace brass
