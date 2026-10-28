#pragma once

// What deterministic_trig.cpp takes from deterministic_math.cpp: the exact
// reduction of an argument too large for fdlibm's medium range.

namespace brass::math::detail {

// x (|x| > 2^19 * pi/2, finite) reduced by pi/2 into y0 + y1, y1 the tail of
// y0, returning the multiple of pi/2 taken mod 4: the Payne-Hanek
// double-double reduction.
int reduce_pio2_huge(double x, double& y0, double& y1) noexcept;

} // namespace brass::math::detail
