#pragma once

#include <brass/brass_export.h>

namespace brass::math {

// Sine and cosine as V8 computes Math.sin and Math.cos: fdlibm's kernels and
// argument reduction (V8's ieee754::sin/cos), so the results are V8's bits.
// Plain double arithmetic with no FP-environment save and restore, and one
// branch on the data past the reduction, which is what makes them faster
// than glibc's sin/cos.
BRASS_API double sin_f64(double x) noexcept;
BRASS_API double cos_f64(double x) noexcept;
// out[0] = sin_f64(x), out[1] = cos_f64(x), bit for bit, from one reduction.
BRASS_API void sincos_f64(double x, double* out) noexcept;

// Arc sine and the arc tangents as V8 computes Math.asin, Math.atan and
// Math.atan2 (fdlibm's e_asin.c, s_atan.c and e_atan2.c): V8's bits, with no
// FP-environment save and restore (fast_arctrig.cpp).
BRASS_API double asin_f64(double x) noexcept;
BRASS_API double atan_f64(double x) noexcept;
BRASS_API double atan2_f64(double y, double x) noexcept;

} // namespace brass::math

extern "C" {

// The same functions by C name, for code that links against them.
BRASS_API double brass_sin_f64(double x);
BRASS_API double brass_cos_f64(double x);
BRASS_API void brass_sincos_f64(double x, double* out);

} // extern "C"
