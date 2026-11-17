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
// Tangent as V8 computes Math.tan (fdlibm's s_tan.c and k_tan.c on the same
// reduction).
BRASS_API double tan_f64(double x) noexcept;

// Arc sine, arc cosine and the arc tangents as V8 computes Math.asin,
// Math.acos, Math.atan and Math.atan2 (fdlibm's e_asin.c, e_acos.c, s_atan.c
// and e_atan2.c): V8's bits, with no FP-environment save and restore
// (fast_arctrig.cpp).
BRASS_API double asin_f64(double x) noexcept;
BRASS_API double acos_f64(double x) noexcept;
BRASS_API double atan_f64(double x) noexcept;
BRASS_API double atan2_f64(double y, double x) noexcept;

// e^x and the natural logarithm as V8 computes Math.exp and Math.log
// (fdlibm's e_exp.c and e_log.c, fast_explog.cpp). The platform's are
// correctly rounded more often, so they differ from V8's in the last bit.
BRASS_API double exp_f64(double x) noexcept;
BRASS_API double log_f64(double x) noexcept;
// x^y as V8 computes Math.pow and ** (fdlibm's e_pow.c, fast_explog.cpp).
// The language's rules for pow(+-1, +-inf) and pow(x, NaN), where C's pow
// answers 1, are the caller's.
BRASS_API double pow_f64(double x, double y) noexcept;

} // namespace brass::math

extern "C" {

// The same functions by C name, for code that links against them.
BRASS_API double brass_sin_f64(double x);
BRASS_API double brass_cos_f64(double x);
BRASS_API void brass_sincos_f64(double x, double* out);

} // extern "C"
