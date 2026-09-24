#pragma once

// Floating-point add/sub/mul/div with the NaN rule every tier follows
// (docs/semantics.md): when an operand is a NaN, the result is the lhs NaN if
// the lhs is one, else the rhs NaN, quieted, with its sign and payload. That
// is what x86 SSE (`addsd dst, src` returns dst's NaN) and AArch64 FADD with
// default-NaN off (the first NaN operand) produce. A C++ `a + b` does not
// pin it: the compiler may commute the operands.

#include <bit>
#include <cmath>
#include <cstdint>

namespace brass::fparith {

inline double quiet(double x) noexcept {
    return std::bit_cast<double>(std::bit_cast<uint64_t>(x) | 0x0008000000000000ull);
}
inline float quiet(float x) noexcept {
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) | 0x00400000u);
}

// Sets `out` to the operand NaN the result carries; false if neither
// operand is a NaN.
template <typename T>
inline bool nan_operand(T a, T b, T& out) noexcept {
    if (std::isnan(a)) { out = quiet(a); return true; }
    if (std::isnan(b)) { out = quiet(b); return true; }
    return false;
}

template <typename T> inline T add(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a + b; }
template <typename T> inline T sub(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a - b; }
template <typename T> inline T mul(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a * b; }
template <typename T> inline T div(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a / b; }

} // namespace brass::fparith
