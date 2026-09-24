#pragma once

// Floating-point add/sub/mul/div with the NaN rule every tier follows
// (docs/semantics.md): when an operand is a NaN, the result is quieted, with
// its sign and payload. On x86 SSE (`addsd dst, src`), the lhs NaN is kept if
// it is one, else the rhs NaN. On AArch64 hardware (FPCR.DN == 0), signaling
// NaNs are prioritized over quiet NaNs, picking the first signaling NaN, or the
// first quiet NaN if neither is signaling. A C++ `a + b` does not pin it: the
// compiler may commute the operands.

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

inline bool is_snan(double x) noexcept {
    uint64_t u = std::bit_cast<uint64_t>(x);
    return ((u & 0x7FF0000000000000ull) == 0x7FF0000000000000ull) &&
           ((u & 0x0008000000000000ull) == 0) &&
           ((u & 0x0007FFFFFFFFFFFFull) != 0);
}

inline bool is_snan(float x) noexcept {
    uint32_t u = std::bit_cast<uint32_t>(x);
    return ((u & 0x7F800000u) == 0x7F800000u) &&
           ((u & 0x00400000u) == 0) &&
           ((u & 0x003FFFFFu) != 0);
}

// Sets `out` to the operand NaN the result carries; false if neither
// operand is a NaN.
template <typename T>
inline bool nan_operand(T a, T b, T& out) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (is_snan(a)) { out = quiet(a); return true; }
    if (is_snan(b)) { out = quiet(b); return true; }
    if (std::isnan(a)) { out = quiet(a); return true; }
    if (std::isnan(b)) { out = quiet(b); return true; }
    return false;
#else
    if (std::isnan(a)) { out = quiet(a); return true; }
    if (std::isnan(b)) { out = quiet(b); return true; }
    return false;
#endif
}

template <typename T> inline T add(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a + b; }
template <typename T> inline T sub(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a - b; }
template <typename T> inline T mul(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a * b; }
template <typename T> inline T div(T a, T b) noexcept { T n; return nan_operand(a, b, n) ? n : a / b; }

} // namespace brass::fparith
