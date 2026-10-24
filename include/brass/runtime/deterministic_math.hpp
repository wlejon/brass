#pragma once

#include <brass/brass_export.h>
#include <cstddef>
#include <cstdint>

namespace brass::math {

// Canonical positive quiet NaN bit pattern for binary64 (0x7ff8000000000000).
inline constexpr uint64_t kCanonicalNaNBits = 0x7ff8000000000000ULL;

// Canonicalizes any NaN bit pattern to kCanonicalNaNBits and leaves all other
// IEEE-754 binary64 bit patterns (including -0.0 and +-Inf) unchanged.
BRASS_API double canonicalize_nan(double x) noexcept;
BRASS_API uint64_t canonical_f64_bits(double x) noexcept;

// Software IEEE-754 binary64 transcendental math evaluated in 106-bit
// double-double arithmetic with Veltkamp/Dekker exact splitting (no host libm
// calls, no hardware FMA contraction). Produces bit-identical results on
// x86_64 and AArch64 across Linux, Windows, and macOS.
BRASS_API double det_sin(double x) noexcept;
BRASS_API double det_cos(double x) noexcept;
BRASS_API double det_tan(double x) noexcept;
BRASS_API double det_asin(double x) noexcept;
BRASS_API double det_acos(double x) noexcept;
BRASS_API double det_atan(double x) noexcept;
BRASS_API double det_atan2(double y, double x) noexcept;
BRASS_API double det_exp(double x) noexcept;
BRASS_API double det_log(double x) noexcept;
BRASS_API double det_pow(double x, double y) noexcept;
BRASS_API double det_hypot(double x, double y) noexcept;
BRASS_API double det_cbrt(double x) noexcept;

// Fast deterministic 64-bit state hashes for rollback/replay desync detection.
// hash_f64_state canonicalizes NaN payloads before mixing so NaN bit-payload
// differences cannot cause false desyncs.
BRASS_API uint64_t hash_bytes_64(const void* data, size_t size, uint64_t seed = 0) noexcept;
BRASS_API uint64_t hash_f64_state(const double* values, size_t count, uint64_t seed = 0) noexcept;

} // namespace brass::math

extern "C" {

BRASS_API double brass_det_sin(double x);
BRASS_API double brass_det_cos(double x);
BRASS_API double brass_det_tan(double x);
BRASS_API double brass_det_asin(double x);
BRASS_API double brass_det_acos(double x);
BRASS_API double brass_det_atan(double x);
BRASS_API double brass_det_atan2(double y, double x);
BRASS_API double brass_det_exp(double x);
BRASS_API double brass_det_log(double x);
BRASS_API double brass_det_pow(double x, double y);
BRASS_API double brass_det_hypot(double x, double y);
BRASS_API double brass_det_cbrt(double x);
BRASS_API uint64_t brass_det_hash_bytes(const void* data, size_t size, uint64_t seed);
BRASS_API uint64_t brass_det_hash_f64_state(const double* values, size_t count, uint64_t seed);

} // extern "C"
