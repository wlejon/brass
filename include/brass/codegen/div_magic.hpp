#pragma once

#include <cstdint>
#include <cstdlib>

namespace brass::codegen {

// Signed 64-bit division and remainder by a constant, as a multiply by a
// "magic" number and a shift instead of an idiv / sdiv (Granlund and
// Montgomery 1994; Hacker's Delight, 2nd ed., 10-1, here at 64 bits). For a
// divisor d >= 2 that is not a power of two (those have their own shift
// sequences), and every int64 n:
//
//   t = high 64 bits of the signed 128-bit product M * n
//   if M < 0 (as int64):  t += n        (the multiplier needs 65 bits)
//   t >>= shift                          (arithmetic)
//   q = t + (n >>> 63)                   (truncates toward zero for n < 0)
//
// is trunc(n / d), and n - q * d the remainder with the sign of n, exactly as
// sdiv / smod answer. Only d > 0 is taken: d = -1 and INT64_MIN / -1 never
// reach this, and d = 0 keeps the instruction that traps or is guarded.
//
// The search below runs in unsigned 64-bit arithmetic only, so it has no
// signed overflow; tests/unit/test_div_magic.cpp checks the identity above
// for each divisor it is used with in the tests against a reference product
// built from 32-bit halves, so it holds on every host.
struct SignedDivMagic {
    int64_t multiplier = 0;
    int shift = 0;
};

// Divisors taken: 3 <= d, not a power of two. (d = 2^k folds to shifts in
// both selectors already, and d = 1 / 2 too.)
inline bool takes_signed_div_magic(int64_t d) noexcept {
    if (d < 3) return false;
    const uint64_t u = static_cast<uint64_t>(d);
    return (u & (u - 1)) != 0;
}

// BRASS_NO_DIV_MAGIC=1 keeps the divide instruction (A/B seam).
inline bool signed_div_magic_enabled() noexcept {
    static const bool enabled = std::getenv("BRASS_NO_DIV_MAGIC") == nullptr;
    return enabled;
}

// Hacker's Delight figure 10-1, with 2^31 / 32 bits replaced by 2^63 / 64.
// Precondition: takes_signed_div_magic(d).
inline SignedDivMagic signed_div_magic64(int64_t d) noexcept {
    const uint64_t two63 = uint64_t{1} << 63;
    const uint64_t ad = static_cast<uint64_t>(d);
    const uint64_t t = two63;  // + (d >>> 63), which is 0 for d > 0
    const uint64_t anc = t - 1 - t % ad;  // |nc|
    int p = 63;
    uint64_t q1 = two63 / anc;
    uint64_t r1 = two63 - q1 * anc;
    uint64_t q2 = two63 / ad;
    uint64_t r2 = two63 - q2 * ad;
    uint64_t delta = 0;
    do {
        ++p;
        q1 = 2 * q1;
        r1 = 2 * r1;
        if (r1 >= anc) {
            ++q1;
            r1 -= anc;
        }
        q2 = 2 * q2;
        r2 = 2 * r2;
        if (r2 >= ad) {
            ++q2;
            r2 -= ad;
        }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    SignedDivMagic m;
    m.multiplier = static_cast<int64_t>(q2 + 1);
    m.shift = p - 64;
    return m;
}

} // namespace brass::codegen
