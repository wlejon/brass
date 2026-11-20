#pragma once

#include <bit>
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

// Signed 32-bit division by a constant d >= 3 (non-power-of-two).
// Product fits in signed 64 bits: (sext(n) * sext(multiplier)) >> 32.
struct SignedDivMagic32 {
    int32_t multiplier = 0;
    int shift = 0;
};

inline bool takes_signed_div_magic32(int32_t d) noexcept {
    if (d < 3) return false;
    const uint32_t u = static_cast<uint32_t>(d);
    return (u & (u - 1u)) != 0u;
}

// Hacker's Delight figure 10-1 at 32 bits, evaluated in uint64_t.
// Precondition: takes_signed_div_magic32(d).
inline SignedDivMagic32 signed_div_magic32(int32_t d) noexcept {
    constexpr uint64_t kTwo31 = uint64_t{1} << 31;
    const uint64_t ad = static_cast<uint32_t>(d);
    const uint64_t anc = kTwo31 - 1ULL - (kTwo31 % ad);
    int p = 31;
    uint64_t q1 = kTwo31 / anc;
    uint64_t r1 = kTwo31 - q1 * anc;
    uint64_t q2 = kTwo31 / ad;
    uint64_t r2 = kTwo31 - q2 * ad;
    uint64_t delta = 0;
    do {
        ++p;
        q1 *= 2ULL;
        r1 *= 2ULL;
        if (r1 >= anc) {
            ++q1;
            r1 -= anc;
        }
        q2 *= 2ULL;
        r2 *= 2ULL;
        if (r2 >= ad) {
            ++q2;
            r2 -= ad;
        }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0ULL));
    SignedDivMagic32 m;
    m.multiplier = static_cast<int32_t>(static_cast<uint32_t>(q2 + 1ULL));
    m.shift = p - 32;
    return m;
}

// Unsigned 32-bit division by a constant d >= 3 (non-power-of-two).
// Uses Granlund-Montgomery pre-shift when possible so `needs_add` is often false.
struct UnsignedDivMagic32 {
    uint32_t multiplier = 0;
    uint32_t pre_shift = 0;
    uint32_t post_shift = 0;
    bool needs_add = false;
};

inline bool takes_unsigned_div_magic32(uint32_t d) noexcept {
    return d >= 3u && (d & (d - 1u)) != 0u;
}

inline UnsignedDivMagic32 unsigned_div_magic32(uint32_t d) noexcept {
    UnsignedDivMagic32 out;
    if (d <= 1u) {
        out.multiplier = 1u;
        return out;
    }
    auto try_no_add = [&](uint32_t pre_sh, UnsignedDivMagic32& res, uint64_t& first_m64, uint32_t& first_s) -> bool {
        const uint32_t dp = d >> pre_sh;
        const uint64_t xmax = 0xFFFFFFFFULL >> pre_sh;
        const uint64_t nc = xmax - ((xmax + 1ULL) % dp);
        for (uint32_t s = 0; s < 32u; ++s) {
            const uint64_t two_p = 1ULL << (32u + s);
            const uint64_t rem = two_p % dp;
            const uint64_t delta = (rem == 0ULL) ? 0ULL : (dp - rem);
            if (nc * delta < two_p) {
                const uint64_t m64 = (two_p + delta) / dp;
                first_m64 = m64;
                first_s = s;
                if (m64 <= 0xFFFFFFFFULL) {
                    res.multiplier = static_cast<uint32_t>(m64);
                    res.pre_shift = pre_sh;
                    res.post_shift = s;
                    res.needs_add = false;
                    return true;
                }
                return false;
            }
        }
        return false;
    };

    uint64_t m64_0 = 0;
    uint32_t s_0 = 0;
    if (try_no_add(0u, out, m64_0, s_0)) return out;
    const uint32_t tz = static_cast<uint32_t>(std::countr_zero(d));
    uint64_t dummy_m = 0;
    uint32_t dummy_s = 0;
    if (tz > 0u && try_no_add(tz, out, dummy_m, dummy_s)) return out;
    out.multiplier = static_cast<uint32_t>(m64_0 - (1ULL << 32));
    out.pre_shift = 0u;
    out.post_shift = s_0;
    out.needs_add = true;
    return out;
}

// Unsigned 64-bit division by a constant d >= 3 (non-power-of-two).
struct UnsignedDivMagic64 {
    uint64_t multiplier = 0;
    uint32_t shift = 0;
    bool needs_add = false;
};

inline bool takes_unsigned_div_magic64(uint64_t d) noexcept {
    return d >= 3ULL && (d & (d - 1ULL)) != 0ULL;
}

// Hacker's Delight figure 10-2 at 64 bits in pure uint64_t arithmetic.
// Precondition: takes_unsigned_div_magic64(d).
inline UnsignedDivMagic64 unsigned_div_magic64(uint64_t d) noexcept {
    constexpr uint64_t kTwo63 = uint64_t{1} << 63;
    const uint64_t nc = ~uint64_t{0} - ((~uint64_t{0} - d + 1ULL) % d);
    uint64_t q1 = kTwo63 / nc;
    uint64_t r1 = kTwo63 - q1 * nc;
    uint64_t q2 = (kTwo63 - 1ULL) / d;
    uint64_t r2 = (kTwo63 - 1ULL) - q2 * d;
    uint32_t p = 63u;
    bool add = false;
    uint64_t delta = 0;
    do {
        ++p;
        if (r1 >= nc - r1) {
            q1 = 2ULL * q1 + 1ULL;
            r1 = 2ULL * r1 - nc;
        } else {
            q1 = 2ULL * q1;
            r1 = 2ULL * r1;
        }
        if ((r2 + 1ULL) >= d - r2) {
            if (q2 >= kTwo63 - 1ULL) add = true;
            q2 = 2ULL * q2 + 1ULL;
            r2 = 2ULL * r2 + 1ULL - d;
        } else {
            if (q2 >= kTwo63) add = true;
            q2 = 2ULL * q2;
            r2 = 2ULL * r2 + 1ULL;
        }
        delta = d - 1ULL - r2;
    } while (p < 128u && (q1 < delta || (q1 == delta && r1 == 0ULL)));
    UnsignedDivMagic64 m;
    m.multiplier = q2 + 1ULL;
    m.shift = p - 64u;
    m.needs_add = add;
    return m;
}

} // namespace brass::codegen
