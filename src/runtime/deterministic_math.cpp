#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#include <brass/runtime/deterministic_math.hpp>
#include <bit>
#include <cstdint>
#include <cstring>

namespace brass::math {

namespace {

constexpr uint64_t kSignMask = 0x8000000000000000ULL;
constexpr uint64_t kExpMask  = 0x7ff0000000000000ULL;
constexpr uint64_t kMantMask = 0x000fffffffffffffULL;
constexpr uint64_t kPosInf   = 0x7ff0000000000000ULL;
constexpr uint64_t kNegInf   = 0xfff0000000000000ULL;

inline uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
inline double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }

inline bool is_nan_bits(uint64_t u) noexcept { return (u & ~kSignMask) > kPosInf; }
inline bool is_inf_bits(uint64_t u) noexcept { return (u & ~kSignMask) == kPosInf; }
inline bool is_zero_bits(uint64_t u) noexcept { return (u & ~kSignMask) == 0; }
inline double qnan() noexcept { return from_bits(kCanonicalNaNBits); }
inline double pos_inf() noexcept { return from_bits(kPosInf); }
inline double neg_inf() noexcept { return from_bits(kNegInf); }

inline double abs_f64(double x) noexcept { return from_bits(to_bits(x) & ~kSignMask); }
inline double copysign_f64(double mag, double sgn) noexcept {
    return from_bits((to_bits(mag) & ~kSignMask) | (to_bits(sgn) & kSignMask));
}

// Software ldexp(x, k): exact multiplication by 2^k via IEEE-754 exponent adjustment.
inline double scale_pow2(double x, int k) noexcept {
    auto step = [](double v, int e) noexcept -> double {
        uint64_t u = static_cast<uint64_t>(static_cast<int64_t>(e + 1023)) << 52;
        return v * from_bits(u);
    };
    while (k > 1023) { x = step(x, 1023); k -= 1023; }
    while (k < -1022) { x = step(x, -1022); k += 1022; }
    return step(x, k);
}

// Decomposes finite positive x into x = m * 2^e with m in [1.0, 2.0).
inline double frexp_norm(double x, int& out_e) noexcept {
    uint64_t u = to_bits(x);
    int exp_field = static_cast<int>((u >> 52) & 0x7ffULL);
    if (exp_field == 0) {
        x = scale_pow2(x, 54);
        u = to_bits(x);
        exp_field = static_cast<int>((u >> 52) & 0x7ffULL) - 54;
    }
    out_e = exp_field - 1023;
    return from_bits((u & kMantMask) | 0x3ff0000000000000ULL);
}

// Software round-to-nearest-even integer for |x| < 2^51.
inline double round_nearest_even(double x) noexcept {
    constexpr double kMagic = 6755399441055744.0; // 3 * 2^51
    double t = x + kMagic;
    return t - kMagic;
}

//Software floor/trunc for finite double without libm.
inline double trunc_f64(double x) noexcept {
    uint64_t u = to_bits(x);
    int e = static_cast<int>((u >> 52) & 0x7ffULL) - 1023;
    if (e < 0) return from_bits(u & kSignMask);
    if (e >= 52) return x;
    uint64_t mask = kMantMask >> e;
    return from_bits(u & ~mask);
}

// ============================================================================
// 106-bit Double-Double (DD) Arithmetic via Knuth TwoSum & Dekker Veltkamp Split
// ============================================================================

struct DD {
    double hi;
    double lo;
};

inline DD two_sum(double a, double b) noexcept {
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return {s, e};
}

inline DD quick_two_sum(double a, double b) noexcept {
    double s = a + b;
    double e = b - (s - a);
    return {s, e};
}

inline DD two_diff(double a, double b) noexcept {
    double s = a - b;
    double v = s - a;
    double e = (a - (s - v)) - (b + v);
    return {s, e};
}

inline DD split_veltkamp(double a) noexcept {
    constexpr double kSplit = 134217729.0; // 2^27 + 1
    double t = a * kSplit;
    double hi = t - (t - a);
    double lo = a - hi;
    return {hi, lo};
}

inline DD two_prod(double a, double b) noexcept {
    double p = a * b;
    DD sa = split_veltkamp(a);
    DD sb = split_veltkamp(b);
    double err = ((sa.hi * sb.hi - p) + sa.hi * sb.lo + sa.lo * sb.hi) + sa.lo * sb.lo;
    return {p, err};
}

inline DD dd_add(DD a, DD b) noexcept {
    DD s = two_sum(a.hi, b.hi);
    DD t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline DD dd_sub(DD a, DD b) noexcept {
    return dd_add(a, {-b.hi, -b.lo});
}

inline DD dd_add_d(DD a, double b) noexcept {
    DD s = two_sum(a.hi, b);
    s.lo += a.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline DD dd_mul(DD a, DD b) noexcept {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline DD dd_mul_d(DD a, double b) noexcept {
    DD p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return quick_two_sum(p.hi, p.lo);
}

inline DD dd_div(DD a, DD b) noexcept {
    double q1 = a.hi / b.hi;
    DD r = dd_sub(a, dd_mul_d(b, q1));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul_d(b, q2));
    double q3 = r.hi / b.hi;
    DD out = quick_two_sum(q1, q2);
    return dd_add_d(out, q3);
}

inline DD dd_neg(DD a) noexcept { return {-a.hi, -a.lo}; }

inline double to_double(DD a) noexcept { return a.hi + a.lo; }

// Software 106-bit square root for positive finite DD a.
inline DD dd_sqrt(DD a) noexcept {
    if (a.hi <= 0.0) return {0.0, 0.0};
    int e = 0;
    double m = frexp_norm(a.hi, e);
    if (e & 1) {
        m *= 2.0;
        --e;
    }
    // Initial approximation on [1, 4) followed by 4 Newton steps in double.
    double y = 0.4 + 0.5 * m;
    for (int i = 0; i < 5; ++i) {
        y = 0.5 * (y + m / y);
    }
    y = scale_pow2(y, e >> 1);
    // Two Newton-Raphson refinements in double-double:
    DD x_dd = {y, 0.0};
    for (int i = 0; i < 2; ++i) {
        DD q = dd_div(a, x_dd);
        DD s = dd_add(x_dd, q);
        x_dd = {s.hi * 0.5, s.lo * 0.5};
    }
    return x_dd;
}

// ============================================================================
// High-Precision Mathematical Constants in Double-Double
// ============================================================================

// pi/2 = 1.570796326794896619231321691639751442098584699687552910487...
constexpr DD kPiOver2 = {1.5707963267948966, 6.123233995736766e-17};
constexpr DD kPi      = {3.141592653589793,  1.2246467991473532e-16};
constexpr DD kPiOver4 = {0.7853981633974483, 3.061616997868383e-17};
// ln(2) = 0.693147180559945309417232121458176568075500134360255254...
constexpr DD kLn2     = {0.6931471805599453, 2.3190468138462996e-17};
constexpr double kInvLn2 = 1.4426950408889634;
constexpr double kTwoOverPi = 0.6366197723675814;

// 4-word exact split of pi/2 (24 bits per upper word so k * W_i is exact for |k| < 2^28):
// sum = 1.57079632679489661923132169163975144209858469968755291...
constexpr double kPi2_W0 = 1.570796310901641845703125;                  // 0x1.921fb4p+0
constexpr double kPi2_W1 = 1.58932547122958567343e-8;                   // 0x1.110b46p-26
constexpr double kPi2_W2 = 6.12323399573676603587e-17 - 1.58932547122958567343e-8 + (1.5707963267948966 - 1.570796310901641845703125);

// Exact hexadecimal float literals for the 5-word split of pi/2:
// pi/2 in hex: 0x1.921fb54442d18469898cc51701b839a2p+0
constexpr double kPi2_Hex0 = 0x1.921fb5p+0;          // 25 bits
constexpr double kPi2_Hex1 = 0x1.110b46p-26;         // 27 bits
constexpr double kPi2_Hex2 = 0x1.1a6263p-54;         // 25 bits
constexpr double kPi2_Hex3 = 0x1.8a2e03p-81;         // 25 bits
constexpr double kPi2_Hex4 = 0x1.c1cd10p-107;        // remaining bits

// 4-word exact hex split of ln(2):
// ln(2) in hex: 0x1.62e42fefa39ef35793c7673007e5ed5ep-1
constexpr double kLn2_Hex0 = 0x1.62e42fp-1;          // 25 bits
constexpr double kLn2_Hex1 = 0x1.df473dp-26;         // 25 bits
constexpr double kLn2_Hex2 = 0x1.cd5e4fp-51;         // 25 bits
constexpr double kLn2_Hex3 = 0x1.d9c01fp-75;

// Payne-Hanek 2/pi table (64-bit chunks of 2/pi = 0.a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebbc561b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f52...)
constexpr uint64_t kTwoOverPiBits[6] = {
    0xa2f9836e4e441529ULL,
    0xfc2757d1f534ddc0ULL,
    0xdb6295993c439041ULL,
    0xfe5163abdebbc561ULL,
    0xb7246e3a424dd2e0ULL,
    0x06492eea09d1921cULL,
};

// Reduces x mod pi/2 into r in [-pi/4, pi/4] in double-double, returning quadrant k mod 4.
inline int reduce_trig_pi2(double x, DD& out_r) noexcept {
    double ax = abs_f64(x);
    if (ax <= 0.7853981633974483) { // |x| <= pi/4
        out_r = {x, 0.0};
        return 0;
    }
    if (ax < 1048576.0) { // |x| < 2^20: exact 5-word hex reduction
        double kd = round_nearest_even(x * kTwoOverPi);
        int64_t k = static_cast<int64_t>(kd);
        double r0 = x - kd * kPi2_Hex0;
        DD r = two_diff(r0, kd * kPi2_Hex1);
        r = dd_sub(r, two_prod(kd, kPi2_Hex2));
        r = dd_sub(r, two_prod(kd, kPi2_Hex3));
        r = dd_add_d(r, -kd * kPi2_Hex4);
        out_r = r;
        return static_cast<int>(k & 3);
    }

    // Large argument Payne-Hanek style reduction using 24-bit chunks of ax
    // or multi-step reduction for |x| >= 2^20.
    int e = 0;
    double m = frexp_norm(ax, e);
    uint64_t mant = (to_bits(m) & kMantMask) | (1ULL << 52); // 53-bit integer, value = mant * 2^(e - 52)
    // We want (mant * 2^(e - 52)) * (2 / pi).
    // Let shift = e - 52. Since ax >= 2^20, e >= 20.
    // Multiply mant by kTwoOverPiBitschunks aligned to the binary point.
    int bit_pos = e - 52; // >= -32
    int word_idx = 0;
    int bit_off = bit_pos;
    while (bit_off <= -64 && word_idx < 4) {
        bit_off += 64;
        ++word_idx;
    }
    // Accumulate 192 bits around the binary point using 64x64->128 products.
    auto mul128 = [](uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) noexcept {
#if defined(__SIZEOF_INT128__)
        __uint128_t p = static_cast<__uint128_t>(a) * b;
        hi = static_cast<uint64_t>(p >> 64);
        lo = static_cast<uint64_t>(p);
#else
        uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
        uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
        uint64_t p0 = a_lo * b_lo;
        uint64_t p1 = a_lo * b_hi;
        uint64_t p2 = a_hi * b_lo;
        uint64_t p3 = a_hi * b_hi;
        uint64_t cy = ((p0 >> 32) + (p1 & 0xffffffffULL) + (p2 & 0xffffffffULL)) >> 32;
        lo = p0 + (p1 << 32) + (p2 << 32);
        hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;
#endif
    };

    // Build 4 words of 2/pi shifted by e - 52 so binary point is between w[0] and w[1].
    auto get_table_word = [&](int idx) noexcept -> uint64_t {
        if (idx < 0 || idx >= 6) return 0ULL;
        return kTwoOverPiBits[idx];
    };
    // In kTwoOverPiBits, the binary point is at the top of kTwoOverPiBits[0].
    // Multiplying by 2^(e - 52) shifts the binary point right by (e - 52) bits.
    //Wait: mant is 53 bits (bit 52 is 1, representing 1.xxxx * 2^52), so mant * 2^(e-52) = x.
    // And table[0] * 2^-64 + table[1] * 2^-128 + ... = 2/pi.
    // So mant * table[i] contributes at 2^(e - 52 - 64*(i+1)).
    uint64_t p_hi[6]{}, p_lo[6]{};
    for (int i = 0; i < 6; ++i) {
        mul128(mant, get_table_word(i), p_hi[i], p_lo[i]);
    }
    // Sum into a 6-word 64-bit array acc[0..6] (big-endian, where acc[0] has weight 2^(e - 52)):
    uint64_t acc[7]{};
    for (int i = 5; i >= 0; --i) {
        // p_lo[i] goes to acc[i+1], p_hi[i] goes to acc[i]
        uint64_t s1 = acc[i + 1] + p_lo[i];
        uint64_t c1 = (s1 < acc[i + 1]) ? 1ULL : 0ULL;
        acc[i + 1] = s1;
        uint64_t s0 = acc[i] + p_hi[i] + c1;
        uint64_t c0 = (s0 < acc[i] || (c1 && s0 == acc[i])) ? 1ULL : 0ULL;
        acc[i] = s0;
        for (int j = i - 1; j >= 0 && c0; --j) {
            uint64_t sj = acc[j] + c0;
            c0 = (sj < acc[j]) ? 1ULL : 0ULL;
            acc[j] = sj;
        }
    }
    // Weight of acc[k] is 2^(e - 52 - 64*k).
    // We want the 2 lowest integer bits (weight 2^0 and 2^1) and the fractional bits (weight < 2^0).
    // Find which bit in acc[] has weight 2^0:
    // Bit b (0..63, where 63 is MSB) of acc[k] has weight 2^(e - 52 - 64*k - (63 - b)) = 2^(e - 115 - 64*k + b).
    // For weight 2^0: 64*k - b = e - 115.
    // Easier: shift the entire bit-stream so the binary point aligns at the top of frac_hi!
    // Number of integer bits above 2^0 in acc[0..6]:
    // MSB of acc[0] has weight 2^(e - 52). So there are (e - 52 + 1) integer bits in acc[0..]!
    int int_bits = e - 51; // since e >= 20, int_bits >= -31.
    // Extract 64 bits ending at weight 2^0 (the integer part mod 2^64) and 128 fractional bits below 2^0:
    auto get_bit = [&](int bit_from_top) noexcept -> uint64_t {
        if (bit_from_top < 0 || bit_from_top >= 7 * 64) return 0ULL;
        int w = bit_from_top >> 6;
        int b = 63 - (bit_from_top & 63);
        return (acc[w] >> b) & 1ULL;
    };
    auto get_u64_at = [&](int start_bit_from_top) noexcept -> uint64_t {
        uint64_t val = 0;
        for (int i = 0; i < 64; ++i) {
            val = (val << 1) | get_bit(start_bit_from_top + i);
        }
        return val;
    };
    uint64_t int_part = get_u64_at(int_bits - 64);
    uint64_t frac_hi  = get_u64_at(int_bits);
    uint64_t frac_lo  = get_u64_at(int_bits + 64);

    // Round to nearest integer: if MSB of frac_hi is 1, increment int_part and subtract 1 from fraction.
    bool neg_frac = false;
    if (frac_hi >= 0x8000000000000000ULL) {
        ++int_part;
        neg_frac = true;
        frac_lo = ~frac_lo + 1ULL;
        frac_hi = ~frac_hi + (frac_lo == 0 ? 1ULL : 0ULL);
    }
    int quad = static_cast<int>(int_part & 3ULL);
    // Convert (frac_hi * 2^-64 + frac_lo * 2^-128) to DD and multiply by pi/2:
    double f_hi = scale_pow2(static_cast<double>(frac_hi >> 11), -53);
    double f_mid = scale_pow2(static_cast<double>(frac_hi & 0x7ffULL), -64) +
                   scale_pow2(static_cast<double>(frac_lo >> 11), -117);
    DD frac_dd = two_sum(f_hi, f_mid);
    if (neg_frac) frac_dd = dd_neg(frac_dd);
    DD r_dd = dd_mul(frac_dd, kPiOver2);
    if (to_bits(x) & kSignMask) {
        r_dd = dd_neg(r_dd);
        quad = (4 - quad) & 3;
    }
    out_r = r_dd;
    return quad;
}

// Evaluates sin(r) and cos(r) for |r| <= pi/4 in 106-bit DD using Taylor series
// up to r^23/23! (truncation error < 2^-89).
inline DD sin_poly_dd(DD r) noexcept {
    DD r2 = dd_mul(r, r);
    // 1 / (2k+1)! in double-double:
    static constexpr DD kSinCoeffs[] = {
        { 1.6666666666666666e-01,  9.2518585385429706e-18}, // 1/3!
        { 8.3333333333333332e-03,  4.6259292692714853e-19}, // 1/5!
        { 1.9841269841269841e-04,  1.1014117307789251e-20}, // 1/7!
        { 2.7557319223985893e-06, -1.8583932740464721e-22}, // 1/9!
        { 2.5052108385441720e-08, -1.4488140709359120e-24}, // 1/11!
        { 1.6059043836821613e-10,  1.2585294588752098e-26}, // 1/13!
        { 7.6471637318198164e-13,  7.0387287773345300e-30}, // 1/15!
        { 2.8114572543455206e-15,  1.6508842730861433e-31}, // 1/17!
        { 8.2206352466243295e-18,  0.0},                    // 1/19!
        { 1.9572941063391263e-20,  0.0},                    // 1/21!
        { 3.8681701706306840e-23,  0.0},                    // 1/23!
    };
    DD acc = kSinCoeffs[10];
    for (int i = 9; i >= 0; --i) {
        acc = dd_mul(acc, r2);
        if (i & 1) {
            acc = dd_sub(kSinCoeffs[i], acc);
        } else {
            acc = dd_sub(kSinCoeffs[i], acc);
        }
    }
    // Wait: let's write Horner's method with signed coefficients explicitly so signs are 100% clear!
    return acc;
}

inline DD eval_sin_dd_reduced(DD r) noexcept {
    DD r2 = dd_mul(r, r);
    static constexpr DD kC[] = {
        {-1.6666666666666666e-01, -9.2518585385429706e-18}, // -1/3!
        { 8.3333333333333332e-03,  4.6259292692714853e-19}, // +1/5!
        {-1.9841269841269841e-04, -1.1014117307789251e-20}, // -1/7!
        { 2.7557319223985893e-06, -1.8583932740464721e-22}, // +1/9!
        {-2.5052108385441720e-08,  1.4488140709359120e-24}, // -1/11!
        { 1.6059043836821613e-10,  1.2585294588752098e-26}, // +1/13!
        {-7.6471637318198164e-13, -7.0387287773345300e-30}, // -1/15!
        { 2.8114572543455206e-15,  1.6508842730861433e-31}, // +1/17!
        {-8.2206352466243295e-18,  0.0},                    // -1/19!
        { 1.9572941063391263e-20,  0.0},                    // +1/21!
        {-3.8681701706306840e-23,  0.0},                    // -1/23!
    };
    DD p = kC[10];
    for (int i = 9; i >= 0; --i) {
        p = dd_add(dd_mul(p, r2), kC[i]);
    }
    DD r3 = dd_mul(r, r2);
    return dd_add(r, dd_mul(r3, p));
}

inline DD eval_cos_dd_reduced(DD r) noexcept {
    DD r2 = dd_mul(r, r);
    static constexpr DD kC[] = {
        {-0.5,                     0.0},                    // -1/2!
        { 4.1666666666666664e-02,  2.3129646346357426e-18}, // +1/4!
        {-1.3888888888888889e-03, -7.7098821154524755e-20}, // -1/6!
        { 2.4801587301587302e-05,  1.3767646634736564e-21}, // +1/8!
        {-2.7557319223985891e-07,  1.8583932740464720e-23}, // -1/10!
        { 2.0876756987868099e-09, -1.2073450591132600e-25}, // +1/12!
        {-1.1470745597729725e-11,  2.0659512752330745e-28}, // -1/14!
        { 4.7794773323873853e-14,  4.3992054858340812e-31}, // +1/16!
        {-1.5619206968586225e-16,  0.0},                    // -1/18!
        { 4.1103176233121648e-19,  0.0},                    // +1/20!
        {-8.8967913924505741e-22,  0.0},                    // -1/22!
        { 1.6117375710961184e-24,  0.0},                    // +1/24!
    };
    DD p = kC[11];
    for (int i = 10; i >= 0; --i) {
        p = dd_add(dd_mul(p, r2), kC[i]);
    }
    return dd_add_d(dd_mul(r2, p), 1.0);
}

// ============================================================================
// Core log(x) and exp(x) in 106-bit Double-Double
// ============================================================================

inline DD eval_log_dd(double x) noexcept {
    int e = 0;
    double m = frexp_norm(x, e);
    // Center m into [sqrt(1/2), sqrt(2)] ~ [0.70710678, 1.41421356]:
    if (m > 1.4142135623730951) {
        m *= 0.5;
        ++e;
    }
    // Further split [0.7071, 1.4142] by multiplying by an exact dyadic reciprocal c
    // so m * c is within 1/16 of 1.0, giving |u| = |(m*c - 1)/(m*c + 1)| <= 1/31 ~ 0.032!
    // Actually even without a table, |u| = |(m - 1)/(m + 1)| <= (sqrt(2)-1)/(sqrt(2)+1) = 0.171572875.
    // 0.171573^31 / 31 = 5.5e-26 (~2^-84). With 16 terms (up to u^31/31), accuracy is 84 bits!
    DD m_dd = {m, 0.0};
    DD num = dd_add_d(m_dd, -1.0);
    DD den = dd_add_d(m_dd, 1.0);
    DD u = dd_div(num, den);
    DD u2 = dd_mul(u, u);

    DD acc = {1.0 / 31.0, 0.0};
    for (int k = 29; k >= 3; k -= 2) {
        DD c_k = dd_div({1.0, 0.0}, {static_cast<double>(k), 0.0});
        acc = dd_add(dd_mul(acc, u2), c_k);
    }
    acc = dd_add_d(dd_mul(acc, u2), 1.0);
    DD ln_m = dd_mul(dd_mul(u, acc), {2.0, 0.0});
    if (e != 0) {
        DD e_ln2 = dd_mul_d(kLn2, static_cast<double>(e));
        return dd_add(e_ln2, ln_m);
    }
    return ln_m;
}

inline double eval_exp_dd_to_f64(DD x_dd) noexcept {
    double x = to_double(x_dd);
    if (x > 709.78271289338397) return pos_inf();
    if (x < -745.13321910194122) return 0.0;

    double kd = round_nearest_even(x * kInvLn2);
    int k = static_cast<int>(kd);

    // r = x_dd - k * ln(2)
    DD r = dd_sub(x_dd, two_prod(kd, kLn2_Hex0));
    r = dd_sub(r, two_prod(kd, kLn2_Hex1));
    r = dd_sub(r, two_prod(kd, kLn2_Hex2));
    r = dd_add_d(r, -kd * kLn2_Hex3);

    // Taylor series for exp(r) on |r| <= ln(2)/2 ~ 0.34657:
    // 0.34657^21 / 21! ~ 4.2e-30 (~2^-97.6)
    static constexpr DD kExpC[] = {
        {0.5,                     0.0},                    // 1/2!
        {1.6666666666666666e-01,  9.2518585385429706e-18}, // 1/3!
        {4.1666666666666664e-02,  2.3129646346357426e-18}, // 1/4!
        {8.3333333333333332e-03,  4.6259292692714853e-19}, // 1/5!
        {1.3888888888888889e-03,  7.7098821154524755e-20}, // 1/6!
        {1.9841269841269841e-04,  1.1014117307789251e-20}, // 1/7!
        {2.4801587301587302e-05,  1.3767646634736564e-21}, // 1/8!
        {2.7557319223985893e-06, -1.8583932740464721e-22}, // 1/9!
        {2.7557319223985891e-07, -1.8583932740464720e-23}, // 1/10!
        {2.5052108385441720e-08, -1.4488140709359120e-24}, // 1/11!
        {2.0876756987868099e-09, -1.2073450591132600e-25}, // 1/12!
        {1.6059043836821613e-10,  1.2585294588752098e-26}, // 1/13!
        {1.1470745597729725e-11, -2.0659512752330745e-28}, // 1/14!
        {7.6471637318198164e-13,  7.0387287773345300e-30}, // 1/15!
        {4.7794773323873853e-14,  4.3992054858340812e-31}, // 1/16!
        {2.8114572543455206e-15,  1.6508842730861433e-31}, // 1/17!
        {1.5619206968586225e-16,  0.0},                    // 1/18!
        {8.2206352466243295e-18,  0.0},                    // 1/19!
        {4.1103176233121648e-19,  0.0},                    // 1/20!
    };
    DD p = kExpC[18];
    for (int i = 17; i >= 0; --i) {
        p = dd_add(dd_mul(p, r), kExpC[i]);
    }
    DD r2 = dd_mul(r, r);
    DD exp_r = dd_add_d(dd_add(r, dd_mul(r2, p)), 1.0);
    return scale_pow2(to_double(exp_r), k);
}

// ============================================================================
// Core atan(z) for z >= 0 in 106-bit Double-Double
// ============================================================================

inline DD eval_atan_pos_dd(DD z) noexcept {
    bool recip = false;
    if (z.hi > 1.0 || (z.hi == 1.0 && z.lo > 0.0)) {
        z = dd_div({1.0, 0.0}, z);
        recip = true;
    }

    // Segment [0, 1] into 8 intervals around c_idx = round(z * 8) / 8, so |w| <= 1/16 = 0.0625.
    // On |w| <= 1/16, w^21 / 21 < 2^-88!
    static constexpr double kCenters[9] = {
        0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
    };
    static constexpr DD kAtanCenters[9] = {
        {0.0, 0.0},
        {0.12435499454676144, -4.0933900103213653e-18}, // atan(1/8)
        {0.24497866312686414,  1.1919822223828298e-17}, // atan(2/8)
        {0.3587706702705722,   6.0680802307381280e-18}, // atan(3/8)
        {0.4636476090008061,   2.2698876406955228e-17}, // atan(4/8)
        {0.5585993153435624,   1.3091683447366858e-17}, // atan(5/8)
        {0.6435011087932844,  -1.3679726823213922e-17}, // atan(6/8)
        {0.7188299996216245,  -1.6968393822185346e-17}, // atan(7/8)
        {0.7853981633974483,   3.0616169978683830e-17}, // atan(1) = pi/4
    };

    int idx = static_cast<int>(z.hi * 8.0 + 0.5);
    if (idx < 0) idx = 0;
    if (idx > 8) idx = 8;
    double c = kCenters[idx];

    // w = (z - c) / (1 + z * c)
    DD num = dd_add_d(z, -c);
    DD den = dd_add_d(dd_mul_d(z, c), 1.0);
    DD w = dd_div(num, den);
    DD w2 = dd_mul(w, w);

    // Series: w * (1 - w^2/3 + w^4/5 - ... - w^18/19 + w^20/21)
    DD acc = {1.0 / 21.0, 0.0};
    for (int k = 19; k >= 3; k -= 2) {
        double sign = ((k & 2) != 0) ? -1.0 : 1.0;
        DD ck = dd_div({sign, 0.0}, {static_cast<double>(k), 0.0});
        acc = dd_add(dd_mul(acc, w2), ck);
    }
    acc = dd_add_d(dd_mul(acc, w2), 1.0);
    DD atan_w = dd_mul(w, acc);
    DD res = dd_add(kAtanCenters[idx], atan_w);
    if (recip) {
        res = dd_sub(kPiOver2, res);
    }
    return res;
}

} // namespace

double canonicalize_nan(double x) noexcept {
    uint64_t u = to_bits(x);
    return is_nan_bits(u) ? qnan() : x;
}

uint64_t canonical_f64_bits(double x) noexcept {
    uint64_t u = to_bits(x);
    return is_nan_bits(u) ? kCanonicalNaNBits : u;
}

double det_sin(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u) || is_inf_bits(u)) return qnan();
    if (is_zero_bits(u)) return x; // preserves +-0.0
    if (abs_f64(x) < 1.4901161193847656e-8) return x; // |x| < 2^-26: sin(x) == x to 53 bits

    DD r{};
    int q = reduce_trig_pi2(x, r);
    DD res{};
    switch (q & 3) {
        case 0: res = eval_sin_dd_reduced(r); break;
        case 1: res = eval_cos_dd_reduced(r); break;
        case 2: res = dd_neg(eval_sin_dd_reduced(r)); break;
        case 3: res = dd_neg(eval_cos_dd_reduced(r)); break;
    }
    return to_double(res);
}

double det_cos(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u) || is_inf_bits(u)) return qnan();
    if (abs_f64(x) < 1.4901161193847656e-8) return 1.0;

    DD r{};
    int q = reduce_trig_pi2(x, r);
    DD res{};
    switch (q & 3) {
        case 0: res = eval_cos_dd_reduced(r); break;
        case 1: res = dd_neg(eval_sin_dd_reduced(r)); break;
        case 2: res = dd_neg(eval_cos_dd_reduced(r)); break;
        case 3: res = eval_sin_dd_reduced(r); break;
    }
    return to_double(res);
}

double det_tan(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u) || is_inf_bits(u)) return qnan();
    if (is_zero_bits(u)) return x;
    if (abs_f64(x) < 1.4901161193847656e-8) return x;

    DD r{};
    int q = reduce_trig_pi2(x, r);
    DD s = eval_sin_dd_reduced(r);
    DD c = eval_cos_dd_reduced(r);
    DD res = ((q & 1) == 0) ? dd_div(s, c) : dd_neg(dd_div(c, s));
    return to_double(res);
}

double det_atan(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    if (is_zero_bits(u)) return x;
    if (is_inf_bits(u)) {
        return (u & kSignMask) ? -to_double(kPiOver2) : to_double(kPiOver2);
    }
    if (abs_f64(x) < 1.4901161193847656e-8) return x;

    DD res = eval_atan_pos_dd({abs_f64(x), 0.0});
    return copysign_f64(to_double(res), x);
}

double det_atan2(double y, double x) noexcept {
    uint64_t uy = to_bits(y);
    uint64_t ux = to_bits(x);
    if (is_nan_bits(uy) || is_nan_bits(ux)) return qnan();

    bool y_neg = (uy & kSignMask) != 0;
    bool x_neg = (ux & kSignMask) != 0;

    if (is_zero_bits(uy)) {
        if (x_neg) return y_neg ? -to_double(kPi) : to_double(kPi);
        return y; // +-0.0
    }
    if (is_zero_bits(ux)) {
        return y_neg ? -to_double(kPiOver2) : to_double(kPiOver2);
    }
    if (is_inf_bits(uy)) {
        if (is_inf_bits(ux)) {
            double mag = x_neg ? to_double(dd_mul_d(kPiOver4, 3.0)) : to_double(kPiOver4);
            return y_neg ? -mag : mag;
        }
        return y_neg ? -to_double(kPiOver2) : to_double(kPiOver2);
    }
    if (is_inf_bits(ux)) {
        if (x_neg) return y_neg ? -to_double(kPi) : to_double(kPi);
        return y_neg ? -0.0 : 0.0;
    }

    DD ay = {abs_f64(y), 0.0};
    DD ax = {abs_f64(x), 0.0};
    DD base = (ay.hi > ax.hi)
        ? dd_sub(kPiOver2, eval_atan_pos_dd(dd_div(ax, ay)))
        : eval_atan_pos_dd(dd_div(ay, ax));
    if (x_neg) {
        base = dd_sub(kPi, base);
    }
    double mag = to_double(base);
    return y_neg ? -mag : mag;
}

double det_asin(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    if (is_zero_bits(u)) return x;
    double ax = abs_f64(x);
    if (ax > 1.0) return qnan();
    if (ax == 1.0) return copysign_f64(to_double(kPiOver2), x);
    if (ax < 1.4901161193847656e-8) return x;

    // asin(x) = atan(x / sqrt((1 - x) * (1 + x)))
    DD x_dd = {ax, 0.0};
    DD one_minus = dd_sub({1.0, 0.0}, x_dd);
    DD one_plus  = dd_add({1.0, 0.0}, x_dd);
    DD root = dd_sqrt(dd_mul(one_minus, one_plus));
    DD res = eval_atan_pos_dd(dd_div(x_dd, root));
    return copysign_f64(to_double(res), x);
}

double det_acos(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    double ax = abs_f64(x);
    if (ax > 1.0) return qnan();
    if (x == 1.0) return 0.0;
    if (x == -1.0) return to_double(kPi);

    DD x_dd = {ax, 0.0};
    DD one_minus = dd_sub({1.0, 0.0}, x_dd);
    DD one_plus  = dd_add({1.0, 0.0}, x_dd);
    DD root = dd_sqrt(dd_mul(one_minus, one_plus));
    DD atan_part = eval_atan_pos_dd(dd_div(x_dd, root));
    DD res = (x >= 0.0) ? dd_sub(kPiOver2, atan_part) : dd_add(kPiOver2, atan_part);
    return to_double(res);
}

double det_exp(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    if (is_zero_bits(u)) return 1.0;
    if (u == kPosInf) return pos_inf();
    if (u == kNegInf) return 0.0;
    return eval_exp_dd_to_f64({x, 0.0});
}

double det_log(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    if (is_zero_bits(u)) return neg_inf();
    if (u & kSignMask) return qnan();
    if (u == kPosInf) return pos_inf();
    if (x == 1.0) return 0.0;
    return to_double(eval_log_dd(x));
}

double det_pow(double x, double y) noexcept {
    uint64_t ux = to_bits(x);
    uint64_t uy = to_bits(y);
    if (is_zero_bits(uy)) return 1.0;
    if (is_nan_bits(ux) || is_nan_bits(uy)) return qnan();
    if (x == 1.0) return is_inf_bits(uy) ? qnan() : 1.0;
    if (x == -1.0 && is_inf_bits(uy)) return qnan();

    // Check if y is an exact integer
    double ty = trunc_f64(y);
    bool y_is_int = (!is_inf_bits(uy) && ty == y);
    bool y_is_odd_int = false;
    if (y_is_int && abs_f64(y) < 9007199254740992.0) {
        int64_t yi = static_cast<int64_t>(y);
        y_is_odd_int = ((yi & 1) != 0);
    }

    if (is_zero_bits(ux)) {
        bool y_neg = (uy & kSignMask) != 0;
        if (y_neg) {
            return (y_is_odd_int && (ux & kSignMask)) ? neg_inf() : pos_inf();
        }
        return (y_is_odd_int && (ux & kSignMask)) ? -0.0 : 0.0;
    }
    if (is_inf_bits(uy)) {
        double ax = abs_f64(x);
        if (ax == 1.0) return qnan();
        bool y_pos = (uy == kPosInf);
        return ((ax > 1.0) == y_pos) ? pos_inf() : 0.0;
    }
    if (is_inf_bits(ux)) {
        bool y_neg = (uy & kSignMask) != 0;
        if (ux == kNegInf) {
            if (y_neg) return y_is_odd_int ? -0.0 : 0.0;
            return y_is_odd_int ? neg_inf() : pos_inf();
        }
        return y_neg ? 0.0 : pos_inf();
    }
    if ((ux & kSignMask) && !y_is_int) return qnan();

    double sign = ((ux & kSignMask) && y_is_odd_int) ? -1.0 : 1.0;
    DD ln_ax = eval_log_dd(abs_f64(x));
    DD prod = dd_mul_d(ln_ax, y);
    return sign * eval_exp_dd_to_f64(prod);
}

double det_hypot(double x, double y) noexcept {
    uint64_t ux = to_bits(x);
    uint64_t uy = to_bits(y);
    if (is_inf_bits(ux) || is_inf_bits(uy)) return pos_inf();
    if (is_nan_bits(ux) || is_nan_bits(uy)) return qnan();

    double ax = abs_f64(x);
    double ay = abs_f64(y);
    if (ax < ay) { double t = ax; ax = ay; ay = t; }
    if (ax == 0.0) return 0.0;

    int ex = 0;
    (void)frexp_norm(ax, ex);
    double sx = scale_pow2(ax, -ex);
    double sy = scale_pow2(ay, -ex);
    DD sum_sq = dd_add(two_prod(sx, sx), two_prod(sy, sy));
    DD root = dd_sqrt(sum_sq);
    return scale_pow2(to_double(root), ex);
}

double det_cbrt(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u)) return qnan();
    if (is_zero_bits(u) || is_inf_bits(u)) return x;

    double ax = abs_f64(x);
    int e = 0;
    double m = frexp_norm(ax, e);
    int rem = e % 3;
    if (rem < 0) rem += 3;
    int e3 = (e - rem) / 3;
    m = scale_pow2(m, rem); // m in [1, 8)

    // Initial approximation + 4 Newton-Raphson steps in double, then 2 in DD:
    double y = 0.5 + 0.3 * m;
    for (int i = 0; i < 6; ++i) {
        y = (2.0 * y + m / (y * y)) / 3.0;
    }
    DD y_dd = {y, 0.0};
    DD m_dd = {m, 0.0};
    for (int i = 0; i < 2; ++i) {
        DD y2 = dd_mul(y_dd, y_dd);
        DD q = dd_div(m_dd, y2);
        DD num = dd_add(dd_mul_d(y_dd, 2.0), q);
        y_dd = dd_div(num, {3.0, 0.0});
    }
    double mag = scale_pow2(to_double(y_dd), e3);
    return copysign_f64(mag, x);
}

uint64_t hash_bytes_64(const void* data, size_t size, uint64_t seed) noexcept {
    // Wyhash / SplitMix64-based deterministic 64-bit hash (endian-independent byte fold).
    auto mix = [](uint64_t z) noexcept -> uint64_t {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = mix(seed ^ (static_cast<uint64_t>(size) * 0x9e3779b97f4a7c15ULL));
    size_t i = 0;
    while (i + 8 <= size) {
        uint64_t word = 0;
        for (size_t b = 0; b < 8; ++b) {
            word |= static_cast<uint64_t>(bytes[i + b]) << (b * 8);
        }
        h = mix(h ^ word);
        i += 8;
    }
    if (i < size) {
        uint64_t tail = 0;
        for (size_t b = 0; i + b < size; ++b) {
            tail |= static_cast<uint64_t>(bytes[i + b]) << (b * 8);
        }
        h = mix(h ^ tail ^ 0xa0761d6478bd642fULL);
    }
    return h;
}

uint64_t hash_f64_state(const double* values, size_t count, uint64_t seed) noexcept {
    auto mix = [](uint64_t z) noexcept -> uint64_t {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    uint64_t h = mix(seed ^ (static_cast<uint64_t>(count) * 0x9e3779b97f4a7c15ULL));
    for (size_t i = 0; i < count; ++i) {
        uint64_t bits = canonical_f64_bits(values[i]);
        // Normalize -0.0 to +0.0 so -0.0 vs +0.0 does not cause spurious state hash drift if desired,
        // or preserve exact IEEE bits except NaN payloads:
        h = mix(h ^ bits);
    }
    return h;
}

} // namespace brass::math

extern "C" {

double brass_det_sin(double x) { return brass::math::det_sin(x); }
double brass_det_cos(double x) { return brass::math::det_cos(x); }
double brass_det_tan(double x) { return brass::math::det_tan(x); }
double brass_det_asin(double x) { return brass::math::det_asin(x); }
double brass_det_acos(double x) { return brass::math::det_acos(x); }
double brass_det_atan(double x) { return brass::math::det_atan(x); }
double brass_det_atan2(double y, double x) { return brass::math::det_atan2(y, x); }
double brass_det_exp(double x) { return brass::math::det_exp(x); }
double brass_det_log(double x) { return brass::math::det_log(x); }
double brass_det_pow(double x, double y) { return brass::math::det_pow(x, y); }
double brass_det_hypot(double x, double y) { return brass::math::det_hypot(x, y); }
double brass_det_cbrt(double x) { return brass::math::det_cbrt(x); }
uint64_t brass_det_hash_bytes(const void* data, size_t size, uint64_t seed) {
    return brass::math::hash_bytes_64(data, size, seed);
}
uint64_t brass_det_hash_f64_state(const double* values, size_t count, uint64_t seed) {
    return brass::math::hash_f64_state(values, count, seed);
}

} // extern "C"
