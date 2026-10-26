#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#include <brass/runtime/deterministic_math.hpp>
#include <bit>
#include <cmath>
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

// 106-bit square root for positive finite DD a. The machine's square root is
// IEEE-754's correctly rounded one, the same bits on every target as `/` is,
// so it seeds the root at 53 bits; one Newton step in double-double from
// there (the residual a - y*y taken exactly with two_prod) doubles that.
// The software iteration this replaced (five divisions in double, then two
// dd_div steps) cost Math.hypot ~250 ns a call.
inline DD dd_sqrt(DD a) noexcept {
    if (a.hi <= 0.0) return {0.0, 0.0};
    const double y = std::sqrt(a.hi);
    const DD r = dd_sub(a, two_prod(y, y));
    return quick_two_sum(y, r.hi / (2.0 * y));
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

// 2/pi in 24-bit digits, 2/pi = sum kTwoOverPi24[i] * 2^(-24 (i + 1)): 1584
// bits, enough to reduce any double (fdlibm's ipio2).
constexpr uint32_t kTwoOverPi24[66] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E,
    0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
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

    // Payne-Hanek: |x| * 2/pi in exact integer arithmetic, base-2^24 digits
    // acc[k] of weight 2^(-24k). Digits above the binary point are multiples
    // of 4 (x's quadrant mod 4 is all that is wanted) except acc[0]'s low
    // bits, so only the table digits that land at or below it are multiplied.
    // A double's distance to a multiple of pi/2 is at least 2^-61 of it, so
    // the eight fraction digits keep 130 significant bits or more.
    constexpr int kFrac = 8;
    constexpr int kGuard = 3;
    int e = 0;
    const double m = frexp_norm(ax, e);
    const uint64_t mant = (to_bits(m) & kMantMask) | (1ULL << 52); // ax = mant * 2^(e - 52)
    // mant * 2^(e - 52) = (mant << t) * 2^(24 q), 0 <= t < 24.
    const int s = e - 52;
    const int q = s >= 0 ? s / 24 : -((-s + 23) / 24);
    const int t = s - 24 * q;
    // mant << t is below 2^76: four 24-bit chunks, chunk j of weight 2^(24 (q + j)).
    uint64_t chunk[4];
    {
        const uint64_t lo = mant << t; // the low 64 bits (t < 24, so only bits >= 64 spill)
        const uint64_t hi = t ? mant >> (64 - t) : 0;
        chunk[0] = lo & 0xffffff;
        chunk[1] = (lo >> 24) & 0xffffff;
        chunk[2] = ((lo >> 48) | (hi << 16)) & 0xffffff;
        chunk[3] = (hi >> 8) & 0xffffff;
    }
    // chunk j times table digit i lands at digit 2^(24 (q + j - i - 1)) = acc[k], k = i + 1 - q - j.
    uint64_t acc[kFrac + kGuard + 1]{};
    for (int k = 0; k <= kFrac + kGuard; ++k) {
        for (int j = 0; j < 4; ++j) {
            const int i = k + q + j - 1;
            if (i >= 0 && i < 66) acc[k] += chunk[j] * kTwoOverPi24[i];
        }
    }
    for (int k = kFrac + kGuard; k > 0; --k) {
        acc[k - 1] += acc[k] >> 24;
        acc[k] &= 0xffffff;
    }
    int quad = static_cast<int>(acc[0] & 3);
    // Round to the nearest quadrant: a fraction of a half or more is 1 - it, negated.
    const bool neg_frac = acc[1] >= 0x800000;
    if (neg_frac) {
        ++quad;
        uint64_t borrow = 0;
        for (int k = kFrac + kGuard; k >= 1; --k) {
            const uint64_t sub = acc[k] + borrow;
            acc[k] = sub ? 0x1000000 - sub : 0;
            borrow = sub ? 1 : 0;
        }
    }
    // The fraction's first nonzero digit and the three after it, as two exact
    // 48-bit doubles.
    int k0 = 1;
    while (k0 < kFrac && acc[k0] == 0) ++k0;
    const double f_hi = scale_pow2(static_cast<double>((acc[k0] << 24) | acc[k0 + 1]), -24 * (k0 + 1));
    const double f_lo = scale_pow2(static_cast<double>((acc[k0 + 2] << 24) | acc[k0 + 3]), -24 * (k0 + 3));
    DD frac_dd = two_sum(f_hi, f_lo);
    if (neg_frac) frac_dd = dd_neg(frac_dd);
    DD r_dd = dd_mul(frac_dd, kPiOver2);
    if (to_bits(x) & kSignMask) {
        r_dd = dd_neg(r_dd);
        quad = -quad;
    }
    out_r = r_dd;
    return quad & 3;
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
// sin and cos: fdlibm's kernels and medium-range reduction (as V8 runs them,
// so Math.sin and Math.cos agree with V8's bit for bit below 2^19 * pi/2).
// Plain doubles with contraction off are as deterministic as the
// double-double path and cost a tenth of it.
// ============================================================================

inline int32_t high_word(double x) noexcept { return static_cast<int32_t>(to_bits(x) >> 32); }

// sin(x + y) for |x + y| <= pi/4, y the tail of x; iy = 0 when there is none.
inline double fdlibm_kernel_sin(double x, double y, int iy) noexcept {
    constexpr double S1 = -1.66666666666666324348e-01;
    constexpr double S2 = 8.33333333332248946124e-03;
    constexpr double S3 = -1.98412698298579493134e-04;
    constexpr double S4 = 2.75573137070700676789e-06;
    constexpr double S5 = -2.50507602534068634195e-08;
    constexpr double S6 = 1.58969099521155010221e-10;
    const int32_t ix = high_word(x) & 0x7fffffff;
    if (ix < 0x3e400000 && static_cast<int>(x) == 0) return x; // |x| < 2^-27
    const double z = x * x;
    const double v = z * x;
    const double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if (iy == 0) return x + v * (S1 + z * r);
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

// cos(x + y) for |x + y| <= pi/4.
inline double fdlibm_kernel_cos(double x, double y) noexcept {
    constexpr double C1 = 4.16666666666666019037e-02;
    constexpr double C2 = -1.38888888888741095749e-03;
    constexpr double C3 = 2.48015872894767294178e-05;
    constexpr double C4 = -2.75573143513906633035e-07;
    constexpr double C5 = 2.08757232129817482790e-09;
    constexpr double C6 = -1.13596475577881948265e-11;
    const int32_t ix = high_word(x) & 0x7fffffff;
    if (ix < 0x3e400000 && static_cast<int>(x) == 0) return 1.0; // |x| < 2^-27
    const double z = x * x;
    const double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if (ix < 0x3fd33333) return 1.0 - (0.5 * z - (z * r - x * y)); // |x| < 0.3
    // qx = x/4, its low word cleared (0.28125 above 0.78125).
    const double qx = ix > 0x3fe90000 ? 0.28125 : from_bits(static_cast<uint64_t>(ix - 0x00200000) << 32);
    const double hz = 0.5 * z - qx;
    const double a = 1.0 - qx;
    return a - (hz - (z * r - x * y));
}

// x reduced by pi/2 into y0 + y1 (|y0 + y1| <= pi/4), returning the multiple
// of pi/2 taken. fdlibm's __ieee754_rem_pio2 below 2^19 * pi/2 (its table of
// near-multiples only skips the cancellation check, which is made always);
// above, the Payne-Hanek double-double reduction.
inline int fdlibm_rem_pio2(double x, double& y0, double& y1) noexcept {
    constexpr double invpio2 = 6.36619772367581382433e-01;
    constexpr double pio2_1 = 1.57079632673412561417e+00;
    constexpr double pio2_1t = 6.07710050650619224932e-11;
    constexpr double pio2_2 = 6.07710050630396597660e-11;
    constexpr double pio2_2t = 2.02226624879595063154e-21;
    constexpr double pio2_3 = 2.02226624871116645580e-21;
    constexpr double pio2_3t = 8.47842766036889956997e-32;
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix < 0x4002d97c) { // |x| < 3pi/4: n = +-1
        if (hx > 0) {
            double z = x - pio2_1;
            if (ix != 0x3ff921fb) {
                y0 = z - pio2_1t;
                y1 = (z - y0) - pio2_1t;
            } else { // near pi/2: 33+33+53 bits of pi
                z -= pio2_2;
                y0 = z - pio2_2t;
                y1 = (z - y0) - pio2_2t;
            }
            return 1;
        }
        double z = x + pio2_1;
        if (ix != 0x3ff921fb) {
            y0 = z + pio2_1t;
            y1 = (z - y0) + pio2_1t;
        } else {
            z += pio2_2;
            y0 = z + pio2_2t;
            y1 = (z - y0) + pio2_2t;
        }
        return -1;
    }
    if (ix <= 0x413921fb) { // |x| <= 2^19 * pi/2
        const double t = abs_f64(x);
        const int32_t n = static_cast<int32_t>(t * invpio2 + 0.5);
        const double fn = static_cast<double>(n);
        double r = t - fn * pio2_1;
        double w = fn * pio2_1t; // good to 85 bits
        const int32_t j = ix >> 20;
        y0 = r - w;
        int32_t i = j - ((high_word(y0) >> 20) & 0x7ff);
        if (i > 16) { // cancellation: a second step, good to 118 bits
            double u = r;
            w = fn * pio2_2;
            r = u - w;
            w = fn * pio2_2t - ((u - r) - w);
            y0 = r - w;
            i = j - ((high_word(y0) >> 20) & 0x7ff);
            if (i > 49) { // a third, 151 bits
                u = r;
                w = fn * pio2_3;
                r = u - w;
                w = fn * pio2_3t - ((u - r) - w);
                y0 = r - w;
            }
        }
        y1 = (r - y0) - w;
        if (hx < 0) {
            y0 = -y0;
            y1 = -y1;
            return -n;
        }
        return n;
    }
    DD r{};
    const int q = reduce_trig_pi2(x, r);
    y0 = r.hi + r.lo;
    y1 = r.lo - (y0 - r.hi);
    return q;
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
    if ((high_word(x) & 0x7fffffff) <= 0x3fe921fb) return fdlibm_kernel_sin(x, 0.0, 0); // |x| <= pi/4

    double y0 = 0.0, y1 = 0.0;
    switch (fdlibm_rem_pio2(x, y0, y1) & 3) {
        case 0: return fdlibm_kernel_sin(y0, y1, 1);
        case 1: return fdlibm_kernel_cos(y0, y1);
        case 2: return -fdlibm_kernel_sin(y0, y1, 1);
        default: return -fdlibm_kernel_cos(y0, y1);
    }
}

double det_cos(double x) noexcept {
    uint64_t u = to_bits(x);
    if (is_nan_bits(u) || is_inf_bits(u)) return qnan();
    if ((high_word(x) & 0x7fffffff) <= 0x3fe921fb) return fdlibm_kernel_cos(x, 0.0); // |x| <= pi/4

    double y0 = 0.0, y1 = 0.0;
    switch (fdlibm_rem_pio2(x, y0, y1) & 3) {
        case 0: return fdlibm_kernel_cos(y0, y1);
        case 1: return -fdlibm_kernel_sin(y0, y1, 1);
        case 2: return -fdlibm_kernel_cos(y0, y1);
        default: return fdlibm_kernel_sin(y0, y1, 1);
    }
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
