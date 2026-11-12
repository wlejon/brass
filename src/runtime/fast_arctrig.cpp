// Contraction off: an FMA rounds once where fdlibm's expressions round twice,
// so a contracted kernel would not give V8's bits.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// asin, atan and atan2: fdlibm's e_asin.c, s_atan.c and e_atan2.c as V8 runs
// them (ieee754::asin/atan/atan2), so Math.asin, Math.atan and Math.atan2
// agree with V8's bit for bit. glibc's atan2 saves, sets and restores the
// rounding mode on every call and works in double-double; these touch no FP
// state. Every expression is fdlibm's, in fdlibm's order, so the results are
// its bits; the special cases are ordered so that a finite, non-zero pair
// (what a rotation matrix hands Euler.setFromRotationMatrix) meets two
// compares before its division.

#include <brass/runtime/fast_trig.hpp>
#include <bit>
#include <cmath>
#include <cstdint>

namespace brass::math {

namespace {

constexpr uint64_t kSignMask = 0x8000000000000000ULL;
constexpr uint64_t kAbsInf = 0x7ff0000000000000ULL;

constexpr uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
constexpr double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }
inline double abs_f64(double x) noexcept { return from_bits(to_bits(x) & ~kSignMask); }

constexpr double kAtanHi[4] = {
    4.63647609000806093515e-01,  // atan(0.5) high
    7.85398163397448278999e-01,  // atan(1.0) high
    9.82793723247329054082e-01,  // atan(1.5) high
    1.57079632679489655800e+00,  // atan(inf) high
};
constexpr double kAtanLo[4] = {
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
};
constexpr double kAT[11] = {
    3.33333333333329318027e-01,  -1.99999999998764832476e-01, 1.42857142725034663711e-01,
    -1.11111104054623557880e-01, 9.09088713343650656196e-02,  -7.69187620504482999495e-02,
    6.66107313738753120669e-02,  -5.83357013379057348645e-02, 4.97687799461593236017e-02,
    -3.65315727442169155270e-02, 1.62858201153657823623e-02,
};

constexpr double kPi = 3.1415926535897931160e+00;
constexpr double kPiLo = 1.2246467991473531772e-16;
constexpr double kPiO2 = 1.5707963267948965580e+00;
constexpr double kPiO4 = 7.8539816339744827900e-01;
constexpr double kTiny = 1.0e-300;

// s_atan.c past its NaN and |x| >= 2^66 cases: `x` finite or infinite below
// that bound, `ix` the high word of |x|, `negative` its sign.
inline double atan_reduced(double x, int32_t ix, bool negative) noexcept {
    int id;
    if (ix < 0x3fdc0000) {  // |x| < 0.4375
        if (ix < 0x3e200000) return x;  // |x| < 2^-29
        id = -1;
    } else {
        x = abs_f64(x);
        if (ix < 0x3ff30000) {      // |x| < 1.1875
            if (ix < 0x3fe60000) {  // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {  // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if (ix < 0x40038000) {  // |x| < 2.4375
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {  // 2.4375 <= |x| < 2^66
            id = 3;
            x = -1.0 / x;
        }
    }
    const double z = x * x;
    const double w = z * z;
    const double s1 = z * (kAT[0] + w * (kAT[2] + w * (kAT[4] + w * (kAT[6] + w * (kAT[8] + w * kAT[10])))));
    const double s2 = w * (kAT[1] + w * (kAT[3] + w * (kAT[5] + w * (kAT[7] + w * kAT[9]))));
    if (id < 0) return x - x * (s1 + s2);
    const double r = kAtanHi[id] - ((x * (s1 + s2) - kAtanLo[id]) - x);
    return negative ? -r : r;
}

} // namespace

double atan_f64(double x) noexcept {
    const uint64_t bits = to_bits(x);
    const uint64_t abs = bits & ~kSignMask;
    const auto ix = static_cast<int32_t>(abs >> 32);
    const bool negative = (bits >> 63) != 0;
    if (ix >= 0x44100000) {  // |x| >= 2^66
        if (abs > kAbsInf) return x + x;  // NaN
        return negative ? -kAtanHi[3] - kAtanLo[3] : kAtanHi[3] + kAtanLo[3];
    }
    return atan_reduced(x, ix, negative);
}

double atan2_f64(double y, double x) noexcept {
    const uint64_t xb = to_bits(x);
    const uint64_t yb = to_bits(y);
    const uint64_t xa = xb & ~kSignMask;
    const uint64_t ya = yb & ~kSignMask;
    // 2 * sign(x) + sign(y)
    uint32_t m = static_cast<uint32_t>(yb >> 63) | (static_cast<uint32_t>(xb >> 62) & 2u);
    // Zero, infinite or NaN in either operand: (a - 1) wraps a zero past
    // every finite value.
    if (xa - 1 >= kAbsInf - 1 || ya - 1 >= kAbsInf - 1) [[unlikely]] {
        if (xa > kAbsInf || ya > kAbsInf) return x + y;  // NaN
        if (xb == to_bits(1.0)) return atan_f64(y);
        if (ya == 0) {
            switch (m) {
                case 0:
                case 1: return y;  // atan(+-0, +anything) = +-0
                case 2: return kPi + kTiny;  // atan(+0, -anything) = pi
                default: return -kPi - kTiny;  // atan(-0, -anything) = -pi
            }
        }
        if (xa == 0) return (m & 1u) ? -kPiO2 - kTiny : kPiO2 + kTiny;
        if (xa == kAbsInf) {
            if (ya == kAbsInf) {
                switch (m) {
                    case 0: return kPiO4 + kTiny;
                    case 1: return -kPiO4 - kTiny;
                    case 2: return 3.0 * kPiO4 + kTiny;
                    default: return -3.0 * kPiO4 - kTiny;
                }
            }
            switch (m) {
                case 0: return 0.0;
                case 1: return -0.0;
                case 2: return kPi + kTiny;
                default: return -kPi - kTiny;
            }
        }
        // y is infinite, x finite and non-zero
        return (m & 1u) ? -kPiO2 - kTiny : kPiO2 + kTiny;
    }
    if (xb == to_bits(1.0)) return atan_f64(y);

    const auto ix = static_cast<int32_t>(xa >> 32);
    const auto iy = static_cast<int32_t>(ya >> 32);
    const int32_t k = (iy - ix) >> 20;
    double z;
    if (k > 60) {  // |y/x| > 2^60
        z = kPiO2 + 0.5 * kPiLo;
        m &= 1u;
    } else if ((m & 2u) != 0 && k < -60) {  // 0 > |y|/x > -2^-60
        z = 0.0;
    } else {
        const double q = abs_f64(y / x);
        const uint64_t qa = to_bits(q);
        const auto iq = static_cast<int32_t>(qa >> 32);
        // q is non-negative and no NaN; at or above 2^66 it is atan's
        // saturated answer.
        z = iq >= 0x44100000 ? kAtanHi[3] + kAtanLo[3] : atan_reduced(q, iq, false);
    }
    switch (m) {
        case 0: return z;
        case 1: return -z;
        case 2: return kPi - (z - kPiLo);
        default: return (z - kPiLo) - kPi;
    }
}

double asin_f64(double x) noexcept {
    constexpr double pio2_hi = 1.57079632679489655800e+00;
    constexpr double pio2_lo = 6.12323399573676603587e-17;
    constexpr double pio4_hi = 7.85398163397448278999e-01;
    constexpr double pS0 = 1.66666666666666657415e-01;
    constexpr double pS1 = -3.25565818622400915405e-01;
    constexpr double pS2 = 2.01212532134862925881e-01;
    constexpr double pS3 = -4.00555345006794114027e-02;
    constexpr double pS4 = 7.91534994289814532176e-04;
    constexpr double pS5 = 3.47933107596021167570e-05;
    constexpr double qS1 = -2.40339491173441421878e+00;
    constexpr double qS2 = 2.02094576023350569471e+00;
    constexpr double qS3 = -6.88283971605453293030e-01;
    constexpr double qS4 = 7.70381505559019352791e-02;

    const uint64_t bits = to_bits(x);
    const uint64_t abs = bits & ~kSignMask;
    const auto ix = static_cast<int32_t>(abs >> 32);
    if (ix >= 0x3ff00000) {  // |x| >= 1
        if (abs == to_bits(1.0)) return x * pio2_hi + x * pio2_lo;  // asin(+-1) = +-pi/2
        return (x - x) / (x - x);  // NaN
    }
    if (ix < 0x3fe00000) {  // |x| < 0.5
        if (ix < 0x3e400000) return x;  // |x| < 2^-27
        const double t = x * x;
        const double p = t * (pS0 + t * (pS1 + t * (pS2 + t * (pS3 + t * (pS4 + t * pS5)))));
        const double q = 1.0 + t * (qS1 + t * (qS2 + t * (qS3 + t * qS4)));
        return x + x * (p / q);
    }
    // 1 > |x| >= 0.5
    double w = 1.0 - abs_f64(x);
    double t = w * 0.5;
    double p = t * (pS0 + t * (pS1 + t * (pS2 + t * (pS3 + t * (pS4 + t * pS5)))));
    double q = 1.0 + t * (qS1 + t * (qS2 + t * (qS3 + t * qS4)));
    const double s = std::sqrt(t);
    if (ix >= 0x3fef3333) {  // |x| > 0.975
        w = p / q;
        t = pio2_hi - (2.0 * (s + s * w) - pio2_lo);
    } else {
        w = from_bits(to_bits(s) & 0xffffffff00000000ULL);
        const double c = (t - w * w) / (s + w);
        const double r = p / q;
        p = 2.0 * s * r - (pio2_lo - 2.0 * c);
        q = pio4_hi - 2.0 * w;
        t = pio4_hi - (p - q);
    }
    return (bits >> 63) != 0 ? -t : t;
}

} // namespace brass::math
