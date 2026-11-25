// Contraction off: an FMA rounds once where fdlibm's expressions round twice,
// so a contracted kernel would not give V8's bits (and would differ between
// a target that fuses by default, such as clang on arm64, and one that does not).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// The rest of Math's transcendentals as V8 computes them: expm1, log1p,
// log2, log10, cbrt, sinh, cosh, tanh, asinh, acosh, atanh (fdlibm, as V8's
// src/base/ieee754.cc ran it before V8 moved to llvm-libc; V8 branch-heads/
// 11.9), and hypot (V8's Math.hypot: scaled by the largest magnitude, the
// squares Kahan-summed). The platform's libm answers these differently from
// one platform to the next in the last bit, so a program that steps a
// simulation on them (a lockstep game) would part ways between machines.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// The original source code covered by the above license above has been
// modified significantly by Google Inc.
// Copyright 2016 the V8 project authors. All rights reserved.
// (BSD-style licence: https://github.com/v8/v8/blob/main/LICENSE)

#include <brass/runtime/fast_trig.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace brass::math {

namespace {

inline uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
inline double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }
inline int32_t high_word(double x) noexcept { return static_cast<int32_t>(to_bits(x) >> 32); }
inline uint32_t low_word(double x) noexcept { return static_cast<uint32_t>(to_bits(x)); }
inline double from_words(uint32_t hi, uint32_t lo) noexcept {
    return from_bits((static_cast<uint64_t>(hi) << 32) | lo);
}
inline double with_high_word(double x, uint32_t hi) noexcept {
    return from_bits((static_cast<uint64_t>(hi) << 32) | (to_bits(x) & 0xffffffffULL));
}
inline double with_low_word(double x, uint32_t lo) noexcept {
    return from_bits((to_bits(x) & 0xffffffff00000000ULL) | lo);
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

constexpr double Lg1 = 6.666666666666735130e-01;  // 3FE55555 55555593
constexpr double Lg2 = 3.999999999940941908e-01;  // 3FD99999 9997FA04
constexpr double Lg3 = 2.857142874366239149e-01;  // 3FD24924 94229359
constexpr double Lg4 = 2.222219843214978396e-01;  // 3FCC71C5 1D8E78AF
constexpr double Lg5 = 1.818357216161805012e-01;  // 3FC74664 96CB03DE
constexpr double Lg6 = 1.531383769920937332e-01;  // 3FC39A09 D078C69F
constexpr double Lg7 = 1.479819860511658591e-01;  // 3FC2F112 DF3E5244

// log(1+f) - f + f*f/2, for f in [sqrt(2)/2 - 1, sqrt(2) - 1] (k_log.h).
inline double k_log1p(double f) noexcept {
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const double R = t2 + t1;
    const double hfsq = 0.5 * f * f;
    return s * (hfsq + R);
}

} // namespace

double expm1_f64(double x) noexcept {
    constexpr double one = 1.0;
    constexpr double tiny = 1.0e-300;
    constexpr double o_threshold = 7.09782712893383973096e+02;  // 0x40862E42, 0xFEFA39EF
    constexpr double ln2_hi = 6.93147180369123816490e-01;       // 0x3FE62E42, 0xFEE00000
    constexpr double ln2_lo = 1.90821492927058770002e-10;       // 0x3DEA39EF, 0x35793C76
    constexpr double invln2 = 1.44269504088896338700e+00;       // 0x3FF71547, 0x652B82FE
    // Scaled Q's: Qn_here = 2**n * Qn_above, for R(2*z) where z = hxs = x*x/2.
    constexpr double Q1 = -3.33333333333331316428e-02;  // BFA11111 111110F4
    constexpr double Q2 = 1.58730158725481460165e-03;   // 3F5A01A0 19FE5585
    constexpr double Q3 = -7.93650757867487942473e-05;  // BF14CE19 9EAADBB7
    constexpr double Q4 = 4.00821782732936239552e-06;   // 3ED0CFCA 86E65239
    constexpr double Q5 = -2.01099218183624371326e-07;  // BE8AFDB7 6E09C32D
    static volatile double huge = 1.0e+300;

    double y, hi, lo, c = 0.0, t, e, hxs, hfx, r1, twopk;
    int32_t k;

    uint32_t hx = static_cast<uint32_t>(high_word(x));
    const uint32_t xsb = hx & 0x80000000;  // sign bit of x
    hx &= 0x7FFFFFFF;                      // high word of |x|

    // filter out huge and non-finite argument
    if (hx >= 0x4043687A) {      // if |x|>=56*ln2
        if (hx >= 0x40862E42) {  // if |x|>=709.78...
            if (hx >= 0x7FF00000) {
                if (((hx & 0xFFFFF) | low_word(x)) != 0) return x + x;  // NaN
                return (xsb == 0) ? x : -1.0;                           // exp(+-inf)={inf,-1}
            }
            if (x > o_threshold) return huge * huge;  // overflow
        }
        if (xsb != 0) {            // x < -56*ln2, return -1.0 with inexact
            if (x + tiny < 0.0)    // raise inexact
                return tiny - one;  // return -1
        }
    }

    // argument reduction
    if (hx > 0x3FD62E42) {      // if  |x| > 0.5 ln2
        if (hx < 0x3FF0A2B2) {  // and |x| < 1.5 ln2
            if (xsb == 0) {
                hi = x - ln2_hi;
                lo = ln2_lo;
                k = 1;
            } else {
                hi = x + ln2_hi;
                lo = -ln2_lo;
                k = -1;
            }
        } else {
            k = static_cast<int32_t>(invln2 * x + ((xsb == 0) ? 0.5 : -0.5));
            t = k;
            hi = x - t * ln2_hi;  // t*ln2_hi is exact here
            lo = t * ln2_lo;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if (hx < 0x3C900000) {  // when |x|<2**-54, return x
        t = huge + x;              // return x with inexact flags when x!=0
        return x - (t - (huge + x));
    } else {
        k = 0;
    }

    // x is now in primary range
    hfx = 0.5 * x;
    hxs = x * hfx;
    r1 = one + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    t = 3.0 - r1 * hfx;
    e = hxs * ((r1 - t) / (6.0 - x * t));
    if (k == 0) return x - (x * e - hxs);  // c is 0
    twopk = from_words(0x3FF00000u + (static_cast<uint32_t>(k) << 20), 0);  // 2^k
    e = (x * (e - c) - c);
    e -= hxs;
    if (k == -1) return 0.5 * (x - e) - 0.5;
    if (k == 1) {
        if (x < -0.25) return -2.0 * (e - (x + 0.5));
        return one + 2.0 * (x - e);
    }
    if (k <= -2 || k > 56) {  // suffice to return exp(x)-1
        y = one - (e - x);
        if (k == 1024) y = y * 2.0 * 8.98846567431158e+307;
        else y = y * twopk;
        return y - one;
    }
    t = one;
    if (k < 20) {
        t = with_high_word(t, 0x3FF00000u - (0x200000u >> k));  // t=1-2^-k
        y = t - (e - x);
        y = y * twopk;
    } else {
        t = with_high_word(t, static_cast<uint32_t>(0x3FF - k) << 20);  // 2^-k
        y = x - (e + t);
        y += one;
        y = y * twopk;
    }
    return y;
}

double log1p_f64(double x) noexcept {
    constexpr double ln2_hi = 6.93147180369123816490e-01;  // 3fe62e42 fee00000
    constexpr double ln2_lo = 1.90821492927058770002e-10;  // 3dea39ef 35793c76
    constexpr double two54 = 1.80143985094819840000e+16;   // 43500000 00000000
    constexpr double Lp1 = 6.666666666666735130e-01;       // 3FE55555 55555593
    constexpr double Lp2 = 3.999999999940941908e-01;       // 3FD99999 9997FA04
    constexpr double Lp3 = 2.857142874366239149e-01;       // 3FD24924 94229359
    constexpr double Lp4 = 2.222219843214978396e-01;       // 3FCC71C5 1D8E78AF
    constexpr double Lp5 = 1.818357216161805012e-01;       // 3FC74664 96CB03DE
    constexpr double Lp6 = 1.531383769920937332e-01;       // 3FC39A09 D078C69F
    constexpr double Lp7 = 1.479819860511658591e-01;       // 3FC2F112 DF3E5244
    constexpr double zero = 0.0;

    double hfsq, f = 0.0, c = 0.0, s, z, R, u;
    int32_t k, hu = 0;

    const int32_t hx = high_word(x);
    const int32_t ax = hx & 0x7FFFFFFF;

    k = 1;
    if (hx < 0x3FDA827A) {        // 1+x < sqrt(2)+
        if (ax >= 0x3FF00000) {   // x <= -1.0
            if (x == -1.0) return -kInf;  // log1p(-1)=-inf
            return kNaN;                  // log1p(x<-1)=NaN
        }
        if (ax < 0x3E200000) {                         // |x| < 2**-29
            if (two54 + x > zero && ax < 0x3C900000)   // |x| < 2**-54
                return x;
            return x - x * x * 0.5;
        }
        if (hx > 0 || hx <= static_cast<int32_t>(0xBFD2BEC4)) {
            k = 0;
            f = x;
            hu = 1;
        }  // sqrt(2)/2- <= 1+x < sqrt(2)+
    }
    if (hx >= 0x7FF00000) return x + x;
    if (k != 0) {
        if (hx < 0x43400000) {
            u = 1.0 + x;
            hu = high_word(u);
            k = (hu >> 20) - 1023;
            c = (k > 0) ? 1.0 - (u - x) : x - (u - 1.0);  // correction term
            c /= u;
        } else {
            u = x;
            hu = high_word(u);
            k = (hu >> 20) - 1023;
            c = 0;
        }
        hu &= 0x000FFFFF;
        if (hu < 0x6A09E) {  // u ~< sqrt(2)
            u = with_high_word(u, static_cast<uint32_t>(hu | 0x3FF00000));  // normalize u
        } else {
            k += 1;
            u = with_high_word(u, static_cast<uint32_t>(hu | 0x3FE00000));  // normalize u/2
            hu = (0x00100000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    hfsq = 0.5 * f * f;
    if (hu == 0) {  // |f| < 2**-20
        if (f == zero) {
            if (k == 0) return zero;
            c += k * ln2_lo;
            return k * ln2_hi + c;
        }
        R = hfsq * (1.0 - 0.66666666666666666 * f);
        if (k == 0) return f - R;
        return k * ln2_hi - ((R - (k * ln2_lo + c)) - f);
    }
    s = f / (2.0 + f);
    z = s * s;
    R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
    if (k == 0) return f - (hfsq - s * (hfsq + R));
    return k * ln2_hi - ((hfsq - (s * (hfsq + R) + (k * ln2_lo + c))) - f);
}

double log2_f64(double x) noexcept {
    constexpr double two54 = 1.80143985094819840000e+16;    // 0x43500000, 0x00000000
    constexpr double ivln2hi = 1.44269504072144627571e+00;  // 0x3FF71547, 0x65200000
    constexpr double ivln2lo = 1.67517131648865118353e-10;  // 0x3DE705FC, 0x2EEFA200

    int32_t hx = high_word(x);
    const uint32_t lx = low_word(x);

    int32_t k = 0;
    if (hx < 0x00100000) {  // x < 2**-1022
        if (((hx & 0x7FFFFFFF) | static_cast<int32_t>(lx)) == 0) return -kInf;  // log(+-0)=-inf
        if (hx < 0) return kNaN;  // log(-#) = NaN
        k -= 54;
        x *= two54;  // subnormal number, scale up x
        hx = high_word(x);
    }
    if (hx >= 0x7FF00000) return x + x;
    if (hx == 0x3FF00000 && lx == 0) return 0.0;  // log(1) = +0
    k += (hx >> 20) - 1023;
    hx &= 0x000FFFFF;
    const int32_t i = (hx + 0x95F64) & 0x100000;
    x = with_high_word(x, static_cast<uint32_t>(hx | (i ^ 0x3FF00000)));  // normalize x or x/2
    k += (i >> 20);
    const double y = static_cast<double>(k);
    const double f = x - 1.0;
    const double hfsq = 0.5 * f * f;
    const double r = k_log1p(f);

    // f-hfsq in extra precision (Dekker), then y added in extra precision too.
    double hi = f - hfsq;
    hi = with_low_word(hi, 0);
    const double lo = (f - hi) - hfsq + r;
    double val_hi = hi * ivln2hi;
    double val_lo = (lo + hi) * ivln2lo + lo * ivln2hi;

    const double w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;

    return val_lo + val_hi;
}

double log10_f64(double x) noexcept {
    constexpr double two54 = 1.80143985094819840000e+16;  // 0x43500000, 0x00000000
    constexpr double ivln10 = 4.34294481903251816668e-01;
    constexpr double log10_2hi = 3.01029995663611771306e-01;  // 0x3FD34413, 0x509F6000
    constexpr double log10_2lo = 3.69423907715893078616e-13;  // 0x3D59FEF3, 0x11F12B36

    int32_t hx = high_word(x);
    uint32_t lx = low_word(x);

    int32_t k = 0;
    if (hx < 0x00100000) {  // x < 2**-1022
        if (((hx & 0x7FFFFFFF) | static_cast<int32_t>(lx)) == 0) return -kInf;  // log(+-0)=-inf
        if (hx < 0) return kNaN;  // log(-#) = NaN
        k -= 54;
        x *= two54;  // subnormal number, scale up x
        hx = high_word(x);
        lx = low_word(x);
    }
    if (hx >= 0x7FF00000) return x + x;
    if (hx == 0x3FF00000 && lx == 0) return 0.0;  // log(1) = +0
    k += (hx >> 20) - 1023;

    const int32_t i = static_cast<int32_t>((static_cast<uint32_t>(k) & 0x80000000u) >> 31);
    hx = (hx & 0x000FFFFF) | ((0x3FF - i) << 20);
    const double y = k + i;
    x = from_words(static_cast<uint32_t>(hx), lx);

    const double z = y * log10_2lo + ivln10 * log_f64(x);
    return z + y * log10_2hi;
}

double cbrt_f64(double x) noexcept {
    constexpr uint32_t B1 = 715094163;  // B1 = (1023-1023/3-0.03306235651)*2**20
    constexpr uint32_t B2 = 696219795;  // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
    // |1/cbrt(x) - p(x)| < 2**-23.5 (~[-7.93e-8, 7.929e-8]).
    constexpr double P0 = 1.87595182427177009643;    // 0x3FFE03E6, 0x0F61E692
    constexpr double P1 = -1.88497979543377169875;   // 0xBFFE28E0, 0x92F02420
    constexpr double P2 = 1.621429720105354466140;   // 0x3FF9F160, 0x4A49D6C2
    constexpr double P3 = -0.758397934778766047437;  // 0xBFE844CB, 0xBEE751D9
    constexpr double P4 = 0.145996192886612446982;   // 0x3FC2B000, 0xD4E4EDD7

    double r, s, t = 0.0, w;

    uint32_t hx = static_cast<uint32_t>(high_word(x));
    const uint32_t low = low_word(x);
    const uint32_t sign = hx & 0x80000000;  // sign= sign(x)
    hx ^= sign;
    if (hx >= 0x7FF00000) return (x + x);  // cbrt(NaN,INF) is itself

    // Rough cbrt to 5 bits, by integer division of the exponent's bits.
    if (hx < 0x00100000) {                 // zero or subnormal?
        if ((hx | low) == 0) return (x);   // cbrt(0) is itself
        t = with_high_word(t, 0x43500000);  // set t= 2**54
        t *= x;
        const uint32_t high = static_cast<uint32_t>(high_word(t));
        t = from_words(sign | ((high & 0x7FFFFFFF) / 3 + B2), 0);
    } else {
        t = from_words(sign | (hx / 3 + B1), 0);
    }

    // New cbrt to 23 bits: cbrt(x) = t*cbrt(x/t**3) ~= t*P(t**3/x).
    r = (t * t) * (t / x);
    t = t * ((P0 + r * (P1 + r * P2)) + ((r * r) * r) * (P3 + r * P4));

    // Round t away from zero to 23 bits.
    uint64_t bits = to_bits(t);
    bits = (bits + 0x80000000) & 0xFFFFFFFFC0000000ULL;
    t = from_bits(bits);

    // one step Newton iteration to 53 bits with error < 0.667 ulps
    s = t * t;              // t*t is exact
    r = x / s;              // error <= 0.5 ulps; |r| < |t|
    w = t + t;              // t+t is exact
    r = (r - t) / (w + r);  // r-t is exact; w+r ~= 3*t
    t = t + t * r;          // error <= 0.5 + 0.5/3 + epsilon

    return (t);
}

double sinh_f64(double x) noexcept {
    constexpr double KSINH_OVERFLOW = 710.4758600739439;
    constexpr double TWO_M28 = 3.725290298461914e-9;  // 2^-28, empty lower half
    constexpr double LOG_MAXD = 709.7822265625;       // 0x40862E42 00000000, empty lower half
    constexpr double shuge = 1.0e307;

    const double h = (x < 0) ? -0.5 : 0.5;
    // |x| in [0, 22]. return sign(x)*0.5*(E+E/(E+1))
    const double ax = std::fabs(x);
    if (ax < 22) {
        // For |x| < 2^-28, sinh(x) = x
        if (ax < TWO_M28) return x;
        const double t = expm1_f64(ax);
        if (ax < 1) return h * (2 * t - t * t / (t + 1));
        return h * (t + t / (t + 1));
    }
    // |x| in [22, log(maxdouble)], return 0.5 * exp(|x|)
    if (ax < LOG_MAXD) return h * exp_f64(ax);
    // |x| in [log(maxdouble), overflowthreshold]
    if (ax <= KSINH_OVERFLOW) {
        const double w = exp_f64(0.5 * ax);
        const double t = h * w;
        return t * w;
    }
    // |x| > overflowthreshold or is NaN: infinity of the sign, or NaN.
    return x * shuge;
}

double cosh_f64(double x) noexcept {
    constexpr double KCOSH_OVERFLOW = 710.4758600739439;
    constexpr double one = 1.0, half = 0.5;
    static volatile double huge = 1.0e+300;

    const int32_t ix = high_word(x) & 0x7FFFFFFF;

    // |x| in [0,0.5*log2], return 1+expm1(|x|)^2/(2*exp(|x|))
    if (ix < 0x3FD62E43) {
        const double t = expm1_f64(std::fabs(x));
        const double w = one + t;
        // For |x| < 2^-55, cosh(x) = 1
        if (ix < 0x3C800000) return w;
        return one + (t * t) / (w + w);
    }
    // |x| in [0.5*log2, 22], return (exp(|x|)+1/exp(|x|)/2
    if (ix < 0x40360000) {
        const double t = exp_f64(std::fabs(x));
        return half * t + half / t;
    }
    // |x| in [22, log(maxdouble)], return half*exp(|x|)
    if (ix < 0x40862E42) return half * exp_f64(std::fabs(x));
    // |x| in [log(maxdouble), overflowthreshold]
    if (std::fabs(x) <= KCOSH_OVERFLOW) {
        const double w = exp_f64(half * std::fabs(x));
        const double t = half * w;
        return t * w;
    }
    // x is INF or NaN
    if (ix >= 0x7FF00000) return x * x;
    // |x| > overflowthreshold.
    return huge * huge;
}

double tanh_f64(double x) noexcept {
    static const volatile double tiny = 1.0e-300;
    constexpr double one = 1.0, two = 2.0, huge = 1.0e300;
    double t, z;

    const int32_t jx = high_word(x);
    const int32_t ix = jx & 0x7FFFFFFF;

    // x is INF or NaN
    if (ix >= 0x7FF00000) {
        if (jx >= 0) return one / x + one;  // tanh(+-inf)=+-1
        return one / x - one;               // tanh(NaN) = NaN
    }

    if (ix < 0x40360000) {                   // |x|<22
        if (ix < 0x3E300000) {               // |x|<2**-28
            if (huge + x > one) return x;    // tanh(tiny) = tiny with inexact
        }
        if (ix >= 0x3FF00000) {  // |x|>=1
            t = expm1_f64(two * std::fabs(x));
            z = one - two / (t + two);
        } else {
            t = expm1_f64(-two * std::fabs(x));
            z = -t / (t + two);
        }
    } else {  // |x| >= 22, return +-1
        z = one - tiny;  // raise inexact flag
    }
    return (jx >= 0) ? z : -z;
}

double asinh_f64(double x) noexcept {
    constexpr double one = 1.00000000000000000000e+00;  // 0x3FF00000, 0x00000000
    constexpr double ln2 = 6.93147180559945286227e-01;  // 0x3FE62E42, 0xFEFA39EF
    constexpr double huge = 1.00000000000000000000e+300;

    double t, w;
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7FFFFFFF;
    if (ix >= 0x7FF00000) return x + x;  // x is inf or NaN
    if (ix < 0x3E300000) {               // |x|<2**-28
        if (huge + x > one) return x;    // return x inexact except 0
    }
    if (ix > 0x41B00000) {  // |x| > 2**28
        w = log_f64(std::fabs(x)) + ln2;
    } else if (ix > 0x40000000) {  // 2**28 > |x| > 2.0
        t = std::fabs(x);
        w = log_f64(2.0 * t + one / (std::sqrt(x * x + one) + t));
    } else {  // 2.0 > |x| > 2**-28
        t = x * x;
        w = log1p_f64(std::fabs(x) + t / (one + std::sqrt(one + t)));
    }
    return hx > 0 ? w : -w;
}

double acosh_f64(double x) noexcept {
    constexpr double one = 1.0;
    constexpr double ln2 = 6.93147180559945286227e-01;  // 0x3FE62E42, 0xFEFA39EF
    double t;
    const int32_t hx = high_word(x);
    const uint32_t lx = low_word(x);
    if (hx < 0x3FF00000) return kNaN;  // x < 1
    if (hx >= 0x41B00000) {            // x > 2**28
        if (hx >= 0x7FF00000) return x + x;  // x is inf of NaN
        return log_f64(x) + ln2;             // acosh(huge)=log(2x)
    }
    if (((hx - 0x3FF00000) | static_cast<int32_t>(lx)) == 0) return 0.0;  // acosh(1) = 0
    if (hx > 0x40000000) {  // 2**28 > x > 2
        t = x * x;
        return log_f64(2.0 * x - one / (x + std::sqrt(t - one)));
    }
    // 1<x<2
    t = x - one;
    return log1p_f64(t + std::sqrt(2.0 * t + t * t));
}

double atanh_f64(double x) noexcept {
    constexpr double one = 1.0, huge = 1e300;
    constexpr double zero = 0.0;

    double t;
    const int32_t hx = high_word(x);
    const uint32_t lx = low_word(x);
    const int32_t ix = hx & 0x7FFFFFFF;
    if ((ix | static_cast<int32_t>((lx | (0u - lx)) >> 31)) > 0x3FF00000) return kNaN;  // |x|>1
    if (ix == 0x3FF00000) return x > 0 ? kInf : -kInf;
    if (ix < 0x3E300000 && (huge + x) > zero) return x;  // x<2**-28
    x = with_high_word(x, static_cast<uint32_t>(ix));
    if (ix < 0x3FE00000) {  // x < 0.5
        t = x + x;
        t = 0.5 * log1p_f64(t + t * x / (one - x));
    } else {
        t = 0.5 * log1p_f64((x + x) / (one - x));
    }
    return hx >= 0 ? t : -t;
}

double hypot_f64(const double* values, unsigned count) noexcept {
    // V8's Math.hypot (builtins/math.tq): an infinity wins over a NaN, a NaN
    // over the rest; otherwise every magnitude over the largest, squared,
    // Kahan-summed in argument order, and the square root scaled back up.
    if (count == 0) return 0.0;
    bool nan = false;
    double max = 0.0;
    for (unsigned i = 0; i < count; ++i) {
        const double v = values[i];
        if (std::isnan(v)) {
            nan = true;
            continue;
        }
        const double a = std::fabs(v);
        if (a > max) max = a;
    }
    if (max == kInf) return kInf;
    if (nan) return kNaN;
    if (max == 0.0) return 0.0;
    double sum = 0.0;
    double compensation = 0.0;
    for (unsigned i = 0; i < count; ++i) {
        const double n = std::fabs(values[i]) / max;
        const double summand = n * n - compensation;
        const double preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    return std::sqrt(sum) * max;
}

double hypot2_f64(double a, double b) noexcept {
    const double v[2] = {a, b};
    return hypot_f64(v, 2);
}

} // namespace brass::math
