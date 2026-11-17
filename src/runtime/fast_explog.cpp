// Contraction off: an FMA rounds once where fdlibm's expressions round twice,
// so a contracted kernel would not give V8's bits.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// exp and log: fdlibm's e_exp.c and e_log.c as V8 runs them (ieee754::exp and
// ieee754::log, including V8's exp(1) == Math.E), so Math.exp and Math.log
// agree with V8's bit for bit. The platform libm's are correctly rounded more
// often than fdlibm's (under one ulp), which is exactly why they differ from
// node in the last bit on a few percent of arguments.

#include <brass/runtime/fast_trig.hpp>
#include <bit>
#include <cmath>
#include <cstdint>

namespace brass::math {

namespace {

constexpr uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
constexpr double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }
inline int32_t high_word(double x) noexcept { return static_cast<int32_t>(to_bits(x) >> 32); }
inline double with_high_word(double x, uint32_t hi) noexcept {
    return from_bits((static_cast<uint64_t>(hi) << 32) | (to_bits(x) & 0xffffffffULL));
}

constexpr double ln2_hi = 6.93147180369123816490e-01;  // 0x3fe62e42 fee00000
constexpr double ln2_lo = 1.90821492927058770002e-10;  // 0x3dea39ef 35793c76

} // namespace

double exp_f64(double x) noexcept {
    constexpr double halF[2] = {0.5, -0.5};
    constexpr double ln2HI[2] = {ln2_hi, -ln2_hi};
    constexpr double ln2LO[2] = {ln2_lo, -ln2_lo};
    constexpr double huge = 1.0e+300;
    constexpr double o_threshold = 7.09782712893383973096e+02;
    constexpr double u_threshold = -7.45133219101941108420e+02;
    constexpr double invln2 = 1.44269504088896338700e+00;
    constexpr double P1 = 1.66666666666666019037e-01;
    constexpr double P2 = -2.77777777770155933842e-03;
    constexpr double P3 = 6.61375632143793436117e-05;
    constexpr double P4 = -1.65339022054652515390e-06;
    constexpr double P5 = 4.13813679705723846039e-08;
    constexpr double E = 2.718281828459045;
    constexpr double twom1000 = 9.33263618503218878990e-302;  // 2^-1000
    constexpr double two1023 = 8.988465674311579539e307;      // 2^1023

    const uint32_t whole = static_cast<uint32_t>(high_word(x));
    const int32_t xsb = static_cast<int32_t>(whole >> 31);
    const uint32_t hx = whole & 0x7fffffff;
    if (hx >= 0x40862e42) {  // |x| >= 709.78...
        if (hx >= 0x7ff00000) {
            if (((hx & 0xfffff) | static_cast<uint32_t>(to_bits(x))) != 0) return x + x;  // NaN
            return xsb == 0 ? x : 0.0;  // exp(+-inf) = {inf, 0}
        }
        if (x > o_threshold) return huge * huge;  // overflow
        if (x < u_threshold) return twom1000 * twom1000;  // underflow
    }
    double hi = 0.0;
    double lo = 0.0;
    int32_t k = 0;
    if (hx > 0x3fd62e42) {  // |x| > 0.5 ln2
        if (hx < 0x3ff0a2b2) {  // and |x| < 1.5 ln2
            if (x == 1.0) return E;  // V8's special case: the reduction gets its last bit wrong
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        } else {
            k = static_cast<int32_t>(invln2 * x + halF[xsb]);
            const double t = k;
            hi = x - t * ln2HI[0];  // t * ln2HI is exact here
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    } else if (hx < 0x3e300000) {  // |x| < 2^-28
        if (huge + x > 1.0) return 1.0 + x;
    }
    // x is now in the primary range
    const double t = x * x;
    const double twopk = k >= -1021 ? from_bits(static_cast<uint64_t>(0x3ff00000 + (k << 20)) << 32)
                                    : from_bits(static_cast<uint64_t>(0x3ff00000 + ((k + 1000) << 20)) << 32);
    const double c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return 1.0 - ((x * c) / (c - 2.0) - x);
    const double y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021) {
        if (k == 1024) return y * 2.0 * two1023;
        return y * twopk;
    }
    return y * twopk * twom1000;
}

double log_f64(double x) noexcept {
    constexpr double two54 = 1.80143985094819840000e+16;
    constexpr double Lg1 = 6.666666666666735130e-01;
    constexpr double Lg2 = 3.999999999940941908e-01;
    constexpr double Lg3 = 2.857142874366239149e-01;
    constexpr double Lg4 = 2.222219843214978396e-01;
    constexpr double Lg5 = 1.818357216161805012e-01;
    constexpr double Lg6 = 1.531383769920937332e-01;
    constexpr double Lg7 = 1.479819860511658591e-01;

    int32_t hx = high_word(x);
    const uint32_t lx = static_cast<uint32_t>(to_bits(x));
    int32_t k = 0;
    if (hx < 0x00100000) {  // x < 2^-1022
        if (((hx & 0x7fffffff) | static_cast<int32_t>(lx)) == 0) return -std::bit_cast<double>(0x7ff0000000000000ULL);
        if (hx < 0) return std::bit_cast<double>(0x7ff8000000000000ULL);  // log(-#) = NaN
        k -= 54;
        x *= two54;  // subnormal: scale up
        hx = high_word(x);
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    int32_t i = (hx + 0x95f64) & 0x100000;
    x = with_high_word(x, static_cast<uint32_t>(hx | (i ^ 0x3ff00000)));  // normalize x or x/2
    k += i >> 20;
    const double f = x - 1.0;
    if ((0x000fffff & (2 + hx)) < 3) {  // -2^-20 <= f < 2^-20
        if (f == 0.0) {
            if (k == 0) return 0.0;
            const double dk = static_cast<double>(k);
            return dk * ln2_hi + dk * ln2_lo;
        }
        const double R = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0) return f - R;
        const double dk = static_cast<double>(k);
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    const double s = f / (2.0 + f);
    const double dk = static_cast<double>(k);
    const double z = s * s;
    i = hx - 0x6147a;
    const double w = z * z;
    const int32_t j = 0x6b851 - hx;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    i |= j;
    const double R = t2 + t1;
    if (i > 0) {
        const double hfsq = 0.5 * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

namespace {

inline uint32_t low_word(double x) noexcept { return static_cast<uint32_t>(to_bits(x)); }
inline double zero_low_word(double x) noexcept { return from_bits(to_bits(x) & 0xffffffff00000000ULL); }
inline double from_words(uint32_t hi, uint32_t lo) noexcept {
    return from_bits((static_cast<uint64_t>(hi) << 32) | lo);
}

} // namespace

// fdlibm's e_pow.c (__ieee754_pow) as V8 runs it (ieee754::legacy::pow):
// log2(|x|) in two pieces, y times it in simulated extra precision split into
// n + y', and 2^n * exp(y' ln 2). The platform's pow is correctly rounded on
// some systems and not on others, so its last bit differs by platform.
double pow_f64(double x, double y) noexcept {
    constexpr double bp[2] = {1.0, 1.5};
    constexpr double dp_h[2] = {0.0, 5.84962487220764160156e-01};  // 0x3FE2B803 40000000
    constexpr double dp_l[2] = {0.0, 1.35003920212974897128e-08};  // 0x3E4CFDEB 43CFD006
    constexpr double two53 = 9007199254740992.0;
    constexpr double huge = 1.0e300;
    constexpr double tiny = 1.0e-300;
    constexpr double L1 = 5.99999999999994648725e-01;
    constexpr double L2 = 4.28571428578550184252e-01;
    constexpr double L3 = 3.33333329818377432918e-01;
    constexpr double L4 = 2.72728123808534006489e-01;
    constexpr double L5 = 2.30660745775561754067e-01;
    constexpr double L6 = 2.06975017800338417784e-01;
    constexpr double P1 = 1.66666666666666019037e-01;
    constexpr double P2 = -2.77777777770155933842e-03;
    constexpr double P3 = 6.61375632143793436117e-05;
    constexpr double P4 = -1.65339022054652515390e-06;
    constexpr double P5 = 4.13813679705723846039e-08;
    constexpr double lg2 = 6.93147180559945286227e-01;
    constexpr double lg2_h = 6.93147182464599609375e-01;
    constexpr double lg2_l = -1.90465429995776804525e-09;
    constexpr double ovt = 8.0085662595372944372e-17;
    constexpr double cp = 9.61796693925975554329e-01;
    constexpr double cp_h = 9.61796700954437255859e-01;
    constexpr double cp_l = -7.02846165095275826516e-09;
    constexpr double ivln2 = 1.44269504088896338700e+00;
    constexpr double ivln2_h = 1.44269502162933349609e+00;
    constexpr double ivln2_l = 1.92596299112661746887e-08;

    const int32_t hx = high_word(x);
    const uint32_t lx = low_word(x);
    const int32_t hy = high_word(y);
    const uint32_t ly = low_word(y);
    int32_t ix = hx & 0x7fffffff;
    const int32_t iy = hy & 0x7fffffff;

    if ((static_cast<uint32_t>(iy) | ly) == 0) return 1.0;  // x**0 = 1
    if (hx == 0x3ff00000 && lx == 0) return x;               // 1**y = 1
    if (ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) || iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0))
        return x + y;  // NaN

    // yisint: 0 y is not an integer, 1 an odd one, 2 an even one (for x < 0).
    int32_t yisint = 0;
    if (hx < 0) {
        if (iy >= 0x43400000) {
            yisint = 2;
        } else if (iy >= 0x3ff00000) {
            const int32_t k = (iy >> 20) - 0x3ff;
            if (k > 20) {
                const uint32_t j = ly >> (52 - k);
                if ((j << (52 - k)) == ly) yisint = 2 - static_cast<int32_t>(j & 1u);
            } else if (ly == 0) {
                const int32_t j = iy >> (20 - k);
                if ((j << (20 - k)) == iy) yisint = 2 - (j & 1);
            }
        }
    }

    if (ly == 0) {
        if (iy == 0x7ff00000) {  // y is +-inf
            if (((ix - 0x3ff00000) | static_cast<int32_t>(lx)) == 0) return 1.0;
            if (ix >= 0x3ff00000) return hy >= 0 ? y : 0.0;
            return hy < 0 ? -y : 0.0;
        }
        if (iy == 0x3ff00000) return hy < 0 ? 1.0 / x : x;  // y is +-1
        if (hy == 0x40000000) return x * x;                  // y is 2
        if (hy == 0x3fe00000 && hx >= 0) return std::sqrt(x);  // y is 0.5, x >= +0
    }

    double ax = std::fabs(x);
    if (lx == 0 && (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000)) {  // +-0, +-inf, +-1
        double z = ax;
        if (hy < 0) z = 1.0 / z;
        if (hx < 0) {
            if (((ix - 0x3ff00000) | yisint) == 0) z = (z - z) / (z - z);  // (-1)**non-int
            else if (yisint == 1) z = -z;
        }
        return z;
    }

    if (((static_cast<int32_t>(static_cast<uint32_t>(hx) >> 31) - 1) | yisint) == 0) return (x - x) / (x - x);

    double t1, t2;
    if (iy > 0x41e00000) {  // |y| > 2^31
        if (iy > 0x43f00000) {  // |y| > 2^64: over- or underflow
            if (ix <= 0x3fefffff) return hy < 0 ? huge * huge : tiny * tiny;
            if (ix >= 0x3ff00000) return hy > 0 ? huge * huge : tiny * tiny;
        }
        if (ix < 0x3fefffff) return hy < 0 ? huge * huge : tiny * tiny;
        if (ix > 0x3ff00000) return hy > 0 ? huge * huge : tiny * tiny;
        // |1 - x| <= 2^-20: log(x) by x - x^2/2 + x^3/3 - x^4/4.
        const double t = ax - 1.0;
        const double w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        const double u = ivln2_h * t;
        const double v = t * ivln2_l - w * ivln2;
        t1 = zero_low_word(u + v);
        t2 = v - (t1 - u);
    } else {
        int32_t n = 0;
        if (ix < 0x00100000) {  // subnormal
            ax *= two53;
            n -= 53;
            ix = high_word(ax);
        }
        n += (ix >> 20) - 0x3ff;
        const int32_t j = ix & 0x000fffff;
        ix = j | 0x3ff00000;
        int k;
        if (j <= 0x3988E) {
            k = 0;  // |x| < sqrt(3/2)
        } else if (j < 0xBB67A) {
            k = 1;  // |x| < sqrt(3)
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = with_high_word(ax, static_cast<uint32_t>(ix));

        // s = s_h + s_l = (x - 1) / (x + 1) or (x - 1.5) / (x + 1.5)
        double u = ax - bp[k];
        double v = 1.0 / (ax + bp[k]);
        const double s = u * v;
        const double s_h = zero_low_word(s);
        double t_h = from_words(static_cast<uint32_t>(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18)), 0);
        double t_l = ax - (t_h - bp[k]);
        const double s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // log(ax)
        double s2 = s * s;
        double r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + s);
        s2 = s_h * s_h;
        t_h = zero_low_word(3.0 + s2 + r);
        t_l = r - ((t_h - 3.0) - s2);
        u = s_h * t_h;
        v = s_l * t_h + t_l * s;
        const double p_h = zero_low_word(u + v);
        const double p_l = v - (p_h - u);
        const double z_h = cp_h * p_h;
        const double z_l = cp_l * p_h + p_l * cp + dp_l[k];
        // log2(ax) = n + dp_h + z_h + z_l
        const double t = static_cast<double>(n);
        t1 = zero_low_word(((z_h + z_l) + dp_h[k]) + t);
        t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
    }

    // The sign of the result: -1 for a negative x to an odd integer.
    double s = 1.0;
    if (((static_cast<int32_t>(static_cast<uint32_t>(hx) >> 31) - 1) | (yisint - 1)) == 0) s = -1.0;

    // (y1 + y2) * (t1 + t2)
    const double y1 = zero_low_word(y);
    const double p_l = (y - y1) * t1 + y * t2;
    double p_h = y1 * t1;
    double z = p_l + p_h;
    int32_t j = high_word(z);
    const int32_t i = static_cast<int32_t>(low_word(z));
    if (j >= 0x40900000) {  // z >= 1024
        if (((j - 0x40900000) | i) != 0) return s * huge * huge;
        if (p_l + ovt > z - p_h) return s * huge * huge;
    } else if ((j & 0x7fffffff) >= 0x4090cc00) {  // z <= -1075
        if (((j - static_cast<int32_t>(0xc090cc00)) | i) != 0) return s * tiny * tiny;
        if (p_l <= z - p_h) return s * tiny * tiny;
    }

    // 2^(p_h + p_l)
    const int32_t ai = j & 0x7fffffff;
    int32_t k = (ai >> 20) - 0x3ff;
    int32_t n = 0;
    if (ai > 0x3fe00000) {  // |z| > 0.5: n = [z + 0.5]
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        const double t = from_words(static_cast<uint32_t>(n & ~(0x000fffff >> k)), 0);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if (j < 0) n = -n;
        p_h -= t;
    }
    double t = zero_low_word(p_l + p_h);
    const double u = t * lg2_h;
    const double v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
    z = u + v;
    const double w = v - (z - u);
    t = z * z;
    t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    const double r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = high_word(z);
    j += static_cast<int32_t>(static_cast<uint32_t>(n) << 20);
    if ((j >> 20) <= 0) z = std::scalbn(z, n);  // subnormal result
    else z = with_high_word(z, static_cast<uint32_t>(j));
    return s * z;
}

} // namespace brass::math
