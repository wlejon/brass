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

} // namespace brass::math
