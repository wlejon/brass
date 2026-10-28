#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// sin and cos: fdlibm's kernels and medium-range reduction (as V8 runs them,
// so Math.sin and Math.cos agree with V8's bit for bit below 2^19 * pi/2).
// Plain doubles with contraction off are as deterministic as the
// double-double path and cost a tenth of it.
//
// fdlibm branches on the data past the reduction: on the quadrant, on the
// sign, on |x| < 3pi/4, and in kernel_cos on |y0| < 0.3. Over arguments that
// do not move in step (three.js's instanced_mesh_churn: an angle per
// instance) each is a coin toss, and the quadrant's and kernel_cos's resolve
// only once the reduction's chain is done, so a miss threw that chain away:
// half a misprediction a call, a third of the calls' cycles. Here the medium
// reduction serves |x| < 3pi/4 too (fn = 1 multiplies exactly, so y0, y1 and
// n are fdlibm's +-1 step's), signs are applied by the bits, qx is chosen in
// the integer registers, and the quadrant's parity is the one branch left:
// one kernel is evaluated, its sign set by the bits. Every expression is
// fdlibm's, so the results are its bits. What the fast path cannot give the
// same bits for (the cancellation steps, x by pi/2's own high word,
// |x| > 2^19 * pi/2) takes fdlibm_rem_pio2 whole, on a branch taken about
// never.
//
// Evaluating both kernels and selecting by the bits removes the parity's
// misses too, but twice the operations wait on the reduction's chain: where
// the branches were predicted (mesh_churn_2k, three_math) the FP scheduler
// filled and the calls cost twice the cycles. What the one path costs is
// fdlibm's +-1 shortcut: arguments there that move smoothly, so that its
// branch was predicted (mesh_churn_2k's Euler halves), pay the medium step's
// conversions, 3% of that bench's cycles.

#include <brass/runtime/deterministic_math.hpp>
#include "deterministic_math_internal.hpp"
#include <bit>
#include <cstdint>

namespace brass::math {

namespace {

constexpr uint64_t kSignMask = 0x8000000000000000ULL;

inline uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
inline double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }
inline int32_t high_word(double x) noexcept { return static_cast<int32_t>(to_bits(x) >> 32); }
inline double abs_f64(double x) noexcept { return from_bits(to_bits(x) & ~kSignMask); }

// A sign applied by the bits, not by a branch on the data. On x86-64 with GCC
// or Clang it is one xorpd on the register the double is in: the intrinsics
// zero its upper lane first and integer operations take it through a general
// register and back, each an operation more waiting on the chain.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
inline double bit_xor(double a, double m) noexcept {
    asm("xorpd %1, %0" : "+x"(a) : "x"(m));
    return a;
}
#else
inline double bit_xor(double a, double m) noexcept { return from_bits(to_bits(a) ^ to_bits(m)); }
#endif
inline uint64_t mask_if(bool c) noexcept { return 0 - static_cast<uint64_t>(c); }

constexpr double S1 = -1.66666666666666324348e-01;
constexpr double S2 = 8.33333333332248946124e-03;
constexpr double S3 = -1.98412698298579493134e-04;
constexpr double S4 = 2.75573137070700676789e-06;
constexpr double S5 = -2.50507602534068634195e-08;
constexpr double S6 = 1.58969099521155010221e-10;

// sin(x) for 2^-27 <= |x| <= pi/4: kernel_sin with no tail (iy = 0).
inline double kernel_sin_small(double x) noexcept {
    const double z = x * x;
    const double v = z * x;
    const double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return x + v * (S1 + z * r);
}

// sin(x + y) for 2^-27 <= |x + y| <= pi/4, y the tail of x (iy = 1).
inline double kernel_sin(double x, double y) noexcept {
    const double z = x * x;
    const double v = z * x;
    const double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

// cos(x + y) for 2^-27 <= |x + y| <= pi/4. qx is x/4 with its low word cleared,
// 0.28125 above 0.78125, and 0 below 0.3, where a = 1.0 - 0 and hz = 0.5 * z - 0
// make the last line fdlibm's 1.0 - (0.5 * z - (z * r - x * y)) exactly. It is
// chosen by masks in the integer registers, off the polynomial's chain.
inline double kernel_cos(double x, double y) noexcept {
    constexpr double C1 = 4.16666666666666019037e-02;
    constexpr double C2 = -1.38888888888741095749e-03;
    constexpr double C3 = 2.48015872894767294178e-05;
    constexpr double C4 = -2.75573143513906633035e-07;
    constexpr double C5 = 2.08757232129817482790e-09;
    constexpr double C6 = -1.13596475577881948265e-11;
    const uint32_t ix = static_cast<uint32_t>(high_word(x)) & 0x7fffffffu;
    uint64_t q = static_cast<uint64_t>(ix - 0x00200000u) << 32;
    q ^= (q ^ 0x3fd2000000000000ULL) & mask_if(ix > 0x3fe90000u); // 0.28125
    q &= ~mask_if(ix < 0x3fd33333u);
    const double qx = from_bits(q);
    const double z = x * x;
    const double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    const double hz = 0.5 * z - qx;
    const double a = 1.0 - qx;
    return a - (hz - (z * r - x * y));
}

// x reduced by pi/2 into y0 + y1 (|y0 + y1| <= pi/4), returning the multiple
// of pi/2 taken. fdlibm's __ieee754_rem_pio2 below 2^19 * pi/2 (its table of
// near-multiples only skips the cancellation check, which is made always);
// above, the Payne-Hanek double-double reduction.
int fdlibm_rem_pio2(double x, double& y0, double& y1) noexcept {
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
    return detail::reduce_pio2_huge(x, y0, y1);
}

// fdlibm_rem_pio2 for pi/4 < |x| <= 2^19 * pi/2 with no data-dependent branch:
// the medium step, its sign applied by the bits. False where fdlibm's answer
// differs: a cancellation step is due, or x is by pi/2 (its +-1 step's
// 33+33+53-bit split, ix == 0x3ff921fb).
inline bool reduce_medium(double x, int32_t hx, int32_t ix, double& y0, double& y1, int32_t& n) noexcept {
    constexpr double invpio2 = 6.36619772367581382433e-01;
    constexpr double pio2_1 = 1.57079632673412561417e+00;
    constexpr double pio2_1t = 6.07710050650619224932e-11;
    const double t = abs_f64(x);
    const int32_t m = static_cast<int32_t>(t * invpio2 + 0.5);
    const double fn = static_cast<double>(m);
    const double r = t - fn * pio2_1;
    const double w = fn * pio2_1t;
    const double a = r - w;
    const int32_t i = (ix >> 20) - ((high_word(a) >> 20) & 0x7ff);
    const double sign = from_bits(to_bits(x) & kSignMask); // -y is y with its sign flipped
    y0 = bit_xor(a, sign);
    y1 = bit_xor((r - a) - w, sign);
    const int32_t s = hx >> 31;
    n = (m ^ s) - s;
    return (i <= 16) & (ix != 0x3ff921fb);
}

// v negated where bit 1 of flip is set: flip = n for sin, whose quadrants
// give s, c, -s, -c, and n + 1 for cos, whose give c, -s, -c, s.
inline double sign_by(double v, int32_t flip) noexcept {
    return bit_xor(v, from_bits(static_cast<uint64_t>(flip & 2) << 62));
}

// Out of line, so the fast path takes no local's address: one would give it
// a stack canary, a dozen instructions on every call.
#if defined(_MSC_VER)
#define BRASS_TRIG_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define BRASS_TRIG_NOINLINE __attribute__((noinline))
#else
#define BRASS_TRIG_NOINLINE
#endif

// fdlibm's kernels answer y0 itself for sin and 1 for cos below 2^-27; y0
// is that small only after a cancellation step.
inline bool tiny(double y0) noexcept { return (high_word(y0) & 0x7fffffff) < 0x3e400000; }

BRASS_TRIG_NOINLINE double sin_slow(double x) noexcept {
    double y0, y1;
    const int32_t n = fdlibm_rem_pio2(x, y0, y1);
    if (tiny(y0)) return sign_by((n & 1) ? 1.0 : y0, n);
    return sign_by((n & 1) ? kernel_cos(y0, y1) : kernel_sin(y0, y1), n);
}

BRASS_TRIG_NOINLINE double cos_slow(double x) noexcept {
    double y0, y1;
    const int32_t n = fdlibm_rem_pio2(x, y0, y1);
    if (tiny(y0)) return sign_by((n & 1) ? y0 : 1.0, n + 1);
    return sign_by((n & 1) ? kernel_sin(y0, y1) : kernel_cos(y0, y1), n + 1);
}

inline double qnan() noexcept { return from_bits(kCanonicalNaNBits); }

} // namespace

double det_sin(double x) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) return ix < 0x3e400000 ? x : kernel_sin_small(x); // |x| <= pi/4
    if (ix >= 0x7ff00000) return qnan();
    double y0, y1;
    int32_t n;
    if (ix > 0x413921fb || !reduce_medium(x, hx, ix, y0, y1, n)) return sin_slow(x);
    return sign_by((n & 1) ? kernel_cos(y0, y1) : kernel_sin(y0, y1), n);
}

double det_cos(double x) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) return ix < 0x3e400000 ? 1.0 : kernel_cos(x, 0.0); // |x| <= pi/4
    if (ix >= 0x7ff00000) return qnan();
    double y0, y1;
    int32_t n;
    if (ix > 0x413921fb || !reduce_medium(x, hx, ix, y0, y1, n)) return cos_slow(x);
    return sign_by((n & 1) ? kernel_sin(y0, y1) : kernel_cos(y0, y1), n + 1);
}

} // namespace brass::math
