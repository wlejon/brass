// Contraction off: an FMA rounds once where fdlibm's expressions round twice,
// so a contracted kernel would not give V8's bits.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// sin and cos: fdlibm's kernels and argument reduction, as V8 runs them
// (ieee754::sin/cos), so Math.sin and Math.cos agree with V8's bit for bit.
// glibc's sin/cos save, set and restore the rounding mode on every call
// (stmxcsr, then a load of what it stored, which stalls) and a sine and a
// cosine of one angle are two calls; these touch no FP state and sincos_f64
// answers both from one reduction.
//
// fdlibm branches on the data past the reduction: on the quadrant, on the
// sign, on |x| < 3pi/4, and in kernel_cos on |y0| < 0.3. Over arguments that
// do not move in step (an angle per instance) each is a coin toss, resolved
// only once the reduction's chain is done, so a miss throws that chain away.
// Here the medium reduction serves |x| < 3pi/4 too (fn = 1 multiplies
// exactly, so y0, y1 and n are fdlibm's +-1 step's), signs are applied by the
// bits, qx is chosen in the integer registers, and the quadrant's parity
// indexes a two-row table of the shared degree-4 Horner coefficients in z =
// y0^2 so sin_f64 and cos_f64 evaluate one polynomial chain and select the
// sin/cos reconstruction in XMM registers without branching on n. sincos_f64
// interleaves both polynomials on one z and swaps/signs the pair by the bits.
// Every expression is fdlibm's, so the results are its bits. What the fast
// path cannot give the same bits for (the cancellation steps, x by pi/2's
// own high word, |x| > 2^19 * pi/2) takes fdlibm_rem_pio2 whole, on a branch
// taken about never.

#include <brass/runtime/fast_trig.hpp>
#include <bit>
#include <cmath>
#include <cstdint>

namespace brass::math {

namespace {

constexpr uint64_t kSignMask = 0x8000000000000000ULL;
constexpr uint64_t kCanonicalNaNBits = 0x7ff8000000000000ULL;

constexpr uint64_t to_bits(double x) noexcept { return std::bit_cast<uint64_t>(x); }
constexpr double from_bits(uint64_t u) noexcept { return std::bit_cast<double>(u); }
inline int32_t high_word(double x) noexcept { return static_cast<int32_t>(to_bits(x) >> 32); }
inline double abs_f64(double x) noexcept { return from_bits(to_bits(x) & ~kSignMask); }
inline double qnan() noexcept { return from_bits(kCanonicalNaNBits); }

// Signs and selections applied by the bits, not by a branch on the data. On
// x86-64 with GCC or Clang, baseline SSE2 lacks a scalar FP blend or cmov, so
// a C++ ternary on doubles compiles to a conditional jump; xorpd/andpd stay on
// the XMM registers the doubles are already in, without zeroing upper lanes or
// round-tripping through general registers.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
inline double bit_xor(double a, double m) noexcept {
    asm("xorpd %1, %0" : "+x"(a) : "x"(m));
    return a;
}
inline double bit_and(double a, double m) noexcept {
    asm("andpd %1, %0" : "+x"(a) : "x"(m));
    return a;
}
inline double select_f64(uint32_t choose_b, double a, double b) noexcept {
    const double m = from_bits(0ULL - static_cast<uint64_t>(choose_b));
    return bit_xor(a, bit_and(bit_xor(a, b), m));
}
#else
inline double bit_xor(double a, double m) noexcept { return from_bits(to_bits(a) ^ to_bits(m)); }
inline double select_f64(uint32_t choose_b, double a, double b) noexcept { return choose_b ? b : a; }
#endif
inline uint64_t mask_if(bool c) noexcept { return 0 - static_cast<uint64_t>(c); }

// ---- The reduction of |x| > 2^19 * pi/2 -------------------------------------
//
// fdlibm's __kernel_rem_pio2 (Payne-Hanek in 24-bit digits of 2/pi), called
// as V8's __ieee754_rem_pio2 calls it: x as three 24-bit digits, two words of
// result.

// 2/pi in 24-bit digits, 2/pi = sum kTwoOverPi24[i] * 2^(-24 (i + 1)): 1584
// bits, enough to reduce any double (fdlibm's ipio2).
constexpr int32_t kTwoOverPi24[66] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E,
    0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
};

// pi/2 in pieces of 24 bits or fewer (fdlibm's PIo2).
constexpr double kPiO2Pieces[8] = {
    from_bits(0x3FF921FB40000000ULL), from_bits(0x3E74442D00000000ULL), from_bits(0x3CF8469880000000ULL),
    from_bits(0x3B78CC5160000000ULL), from_bits(0x39F01B8380000000ULL), from_bits(0x387A252040000000ULL),
    from_bits(0x36E3822280000000ULL), from_bits(0x3569F31D00000000ULL),
};

constexpr double two24 = 1.67772160000000000000e+07;
constexpr double twon24 = 5.96046447753906250000e-08;

// x * 2^n, exact for the normal results here.
inline double scalbn_f64(double x, int n) noexcept { return std::ldexp(x, n); }

// fdlibm's __kernel_rem_pio2 for prec = 2 (jk = 4): x[0..nx) are the 24-bit
// digits of the argument scaled by 2^-e0, y[0] + y[1] the remainder, and the
// result the quadrant mod 8.
int kernel_rem_pio2(const double* x, double* y, int e0, int nx) noexcept {
    constexpr int jk = 4;
    constexpr int jp = jk;
    int32_t iq[20];
    double f[20], fq[20], q[20];

    const int jx = nx - 1;
    int jv = (e0 - 3) / 24;
    if (jv < 0) jv = 0;
    int q0 = e0 - 24 * (jv + 1);

    // f[0..jx+jk], f[jx+jk] = ipio2[jv+jk].
    int j = jv - jx;
    const int m = jx + jk;
    for (int i = 0; i <= m; i++, j++) f[i] = (j < 0) ? 0.0 : static_cast<double>(kTwoOverPi24[j]);

    // q[0..jk].
    for (int i = 0; i <= jk; i++) {
        double fw = 0.0;
        for (j = 0; j <= jx; j++) fw += x[j] * f[jx + i - j];
        q[i] = fw;
    }

    int jz = jk;
    int n;
    int ih;
    double z;
    for (;;) {
        // Distill q[] into iq[], reversed.
        int i = 0;
        z = q[jz];
        for (j = jz; j > 0; i++, j--) {
            const double fw = static_cast<double>(static_cast<int32_t>(twon24 * z));
            iq[i] = static_cast<int32_t>(z - two24 * fw);
            z = q[j - 1] + fw;
        }

        z = scalbn_f64(z, q0);           // the value of z
        z -= 8.0 * std::floor(z * 0.125); // trim off the integer >= 8
        n = static_cast<int32_t>(z);
        z -= static_cast<double>(n);
        ih = 0;
        if (q0 > 0) { // iq[jz-1] decides n
            i = (iq[jz - 1] >> (24 - q0));
            n += i;
            iq[jz - 1] -= i << (24 - q0);
            ih = iq[jz - 1] >> (23 - q0);
        } else if (q0 == 0) {
            ih = iq[jz - 1] >> 23;
        } else if (z >= 0.5) {
            ih = 2;
        }

        if (ih > 0) { // q > 0.5: take 1 - q
            n += 1;
            int32_t carry = 0;
            for (i = 0; i < jz; i++) {
                j = iq[i];
                if (carry == 0) {
                    if (j != 0) {
                        carry = 1;
                        iq[i] = 0x1000000 - j;
                    }
                } else {
                    iq[i] = 0xffffff - j;
                }
            }
            if (q0 > 0) {
                switch (q0) {
                    case 1: iq[jz - 1] &= 0x7fffff; break;
                    case 2: iq[jz - 1] &= 0x3fffff; break;
                }
            }
            if (ih == 2) {
                z = 1.0 - z;
                if (carry != 0) z -= scalbn_f64(1.0, q0);
            }
        }

        // Recompute with more digits when the fraction cancelled to zero.
        if (z == 0.0) {
            j = 0;
            for (i = jz - 1; i >= jk; i--) j |= iq[i];
            if (j == 0) {
                int k = 1;
                while (jk >= k && iq[jk - k] == 0) k++; // k = the terms needed
                for (i = jz + 1; i <= jz + k; i++) {
                    f[jx + i] = kTwoOverPi24[jv + i];
                    double fw = 0.0;
                    for (j = 0; j <= jx; j++) fw += x[j] * f[jx + i - j];
                    q[i] = fw;
                }
                jz += k;
                continue;
            }
        }
        break;
    }

    // Chop off zero terms, or break z into 24-bit digits.
    if (z == 0.0) {
        jz -= 1;
        q0 -= 24;
        while (iq[jz] == 0) {
            jz--;
            q0 -= 24;
        }
    } else {
        z = scalbn_f64(z, -q0);
        if (z >= two24) {
            const double fw = static_cast<double>(static_cast<int32_t>(twon24 * z));
            iq[jz] = static_cast<int32_t>(z - two24 * fw);
            jz += 1;
            q0 += 24;
            iq[jz] = static_cast<int32_t>(fw);
        } else {
            iq[jz] = static_cast<int32_t>(z);
        }
    }

    // The digits as doubles.
    double fw = scalbn_f64(1.0, q0);
    for (int i = jz; i >= 0; i--) {
        q[i] = fw * iq[i];
        fw *= twon24;
    }

    // PIo2[0..jp] * q[jz..0].
    for (int i = jz; i >= 0; i--) {
        fw = 0.0;
        for (int k = 0; k <= jp && k <= jz - i; k++) fw += kPiO2Pieces[k] * q[i + k];
        fq[jz - i] = fw;
    }

    // fq[] compressed into y[0] + y[1].
    fw = 0.0;
    for (int i = jz; i >= 0; i--) fw += fq[i];
    y[0] = (ih == 0) ? fw : -fw;
    fw = fq[0] - fw;
    for (int i = 1; i <= jz; i++) fw += fq[i];
    y[1] = (ih == 0) ? fw : -fw;
    return n & 7;
}

// x (|x| > 2^19 * pi/2, finite) reduced by pi/2 into y0 + y1, y1 the tail of
// y0, returning the multiple of pi/2 taken: __ieee754_rem_pio2's large case.
int reduce_pio2_huge(double x, double& y0, double& y1) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    // z = |x| scaled by 2^-e0 into [2^23, 2^24), e0 = ilogb(x) - 23.
    const int e0 = (ix >> 20) - 1046;
    double z = from_bits((static_cast<uint64_t>(static_cast<uint32_t>(ix - (e0 << 20))) << 32) |
                         (to_bits(x) & 0xffffffffULL));
    double tx[3];
    for (int i = 0; i < 2; i++) {
        tx[i] = static_cast<double>(static_cast<int32_t>(z));
        z = (z - tx[i]) * two24;
    }
    tx[2] = z;
    int nx = 3;
    while (tx[nx - 1] == 0.0) nx--; // skip zero terms
    double y[2];
    const int n = kernel_rem_pio2(tx, y, e0, nx);
    if (hx < 0) {
        y0 = -y[0];
        y1 = -y[1];
        return -n;
    }
    y0 = y[0];
    y1 = y[1];
    return n;
}

// ---- fdlibm's kernels and medium reduction ----------------------------------

constexpr double S1 = -1.66666666666666324348e-01;
constexpr double S2 = 8.33333333332248946124e-03;
constexpr double S3 = -1.98412698298579493134e-04;
constexpr double S4 = 2.75573137070700676789e-06;
constexpr double S5 = -2.50507602534068634195e-08;
constexpr double S6 = 1.58969099521155010221e-10;

constexpr double C1 = 4.16666666666666019037e-02;
constexpr double C2 = -1.38888888888741095749e-03;
constexpr double C3 = 2.48015872894767294178e-05;
constexpr double C4 = -2.75573143513906633035e-07;
constexpr double C5 = 2.08757232129817482790e-09;
constexpr double C6 = -1.13596475577881948265e-11;

// Row 0 is kernel_sin's S2..S6, row 1 is kernel_cos's C2..C6: both evaluate
// the same degree-4 Horner chain in z = y0^2, so indexing by the quadrant's
// parity evaluates one polynomial instead of branching on n & 1.
alignas(64) constexpr double kPoly[2][5] = {
    {S2, S3, S4, S5, S6},
    {C2, C3, C4, C5, C6},
};

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
inline double cos_qx(double x) noexcept {
    const uint32_t ix = static_cast<uint32_t>(high_word(x)) & 0x7fffffffu;
    uint64_t q = static_cast<uint64_t>(ix - 0x00200000u) << 32;
    q ^= (q ^ 0x3fd2000000000000ULL) & mask_if(ix > 0x3fe90000u); // 0.28125
    q &= ~mask_if(ix < 0x3fd33333u);
    return from_bits(q);
}

inline double kernel_cos(double x, double y) noexcept {
    const double qx = cos_qx(x);
    const double z = x * x;
    const double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    const double hz = 0.5 * z - qx;
    const double a = 1.0 - qx;
    return a - (hz - (z * r - x * y));
}

// x reduced by pi/2 into y0 + y1 (|y0 + y1| <= pi/4), returning the multiple
// of pi/2 taken. fdlibm's __ieee754_rem_pio2 below 2^19 * pi/2 (its table of
// near-multiples only skips the cancellation check, which is made always);
// above, reduce_pio2_huge.
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
    return reduce_pio2_huge(x, y0, y1);
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
inline double sign_by(double v, uint32_t flip) noexcept {
    return bit_xor(v, from_bits(static_cast<uint64_t>(flip & 2u) << 62));
}

// Evaluates kernel_sin(y0, y1) when use_cos == 0 or kernel_cos(y0, y1) when
// use_cos == 1 without branching on use_cos: one degree-4 Horner chain runs on
// kPoly[use_cos], both short tail reconstructions finish in parallel, and the
// result is selected and signed in XMM registers.
inline double eval_quadrant(double y0, double y1, uint32_t use_cos, uint32_t flip) noexcept {
    const double* K = kPoly[use_cos];
    const double qx = cos_qx(y0);
    const double z = y0 * y0;
    const double v = z * y0;
    const double p = K[0] + z * (K[1] + z * (K[2] + z * (K[3] + z * K[4])));
    const double sin_val = y0 - ((z * (0.5 * y1 - v * p) - y1) - v * S1);
    const double rc = z * (C1 + z * p);
    const double hz = 0.5 * z - qx;
    const double a = 1.0 - qx;
    const double cos_val = a - (hz - (z * rc - y0 * y1));
    return sign_by(select_f64(use_cos, sin_val, cos_val), flip);
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
    const uint32_t un = static_cast<uint32_t>(fdlibm_rem_pio2(x, y0, y1));
    if (tiny(y0)) return sign_by((un & 1u) ? 1.0 : y0, un);
    return sign_by((un & 1u) ? kernel_cos(y0, y1) : kernel_sin(y0, y1), un);
}

BRASS_TRIG_NOINLINE double cos_slow(double x) noexcept {
    double y0, y1;
    const uint32_t un = static_cast<uint32_t>(fdlibm_rem_pio2(x, y0, y1));
    if (tiny(y0)) return sign_by((un & 1u) ? y0 : 1.0, un + 1u);
    return sign_by((un & 1u) ? kernel_sin(y0, y1) : kernel_cos(y0, y1), un + 1u);
}

} // namespace

double sin_f64(double x) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) return ix < 0x3e400000 ? x : kernel_sin_small(x); // |x| <= pi/4
    if (ix >= 0x7ff00000) return qnan();
    double y0, y1;
    int32_t n;
    if (ix > 0x413921fb || !reduce_medium(x, hx, ix, y0, y1, n)) return sin_slow(x);
    const uint32_t un = static_cast<uint32_t>(n);
    return eval_quadrant(y0, y1, un & 1u, un);
}

double cos_f64(double x) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) return ix < 0x3e400000 ? 1.0 : kernel_cos(x, 0.0); // |x| <= pi/4
    if (ix >= 0x7ff00000) return qnan();
    double y0, y1;
    int32_t n;
    if (ix > 0x413921fb || !reduce_medium(x, hx, ix, y0, y1, n)) return cos_slow(x);
    const uint32_t un = static_cast<uint32_t>(n);
    return eval_quadrant(y0, y1, (un ^ 1u) & 1u, un + 1u);
}

// sin_f64(x) and cos_f64(x) from one reduction: each branch below is the two
// functions' own for that range, so the results are their bits.
void sincos_f64(double x, double* out) noexcept {
    const int32_t hx = high_word(x);
    const int32_t ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) { // |x| <= pi/4
        if (ix < 0x3e400000) {
            out[0] = x;
            out[1] = 1.0;
        } else {
            out[0] = kernel_sin_small(x);
            out[1] = kernel_cos(x, 0.0);
        }
        return;
    }
    if (ix >= 0x7ff00000) {
        out[0] = qnan();
        out[1] = qnan();
        return;
    }
    double y0, y1;
    int32_t n;
    if (ix > 0x413921fb || !reduce_medium(x, hx, ix, y0, y1, n)) {
        out[0] = sin_slow(x);
        out[1] = cos_slow(x);
        return;
    }
    const uint32_t un = static_cast<uint32_t>(n);
    const double qx = cos_qx(y0);
    const double z = y0 * y0;
    const double v = z * y0;
    double ps = S5 + z * S6;
    double pc = C5 + z * C6;
    ps = S4 + z * ps;
    pc = C4 + z * pc;
    ps = S3 + z * ps;
    pc = C3 + z * pc;
    ps = S2 + z * ps;
    pc = C2 + z * pc;
    const double rc = z * (C1 + z * pc);
    const double ks = y0 - ((z * (0.5 * y1 - v * ps) - y1) - v * S1);
    const double hz = 0.5 * z - qx;
    const double a = 1.0 - qx;
    const double kc = a - (hz - (z * rc - y0 * y1));
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    const double swap = bit_and(bit_xor(ks, kc), from_bits(0ULL - static_cast<uint64_t>(un & 1u)));
    out[0] = sign_by(bit_xor(ks, swap), un);
    out[1] = sign_by(bit_xor(kc, swap), un + 1u);
#else
    const uint32_t odd = un & 1u;
    out[0] = sign_by(odd ? kc : ks, un);
    out[1] = sign_by(odd ? ks : kc, un + 1u);
#endif
}

} // namespace brass::math

extern "C" {

double brass_sin_f64(double x) { return brass::math::sin_f64(x); }
double brass_cos_f64(double x) { return brass::math::cos_f64(x); }
void brass_sincos_f64(double x, double* out) { brass::math::sincos_f64(x, out); }

} // extern "C"
