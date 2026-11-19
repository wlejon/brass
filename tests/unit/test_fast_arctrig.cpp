#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
// The arc-trig runtime (fast_trig.hpp, fast_arctrig.cpp): V8's bits for
// asin, atan and atan2, with fdlibm's special cases.

#include "test_framework.hpp"
#include <bit>
#include <brass/runtime/fast_trig.hpp>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace brass;
using namespace brass::test;

namespace {

uint64_t bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

struct Atan2Golden { double y, x; uint64_t bits; };
struct UnaryGolden { double x; uint64_t bits; };

// node 22: Math.atan2(y, x), Math.atan(x), Math.asin(x).
constexpr Atan2Golden kAtan2Goldens[] = {
    {0.3, 0.7, 0x3fd9e9bf3d20dc71ULL},
    {-0.3, 0.7, 0xbfd9e9bf3d20dc71ULL},
    {0.3, -0.7, 0x4005e4c36ca0118aULL},
    {-0.3, -0.7, 0xc005e4c36ca0118aULL},
    {1.0, 1.0, 0x3fe921fb54442d18ULL},
    {2.5, 1.0, 0x3ff30b6d796a4da8ULL},
    {0.5, 1.0, 0x3fddac670561bb4fULL},
    {1e-10, 3.0, 0x3dc2533fe68fd3d2ULL},
    {5.0, 1e-12, 0x3ff921fb54442994ULL},
    {-0.8414709848078965, 0.5403023058681398, 0xbff0000000000000ULL},
    {0.6, 0.9, 0x3fe2d0ead6066395ULL},
    {1.3, 0.7, 0x3ff13acc43a9c24eULL},
    {7.0, 2.0, 0x3ff4ae10fc6589a5ULL},
    {1e+80, 1e-80, 0x3ff921fb54442d18ULL},
    {1e-80, -1e+80, 0x400921fb54442d18ULL},
    {0.123456789, 0.987654321, 0x3fbfd5ba95db0a6bULL},
};
constexpr UnaryGolden kAtanGoldens[] = {
    {0.1, 0x3fb983e282e2cc4dULL},     {0.4375, 0x3fda64eec3cc23fdULL}, {0.5, 0x3fddac670561bb4fULL},
    {0.6, 0x3fe14b1dd5f90ce1ULL},     {0.6875, 0x3fe345f01cce37bbULL}, {1.0, 0x3fe921fb54442d18ULL},
    {1.1875, 0x3febde70ed439fe7ULL},  {1.5, 0x3fef730bd281f69bULL},    {2.4375, 0x3ff2e75728833a54ULL},
    {3.0, 0x3ff3fc176b7a8560ULL},     {-0.25, 0xbfcf5b75f92c80ddULL},  {-0.9, 0xbfe77338a80603beULL},
    {-5.0, 0xbff5f97315254857ULL},    {1e-10, 0x3ddb7cdfd9d7bdbbULL},  {1e20, 0x3ff921fb54442d18ULL},
    {1e+30, 0x3ff921fb54442d18ULL},   {-1e+30, 0xbff921fb54442d18ULL},
};
constexpr UnaryGolden kAsinGoldens[] = {
    {0.1, 0x3fb9a49276037884ULL},   {0.3, 0x3fd380159e14f6ffULL},      {0.4999, 0x3fe0c06011bb06b4ULL},
    {0.5, 0x3fe0c152382d7366ULL},   {0.6, 0x3fe4978fa3269ee1ULL},      {0.8, 0x3fedac670561bb50ULL},
    {0.975, 0x3ff58c2b5ce0c3e5ULL}, {0.98, 0x3ff5ed690583be07ULL},     {0.999999, 0x3ff91c306b2c13adULL},
    {1.0, 0x3ff921fb54442d18ULL},   {-1.0, 0xbff921fb54442d18ULL},     {-0.25, 0xbfd02be9ce0b87cdULL},
    {-0.7, 0xbfe8d00e692afd95ULL},  {-0.99, 0xbff6de3c6f33d51dULL},    {1e-9, 0x3e112e0be826d695ULL},
};

} // namespace

TEST_CASE("Fast arc trig - asin, atan and atan2 are V8's bits") {
    for (const auto& g : kAtan2Goldens) CHECK_EQ(bits(math::atan2_f64(g.y, g.x)), g.bits);
    for (const auto& g : kAtanGoldens) CHECK_EQ(bits(math::atan_f64(g.x)), g.bits);
    for (const auto& g : kAsinGoldens) CHECK_EQ(bits(math::asin_f64(g.x)), g.bits);
}

TEST_CASE("Fast arc trig - zeros, infinities and NaN") {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double pi = 3.141592653589793;
    CHECK_EQ(bits(math::atan2_f64(0.0, 1.0)), bits(0.0));
    CHECK_EQ(bits(math::atan2_f64(-0.0, 1.0)), bits(-0.0));
    CHECK_EQ(bits(math::atan2_f64(0.0, 5.0)), bits(0.0));
    CHECK_EQ(bits(math::atan2_f64(-0.0, 5.0)), bits(-0.0));
    CHECK_EQ(bits(math::atan2_f64(0.0, -5.0)), bits(pi));
    CHECK_EQ(bits(math::atan2_f64(-0.0, -5.0)), bits(-pi));
    CHECK_EQ(bits(math::atan2_f64(0.0, -0.0)), bits(pi));
    CHECK_EQ(bits(math::atan2_f64(0.0, 0.0)), bits(0.0));
    CHECK_EQ(bits(math::atan2_f64(3.0, 0.0)), bits(pi / 2));
    CHECK_EQ(bits(math::atan2_f64(-3.0, -0.0)), bits(-pi / 2));
    CHECK_EQ(bits(math::atan2_f64(inf, inf)), bits(pi / 4));
    CHECK_EQ(bits(math::atan2_f64(-inf, inf)), bits(-pi / 4));
    CHECK_EQ(bits(math::atan2_f64(inf, -inf)), bits(3 * pi / 4));
    CHECK_EQ(bits(math::atan2_f64(-inf, -inf)), bits(-3 * pi / 4));
    CHECK_EQ(bits(math::atan2_f64(2.0, inf)), bits(0.0));
    CHECK_EQ(bits(math::atan2_f64(-2.0, inf)), bits(-0.0));
    CHECK_EQ(bits(math::atan2_f64(2.0, -inf)), bits(pi));
    CHECK_EQ(bits(math::atan2_f64(-2.0, -inf)), bits(-pi));
    CHECK_EQ(bits(math::atan2_f64(inf, 2.0)), bits(pi / 2));
    CHECK_EQ(bits(math::atan2_f64(-inf, -2.0)), bits(-pi / 2));
    CHECK(math::atan2_f64(qnan, 1.0) != math::atan2_f64(qnan, 1.0));
    CHECK(math::atan2_f64(1.0, qnan) != math::atan2_f64(1.0, qnan));
    CHECK_EQ(bits(math::atan_f64(0.0)), bits(0.0));
    CHECK_EQ(bits(math::atan_f64(-0.0)), bits(-0.0));
    CHECK_EQ(bits(math::atan_f64(inf)), bits(pi / 2));
    CHECK_EQ(bits(math::atan_f64(-inf)), bits(-pi / 2));
    CHECK(math::atan_f64(qnan) != math::atan_f64(qnan));
    CHECK_EQ(bits(math::asin_f64(0.0)), bits(0.0));
    CHECK_EQ(bits(math::asin_f64(-0.0)), bits(-0.0));
    CHECK(math::asin_f64(1.5) != math::asin_f64(1.5));
    CHECK(math::asin_f64(-inf) != math::asin_f64(-inf));
    CHECK(math::asin_f64(qnan) != math::asin_f64(qnan));
}

namespace {

double ref_atan(double x) noexcept {
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    constexpr uint64_t kAbsInf = 0x7ff0000000000000ULL;
    constexpr double kAtanHi[4] = {4.63647609000806093515e-01, 7.85398163397448278999e-01,
                                   9.82793723247329054082e-01, 1.57079632679489655800e+00};
    constexpr double kAtanLo[4] = {2.26987774529616870924e-17, 3.06161699786838301793e-17,
                                   1.39033110312309984516e-17, 6.12323399573676603587e-17};
    constexpr double kAT[11] = {
        3.33333333333329318027e-01,  -1.99999999998764832476e-01, 1.42857142725034663711e-01,
        -1.11111104054623557880e-01, 9.09088713343650656196e-02,  -7.69187620504482999495e-02,
        6.66107313738753120669e-02,  -5.83357013379057348645e-02, 4.97687799461593236017e-02,
        -3.65315727442169155270e-02, 1.62858201153657823623e-02,
    };
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const uint64_t a = b & ~kSignMask;
    const auto ix = static_cast<int32_t>(a >> 32);
    const bool negative = (b >> 63) != 0;
    if (ix >= 0x44100000) {
        if (a > kAbsInf) return x + x;
        return negative ? -kAtanHi[3] - kAtanLo[3] : kAtanHi[3] + kAtanLo[3];
    }
    int id;
    if (ix < 0x3fdc0000) {
        if (ix < 0x3e200000) return x;
        id = -1;
    } else {
        x = std::bit_cast<double>(a);
        if (ix < 0x3ff30000) {
            if (ix < 0x3fe60000) {
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if (ix < 0x40038000) {
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
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

double ref_atan2(double y, double x) noexcept {
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    constexpr uint64_t kAbsInf = 0x7ff0000000000000ULL;
    constexpr double kPi = 3.1415926535897931160e+00;
    constexpr double kPiLo = 1.2246467991473531772e-16;
    constexpr double kPiO2 = 1.5707963267948965580e+00;
    constexpr double kPiO4 = 7.8539816339744827900e-01;
    constexpr double kTiny = 1.0e-300;
    const uint64_t xb = std::bit_cast<uint64_t>(x);
    const uint64_t yb = std::bit_cast<uint64_t>(y);
    const uint64_t xa = xb & ~kSignMask;
    const uint64_t ya = yb & ~kSignMask;
    uint32_t m = static_cast<uint32_t>(yb >> 63) | (static_cast<uint32_t>(xb >> 62) & 2u);
    if (xa - 1 >= kAbsInf - 1 || ya - 1 >= kAbsInf - 1) {
        if (xa > kAbsInf || ya > kAbsInf) return x + y;
        if (xb == std::bit_cast<uint64_t>(1.0)) return ref_atan(y);
        if (ya == 0) {
            switch (m) {
                case 0:
                case 1: return y;
                case 2: return kPi + kTiny;
                default: return -kPi - kTiny;
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
        return (m & 1u) ? -kPiO2 - kTiny : kPiO2 + kTiny;
    }
    if (xb == std::bit_cast<uint64_t>(1.0)) return ref_atan(y);
    const auto ix = static_cast<int32_t>(xa >> 32);
    const auto iy = static_cast<int32_t>(ya >> 32);
    const int32_t k = (iy - ix) >> 20;
    double z;
    if (k > 60) {
        z = kPiO2 + 0.5 * kPiLo;
        m &= 1u;
    } else if ((m & 2u) != 0 && k < -60) {
        z = 0.0;
    } else {
        const double q = std::bit_cast<double>(std::bit_cast<uint64_t>(y / x) & ~kSignMask);
        z = ref_atan(q);
    }
    switch (m) {
        case 0: return z;
        case 1: return -z;
        case 2: return kPi - (z - kPiLo);
        default: return (z - kPiLo) - kPi;
    }
}

}  // namespace

TEST_CASE("Fast arc trig - branchless matches reference fdlibm bit-for-bit across all intervals and quadrants") {
    constexpr uint32_t kHiBounds[] = {
        0x3e1fffffU, 0x3e200000U, 0x3e200001U,
        0x3fdbffffU, 0x3fdc0000U, 0x3fdc0001U,
        0x3fe5ffffU, 0x3fe60000U, 0x3fe60001U,
        0x3ff2ffffU, 0x3ff30000U, 0x3ff30001U,
        0x40037fffU, 0x40038000U, 0x40038001U,
        0x440fffffU, 0x44100000U, 0x44100001U,
    };
    for (uint32_t hi : kHiBounds) {
        for (uint64_t lo : {0x00000000ULL, 0x80000000ULL, 0xffffffffULL}) {
            const uint64_t mag = (uint64_t{hi} << 32) | lo;
            for (uint64_t sign : {uint64_t{0}, uint64_t{1} << 63}) {
                const double v = std::bit_cast<double>(mag | sign);
                CHECK_EQ(bits(math::atan_f64(v)), bits(ref_atan(v)));
            }
        }
    }

    uint64_t state = 0x123456789abcdef0ULL;
    auto next_u64 = [&]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (int i = 0; i < 4500; ++i) {
        const double sx = std::bit_cast<double>(next_u64());
        const double sy = std::bit_cast<double>(next_u64());
        CHECK_EQ(bits(math::atan_f64(sx)), bits(ref_atan(sx)));
        if (std::bit_cast<uint64_t>(sx) == std::bit_cast<uint64_t>(sx) &&
            !((std::bit_cast<uint64_t>(sx) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL ||
              (std::bit_cast<uint64_t>(sy) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)) {
            CHECK_EQ(bits(math::atan2_f64(sy, sx)), bits(ref_atan2(sy, sx)));
        }
    }
    for (int q = 0; q < 4; ++q) {
        const uint64_t sy_sign = (q & 1) ? (uint64_t{1} << 63) : 0ULL;
        const uint64_t sx_sign = (q & 2) ? (uint64_t{1} << 63) : 0ULL;
        for (double magY : {1e-25, 0.25, 0.4375, 0.7, 1.0, 1.1875, 1.9, 2.4375, 10.0, 1e25}) {
            for (double magX : {1e-25, 0.3, 0.7, 1.0, 1.5, 2.0, 1e25}) {
                const double yv = std::bit_cast<double>(bits(magY) | sy_sign);
                const double xv = std::bit_cast<double>(bits(magX) | sx_sign);
                CHECK_EQ(bits(math::atan2_f64(yv, xv)), bits(ref_atan2(yv, xv)));
                CHECK_EQ(bits(math::atan_f64(yv)), bits(ref_atan(yv)));
            }
        }
    }
}
