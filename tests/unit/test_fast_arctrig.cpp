// The arc-trig runtime (fast_trig.hpp, fast_arctrig.cpp): V8's bits for
// asin, atan and atan2, with fdlibm's special cases.

#include "test_framework.hpp"
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
