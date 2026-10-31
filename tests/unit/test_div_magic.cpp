#include "brass/core/env.hpp"
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/div_magic.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstdint>
#include <string>
#include <vector>

// Signed division / remainder of an i64 by a constant as a multiply by a
// magic number (div_magic.hpp, x64_isel_shift_div.cpp lower_signed_div_magic,
// aarch64_isel_alu.cpp lower_div_mod). Three halves:
//   - the identity itself, evaluated in portable unsigned arithmetic (the
//     high half of the product from 32-bit limbs), so it is checked on any
//     host for every divisor below, which is what the AArch64 sequence
//     computes as well;
//   - both selectors run explicitly on smod / sdiv by a constant: smulh and
//     no divide, or the divide under BRASS_NO_DIV_MAGIC;
//   - the JIT on this host against C++ `/` and `%` (d > 0, so neither
//     overflows), over the int64 edges.

using namespace brass;
using namespace brass::codegen;

namespace {

uint64_t mulhu(uint64_t a, uint64_t b) {
    const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32;
    const uint64_t b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// High 64 bits of the signed 128-bit product, modulo 2^64.
uint64_t mulhs(int64_t a, int64_t b) {
    const uint64_t ua = static_cast<uint64_t>(a), ub = static_cast<uint64_t>(b);
    uint64_t h = mulhu(ua, ub);
    if (a < 0) h -= ub;
    if (b < 0) h -= ua;
    return h;
}

uint64_t asr(uint64_t v, int s) {
    if (s == 0) return v;
    const uint64_t fill = (v >> 63) ? ~(~uint64_t{0} >> s) : 0;
    return (v >> s) | fill;
}

// What the lowered sequence computes.
int64_t magic_quotient(int64_t n, int64_t d) {
    const SignedDivMagic mg = signed_div_magic64(d);
    uint64_t t = mulhs(mg.multiplier, n);
    if (mg.multiplier < 0) t += static_cast<uint64_t>(n);
    t = asr(t, mg.shift);
    t += static_cast<uint64_t>(n) >> 63;
    return static_cast<int64_t>(t);
}

const std::vector<int64_t>& divisors() {
    static const std::vector<int64_t> d = {
        3, 5, 6, 7, 9, 10, 11, 12, 13, 25, 100, 125, 641, 1000, 16807, 6700417,
        1000000007, 2147483646, 2147483647, 2147483649LL, 4294967295LL, 4294967297LL,
        0x5555555555555555LL, 0x4000000000000001LL, 0x3FFFFFFFFFFFFFFFLL,
        0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFFLL, 3LL << 40, 9007199254740993LL,
    };
    return d;
}

std::vector<int64_t> dividends(int64_t d) {
    const uint64_t ud = static_cast<uint64_t>(d);
    std::vector<uint64_t> u = {
        0, 1, ~uint64_t{0}, 2, ~uint64_t{1},
        uint64_t{1} << 63, (uint64_t{1} << 63) + 1, (uint64_t{1} << 63) - 1, (uint64_t{1} << 63) - 2,
        ud, ud - 1, ud + 1, 0 - ud, 0 - ud + 1, 0 - ud - 1, 2 * ud, 0 - 2 * ud, 2 * ud - 1, 0 - (2 * ud - 1),
        uint64_t{1} << 53, (uint64_t{1} << 53) - 1, 0 - (uint64_t{1} << 53),
        uint64_t{0x7FFFFFFF} * 16807, uint64_t{2147483647}, uint64_t{2147483648}, 0 - uint64_t{2147483648},
    };
    // A few multiples of d near the int64 ends (kept below 2^63 in magnitude).
    const uint64_t k = (uint64_t{1} << 63) / ud;
    u.push_back(k * ud);
    u.push_back(k * ud - 1);
    u.push_back(0 - k * ud);
    u.push_back(0 - k * ud + 1);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 64; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        u.push_back(x);
        u.push_back(x >> (i % 63));
    }
    std::vector<int64_t> out;
    for (uint64_t v : u) out.push_back(static_cast<int64_t>(v));
    return out;
}

bool seam_off() { return !brass::env_value("BRASS_NO_DIV_MAGIC").empty(); }

std::string module_text(int64_t d) {
    std::string ds = std::to_string(d);
    return "module @dm\n\n"
           "func @rem(%n: i64) -> i64 {\nentry:\n  %d = iconst.i64 " + ds +
           "\n  %r = smod.i64 %n, %d\n  ret %r\n}\n\n"
           "func @quo(%n: i64) -> i64 {\nentry:\n  %d = iconst.i64 " + ds +
           "\n  %q = sdiv.i64 %n, %d\n  ret %q\n}\n";
}

template <typename Lir>
void count_ops(const Lir& lir, size_t& smulh, size_t& divs) {
    smulh = 0;
    divs = 0;
    for (const auto& bb : lir.blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst->opcode == LirOpcode::Smulh) ++smulh;
            if (inst->opcode == LirOpcode::Idiv || inst->opcode == LirOpcode::Idiv32) ++divs;
        }
    }
}

}  // namespace

TEST_CASE("Div magic - the identity holds for every divisor and int64 edge (portable arithmetic)") {
    size_t checked = 0;
    for (int64_t d : divisors()) {
        REQUIRE(takes_signed_div_magic(d));
        for (int64_t n : dividends(d)) {
            CHECK_EQ(magic_quotient(n, d), n / d);
            ++checked;
        }
    }
    CHECK(checked > 1000);
    CHECK(!takes_signed_div_magic(0));
    CHECK(!takes_signed_div_magic(1));
    CHECK(!takes_signed_div_magic(2));
    CHECK(!takes_signed_div_magic(4));
    CHECK(!takes_signed_div_magic(-3));
    CHECK(!takes_signed_div_magic(-1));
    CHECK(!takes_signed_div_magic(INT64_MIN));
    CHECK(!takes_signed_div_magic(int64_t{1} << 62));
}

TEST_CASE("Div magic - known multipliers") {
    // Hacker's Delight table 10-2 (64-bit column) and GCC's output.
    CHECK_EQ(signed_div_magic64(3).multiplier, int64_t{0x5555555555555556LL});
    CHECK_EQ(signed_div_magic64(3).shift, 0);
    CHECK_EQ(signed_div_magic64(5).multiplier, int64_t{0x6666666666666667LL});
    CHECK_EQ(signed_div_magic64(5).shift, 1);
    CHECK_EQ(signed_div_magic64(7).multiplier, int64_t{0x4924924924924925LL});
    CHECK_EQ(signed_div_magic64(7).shift, 1);
}

TEST_CASE("Div magic - x64 selector: smulh, no idiv") {
    auto mod = parse_module(module_text(2147483647));
    REQUIRE(mod != nullptr);
    for (Function* fn : mod->functions()) {
        fn->rebuild_cfg_predecessors();
        x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        size_t smulh = 0, divs = 0;
        count_ops(*lir, smulh, divs);
        CHECK_EQ(smulh, seam_off() ? size_t(0) : size_t(1));
        CHECK_EQ(divs, seam_off() ? size_t(1) : size_t(0));
    }
}

TEST_CASE("Div magic - AArch64 selector: smulh, no sdiv") {
    for (int64_t d : {int64_t{7}, int64_t{2147483647}, int64_t{0x7FFFFFFFFFFFFFFFLL}}) {
        auto mod = parse_module(module_text(d));
        REQUIRE(mod != nullptr);
        for (Function* fn : mod->functions()) {
            fn->rebuild_cfg_predecessors();
            aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
            auto lir = isel.lower(*fn);
            REQUIRE(lir != nullptr);
            size_t smulh = 0, divs = 0;
            count_ops(*lir, smulh, divs);
            CHECK_EQ(smulh, seam_off() ? size_t(0) : size_t(1));
            CHECK_EQ(divs, seam_off() ? size_t(1) : size_t(0));
        }
    }
}

TEST_CASE("Div magic - JIT quotient and remainder match / and % over the int64 edges") {
    for (int64_t d : divisors()) {
        auto mod = parse_module(module_text(d));
        REQUIRE(mod != nullptr);
        JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(*mod));
        auto rem = jit.get_function_ptr<int64_t (*)(int64_t)>("rem");
        auto quo = jit.get_function_ptr<int64_t (*)(int64_t)>("quo");
        REQUIRE(rem != nullptr);
        REQUIRE(quo != nullptr);
        for (int64_t n : dividends(d)) {
            CHECK_EQ(quo(n), n / d);
            CHECK_EQ(rem(n), n % d);
        }
    }
}
