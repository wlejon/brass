#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>
#include <limits>

// An overflow check against a constant takes the constant as an immediate
// (x64_isel_overflow.cpp): `i + 1` is `add r, 1; jo`, `i * 3` is
// `imul t, r, 3; jo`, with no register or slot held for the constant, both
// as a branch (the sum taken from the check's register) and as a 0/1 value;
// and an equality with INT64_MIN is `cmp x, 1` and the overflow flag.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// f: a branch on sadd_overflow(i, 1) whose passing side computes the sum.
// g: a branch on smul_overflow(3, i) (constant first) with the product.
// h: ssub_overflow(i, 7) and smul_overflow(i, -5) as 0/1 values.
// k: a 64-bit uadd_overflow(i, 100) branch.
const char* kSrc = R"(module @m
func @f(%i: i32) -> i64 {
entry:
  %one = iconst.i32 1
  %o = sadd_overflow.i32 %i, %one
  br_if %o, ovf, ok
ok:
  %s = add.i32 %i, %one
  %w = sext.i64 %s
  ret %w
ovf:
  %m = iconst.i64 -1
  ret %m
}
func @g(%i: i32) -> i64 {
entry:
  %three = iconst.i32 3
  %o = smul_overflow.i32 %three, %i
  br_if %o, ovf, ok
ok:
  %p = mul.i32 %three, %i
  %w = sext.i64 %p
  ret %w
ovf:
  %m = iconst.i64 -1
  ret %m
}
func @h(%i: i32) -> i32 {
entry:
  %seven = iconst.i32 7
  %k = iconst.i32 -5
  %a = ssub_overflow.i32 %i, %seven
  %b = smul_overflow.i32 %i, %k
  %two = iconst.i32 2
  %bb = mul.i32 %b, %two
  %r = or.i32 %a, %bb
  ret %r
}
func @k(%i: i64) -> i64 {
entry:
  %c = iconst.i64 100
  %o = uadd_overflow.i64 %i, %c
  br_if %o, ovf, ok
ok:
  %s = add.i64 %i, %c
  ret %s
ovf:
  %m = iconst.i64 -1
  ret %m
}
)";

}  // namespace

TEST_CASE("x64 overflow imm - a constant operand of an overflow check is an immediate") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    for (Function* fn : mod->functions()) {
        fn->rebuild_cfg_predecessors();
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        size_t arith_imm = 0;
        size_t const_movs = 0;
        for (const auto& bb : lir->blocks) {
            for (const auto& inst : bb->instructions) {
                const LirOpcode op = inst->opcode;
                const bool arith = op == LirOpcode::Add || op == LirOpcode::Add32 || op == LirOpcode::Sub ||
                                   op == LirOpcode::Sub32 || op == LirOpcode::Imul || op == LirOpcode::Imul32;
                if (arith && !inst->uses.empty() && inst->uses.back().is_imm_int()) ++arith_imm;
                // A constant materialized into a register for the check.
                if ((op == LirOpcode::Mov || op == LirOpcode::Mov32 || op == LirOpcode::Movabs) &&
                    inst->uses.size() == 1 && inst->uses[0].is_imm_int() && inst->uses[0].imm_int != -1 &&
                    inst->uses[0].imm_int != 0) {
                    ++const_movs;
                }
            }
        }
        CHECK(arith_imm >= 1ULL);
        CHECK_EQ(const_movs, 0ULL);
    }
}

TEST_CASE("x64 overflow imm - immediates answer as the register forms did, at the limits") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int32_t)>("f");
    auto g = jit.get_function_ptr<int64_t (*)(int32_t)>("g");
    auto h = jit.get_function_ptr<int32_t (*)(int32_t)>("h");
    auto k = jit.get_function_ptr<int64_t (*)(int64_t)>("k");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    REQUIRE(h != nullptr);
    REQUIRE(k != nullptr);
    const int32_t imax = std::numeric_limits<int32_t>::max();
    const int32_t imin = std::numeric_limits<int32_t>::min();
    for (int32_t i : {imin, imin + 6, -1000000000, -715827883, -715827882, -1, 0, 1, 715827882, 715827883, imax - 1, imax}) {
        const int64_t w = i;
        CHECK_EQ(f(i), w + 1 > imax ? -1 : w + 1);
        CHECK_EQ(g(i), (w * 3 > imax || w * 3 < imin) ? -1 : w * 3);
        const bool sub_ovf = w - 7 < imin;
        const bool mul_ovf = w * -5 > imax || w * -5 < imin;
        CHECK_EQ(h(i), (sub_ovf ? 1 : 0) | (mul_ovf ? 2 : 0));
    }
    const int64_t umax = -1;
    for (int64_t i : {int64_t{0}, int64_t{5}, umax - 100, umax - 99, umax}) {
        const bool ovf = static_cast<uint64_t>(i) + 100 < static_cast<uint64_t>(i);
        CHECK_EQ(k(i), ovf ? -1 : static_cast<int64_t>(static_cast<uint64_t>(i) + 100));
    }
}

namespace {

// `eq` / `ne` of an i64 against INT64_MIN (cvttsd2si's answer for NaN, the
// infinities and |x| >= 2^63) as a branch and as a guard-like select: `cmp
// x, 1` and the overflow flag, no movabs of the constant.
const char* kMinSrc = R"(module @m
func @f(%x: f64) -> i64 {
entry:
  %i = fptosi_i64 %x
  %min = iconst.i64 -9223372036854775808
  %e = eq.i64 %i, %min
  br_if %e, slow, fast
fast:
  ret %i
slow:
  %z = iconst.i64 7
  ret %z
}
func @g(%a: i64, %b: i64) -> i64 {
entry:
  %min = iconst.i64 -9223372036854775808
  %e = ne.i64 %min, %a
  %one = iconst.i64 1
  %c = iconst.i64 2
  %lt = slt.i64 %b, %c
  %both = and.i32 %e, %lt
  br_if %both, yes, no
yes:
  ret %one
no:
  %zero = iconst.i64 0
  ret %zero
}
)";

}  // namespace

TEST_CASE("x64 int64_min compare - an equality with INT64_MIN is cmp x, 1 and the overflow flag") {
    auto mod = parse_module(kMinSrc);
    REQUIRE(mod != nullptr);
    for (Function* fn : mod->functions()) {
        fn->rebuild_cfg_predecessors();
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        size_t movabs = 0;
        size_t jo = 0;
        for (const auto& bb : lir->blocks) {
            for (const auto& inst : bb->instructions) {
                movabs += inst->opcode == LirOpcode::Movabs;
                jo += inst->opcode == LirOpcode::Jcc &&
                      (inst->condition == Condition::O || inst->condition == Condition::NO);
            }
        }
        CHECK_EQ(movabs, 0ULL);
        CHECK(jo >= 1ULL);
    }
}

TEST_CASE("x64 int64_min compare - answers at INT64_MIN and its neighbours") {
    auto mod = parse_module(kMinSrc);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(double)>("f");
    auto g = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("g");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_EQ(f(nan), 7);
    CHECK_EQ(f(inf), 7);
    CHECK_EQ(f(-inf), 7);
    CHECK_EQ(f(1e19), 7);
    CHECK_EQ(f(-9223372036854775808.0), 7);
    CHECK_EQ(f(3.7), 3);
    CHECK_EQ(f(-3.7), -3);
    CHECK_EQ(f(-9223372036854774784.0), -9223372036854774784LL);
    const int64_t imin = std::numeric_limits<int64_t>::min();
    for (int64_t a : {imin, imin + 1, int64_t{-1}, int64_t{0}, std::numeric_limits<int64_t>::max()}) {
        for (int64_t b : {int64_t{-5}, int64_t{1}, int64_t{2}, int64_t{9}}) {
            CHECK_EQ(g(a, b), (a != imin && b < 2) ? 1 : 0);
        }
    }
}
