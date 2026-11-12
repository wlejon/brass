#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

// A br_if on an OR of comparisons, on an `and` / `or` tree of them, and a
// float `eq` / `ne` leaf (x64_isel_and_chain.cpp): each leaf is its flags and
// a jump, with no setcc, and answers what the value form answered, NaN
// included.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// f: a tag test, `t == 5 || t == 9 || t <u 2`.
// g: an AND with a float eq in the middle, as an integral-index test is.
// h: an OR with a float ne first and a float eq last.
// k: an OR under an AND and an AND under an OR, `((t == 5 || t == 9) && t == u) || (u <s 0 && t <s u)`.
const char* kSrc = R"(module @m
func @f(%t: i64) -> i64 {
entry:
  %five = iconst.i64 5
  %nine = iconst.i64 9
  %two = iconst.i64 2
  %a = eq.i64 %t, %five
  %b = eq.i64 %t, %nine
  %c = ult.i64 %t, %two
  %ab = or.i32 %a, %b
  %abc = or.i32 %ab, %c
  br_if %abc, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
func @g(%x: f64, %y: f64, %n: i64) -> i64 {
entry:
  %ten = iconst.i64 10
  %lt = slt.i64 %n, %ten
  %same = eq.f64 %x, %y
  %zero = iconst.i64 0
  %pos = sgt.i64 %n, %zero
  %a = and.i32 %lt, %same
  %b = and.i32 %a, %pos
  br_if %b, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
func @k(%t: i64, %u: i64) -> i64 {
entry:
  %five = iconst.i64 5
  %nine = iconst.i64 9
  %zero = iconst.i64 0
  %a = eq.i64 %t, %five
  %b = eq.i64 %t, %nine
  %ab = or.i32 %a, %b
  %same = eq.i64 %t, %u
  %left = and.i32 %ab, %same
  %neg = slt.i64 %u, %zero
  %below = slt.i64 %t, %u
  %right = and.i32 %neg, %below
  %all = or.i32 %left, %right
  br_if %all, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
func @h(%x: f64, %y: f64, %z: f64) -> i64 {
entry:
  %differ = ne.f64 %x, %y
  %same = eq.f64 %y, %z
  %a = or.i32 %differ, %same
  br_if %a, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
)";

}  // namespace

TEST_CASE("x64 chains - an OR of comparisons and a float eq leaf lower without setcc") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    for (Function* fn : mod->functions()) {
        fn->rebuild_cfg_predecessors();
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        size_t setcc = 0;
        size_t jcc = 0;
        for (const auto& bb : lir->blocks) {
            for (const auto& inst : bb->instructions) {
                setcc += inst->opcode == LirOpcode::Setcc;
                jcc += inst->opcode == LirOpcode::Jcc;
            }
        }
        CHECK_EQ(setcc, 0ULL);
        CHECK(jcc >= 3ULL);
    }
}

TEST_CASE("x64 chains - OR chains and float leaves answer through the JIT, NaN included") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t)>("f");
    auto g = jit.get_function_ptr<int64_t (*)(double, double, int64_t)>("g");
    auto h = jit.get_function_ptr<int64_t (*)(double, double, double)>("h");
    auto k = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("k");
    REQUIRE(k != nullptr);
    for (int64_t t : {int64_t{-7}, int64_t{-2}, int64_t{0}, int64_t{5}, int64_t{9}}) {
        for (int64_t u : {int64_t{-7}, int64_t{-1}, int64_t{5}, int64_t{9}, int64_t{11}}) {
            CHECK_EQ(k(t, u), (((t == 5 || t == 9) && t == u) || (u < 0 && t < u)) ? 1 : 0);
        }
    }
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    REQUIRE(h != nullptr);
    for (int64_t t : {int64_t{-1}, int64_t{0}, int64_t{1}, int64_t{2}, int64_t{5}, int64_t{7}, int64_t{9}, int64_t{10}}) {
        CHECK_EQ(f(t), (t == 5 || t == 9 || static_cast<uint64_t>(t) < 2) ? 1 : 0);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double vals[] = {0.0, -0.0, 1.5, 2.0, nan};
    for (double x : vals) {
        for (double y : vals) {
            for (int64_t n : {int64_t{-3}, int64_t{0}, int64_t{4}, int64_t{10}}) {
                CHECK_EQ(g(x, y, n), (n < 10 && x == y && n > 0) ? 1 : 0);
            }
            for (double z : vals) {
                CHECK_EQ(h(x, y, z), (x != y || y == z) ? 1 : 0);
            }
        }
    }
}
