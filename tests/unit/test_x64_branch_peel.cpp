#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

// Branch peeling (x64_isel_branch.cpp, find_peeled_branch): a br_if whose
// condition is boolean glue around a comparison (`and c, 1`, `ne c, 0`,
// `eq c, 0`) branches on the comparison's flags, the `eq c, 0` form with the
// condition inverted, and a float `eq` / `ne` branches as ucomis with a
// parity jump. The lowering cases pin that no setcc is left; the JIT cases
// that the answers hold at the values where an inverted float condition
// could go wrong: NaN, -0 against 0, infinities.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// Each block tests one glue form of an integer compare of %x and %y and
// adds its bit to the answer.
const char* kIntGlue = R"(module @peel

func @f(%x: i64, %y: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br b1(%z)
b1(%s1: i64):
  %c1 = ult.i64 %x, %y
  %one1 = iconst.i32 1
  %a1 = and.i32 %c1, %one1
  br_if %a1, t1, b2(%s1)
t1:
  %k1 = iconst.i64 1
  %n1 = add.i64 %s1, %k1
  br b2(%n1)
b2(%s2: i64):
  %c2 = eq.i64 %x, %y
  %zero2 = iconst.i32 0
  %e2 = eq.i32 %c2, %zero2
  br_if %e2, t2, b3(%s2)
t2:
  %k2 = iconst.i64 2
  %n2 = add.i64 %s2, %k2
  br b3(%n2)
b3(%s3: i64):
  %c3 = sle.i64 %x, %y
  %zero3 = iconst.i32 0
  %e3 = ne.i32 %c3, %zero3
  br_if %e3, t3, b4(%s3)
t3:
  %k3 = iconst.i64 4
  %n3 = add.i64 %s3, %k3
  br b4(%n3)
b4(%s4: i64):
  %c4 = sgt.i64 %x, %y
  %zero4 = iconst.i32 0
  %e4 = eq.i32 %c4, %zero4
  %zero5 = iconst.i32 0
  %e5 = eq.i32 %e4, %zero5
  br_if %e5, t4, out(%s4)
t4:
  %k4 = iconst.i64 8
  %n4 = add.i64 %s4, %k4
  br out(%n4)
out(%r: i64):
  ret %r
}
)";

int64_t int_reference(int64_t x, int64_t y) {
    int64_t s = 0;
    if (static_cast<uint64_t>(x) < static_cast<uint64_t>(y)) s += 1;
    if (!(x == y)) s += 2;
    if (x <= y) s += 4;
    if (x > y) s += 8;
    return s;
}

// Float eq / ne as the branch, bare and negated.
const char* kFloatEq = R"(module @peelf

func @g(%x: f64, %y: f64) -> i64 {
entry:
  %z = iconst.i64 0
  br b0(%z)
b0(%s0: i64):
  %c0 = eq.f64 %x, %y
  br_if %c0, t0, m0
t0:
  %k0 = iconst.i64 1
  %n0 = add.i64 %s0, %k0
  br b1(%n0)
m0:
  br b1(%s0)
b1(%s1: i64):
  %c1 = ne.f64 %x, %y
  br_if %c1, t1, m1
t1:
  %k1 = iconst.i64 2
  %n1 = add.i64 %s1, %k1
  br b2(%n1)
m1:
  br b2(%s1)
b2(%s2: i64):
  %c2 = eq.f64 %x, %y
  %zero2 = iconst.i32 0
  %e2 = eq.i32 %c2, %zero2
  br_if %e2, t2, m2
t2:
  %k2 = iconst.i64 4
  %n2 = add.i64 %s2, %k2
  br b3(%n2)
m2:
  br b3(%s2)
b3(%s3: i64):
  %c3 = ne.f64 %x, %y
  %zero3 = iconst.i32 0
  %e3 = eq.i32 %c3, %zero3
  br_if %e3, t3, m3
t3:
  %k3 = iconst.i64 8
  %n3 = add.i64 %s3, %k3
  br out(%n3)
m3:
  br out(%s3)
out(%r: i64):
  ret %r
}
)";

int64_t float_reference(double x, double y) {
    int64_t s = 0;
    if (x == y) s += 1;
    if (x != y) s += 2;
    if (!(x == y)) s += 4;
    if (!(x != y)) s += 8;
    return s;
}

size_t count_setcc(Module& mod) {
    Function* fn = mod.functions().front();
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    size_t n = 0;
    for (const auto& bb : lir->blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst && inst->opcode == LirOpcode::Setcc) ++n;
        }
    }
    return n;
}

}  // namespace

TEST_CASE("x64 branch peel - glue around an integer compare leaves no setcc") {
    auto mod = parse_module(kIntGlue);
    REQUIRE(mod != nullptr);
    CHECK_EQ(count_setcc(*mod), size_t(0));
}

TEST_CASE("x64 branch peel - float eq and ne branches leave no setcc") {
    auto mod = parse_module(kFloatEq);
    REQUIRE(mod != nullptr);
    CHECK_EQ(count_setcc(*mod), size_t(0));
}

TEST_CASE("x64 branch peel - peeled integer branches compute the same answers through the JIT") {
    auto mod = parse_module(kIntGlue);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("f");
    REQUIRE(fn != nullptr);
    const int64_t vals[] = {-9, -1, 0, 1, 7, INT64_MIN, INT64_MAX};
    for (int64_t x : vals) {
        for (int64_t y : vals) CHECK_EQ(fn(x, y), int_reference(x, y));
    }
}

TEST_CASE("x64 branch peel - float eq and ne branches hold at NaN and signed zeros through the JIT") {
    auto mod = parse_module(kFloatEq);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(double, double)>("g");
    REQUIRE(fn != nullptr);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double vals[] = {0.0, -0.0, nan, -nan, 1.5, -1.5, inf, -inf, 5e-324};
    for (double x : vals) {
        for (double y : vals) CHECK_EQ(fn(x, y), float_reference(x, y));
    }
}
