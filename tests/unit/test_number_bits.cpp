#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <cmath>
#include <limits>
#include <string_view>

using namespace brass;

namespace {

size_t count_ops(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

int64_t run(Function& fn, int64_t n, double x) {
    Interpreter interp;
    return interp.run(fn, {RuntimeValue::from_i64(n), RuntimeValue::from_f64(x)}).as_i64();
}

// A NaN-boxing front end's loop: the sum of n x's carried round as a boxed
// double (a NaN boxed as the canonical one), each step checking that the
// sum is a number (its bits at most -inf's) and counting the checks that
// fail, returned with the boxed sum: `ok` the check of a fresh box, `okp`
// that of the loop's parameter.
constexpr std::string_view kSrc = R"(module @m
func @sum(%n: i64, %x: f64) -> i64 {
entry:
  %zero = iconst.i64 0
  br head(%zero, %zero, %zero)
head(%i: i64, %acc: i64, %bad: i64):
  %max = iconst.i64 -4503599627370496
  %okp = ule.i64 %acc, %max
  %in = slt.i64 %i, %n
  br_if %in, body, out(%acc, %bad)
body:
  %f = bitcast.f64.i64 %acc
  %g = add.f64 %f, %x
  %bits = bitcast.i64.f64 %g
  %isn = ne.f64 %g, %g
  %canon = iconst.i64 9221120237041090560
  %box = select.i64 %isn, %canon, %bits
  %ok = ule.i64 %box, %max
  %both = and.i32 %ok, %okp
  %one = iconst.i64 1
  %fail = xor.i32 %both, %one
  %failx = zext.i64 %fail
  %bad1 = add.i64 %bad, %failx
  %i1 = add.i64 %i, %one
  br head(%i1, %box, %bad1)
out(%r: i64, %b: i64):
  %r2 = xor.i64 %r, %b
  ret %r2
}
func @raw(%n: i64, %x: f64) -> i64 {
entry:
  br head(%n)
head(%v: i64):
  %max = iconst.i64 -4503599627370496
  %ok = ule.i64 %v, %max
  %okx = zext.i64 %ok
  %f = bitcast.f64.i64 %v
  %z = fconst.f64 0.0
  %pos = sgt.f64 %f, %z
  br_if %pos, out(%okx), out(%okx)
out(%r: i64):
  ret %r
}
)";

} // namespace

TEST_CASE("Number bits - a boxed double's number check folds, round a loop too") {
    DiagnosticReporter diag;
    auto ref = parse_module(kSrc, &diag);
    auto mod = parse_module(kSrc, &diag);
    REQUIRE(ref != nullptr);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("sum");
    Function* ref_fn = ref->get_function("sum");
    REQUIRE(fn != nullptr);
    CHECK(scalar_cleanup(*fn));
    CHECK(verify_function(*fn));
    CHECK_EQ(count_ops(*fn, Opcode::ule), 0u);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t n : {0, 1, 5}) {
        for (double x : {0.5, -3.25, nan, -nan, std::numeric_limits<double>::infinity()}) {
            CHECK_EQ(run(*fn, n, x), run(*ref_fn, n, x));
        }
    }
}

TEST_CASE("Number bits - a loop's boxed double is carried as the double") {
    DiagnosticReporter diag;
    auto ref = parse_module(kSrc, &diag);
    auto mod = parse_module(kSrc, &diag);
    REQUIRE(ref != nullptr);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("sum");
    Function* ref_fn = ref->get_function("sum");
    CHECK(unbox_double_params(*fn));
    CHECK(verify_function(*fn));
    scalar_cleanup(*fn);
    CHECK(verify_function(*fn));
    // The sum's unbox is gone: the loop adds doubles.
    CHECK_EQ(count_ops(*fn, Opcode::bitcast_f64_i64), 0u);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t n : {0, 1, 5}) {
        for (double x : {0.5, -3.25, nan, -nan, std::numeric_limits<double>::infinity()}) {
            CHECK_EQ(run(*fn, n, x), run(*ref_fn, n, x));
        }
    }
}

TEST_CASE("Number bits - an i64 that may be any bits stays as it is") {
    DiagnosticReporter diag;
    auto mod = parse_module(kSrc, &diag);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("raw");
    CHECK_FALSE(unbox_double_params(*fn));
    scalar_cleanup(*fn);
    CHECK_EQ(count_ops(*fn, Opcode::ule), 1u);
    CHECK_EQ(run(*fn, -1, 0.0), 0);
    CHECK_EQ(run(*fn, 1, 0.0), 1);
}
