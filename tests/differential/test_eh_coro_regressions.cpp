// Exception/coroutine regressions from sweep 4, run in the interpreter and in
// tier 2 (the x64 JIT) from MIR text.
//
// S4-1: the Calls x64 isel emitted for invoke and coro_create/resume/destroy
// declared no caller-saved clobbers (and invoke no RAX/XMM0 result def), so
// values live across them sat in registers the callee overwrote.
//
// S4-3: coro_create's operands are the coroutine's arguments; x64 isel read
// them as a slot count and pointer mask and never stored the arguments.
//
// S4-4: a coroutine body that had not been through CoroTransformPass ran with
// silently wrong results (a second resume returned 0, values live across a
// suspend were lost). Bodies now run only lowered; anything else is an error.

#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <memory>
#include <string_view>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> parse_or_fail(std::string_view src, bool lower_coroutines = true) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    if (lower_coroutines) {
        CoroTransformPass().run_on_module(*mod);
        DiagnosticReporter post;
        REQUIRE(verify_module(*mod, &post));
    }
    return mod;
}

int64_t interp_run(const Module& mod, std::string_view fn, int64_t arg) {
    Interpreter interp;
    return interp.run(mod, fn, {RuntimeValue::from_i64(arg)}).as_i64();
}

int64_t jit_run(Module& mod, std::string_view fn, int64_t arg) {
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    return jit.invoke(fn, {RuntimeValue::from_i64(arg)}).as_i64();
}

// Interpreter and tier 2 agree, and on the expected value.
void check_both(Module& mod, std::string_view fn, int64_t arg, int64_t expected) {
    CHECK_EQ(interp_run(mod, fn, arg), expected);
    CHECK_EQ(jit_run(mod, fn, arg), expected);
}

constexpr std::string_view kEh = R"(
func @work(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 0
  %2 = slt.i64 %0, %1
  br_if %2, bb1, bb2

bb1:
  throw %0

bb2:
  %3 = iconst.i64 3
  %4 = mul.i64 %0, %3
  ret %4
}

func @d_add(%0: i64) -> i64 {
bb0:
  %8 = iconst.i64 7
  %9 = add.i64 %0, %8
  %12 = invoke.i64 @work(%9), bb3, bb4

bb3:
  ret %12

bb4:
  %16 = landing_pad
  ret %16
}

func @f_add(%0: i64) -> i64 {
bb0:
  %8 = iconst.i64 7
  %9 = add.i64 %0, %8
  %12 = invoke.i64 @work(%9), bb3, bb4

bb3:
  %13 = add.i64 %12, %9
  ret %13

bb4:
  %16 = landing_pad
  %17 = add.i64 %16, %9
  ret %17
}

func @loop_eh(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 0
  %2 = iconst.i64 0
  br bb1(%1, %2)

bb1(%3: i64, %4: i64):
  %5 = slt.i64 %3, %0
  br_if %5, bb2, bb5

bb2:
  %6 = iconst.i64 3
  %7 = smod.i64 %3, %6
  %8 = iconst.i64 0
  %9 = eq.i64 %7, %8
  %10 = sub.i64 %8, %3
  %11 = select.i64 %9, %10, %3
  %12 = invoke.i64 @work(%11), bb3, bb4

bb3:
  %13 = add.i64 %4, %12
  %14 = iconst.i64 1
  %15 = add.i64 %3, %14
  br bb1(%15, %13)

bb4:
  %16 = landing_pad
  %17 = add.i64 %4, %16
  %18 = iconst.i64 1
  %19 = add.i64 %3, %18
  %20 = iconst.i64 100000
  %21 = add.i64 %17, %20
  br bb1(%19, %21)

bb5:
  ret %4
}
)";

// gen yields 1, then 1 + r1, then returns (1 + r1) + r2 + arg.
constexpr std::string_view kCoro1 = R"(
func @gen(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 1
  %2 = coro_suspend.i64 %1, 1
  %3 = add.i64 %1, %2
  %4 = coro_suspend.i64 %3, 2
  %5 = add.i64 %3, %4
  %6 = add.i64 %5, %0
  ret %6
}

func @main(%0: i64) -> i64 {
bb0:
  %1 = coro_create @gen(%0)
  %2 = iconst.i64 0
  %3 = coro_resume.i64 %1, %2
  %4 = iconst.i64 10
  %5 = coro_resume.i64 %1, %4
  %6 = iconst.i64 20
  %7 = coro_resume.i64 %1, %6
  coro_destroy %1
  %8 = iconst.i64 1000000
  %9 = mul.i64 %3, %8
  %10 = iconst.i64 1000
  %11 = mul.i64 %5, %10
  %12 = add.i64 %9, %11
  %13 = add.i64 %12, %7
  ret %13
}
)";

constexpr std::string_view kCoro3 = R"(
func @gen(%0: gcref) -> i64 {
bb0:
  %1 = iconst.i64 42
  ret %1
}

func @only_res(%0: i64) -> i64 {
bb0:
  %1 = coro_create @gen()
  %2 = iconst.i64 0
  %3 = coro_resume.i64 %1, %2
  coro_destroy %1
  ret %3
}

func @only_live(%0: i64) -> i64 {
bb0:
  %20 = iconst.i64 7
  %21 = add.i64 %0, %20
  %22 = mul.i64 %21, %21
  %1 = coro_create @gen()
  coro_destroy %1
  %5 = add.i64 %21, %22
  ret %5
}
)";

} // namespace

TEST_CASE("EH regression S4-1 - values live across an invoke survive the call in tier 2") {
    auto mod = parse_or_fail(kEh);
    check_both(*mod, "d_add", 0, 21);
    check_both(*mod, "f_add", 0, 28);   // 21 + 7, %9 live across the invoke
    check_both(*mod, "f_add", -9, -4);  // work(-2) throws -2; pad adds %9 = -2
    check_both(*mod, "loop_eh", 5, 100018);
}

TEST_CASE("Coro regression S4-1 - values live across coro_create, resume and destroy survive in tier 2") {
    auto mod = parse_or_fail(kCoro3);
    check_both(*mod, "only_res", 3, 42);
    check_both(*mod, "only_live", 3, 110);
}

TEST_CASE("Coro regression S4-3 S4-4 - coro_create passes its arguments and lowered coroutines agree across tiers") {
    auto mod = parse_or_fail(kCoro1);
    check_both(*mod, "main", 5, 1011036);   // yields 1, 11, then 11 + 20 + 5
    check_both(*mod, "main", -5, 1011026);
}

TEST_CASE("Coro regression S4-4 - an unlowered coroutine body is an error and not a wrong answer") {
    auto mod = parse_or_fail(kCoro1, /*lower_coroutines=*/false);

    bool interp_threw = false;
    try {
        (void)interp_run(*mod, "main", 5);
    } catch (const InterpreterException&) {
        interp_threw = true;
    }
    CHECK(interp_threw);

    bool jit_threw = false;
    try {
        codegen::JitExecutionEngine jit;
        (void)jit.compile_and_load(*mod);
    } catch (const codegen::UnsupportedOperation&) {
        jit_threw = true;
    }
    CHECK(jit_threw);
}
