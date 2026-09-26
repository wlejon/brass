// Coroutines across tiers: every placement of a body and of the code that
// creates and resumes it (Interpreter, FastInterpreter, Tier 1, Tier 2)
// agrees with the all-Interpreter run, including the resume mode and
// exceptions thrown and caught across suspend points; a body tiers up from
// resumes alone; the pipeline lowers coroutines itself.
#include "coro_tier_harness.hpp"
#include <brass/runtime/coroutine.hpp>
#include <string>

using namespace brass;
using namespace brass::coro_test;
using namespace brass::runtime;

namespace {

// @gen yields p, then reads the resume mode: Throw throws the value, Return
// returns it, Next yields value + p and returns the next value + that.
// @drv resumes it once plainly, then with (100, mode) through @resume_m, an
// invoke whose pad catches what the body throws.
constexpr std::string_view kModes = R"(module @modes
func @gen(%f: gcref, %p: i64) -> i64 {
b0:
  %v1 = coro_suspend.i64 %p, 1
  %m1 = load.i32 %f, 40
  %one = iconst.i32 1
  %is_throw = eq.i32 %m1, %one
  br_if %is_throw, do_throw, chk_ret
do_throw:
  throw %v1
chk_ret:
  %two = iconst.i32 2
  %is_ret = eq.i32 %m1, %two
  br_if %is_ret, do_ret, cont
do_ret:
  ret %v1
cont:
  %s = add.i64 %v1, %p
  %v2 = coro_suspend.i64 %s, 2
  %r = add.i64 %v2, %s
  ret %r
}

func @resume_m(%c: gcref, %v: i64, %m: i32) -> i64 {
b0:
  %r = coro_resume.i64 %c, %v, %m
  ret %r
}

func @drv(%p: i64, %mode: i32) -> i64 {
b0:
  %c = coro_create @gen(%p)
  %z = iconst.i64 0
  %y0 = coro_resume.i64 %c, %z
  %v = iconst.i64 100
  %y1 = invoke.i64 @resume_m(%c, %v, %mode), ok, pad
ok:
  %k = iconst.i64 1000
  %t = mul.i64 %y0, %k
  %u = add.i64 %t, %y1
  ret %u
pad:
  %e = landing_pad
  %big = iconst.i64 1000000
  %w = add.i64 %e, %big
  ret %w
}
)";

// Try/catch regions split by suspends: @ehgen suspends inside the protected
// region (between two invokes of @step, which throws a negative argument)
// and again inside its handler, which uses a value live across both.
constexpr std::string_view kEh = R"(module @eh
func @step(%x: i64) -> i64 {
b0:
  %z = iconst.i64 0
  %neg = slt.i64 %x, %z
  br_if %neg, bad, good
bad:
  throw %x
good:
  %three = iconst.i64 3
  %r = mul.i64 %x, %three
  ret %r
}

func @ehgen(%p: i64) -> i64 {
b0:
  %a = invoke.i64 @step(%p), c1, pad
c1:
  %y = coro_suspend.i64 %a, 1
  %b = invoke.i64 @step(%y), c2, pad
c2:
  %y2 = coro_suspend.i64 %b, 2
  ret %y2
pad:
  %e = landing_pad
  %k = iconst.i64 10000
  %h = add.i64 %e, %k
  %h2 = add.i64 %h, %p
  %y3 = coro_suspend.i64 %h2, 3
  %fin = add.i64 %y3, %p
  ret %fin
}

func @ehdrv(%p: i64, %second: i64) -> i64 {
b0:
  %c = coro_create @ehgen(%p)
  %z = iconst.i64 0
  %y0 = coro_resume.i64 %c, %z
  %y1 = coro_resume.i64 %c, %second
  %seven = iconst.i64 7
  %y2 = coro_resume.i64 %c, %seven
  coro_destroy %c
  %hk = iconst.i64 100000
  %t0 = mul.i64 %y0, %hk
  %t1 = add.i64 %t0, %y1
  %h = iconst.i64 100
  %t2 = mul.i64 %t1, %h
  %t3 = add.i64 %t2, %y2
  ret %t3
}
)";

// A generator in a loop, resumed n times from a loop.
constexpr std::string_view kCounter = R"(module @counter
func @counter(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %in = coro_suspend.i64 %acc, 1
  %sq = mul.i64 %in, %in
  %acc2 = add.i64 %acc, %sq
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %more = slt.i64 %i2, %n
  br_if %more, loop(%i2, %acc2), done(%acc2)
done(%r: i64):
  ret %r
}

func @drive(%n: i64) -> i64 {
b0:
  %c = coro_create @counter(%n)
  %z = iconst.i64 0
  %first = coro_resume.i64 %c, %z
  br loop(%z, %first)
loop(%i: i64, %last: i64):
  %v = coro_resume.i64 %c, %i
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %more = slt.i64 %i2, %n
  br_if %more, loop(%i2, %v), done(%v)
done(%r: i64):
  coro_destroy %c
  ret %r
}
)";

int64_t run_placed(std::string_view src, std::string_view body, Tier body_tier, std::string_view driver,
                   Tier driver_tier, const std::vector<RuntimeValue>& args) {
    auto mod = parse_program(src);
    test::BoundHeap heap;
    Program prog(*mod);
    prog.place(body, body_tier);
    prog.place(driver, driver_tier);
    return prog.call(driver, driver_tier, args).as_i64();
}

void check_matrix(std::string_view src, std::string_view body, std::string_view driver,
                  const std::vector<RuntimeValue>& args, int64_t want) {
    for (Tier bt : kTiers) {
        for (Tier dt : kTiers) {
            const int64_t got = run_placed(src, body, bt, driver, dt, args);
            if (got != want) {
                std::printf("body %s, driver %s: %lld, want %lld\n", tier_name(bt), tier_name(dt),
                            static_cast<long long>(got), static_cast<long long>(want));
            }
            CHECK_EQ(got, want);
        }
    }
}

} // namespace

TEST_CASE("CoroTiers - resume mode Next agrees in every placement of body and driver") {
    check_matrix(kModes, "gen", "drv", {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)}, 5 * 1000 + 105);
}

TEST_CASE("CoroTiers - resume mode Throw throws at the suspend, caught by the resumer, in every placement") {
    check_matrix(kModes, "gen", "drv", {RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)}, 1000000 + 100);
}

TEST_CASE("CoroTiers - resume mode Return returns the value in every placement") {
    check_matrix(kModes, "gen", "drv", {RuntimeValue::from_i64(5), RuntimeValue::from_i32(2)}, 5 * 1000 + 100);
}

TEST_CASE("CoroTiers - exceptions across suspend points (try region and handler split) in every placement") {
    // Resumed with 4: no throw. y0 = 15, y1 = 12, y2 = 7.
    check_matrix(kEh, "ehgen", "ehdrv", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(4)},
                 (15LL * 100000 + 12) * 100 + 7);
    // Resumed with -4: @step throws -4 after the suspend; the handler yields
    // -4 + 10000 + 5 and returns 7 + 5.
    check_matrix(kEh, "ehgen", "ehdrv", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(-4)},
                 (15LL * 100000 + 10001) * 100 + 12);
}

TEST_CASE("CoroTiers - a generator loop agrees in every placement") {
    const int64_t want = run_placed(kCounter, "counter", Tier::Interp, "drive", Tier::Interp,
                                    {RuntimeValue::from_i64(40)});
    check_matrix(kCounter, "counter", "drive", {RuntimeValue::from_i64(40)}, want);
}

TEST_CASE("CoroTiers - a body resumed often tiers up to Tier 1 then Tier 2, frames and all") {
    auto mod = parse_program(kCounter);
    test::BoundHeap heap;
    TieringConfig cfg = manual_tiering();
    cfg.invocation_tier1_threshold = 3;
    cfg.invocation_tier2_threshold = 10;
    Program prog(*mod, cfg);
    const int64_t want = run_placed(kCounter, "counter", Tier::Interp, "drive", Tier::Interp,
                                    {RuntimeValue::from_i64(60)});
    // The frame is created and resumed by the Interpreter; only the resumes
    // count toward the body's tier-up.
    CHECK_EQ(prog.call("drive", Tier::Interp, {RuntimeValue::from_i64(60)}).as_i64(), want);
    FunctionHandle* body = prog.table.find("counter");
    REQUIRE(body != nullptr);
    CHECK(body->native_entry() != nullptr);
    CHECK_EQ(body->tier(), TierLevel::Tier2_Optimized);
    // A new frame starts in the tier the body has reached.
    CHECK_EQ(prog.call("drive", Tier::Fast, {RuntimeValue::from_i64(60)}).as_i64(), want);
}

TEST_CASE("CoroTiers - Tier 1 compiles a function that creates and resumes a coroutine") {
    auto mod = parse_program(kModes);
    test::BoundHeap heap;
    Program prog(*mod);
    const Function* drv = mod->get_function("drv");
    CHECK(prog.table.pipeline().baseline_compiler().passes_prescan(*drv, Target::host()));
    prog.place("drv", Tier::Base);
    CHECK_EQ(prog.table.find("drv")->tier(), TierLevel::Tier1_Baseline);
}

TEST_CASE("CoroTiers - the pipeline lowers an unlowered module before running it") {
    DiagnosticReporter diag;
    auto mod = parse_module(kCounter, &diag);
    REQUIRE(mod != nullptr);
    CHECK(!is_lowered_coro_body(*mod->get_function("counter")));
    test::BoundHeap heap;
    FunctionDispatchTable table;
    table.pipeline().initialize(manual_tiering());
    const RuntimeValue r = table.pipeline().execute(*mod, "drive", {RuntimeValue::from_i64(40)});
    CHECK(is_lowered_coro_body(*mod->get_function("counter")));
    CHECK_EQ(r.as_i64(), run_placed(kCounter, "counter", Tier::Interp, "drive", Tier::Interp,
                                    {RuntimeValue::from_i64(40)}));
    // Lowering again changes nothing.
    CHECK(!lower_coroutines(*mod));
}
