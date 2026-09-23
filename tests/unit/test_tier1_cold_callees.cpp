// Tier-1 code calling a module function that has never run: the callee has
// no native entry when its caller tiers up, and the lazy stub the caller
// calls it through must still reach it (it used to trap with "call to
// unresolved symbol"). The tier-up compiles such callees first; a callee
// the baseline tier rejects keeps its caller in Tier 0.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64)

namespace {

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n" << src << "\n";
    REQUIRE(ok);
    return mod;
}

TieringConfig tier1_at_two(bool fast) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    return cfg;
}

// @tc_f(x) calls @tc_cold only when x > 100, so @tc_f tiers up first.
const char* kColdSrc = R"(module @cold
func @tc_cold(%n: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @tc_f(%x: i64) -> i64 {
entry:
  %lim = iconst.i64 100
  %c = sgt.i64 %x, %lim
  br_if %c, hot, done
hot:
  %s = call.i64 @tc_cold(%x)
  ret %s
done:
  ret %x
}
)";

const int64_t kColdArgs[] = {1, 2, 3, 4, 5, 200};

int64_t cold_expected(int64_t x) { return x > 100 ? x + 1 : x; }

} // namespace

TEST_CASE("Tier-1 cold callees - Interpreter reaches a callee that never ran") {
    auto mod = parse_or_fail(kColdSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("tc_f");
    for (int64_t x : kColdArgs) {
        CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), cold_expected(x));
    }
    FunctionHandle* h = prog.find("tc_f");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier1_Baseline);
    // Compiled with its caller, before it ever ran.
    FunctionHandle* cold = prog.find("tc_cold");
    REQUIRE(cold != nullptr);
    CHECK(cold->native_entry() != nullptr);
}

TEST_CASE("Tier-1 cold callees - MultiTierPipeline::execute reaches a callee that never ran") {
    auto mod = parse_or_fail(kColdSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    for (int64_t x : kColdArgs) {
        const RuntimeValue r = prog.pipeline().execute(*mod, "tc_f", {RuntimeValue::from_i64(x)});
        CHECK_EQ(r.as_i64(), cold_expected(x));
    }
    FunctionHandle* h = prog.find("tc_f");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier1_Baseline);
}

TEST_CASE("Tier-1 cold callees - FastInterpreter and Interpreter share a program") {
    // @tc_sum has run once (under the FastInterpreter) when the Interpreter
    // tiers @tc_mf up.
    const char* src = R"(module @mixed
func @tc_sum(%n: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @tc_mf(%x: i64) -> i64 {
entry:
  %s = call.i64 @tc_sum(%x)
  ret %s
}
)";
    for (const bool owned : {false, true}) {
        auto mod = parse_or_fail(src);
        std::optional<FunctionDispatchTable> own;
        FunctionDispatchTable* prog = &FunctionDispatchTable::instance();
        if (owned) {
            own.emplace();
            prog = &*own;
        }
        prog->pipeline().initialize(tier1_at_two(true));
        {
            FastInterpreter fast;
            fast.set_dispatch_table(prog);
            Interpreter interp;
            interp.set_dispatch_table(prog);
            interp.set_module(mod.get());
            Function* f = mod->get_function("tc_mf");
            for (int k = 0; k < 6; ++k) {
                const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(41)};
                CHECK_EQ(fast.run(*f, args).as_i64(), int64_t{42});
                CHECK_EQ(interp.run(*f, args).as_i64(), int64_t{42});
            }
        }
        if (!owned) {
            prog->pipeline().shutdown();
            prog->tiering().set_active_module(nullptr);
            prog->forget_module(*mod);
        }
    }
}

TEST_CASE("Tier-1 cold callees - a cold call cycle tiers up whole") {
    // @tc_even and @tc_odd call each other; neither has run when @tc_even
    // tiers up.
    auto mod = parse_or_fail(R"(module @cycle
func @tc_even(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %done = eq.i64 %n, %z
  br_if %done, yes, rec
yes:
  %one = iconst.i64 1
  ret %one
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @tc_odd(%m)
  ret %r
}
func @tc_odd(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %done = eq.i64 %n, %z
  br_if %done, no, rec
no:
  ret %z
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @tc_even(%m)
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    REQUIRE(prog.pipeline().compile_and_install_tier1("tc_even", mod->get_function("tc_even")));
    FunctionHandle* even = prog.find("tc_even");
    FunctionHandle* odd = prog.find("tc_odd");
    REQUIRE(even != nullptr);
    REQUIRE(odd != nullptr);
    REQUIRE(even->native_entry() != nullptr);
    REQUIRE(odd->native_entry() != nullptr);
    using I64Fn = int64_t (*)(int64_t);
    auto even_fn = reinterpret_cast<I64Fn>(even->native_entry());
    CHECK_EQ(even_fn(10), int64_t{1});
    CHECK_EQ(even_fn(7), int64_t{0});
}

TEST_CASE("Tier-1 cold callees - a callee the baseline tier rejects keeps its caller in Tier 0") {
    // @tc_rej uses an opcode the x64 baseline tier does not compile.
    auto mod = parse_or_fail(R"(module @rej
func @tc_rej(%n: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @tc_rf(%x: i64) -> i64 {
entry:
  %lim = iconst.i64 100
  %c = sgt.i64 %x, %lim
  br_if %c, hot, done
hot:
  %s = call.i64 @tc_rej(%x)
  ret %s
done:
  ret %x
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("tc_rf");
    for (int64_t x : kColdArgs) {
        CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), cold_expected(x));
    }
    CHECK(prog.pipeline().is_baseline_rejected("tc_rej"));
    CHECK(prog.pipeline().is_baseline_rejected("tc_rf"));
    FunctionHandle* h = prog.find("tc_rf");
    REQUIRE(h != nullptr);
    CHECK(h->native_entry() == nullptr);
}

#endif
