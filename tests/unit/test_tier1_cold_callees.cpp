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
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

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
    // @tc_rej uses an opcode the baseline tier does not compile.
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

TEST_CASE("Tier-1 cold callees - a call cycle whose root is rejected installs none of it") {
    // @tc_cf calls @tc_cg and @tc_crej (rejected); @tc_cg calls @tc_cf.
    // Tiering @tc_cf compiles @tc_cg first, whose code calls @tc_cf through
    // a stub. @tc_cf is then rejected, so @tc_cg must not be installed: it
    // used to be, and its first call trapped on the stub.
    auto mod = parse_or_fail(R"(module @cyrej
func @tc_crej(%n: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @tc_cf(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %done = sle.i64 %n, %z
  br_if %done, base, rec
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @tc_cg(%m)
  ret %r
base:
  %r0 = call.i64 @tc_crej(%n)
  ret %r0
}
func @tc_cg(%n: i64) -> i64 {
entry:
  %r = call.i64 @tc_cf(%n)
  %t = iconst.i64 10
  %s = add.i64 %r, %t
  ret %s
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("tc_cf");
    Function* g = mod->get_function("tc_cg");
    for (int64_t n : {3, 3, 3, 3, 5, 0, 7}) {
        CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(n)}).as_i64(), 10 * n + 1);
    }
    // @tc_cg tiers up on its own count too, and is rejected as a caller of
    // a rejected function.
    for (int64_t n : {2, 4, 6}) {
        CHECK_EQ(interp.run(*g, {RuntimeValue::from_i64(n)}).as_i64(), 10 * n + 11);
    }
    CHECK(prog.pipeline().is_baseline_rejected("tc_cf"));
    CHECK(prog.pipeline().is_baseline_rejected("tc_cg"));
    for (const char* name : {"tc_cf", "tc_cg"}) {
        FunctionHandle* h = prog.find(name);
        REQUIRE(h != nullptr);
        CHECK(h->native_entry() == nullptr);
        CHECK_EQ(h->tier(), TierLevel::Tier0_Interpreter);
    }
}

TEST_CASE("Tier-1 cold callees - a cycle nested in a longer chain is installed whole") {
    // @tc_na -> @tc_nb -> @tc_nc -> @tc_nb (a cycle below the root) and
    // @tc_nc -> @tc_na (back to the root). All compile: all are installed.
    auto mod = parse_or_fail(R"(module @nested
func @tc_na(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %done = sle.i64 %n, %z
  br_if %done, base, rec
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @tc_nb(%m)
  ret %r
base:
  ret %z
}
func @tc_nb(%n: i64) -> i64 {
entry:
  %r = call.i64 @tc_nc(%n)
  %one = iconst.i64 1
  %s = add.i64 %r, %one
  ret %s
}
func @tc_nc(%n: i64) -> i64 {
entry:
  %lowbit = iconst.i64 1
  %odd = and.i64 %n, %lowbit
  %z = iconst.i64 0
  %even = eq.i64 %odd, %z
  br_if %even, viaa, viab
viaa:
  %ra = call.i64 @tc_na(%n)
  ret %ra
viab:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %rb = call.i64 @tc_nb(%m)
  ret %rb
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* a = mod->get_function("tc_na");
    // Reference results from a program that never tiers up.
    std::vector<int64_t> expected;
    {
        FunctionDispatchTable ref;
        TieringConfig cfg = tier1_at_two(false);
        cfg.invocation_tier1_threshold = 1000000;
        ref.pipeline().initialize(cfg);
        Interpreter ri;
        ri.set_dispatch_table(&ref);
        ri.set_module(mod.get());
        for (int64_t n = 0; n < 8; ++n) expected.push_back(ri.run(*a, {RuntimeValue::from_i64(n)}).as_i64());
    }
    for (int round = 0; round < 3; ++round) {
        for (int64_t n = 0; n < 8; ++n) {
            CHECK_EQ(interp.run(*a, {RuntimeValue::from_i64(n)}).as_i64(), expected[static_cast<size_t>(n)]);
        }
    }
    for (const char* name : {"tc_na", "tc_nb", "tc_nc"}) {
        FunctionHandle* h = prog.find(name);
        REQUIRE(h != nullptr);
        CHECK(h->native_entry() != nullptr);
    }
}

TEST_CASE("Tier-1 cold callees - an attempt pending on another thread's compile is retried") {
    // @tc_pw's tier-up needs @tc_pleaf, which another thread is compiling
    // (held there by the compile hook): the attempt at the threshold cannot
    // finish. It used to be the only one, leaving @tc_pw in Tier 0 for good.
    auto mod = parse_or_fail(R"(module @pending
func @tc_pleaf(%n: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @tc_pw(%x: i64) -> i64 {
entry:
  %s = call.i64 @tc_pleaf(%x)
  %s2 = add.i64 %s, %x
  ret %s2
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at_two(false));
    std::mutex m;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
    prog.pipeline().set_tier1_compile_hook([&](std::string_view name) {
        if (name != "tc_pleaf") return;
        std::unique_lock<std::mutex> lock(m);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    });
    std::thread other([&] {
        CHECK(prog.pipeline().compile_and_install_tier1("tc_pleaf", mod->get_function("tc_pleaf")));
    });
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return entered; });
    }
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* w = mod->get_function("tc_pw");
    auto run = [&](int64_t x) { CHECK_EQ(interp.run(*w, {RuntimeValue::from_i64(x)}).as_i64(), 2 * x + 1); };
    run(1);
    run(2); // the threshold: @tc_pleaf is in progress on the other thread
    FunctionHandle* h = prog.find("tc_pw");
    REQUIRE(h != nullptr);
    CHECK(h->native_entry() == nullptr);
    CHECK(!prog.pipeline().is_baseline_rejected("tc_pw"));
    {
        std::lock_guard<std::mutex> lock(m);
        released = true;
    }
    cv.notify_all();
    other.join();
    for (int64_t x = 3; x < 10; ++x) run(x);
    CHECK(h->native_entry() != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier1_Baseline);
    CHECK(prog.tiering().get_feedback("tc_pw").tier1_retries() >= 1);
}

#endif
