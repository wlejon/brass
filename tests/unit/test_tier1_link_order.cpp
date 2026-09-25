// Tier-1 tier-up links every module function its code reaches through a
// lazy stub before the code is installed. These cover how it gets there:
// no code is installed while a callee's native entry is unpublished on
// another thread, a long chain of cold callees takes no native stack, and
// a func_addr target is a callee like any other.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

TieringConfig tier1_at(uint64_t threshold) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = threshold;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

// @lo_pa and @lo_pb call each other, and both call @lo_leaf.
const char* kPairSrc = R"(module @pair
func @lo_pa(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %d = sle.i64 %n, %z
  br_if %d, base, rec
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @lo_pb(%m)
  %t = iconst.i64 3
  %s = add.i64 %r, %t
  ret %s
base:
  %q = call.i64 @lo_leaf(%n)
  ret %q
}
func @lo_pb(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %d = sle.i64 %n, %z
  br_if %d, base, rec
rec:
  %c1 = iconst.i64 1
  %m = sub.i64 %n, %c1
  %r = call.i64 @lo_pa(%m)
  %t = iconst.i64 5
  %s = mul.i64 %r, %t
  ret %s
base:
  %q = call.i64 @lo_leaf(%n)
  ret %q
}
func @lo_leaf(%n: i64) -> i64 {
b0:
  %one = iconst.i64 11
  %r = add.i64 %n, %one
  ret %r
}
)";

int64_t pair_expected(bool a, int64_t n) {
    if (n <= 0) return n + 11;
    return a ? pair_expected(false, n - 1) + 3 : pair_expected(true, n - 1) * 5;
}

} // namespace

TEST_CASE("Tier-1 link order - no code is installed before a callee's entry is published") {
    // Another thread has compiled and registered @lo_leaf (it is found by
    // find_baseline_compiled) but not yet published its native entry, held
    // there by the install hook. @lo_pa's tier-up used to count @lo_leaf as
    // compiled and install code whose stub to it could not resolve: the
    // first call trapped with "call to unresolved symbol 'lo_leaf'".
    auto mod = parse_or_fail(kPairSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at(1000000));
    std::mutex m;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
    prog.pipeline().set_tier1_install_hook([&](std::string_view name) {
        if (name != "lo_leaf") return;
        std::unique_lock<std::mutex> lock(m);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    });
    std::thread other([&] {
        CHECK(prog.pipeline().compile_and_install_tier1("lo_leaf", mod->get_function("lo_leaf")));
    });
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return entered; });
    }
    FunctionHandle* leaf = prog.find("lo_leaf");
    REQUIRE(leaf != nullptr);
    CHECK(prog.pipeline().find_baseline_compiled("lo_leaf") != nullptr);
    CHECK(leaf->native_entry() == nullptr);

    CHECK(!prog.pipeline().compile_and_install_tier1("lo_pa", mod->get_function("lo_pa")));
    FunctionHandle* pa = prog.find("lo_pa");
    FunctionHandle* pb = prog.find("lo_pb");
    REQUIRE(pa != nullptr);
    REQUIRE(pb != nullptr);
    CHECK(pa->native_entry() == nullptr);
    CHECK(pb->native_entry() == nullptr);
    CHECK(!prog.pipeline().is_baseline_rejected("lo_pa"));
    CHECK(!prog.pipeline().is_baseline_rejected("lo_pb"));

    {
        std::lock_guard<std::mutex> lock(m);
        released = true;
    }
    cv.notify_all();
    other.join();
    CHECK(leaf->native_entry() != nullptr);

    // Retried with @lo_leaf published: the cycle is installed and runs.
    CHECK(prog.pipeline().compile_and_install_tier1("lo_pa", mod->get_function("lo_pa")));
    CHECK(pa->native_entry() != nullptr);
    CHECK(pb->native_entry() != nullptr);
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    for (int64_t n = 0; n < 10; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("lo_pa"), {RuntimeValue::from_i64(n)}).as_i64(),
                 pair_expected(true, n));
        CHECK_EQ(interp.run(*mod->get_function("lo_pb"), {RuntimeValue::from_i64(n)}).as_i64(),
                 pair_expected(false, n));
    }
}

TEST_CASE("Tier-1 link order - two threads tier up a cycle sharing a callee") {
    // The race above, unforced: one thread runs @lo_pa, one @lo_pb.
    auto mod = parse_or_fail(kPairSrc);
    for (int p = 0; p < 40; ++p) {
        FunctionDispatchTable prog;
        prog.pipeline().initialize(tier1_at(2 + static_cast<uint64_t>(p % 3)));
        std::atomic<int> go{0};
        std::atomic<int> bad{0};
        auto worker = [&](const char* fn, bool a) {
            Interpreter interp;
            interp.set_dispatch_table(&prog);
            interp.set_module(mod.get());
            go.fetch_add(1);
            while (go.load() < 2) {
            }
            for (int it = 0; it < 100; ++it) {
                const int64_t n = it % 10;
                if (interp.run(*mod->get_function(fn), {RuntimeValue::from_i64(n)}).as_i64() != pair_expected(a, n)) {
                    bad.fetch_add(1);
                }
            }
        };
        std::thread ta(worker, "lo_pa", true);
        std::thread tb(worker, "lo_pb", false);
        ta.join();
        tb.join();
        CHECK_EQ(bad.load(), 0);
    }
}

TEST_CASE("Tier-1 link order - a 5000-function cold chain tiers up") {
    // @lc_i calls @lc_{i+1} only for n > 1e9, so only @lc_0 ever runs. Its
    // tier-up compiles the whole chain; that used to recurse once per
    // function and overflowed the native stack near 950.
    constexpr int kN = 5000;
    std::string src = "module @chain\n";
    for (int i = 0; i < kN; ++i) {
        const std::string id = std::to_string(i);
        src += "func @lc_" + id + "(%n: i64) -> i64 {\nentry:\n  %lim = iconst.i64 1000000000\n"
               "  %c = sgt.i64 %n, %lim\n  br_if %c, hot, cold\n";
        if (i + 1 < kN) src += "hot:\n  %r = call.i64 @lc_" + std::to_string(i + 1) + "(%n)\n  ret %r\n";
        else src += "hot:\n  ret %n\n";
        src += "cold:\n  %ii = iconst.i64 " + id + "\n  %s = add.i64 %n, %ii\n  ret %s\n}\n";
    }
    auto mod = parse_or_fail(src);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at(2));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    for (int it = 0; it < 4; ++it) {
        CHECK_EQ(interp.run(*mod->get_function("lc_0"), {RuntimeValue::from_i64(5)}).as_i64(), 5);
    }
    // The far end of the chain runs, through every stub, in Tier 1.
    CHECK_EQ(interp.run(*mod->get_function("lc_0"), {RuntimeValue::from_i64(2000000000)}).as_i64(), 2000000000);
    for (const char* name : {"lc_0", "lc_1", "lc_2500", "lc_4999"}) {
        FunctionHandle* h = prog.find(name);
        REQUIRE(h != nullptr);
        CHECK(h->native_entry() != nullptr);
    }
    CHECK_EQ(prog.pipeline().stats().tier1_compilations.load(), static_cast<uint64_t>(kN));
}

TEST_CASE("Tier-1 link order - func_addr of a function that never ran") {
    // @fa_main takes @fa_t's address and @fa_call calls through it; @fa_t
    // never runs before both are in Tier 1. It was not counted as a
    // callee, so it was never compiled, and the call through the pointer
    // trapped with "call to unresolved symbol 'fa_t'".
    auto mod = parse_or_fail(R"(module @fptr
func @fa_t(%n: i64) -> i64 {
b0:
  %c = iconst.i64 100
  %r = add.i64 %n, %c
  ret %r
}
func @fa_call(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
func @fa_main(%n: i64) -> i64 {
b0:
  %p = func_addr @fa_t
  %r = call.i64 @fa_call(%p, %n)
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at(1000000));
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    // Both caller and callee of the pointer in Tier 1 before it is taken.
    REQUIRE(prog.pipeline().compile_and_install_tier1("fa_main", mod->get_function("fa_main")));
    FunctionHandle* call = prog.find("fa_call");
    FunctionHandle* target = prog.find("fa_t");
    REQUIRE(call != nullptr);
    REQUIRE(target != nullptr);
    CHECK(call->native_entry() != nullptr);
    // Not compiled until called through: code may take addresses it never
    // calls. Its stub compiles it on the first call.
    CHECK(target->native_entry() == nullptr);
    for (int64_t n = 0; n < 6; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("fa_main"), {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
    }
    CHECK(target->native_entry() != nullptr);
    CHECK_EQ(target->tier(), TierLevel::Tier1_Baseline);
}

TEST_CASE("Tier-1 link order - func_addr of a function the baseline tier rejects") {
    // @fr_t uses a vector type, which the baseline tier does not compile, and takes an
    // f32, which a native-to-Tier-0 bridge does not pass: code taking its
    // address could not call it, so it stays in Tier 0 too. (One with a
    // bridgeable signature is called through the bridge instead:
    // test_function_pointer_tiers.cpp.)
    auto mod = parse_or_fail(R"(module @fptr_rej
func @fr_t(%n: i64, %x: f32) -> i64 {
b0:
  %v = vzero.f64x4
  throw %n
}
func @fr_main(%n: i64) -> ptr {
b0:
  %p = func_addr @fr_t
  ret %p
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tier1_at(1000000));
    CHECK(!prog.pipeline().compile_and_install_tier1("fr_main", mod->get_function("fr_main")));
    CHECK(prog.pipeline().is_baseline_rejected("fr_main"));
    FunctionHandle* h = prog.find("fr_main");
    REQUIRE(h != nullptr);
    CHECK(h->native_entry() == nullptr);
}

#endif
