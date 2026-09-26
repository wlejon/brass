// Tier-ups off the program's thread. With enable_background_tier1 a function
// reaching the tier-1 threshold keeps running in Tier 0 while a CompilePool
// worker baseline-compiles it and every function it reaches, and the next
// call after the install runs the native code. With enable_background_compile
// in a running program, the copy a tier-2 compile works on is taken on the
// worker too: the program's thread reads only the call targets.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tier_timeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <string>

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

const char* kSrc = R"(module @bg
func @bg_leaf(%n: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @bg_f(%x: i64) -> i64 {
entry:
  %lim = iconst.i64 100
  %c = sgt.i64 %x, %lim
  br_if %c, hot, done
hot:
  %s = call.i64 @bg_leaf(%x)
  ret %s
done:
  %t = mul.i64 %x, %x
  ret %t
}
)";

int64_t expected(int64_t x) { return x > 100 ? x + 1 : x * x; }

const char* kFptrSrc = R"(module @bgp
func @bp_t(%n: i64) -> i64 {
b0:
  %c = iconst.i64 100
  %r = add.i64 %n, %c
  ret %r
}
func @bp_call(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
func @bp_main(%n: i64) -> i64 {
b0:
  %p = func_addr @bp_t
  %r = call.i64 @bp_call(%p, %n)
  ret %r
}
)";

bool saw_event(TierEventKind kind, std::string_view name, bool on_worker) {
    for (const TierEvent& e : tier_timeline_snapshot()) {
        if (e.kind == kind && e.name == name && e.on_worker == on_worker) return true;
    }
    return false;
}

struct TimelineOn {
    TimelineOn() { enable_tier_timeline(true); }
    ~TimelineOn() { enable_tier_timeline(false); }
};

} // namespace

TEST_CASE("Background tier-1 - a hot function compiles on a worker and runs Tier 0 meanwhile") {
    TimelineOn timeline;
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_tier1 = true;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);

    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("bg_f");
    for (int64_t x : {1, 2, 3, 4}) CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), expected(x));

    CompilePool::shared().wait_owner(&prog.pipeline());
    FunctionHandle* h = prog.find("bg_f");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier1_Baseline);
    CHECK(h->native_entry() != nullptr);
    FunctionHandle* leaf = prog.find("bg_leaf");
    REQUIRE(leaf != nullptr);
    CHECK(leaf->native_entry() != nullptr);
    CHECK(saw_event(TierEventKind::Tier1Compile, "bg_f", true));
    CHECK_FALSE(saw_event(TierEventKind::Tier1Compile, "bg_f", false));

    for (int64_t x : {5, 200, 7, 300}) CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
}

TEST_CASE("Background tier-1 - native code calling a cold function runs it in Tier 0 until it compiles") {
    TimelineOn timeline;
    auto mod = parse_or_fail(kFptrSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_tier1 = true;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);

    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("bp_call", mod->get_function("bp_call")));
    Function* main = mod->get_function("bp_main");
    for (int64_t n = 0; n < 4; ++n) CHECK_EQ(interp.run(*main, {RuntimeValue::from_i64(n)}).as_i64(), n + 100);

    CompilePool::shared().wait_owner(&prog.pipeline());
    FunctionHandle* t = prog.find("bp_t");
    REQUIRE(t != nullptr);
    CHECK(t->native_entry() != nullptr);
    CHECK(prog.pipeline().baseline_compiler().lazy_symbols()->resolved_target("bp_t") == t->native_entry());
    CHECK(saw_event(TierEventKind::Tier1Compile, "bp_t", true));
    CHECK_FALSE(saw_event(TierEventKind::Tier1Compile, "bp_t", false));
    for (int64_t n = 4; n < 8; ++n) CHECK_EQ(interp.run(*main, {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
}

TEST_CASE("Background tier-1 - off by default, the tier-up is on the program's thread") {
    TimelineOn timeline;
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);

    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("bg_f");
    for (int64_t x : {1, 2, 3}) CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
    FunctionHandle* h = prog.find("bg_f");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier1_Baseline);
    CHECK(saw_event(TierEventKind::Tier1Compile, "bg_f", false));
}

TEST_CASE("Background tier-2 - the copy is taken on the worker") {
    TimelineOn timeline;
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 4;
    cfg.enable_background_compile = true;
    cfg.enable_background_tier1 = true;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);

    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* f = mod->get_function("bg_f");
    FunctionHandle* leaf = nullptr;
    for (int round = 0; round < 200; ++round) {
        for (int64_t x : {1, 200, 3, 400}) CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
        CompilePool::shared().wait_owner(&prog.pipeline());
        prog.pipeline().background_compiler().wait_idle();
        leaf = prog.find("bg_leaf");
        if (leaf && leaf->tier() == TierLevel::Tier2_Optimized) break;
    }
    REQUIRE(leaf != nullptr);
    CHECK_EQ(leaf->tier(), TierLevel::Tier2_Optimized);
    CHECK(saw_event(TierEventKind::Tier2Enqueue, "bg_leaf", true));
    CHECK(saw_event(TierEventKind::Tier2Compile, "bg_leaf", true));
    for (int64_t x : {9, 900}) CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
}

#endif
