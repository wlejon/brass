// A program has one function-pointer representation across tiers: the
// function's lazy stub, which func_addr yields in Tier 0 and Tier 1 alike,
// which native code calls directly (compiling the function, or bridging
// into Tier 0 when the baseline tier rejects it), and which Tier 0 maps
// back to the function. Also: reinstalling tier-2 code over live tier-2
// code keeps the replaced code alive for callers bound to it.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
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
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

TieringConfig no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

const char* kFptrSrc = R"(module @fpt
func @pt_t(%n: i64) -> i64 {
b0:
  %c = iconst.i64 100
  %r = add.i64 %n, %c
  ret %r
}
func @pt_call(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
func @pt_main(%n: i64) -> i64 {
b0:
  %p = func_addr @pt_t
  %r = call.i64 @pt_call(%p, %n)
  ret %r
}
func @pt_get() -> ptr {
b0:
  %p = func_addr @pt_t
  ret %p
}
func @pt_main2(%n: i64) -> i64 {
b0:
  %p = call.ptr @pt_get()
  %r = call_indirect.i64 %p(%n)
  ret %r
}
)";

uint32_t call_indirect_site(const Function& fn) {
    for (const auto* bb : fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::call_indirect) return inst->site_id();
        }
    }
    REQUIRE(false);
    return 0;
}

} // namespace

TEST_CASE("Function pointers - a Tier-0 pointer called from Tier-1 code") {
    auto mod = parse_or_fail(kFptrSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    // @pt_main stays interpreted; @pt_call, which calls through the pointer
    // natively, and @pt_t are in Tier 1.
    REQUIRE(prog.pipeline().compile_and_install_tier1("pt_call", mod->get_function("pt_call")));
    REQUIRE(prog.pipeline().compile_and_install_tier1("pt_t", mod->get_function("pt_t")));
    for (int64_t n = 0; n < 6; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("pt_main"), {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
    }
    FunctionHandle* main = prog.find("pt_main");
    REQUIRE(main != nullptr);
    CHECK(main->native_entry() == nullptr);
}

TEST_CASE("Function pointers - a Tier-0 pointer to a cold function called from Tier-1 code") {
    auto mod = parse_or_fail(kFptrSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("pt_call", mod->get_function("pt_call")));
    // @pt_t never ran: the stub compiles it on the first native call.
    for (int64_t n = 0; n < 4; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("pt_main"), {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
    }
    FunctionHandle* t = prog.find("pt_t");
    REQUIRE(t != nullptr);
    CHECK(t->native_entry() != nullptr);
}

TEST_CASE("Function pointers - a Tier-1 pointer called from Tier 0, feedback keyed on one address") {
    auto mod = parse_or_fail(kFptrSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    // In Tier 0, then with @pt_get in Tier 1 handing the pointer to the
    // interpreted @pt_main2.
    CHECK_EQ(interp.run(*mod->get_function("pt_main2"), {RuntimeValue::from_i64(1)}).as_i64(), 101);
    REQUIRE(prog.pipeline().compile_and_install_tier1("pt_get", mod->get_function("pt_get")));
    FunctionHandle* get = prog.find("pt_get");
    REQUIRE(get != nullptr);
    REQUIRE(get->native_entry() != nullptr);
    for (int64_t n = 2; n < 8; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("pt_main2"), {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
    }
    // Both tiers' pointers are the same address, so the site saw one target.
    const uintptr_t canonical =
        reinterpret_cast<uintptr_t>(prog.pipeline().function_address("pt_t", mod->get_function("pt_t")));
    const uint32_t site = call_indirect_site(*mod->get_function("pt_main2"));
    const FeedbackSlot* slot = prog.tiering().type_feedback().get_or_create("pt_main2").find_slot(site);
    REQUIRE(slot != nullptr);
    REQUIRE(slot->targets.size() == 1u);
    CHECK_EQ(slot->targets[0].target_addr, canonical);
    CHECK_EQ(slot->targets[0].count, 7u);
}

TEST_CASE("Function pointers - Tier-1 code calls a function the baseline tier rejects") {
    // @rj_t and @rj_f use a vector type the baseline tier rejects: code
    // calling them through a pointer goes through a native-to-Tier-0 bridge.
    auto mod = parse_or_fail(R"(module @fpt_rej
func @rj_t(%n: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %c = iconst.i64 7
  %r = add.i64 %n, %c
  ret %r
}
func @rj_f(%x: f64, %k: i32) -> f64 {
b0:
  %v = vzero.f64x4
  %y = add.f64 %x, %x
  ret %y
}
func @rj_call(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
func @rj_main(%n: i64) -> i64 {
b0:
  %p = func_addr @rj_t
  %r = call.i64 @rj_call(%p, %n)
  ret %r
}
func @rj_fmain(%x: f64) -> f64 {
b0:
  %p = func_addr @rj_f
  %k = iconst.i32 3
  %r = call_indirect.f64 %p(%x, %k)
  ret %r
}
func @rj_ind(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("rj_main", mod->get_function("rj_main")));
    REQUIRE(prog.pipeline().compile_and_install_tier1("rj_fmain", mod->get_function("rj_fmain")));
    for (int64_t n = 0; n < 5; ++n) {
        CHECK_EQ(interp.run(*mod->get_function("rj_main"), {RuntimeValue::from_i64(n)}).as_i64(), n + 7);
        CHECK_EQ(interp.run(*mod->get_function("rj_fmain"), {RuntimeValue::from_f64(n + 0.25)}).as_f64(),
                 2.0 * (n + 0.25));
    }
    CHECK(prog.pipeline().is_baseline_rejected("rj_t"));
    CHECK(prog.pipeline().is_baseline_rejected("rj_f"));
    FunctionHandle* t = prog.find("rj_t");
    REQUIRE(t != nullptr);
    CHECK(t->native_entry() == nullptr);
    // The interpreted caller of a Tier-1 pointer to it runs it in Tier 0.
    auto* p = prog.pipeline().function_address("rj_t", mod->get_function("rj_t"));
    CHECK_EQ(interp.run(*mod->get_function("rj_ind"),
                        {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(p)), RuntimeValue::from_i64(5)})
                 .as_i64(),
             12);
}

TEST_CASE("Function pointers - a bridged callee's exception reaches the Tier-0 and host handlers") {
    // @ex_t throws (so the baseline tier rejects it); Tier-1 @ex_main calls
    // it through its address, from under a Tier-0 invoke and from the host.
    auto mod = parse_or_fail(R"(module @fpt_throw
func @ex_t(%n: i64) -> i64 {
entry:
  %lim = iconst.i64 3
  %small = slt.i64 %n, %lim
  br_if %small, ok, bad
ok:
  %c = iconst.i64 100
  %r = add.i64 %n, %c
  ret %r
bad:
  throw %n
}
func @ex_call(%p: ptr, %n: i64) -> i64 {
b0:
  %r = call_indirect.i64 %p(%n)
  ret %r
}
func @ex_main(%n: i64) -> i64 {
b0:
  %p = func_addr @ex_t
  %r = call.i64 @ex_call(%p, %n)
  ret %r
}
func @ex_catch(%n: i64) -> i64 {
b0:
  %v = invoke.i64 @ex_main(%n), b1, b2
b1:
  ret %v
b2:
  %e = landing_pad
  %k = iconst.i64 1000
  %r = add.i64 %e, %k
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("ex_main", mod->get_function("ex_main")));
    FunctionHandle* main = prog.find("ex_main");
    REQUIRE(main != nullptr);
    REQUIRE(main->native_entry() != nullptr);
    for (int round = 0; round < 3; ++round) {
        for (int64_t n = 0; n < 6; ++n) {
            // Caught by the Tier-0 invoke above the native frames.
            CHECK_EQ(interp.run(*mod->get_function("ex_catch"), {RuntimeValue::from_i64(n)}).as_i64(),
                     n < 3 ? n + 100 : n + 1000);
            // Caught by the host.
            bool caught = false;
            try {
                CHECK_EQ(interp.run(*mod->get_function("ex_main"), {RuntimeValue::from_i64(n)}).as_i64(), n + 100);
            } catch (const InterpreterThrownException& e) {
                caught = true;
                CHECK_EQ(e.value().as_i64(), n);
            }
            CHECK_EQ(caught, n >= 3);
        }
    }
    CHECK(prog.pipeline().is_baseline_rejected("ex_t"));
}

TEST_CASE("Tier-2 reinstall - replacing live tier-2 code keeps it for callers bound to it") {
    auto mod = parse_or_fail(R"(module @t2re
func @rt_leaf(%x: i64) -> i64 {
b0:
  %c = iconst.i64 3
  %r = mul.i64 %x, %c
  ret %r
}
func @rt_helper(%x: i64) -> i64 {
b0:
  %a = call.i64 @rt_leaf(%x)
  %one = iconst.i64 1
  %r = add.i64 %a, %one
  ret %r
}
func @rt_main(%x: i64) -> i64 {
b0:
  %h = call.i64 @rt_helper(%x)
  %two = iconst.i64 2
  %r = add.i64 %h, %two
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    prog.tiering().set_active_module(mod.get());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    CodeInstaller installer(prog);
    FunctionHandle* helper = prog.get_or_create("rt_helper", mod->get_function("rt_helper"));
    REQUIRE(installer.install_tier2(*helper, *mod, "rt_helper").success);
    // @rt_main's Tier-1 code binds @rt_helper's tier-2 entry directly.
    REQUIRE(prog.pipeline().compile_and_install_tier1("rt_main", mod->get_function("rt_main")));
    for (int round = 0; round < 3; ++round) {
        for (int64_t x = 0; x < 4; ++x) {
            CHECK_EQ(interp.run(*mod->get_function("rt_main"), {RuntimeValue::from_i64(x)}).as_i64(), 3 * x + 3);
        }
        REQUIRE(installer.install_tier2(*helper, *mod, "rt_helper").success);
        CHECK_EQ(helper->tier(), TierLevel::Tier2_Optimized);
        CHECK_EQ(helper->retired_engine_count(), static_cast<size_t>(round + 1));
    }
    for (int64_t x = 0; x < 4; ++x) {
        CHECK_EQ(interp.run(*mod->get_function("rt_main"), {RuntimeValue::from_i64(x)}).as_i64(), 3 * x + 3);
    }
}

#endif
