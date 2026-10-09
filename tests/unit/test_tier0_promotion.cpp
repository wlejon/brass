// A function the baseline tier rejects has no tier-1 code to tier up from.
// It goes to Tier 2 straight from Tier 0 once it is as hot as tier-1 code
// is when it asks for Tier 2, a failed guard of that code finishes the call
// back in Tier 0, and after an invalidation it tiers up again.
#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <string>
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

// Tier 1 at the second call, Tier 2 at the tenth, all on the calling thread.
TieringConfig promote_at_ten() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 10;
    cfg.enable_background_compile = false;
    cfg.enable_osr = false;
    cfg.deopt_threshold = 2;
    cfg.set_use_fast_interpreter(true);
    return cfg;
}

// The 256-bit vector is dead, but the baseline tier rejects the function for
// it; tier 2 drops it.
const char* kRejectedSrc = R"(module @t0p
func @t0p_hot(%n: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %three = iconst.i64 3
  %m = mul.i64 %n, %three
  %one = iconst.i64 1
  %r = add.i64 %m, %one
  ret %r
}
)";

// func @t0p_guarded(%x: i64, %tag: i32) -> i64
//   %v = vzero.f64x4                        (tier 1 rejects it)
//   %a = mul %x, 3
//   guard (eq %tag, 1), [%a, %x]  (resume id 7)
//   ret %a + 1
// resume 7 (%a, %x): ret %a + %x * 100
Function* build_guarded(Module& mod) {
    Function* fn = mod.create_function("t0p_guarded", Type::i64(), {Type::i64(), Type::i32()});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    Value* x = b.add_param(Type::i64());
    Value* tag = b.add_param(Type::i32());
    b.build_vzero(Type::f64x4());
    Value* a = b.build_mul(x, b.build_iconst_i64(3));
    Value* c = b.build_eq(tag, b.build_iconst_i32(1));
    Instruction* g = b.build_guard(c, "t0p_no_stub", {a, x});
    g->set_resume_id(7);
    b.build_ret(b.build_add(a, b.build_iconst_i64(1)));
    BasicBlock* slow = b.append_block("resume7");
    b.position_at_end(slow);
    Value* ra = b.add_param(Type::i64());
    Value* rx = b.add_param(Type::i64());
    b.build_ret(b.build_add(ra, b.build_mul(rx, b.build_iconst_i64(100))));
    fn->add_resume_point(7, slow);
    return fn;
}

int64_t guarded_expected(int64_t x, int32_t tag) { return tag == 1 ? 3 * x + 1 : 3 * x + 100 * x; }

} // namespace

TEST_CASE("Tier-0 promotion - a function tier 1 rejects reaches tier 2 from tier 0") {
    auto mod = parse_or_fail(kRejectedSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(promote_at_ten());
    FastInterpreter fast;
    fast.set_dispatch_table(&prog);
    fast.set_module(mod.get());
    Function* f = mod->get_function("t0p_hot");
    for (int64_t n = 0; n < 40; ++n) {
        CHECK_EQ(fast.run(*f, {RuntimeValue::from_i64(n)}).as_i64(), 3 * n + 1);
    }
    MultiTierPipeline& p = prog.pipeline();
    CHECK(p.is_baseline_rejected("t0p_hot"));
    CHECK(p.baseline_rejection("t0p_hot").find("256-bit vector") != std::string::npos);
    FunctionHandle* h = prog.find("t0p_hot");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
    CHECK(h->baseline_function() == nullptr);
    CHECK(prog.tiering().get_feedback("t0p_hot").tier1_rejected());
    CHECK_EQ(prog.tiering().get_feedback("t0p_hot").current_tier(), TierLevel::Tier2_Optimized);
    const MultiTierStats s = p.stats();
    CHECK_EQ(s.tier1_rejections.load(), 1u);
    CHECK_EQ(s.tier0_promotions.load(), 1u);
    const std::vector<BaselineRejection> all = p.baseline_rejections();
    REQUIRE_EQ(all.size(), 1u);
    CHECK_EQ(all[0].name, std::string("t0p_hot"));
}

TEST_CASE("Tier-0 promotion - at the call tier-1 code would have asked for tier 2") {
    auto mod = parse_or_fail(kRejectedSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(promote_at_ten());
    FastInterpreter fast;
    fast.set_dispatch_table(&prog);
    fast.set_module(mod.get());
    Function* f = mod->get_function("t0p_hot");
    for (int64_t n = 0; n < 9; ++n) CHECK_EQ(fast.run(*f, {RuntimeValue::from_i64(n)}).as_i64(), 3 * n + 1);
    FunctionHandle* h = prog.find("t0p_hot");
    REQUIRE(h != nullptr);
    CHECK(prog.pipeline().is_baseline_rejected("t0p_hot"));
    CHECK_EQ(h->tier(), TierLevel::Tier0_Interpreter);
    CHECK(h->native_entry() == nullptr);
    // The tenth call reaches the threshold: compiled there, and that call
    // already runs the tier-2 code.
    CHECK_EQ(fast.run(*f, {RuntimeValue::from_i64(9)}).as_i64(), 28);
    CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
}

TEST_CASE("Tier-0 promotion - a capped pipeline keeps a rejected function interpreted") {
    auto mod = parse_or_fail(kRejectedSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg = promote_at_ten();
    cfg.max_tier = TierLevel::Tier1_Baseline;
    prog.pipeline().initialize(cfg);
    FastInterpreter fast;
    fast.set_dispatch_table(&prog);
    fast.set_module(mod.get());
    Function* f = mod->get_function("t0p_hot");
    for (int64_t n = 0; n < 40; ++n) CHECK_EQ(fast.run(*f, {RuntimeValue::from_i64(n)}).as_i64(), 3 * n + 1);
    FunctionHandle* h = prog.find("t0p_hot");
    REQUIRE(h != nullptr);
    CHECK_EQ(h->tier(), TierLevel::Tier0_Interpreter);
    CHECK_EQ(prog.pipeline().stats().tier0_promotions.load(), 0u);
}

TEST_CASE("Tier-0 promotion - a deopt lands in the interpreter and the function re-tiers") {
    Module mod("t0p_deopt");
    Function* fn = build_guarded(mod);
    FunctionDispatchTable prog;
    MultiTierPipeline& p = prog.pipeline();
    p.initialize(promote_at_ten());
    // A front pass makes an invalidated function eligible to recompile (as
    // bronze's speculation pass does); this one leaves the copy as it is.
    p.set_tier2_front_pass([](Module&) {});
    // Bound to its Function, as MultiTierPipeline::execute binds a program's
    // handles: tier-2 code deoptimizes into the Function its handle names.
    prog.get_or_create("t0p_guarded", fn);
    FastInterpreter fast;
    fast.set_dispatch_table(&prog);
    fast.set_module(&mod);
    auto call = [&](int64_t x, int32_t tag) {
        const RuntimeValue r = fast.run(*fn, {RuntimeValue::from_i64(x), RuntimeValue::from_i32(tag)});
        CHECK_EQ(r.as_i64(), guarded_expected(x, tag));
    };

    for (int64_t x = 0; x < 12; ++x) call(x, 1);
    FunctionHandle* h = prog.find("t0p_guarded");
    REQUIRE(h != nullptr);
    REQUIRE(p.is_baseline_rejected("t0p_guarded"));
    REQUIRE_EQ(h->tier(), TierLevel::Tier2_Optimized);
    void* first_code = h->native_entry();

    // The guard fails in tier-2 code: each call finishes in Tier 0 with the
    // interpreter's answer, and the second failure drops the code. With no
    // tier-1 code under it the function is back in the interpreter.
    const uint64_t deopts0 = p.tier2_deopts();
    const uint64_t inval0 = p.tier2_invalidations();
    call(50, 0);
    CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
    call(51, 0);
    CHECK_EQ(p.tier2_deopts() - deopts0, 2u);
    CHECK_EQ(p.tier2_invalidations() - inval0, 1u);
    CHECK_EQ(h->tier(), TierLevel::Tier0_Interpreter);
    CHECK(h->native_entry() == nullptr);
    TieringFeedback& fb = prog.tiering().get_feedback("t0p_guarded");
    CHECK_EQ(fb.current_tier(), TierLevel::Tier0_Interpreter);
    CHECK(!fb.is_bailout_set());

    // Its next call is its next promotion check: it is compiled again from
    // there, and that call already runs the new code.
    call(52, 1);
    CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
    CHECK(h->native_entry() != nullptr);
    CHECK(h->native_entry() != first_code);
    CHECK_EQ(fb.reoptimizations(), 1u);
    CHECK_EQ(p.stats().tier0_promotions.load(), 2u);
    for (int64_t x = 60; x < 70; ++x) call(x, 1);
    CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
}

TEST_CASE("Tier-0 promotion - a caller waits for a rejected callee no bridge can call") {
    // @t0p_wide has nine parameters (a native-to-Tier-0 bridge takes eight)
    // and tier 1 rejects it, so until it has tier-2 code of its own nothing
    // native can call it: @t0p_caller's tier-2 code must not link to it
    // before then, and both reach tier 2.
    auto mod = parse_or_fail(R"(module @t0pw
func @t0p_wide(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64, %g: i64, %h: i64, %i: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %s1 = add.i64 %a, %b
  %s2 = add.i64 %s1, %c
  %s3 = add.i64 %s2, %d
  %s4 = add.i64 %s3, %e
  %s5 = add.i64 %s4, %f
  %s6 = add.i64 %s5, %g
  %s7 = add.i64 %s6, %h
  %s8 = add.i64 %s7, %i
  ret %s8
}
func @t0p_caller(%x: i64) -> i64 {
b0:
  %v = vzero.f64x4
  %r = call.i64 @t0p_wide(%x, %x, %x, %x, %x, %x, %x, %x, %x)
  ret %r
}
)");
    FunctionDispatchTable prog;
    prog.pipeline().initialize(promote_at_ten());
    FastInterpreter fast;
    fast.set_dispatch_table(&prog);
    fast.set_module(mod.get());
    Function* caller = mod->get_function("t0p_caller");
    for (int64_t x = 0; x < 60; ++x) CHECK_EQ(fast.run(*caller, {RuntimeValue::from_i64(x)}).as_i64(), 9 * x);
    for (const char* name : {"t0p_wide", "t0p_caller"}) {
        CHECK(prog.pipeline().is_baseline_rejected(name));
        FunctionHandle* h = prog.find(name);
        REQUIRE(h != nullptr);
        CHECK_EQ(h->tier(), TierLevel::Tier2_Optimized);
        CHECK(!h->tier2_rejected());
    }
    CHECK(prog.pipeline().baseline_rejection("t0p_caller").find("256-bit vector") != std::string::npos);
}

#endif
