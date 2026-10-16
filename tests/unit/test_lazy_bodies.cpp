// Lazy function bodies (Function::mark_lazy, Module::set_body_provider): a
// function whose body its producer builds only when something first needs
// it. Tier 0 builds it on the first call; a tier-up of its caller links it
// through its stub without building it; module-wide scans skip it.
#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// @lb_f(x) calls @lb_cold(x) = x + 1 only when x > 100.
const char* kSrc = R"(module @lazy
func @lb_cold(%n: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %n, %one
  ret %r
}
func @lb_f(%x: i64) -> i64 {
entry:
  %lim = iconst.i64 100
  %c = sgt.i64 %x, %lim
  br_if %c, hot, done
hot:
  %s = call.i64 @lb_cold(%x)
  ret %s
done:
  ret %x
}
)";

// The module with @lb_cold's body taken out and left to a provider that
// builds it again, counting its calls in `built`.
std::unique_ptr<Module> lazy_module(int& built) {
    DiagnosticReporter diag;
    auto mod = parse_module(kSrc, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    Function* cold = mod->get_function("lb_cold");
    REQUIRE(cold != nullptr);
    while (!cold->blocks().empty()) cold->remove_block(cold->blocks().back());
    cold->mark_lazy();
    mod->set_body_provider([&built](Function& fn) {
        ++built;
        Builder b(fn);
        BasicBlock* bb = b.append_block("b0");
        Value* n = b.add_block_param(bb, Type::i64());
        b.position_at_end(bb);
        b.build_ret(b.build_add(n, b.build_iconst_i64(1)));
        fn.rebuild_cfg_predecessors();
        return true;
    });
    return mod;
}

TieringConfig config(TierLevel max_tier) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.max_tier = max_tier;
    cfg.set_use_fast_interpreter(true);
    return cfg;
}

int64_t expected(int64_t x) { return x > 100 ? x + 1 : x; }

} // namespace

TEST_CASE("Lazy bodies - declared, not built: verification and coroutine lowering skip them") {
    int built = 0;
    auto mod = lazy_module(built);
    const Function* cold = mod->get_function("lb_cold");
    CHECK(cold->is_lazy());
    CHECK(cold->has_body());
    CHECK(!cold->body_ready());
    DiagnosticReporter diag;
    CHECK(verify_module(*mod, &diag));
    lower_coroutines(*mod);
    CHECK_EQ(built, 0);
    CHECK_EQ(mod->materialized_count(), 0u);
}

TEST_CASE("Lazy bodies - Tier 0 builds a body on its first call, once") {
    int built = 0;
    auto mod = lazy_module(built);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(config(TierLevel::Tier0_Interpreter));
    for (int64_t x : {1, 2, 200, 300}) {
        const bool cold_runs = x > 100;
        const int before = built;
        CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
        if (!cold_runs) CHECK_EQ(built, before);
    }
    CHECK_EQ(built, 1);
    CHECK_EQ(mod->materialized_count(), 1u);
    CHECK(mod->get_function("lb_cold")->body_ready());
}

TEST_CASE("Lazy bodies - a caller's tier-up leaves a callee that never ran unbuilt") {
    int built = 0;
    auto mod = lazy_module(built);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(config(TierLevel::Tier1_Baseline));
    for (int64_t x : {1, 2, 3, 4}) {
        CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
    }
    FunctionHandle* f = prog.find("lb_f");
    REQUIRE(f != nullptr);
    CHECK_EQ(f->tier(), TierLevel::Tier1_Baseline);
    CHECK_EQ(built, 0);
    // The tier-1 caller reaches it through its stub, which builds and
    // compiles it (or bridges into Tier 0) on that first call.
    for (int64_t x : {200, 300}) {
        CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
    }
    CHECK_EQ(built, 1);
}

TEST_CASE("Lazy bodies - a built body is released, and built again when next read") {
    int built = 0;
    auto mod = lazy_module(built);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(config(TierLevel::Tier0_Interpreter));
    CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(200)}).as_i64(), expected(200));
    const Function* cold = mod->get_function("lb_cold");
    CHECK_EQ(built, 1);
    CHECK(cold->body_ready());
    CHECK(cold->body_arena() != nullptr);
    CHECK(cold->built_before());
    REQUIRE(cold->entry_block() != nullptr);
    const uint32_t first_block_id = cold->entry_block()->id();
    const uint32_t first_param_id = cold->entry_block()->param(0)->id();

    // Only the provider's bodies go: @lb_f was parsed into the module arena.
    CHECK(mod->release_bodies() > 0);
    CHECK_EQ(mod->released_count(), 1u);
    CHECK(cold->is_lazy());
    CHECK(cold->has_body());
    CHECK(cold->blocks().empty());
    CHECK(cold->body_arena() == nullptr);
    CHECK(mod->get_function("lb_f")->body_ready());

    // Tier 0 runs @lb_cold's bytecode, compiled before the release.
    CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(300)}).as_i64(), expected(300));
    CHECK_EQ(built, 1);

    // A reader builds it again, numbered as it was first.
    {
        Module::BodyRead read(*cold);
        CHECK_EQ(built, 2);
        CHECK(cold->body_ready());
        REQUIRE(cold->entry_block() != nullptr);
        CHECK_EQ(cold->entry_block()->id(), first_block_id);
        CHECK_EQ(cold->entry_block()->param(0)->id(), first_param_id);
        // Nothing is released while a read is held.
        CHECK_EQ(mod->release_bodies(), 0u);
        CHECK(cold->body_ready());
    }
    CHECK(mod->release_bodies() > 0);
    CHECK(cold->is_lazy());
}

TEST_CASE("Lazy bodies - a pinned body is never released") {
    int built = 0;
    auto mod = lazy_module(built);
    const Function* cold = mod->get_function("lb_cold");
    mod->materialize(*cold);
    cold->pin_body();
    CHECK_EQ(mod->release_bodies(), 0u);
    CHECK(cold->body_ready());
}

TEST_CASE("Lazy bodies - a caller's tier-up compiles a released callee, which has run") {
    int built = 0;
    auto mod = lazy_module(built);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(config(TierLevel::Tier1_Baseline));
    CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(200)}).as_i64(), expected(200));
    CHECK(mod->release_bodies() > 0);
    for (int64_t x : {1, 2, 3, 400}) {
        CHECK_EQ(prog.pipeline().execute(*mod, "lb_f", {RuntimeValue::from_i64(x)}).as_i64(), expected(x));
    }
    FunctionHandle* f = prog.find("lb_f");
    REQUIRE(f != nullptr);
    CHECK_EQ(f->tier(), TierLevel::Tier1_Baseline);
    CHECK_EQ(built, 2);
}

#endif
