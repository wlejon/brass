// Native frames under re-entered Tier-0 code are GC roots: Tier-1 code holds
// a gcref across a call through a pointer to a function the baseline tier
// rejects, so the call goes through the native-to-Tier-0 bridge into the
// active interpreter; the callee allocates in Tier 0 until the interpreter's
// copying collector runs. The Tier-1 frame's reference must be updated.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

#if defined(_M_X64) || defined(__x86_64__)

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

// @gb_churn uses a vector op, so the baseline tier rejects it and Tier-1
// code reaches it through the bridge.
const char* kSrc = R"(module @gnf
func @gb_churn(%n: i64) -> i64 {
entry:
  %v = vzero.f64x4
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 64
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %big = iconst.i64 777
  store.i64 %t, 16, %big
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %n
}
func @gb_hold(%p: ptr, %n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %r = call_indirect.i64 %p(%n)
  %v = load.i64 %o, 16
  ret %v
}
func @gb_main(%n: i64) -> i64 {
b0:
  %p = func_addr @gb_churn
  %r = call.i64 @gb_hold(%p, %n)
  ret %r
}
)";

void run_bridge_gc_case(bool stress) {
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    prog.pipeline().initialize(cfg);
    Interpreter interp(4096);
    interp.gc().set_stress_mode(stress);
    struct ActiveGc {
        explicit ActiveGc(MiniCheneyGC* gc) : prev(brass_get_active_gc()) { brass_set_active_gc(gc); }
        ~ActiveGc() { brass_set_active_gc(prev); }
        MiniCheneyGC* prev;
    } active_gc(&interp.gc());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("gb_hold", mod->get_function("gb_hold")));
    for (int64_t n : {1, 10, 100, 1000}) {
        const size_t before = interp.gc().collection_count();
        CHECK_EQ(interp.run(*mod->get_function("gb_main"), {RuntimeValue::from_i64(n)}).as_i64(), 42);
        if (n == 1000 || stress) CHECK(interp.gc().collection_count() > before);
    }
    CHECK(prog.pipeline().is_baseline_rejected("gb_churn"));
}

} // namespace

TEST_CASE("GC native frames - a Tier-1 gcref survives a collection in a bridged Tier-0 callee") {
    run_bridge_gc_case(false);
}

TEST_CASE("GC native frames - a Tier-1 gcref survives a collection per allocation in a bridged callee") {
    run_bridge_gc_case(true);
}

#endif
