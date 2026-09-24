// Regressions from bug sweep 34:
// - A MIR throw from a function run in Tier 0 through the native-to-Tier-0
//   bridge left the bridge as a C++ InterpreterThrownException, which the
//   landing pads of the bridge's native (tier-2) callers never see: it
//   escaped to the host (host, gc) or to an outer Tier-0 pad (interp). The
//   bridge now raises it again natively once its scopes are gone.
// - Coroutine frames were registered in one process-wide list, shared by
//   every heap. A frame that never finished (its body threw, or it was
//   abandoned while suspended) stayed there after its heap died, and the next
//   heap's collection read it (drive, abandon). Each heap now owns its
//   frames' registry, and a body that throws finishes its frame.
#include "test_framework.hpp"
#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

// @t0 has a `throw`, so the baseline tier rejects it and its lazy stub binds
// the native-to-Tier-0 bridge. Tier-2 @outer invokes @hop (pad: +1000),
// which calls @t0 through its address.
const char* kBridgeSrc = R"(module @s34a
func @t0(%x: i64) -> i64 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  throw %x
ok:
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
func @hop(%x: i64, %p: ptr) -> i64 {
b0:
  %r = call_indirect.i64 %p(%x)
  ret %r
}
func @outer(%x: i64, %p: ptr) -> i64 {
b0:
  %v = invoke.i64 @hop(%x, %p), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.i64
  %k = iconst.i64 1000
  %r = add.i64 %e, %k
  ret %r
}
func @main(%x: i64) -> i64 {
b0:
  %p = func_addr @t0
  %v = invoke.i64 @outer(%x, %p), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.i64
  %k = iconst.i64 1000000
  %r = add.i64 %e, %k
  ret %r
}
func @churn(%n: i64) -> i64 {
entry:
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
func @t0g(%x: i64) -> i64 {
b0:
  %n = iconst.i64 100000
  %c = call.i64 @churn(%n)
  %t2 = iconst.i64 1500
  %b2 = eq.i64 %x, %t2
  br_if %b2, thr2, ok2
thr2:
  throw %x
ok2:
  ret %x
}
func @outerg(%x: i64, %p: ptr) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  %v = invoke.i64 @hop(%x, %p), ok, bad
ok:
  %a = load.i64 %o, 16
  %s = add.i64 %a, %v
  ret %s
bad:
  %e = landing_pad.i64
  %a2 = load.i64 %o, 16
  %k = iconst.i64 1000
  %r = add.i64 %a2, %k
  ret %r
}
)";

struct ActiveGc {
    explicit ActiveGc(MiniCheneyGC* gc) noexcept : prev(brass_get_active_gc()) { brass_set_active_gc(gc); }
    ~ActiveGc() { brass_set_active_gc(prev); }
    MiniCheneyGC* prev;
};

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto m = parse_module(src, &diag);
    REQUIRE(m != nullptr);
    REQUIRE(verify_module(*m, &diag));
    return m;
}

enum class BridgeMode { host, interp, gc };

// The results for x = 10 and x = 1500; an exception escaping is a failure.
std::vector<int64_t> run_bridge(BridgeMode mode) {
    auto mod = parse_ok(kBridgeSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    prog.pipeline().initialize(cfg);
    prog.pipeline().tiering().set_active_module(mod.get());
    const bool g = mode == BridgeMode::gc;
    const char* outer = g ? "outerg" : "outer";
    {
        CodeInstaller installer(prog);
        FunctionHandle* h = prog.get_or_create(outer, mod->get_function(outer));
        REQUIRE(installer.install_tier2(*h, *mod, outer).success);
    }
    // Created after the install, which publishes the module's other
    // functions only to handles that already exist: these stay in Tier 0.
    for (const char* f : {"t0", "churn", "t0g", "main"}) prog.get_or_create(f, mod->get_function(f));
    MiniCheneyGC heap(1 << 20);
    ActiveGc active(&heap);
    const char* callee = g ? "t0g" : "t0";
    void* p = prog.pipeline().function_address(callee, mod->get_function(callee));
    std::vector<int64_t> out;
    for (int64_t x : {int64_t(10), int64_t(1500)}) {
        int64_t r = -1;
        try {
            if (mode == BridgeMode::interp) {
                Interpreter in;
                in.set_dispatch_table(&prog);
                in.set_module(mod.get());
                ActiveGc interp_heap(&in.gc());
                r = in.run(*mod->get_function("main"), {RuntimeValue::from_i64(x)}).as_i64();
            } else {
                ProgramScope scope(prog);
                r = prog.find(outer)->call_native({RuntimeValue::from_i64(x), RuntimeValue::from_ptr(p)}).as_i64();
            }
        } catch (const InterpreterThrownException& e) {
            r = -2;
            std::printf("  x=%lld: InterpreterThrownException %lld escaped\n", (long long)x,
                        (long long)e.value().as_i64());
        } catch (const BrassException& e) {
            r = -3;
            std::printf("  x=%lld: BrassException %lld escaped\n", (long long)x, (long long)e.value().raw());
        }
        out.push_back(r);
    }
    // The call went through the bridge: @callee never got native code.
    CHECK(prog.find(callee)->native_entry() == nullptr);
    CHECK(prog.pipeline().tiering().get_feedback(std::string("brass.tier0_bridge.") + callee).invocation_count() == 2);
    CHECK(brass_innermost_entry_scope() == nullptr);
    return out;
}

// B yields n, then throws n on its next resume. @drive's A catches B's throw
// in a pad and suspends there; @abandon leaves B suspended.
const char* kCoroSrc = R"(module @s34c
func @bthrow(%n: i64) -> i64 {
b0:
  %y = coro_suspend.i64 %n, 1
  throw %n
}
func @res(%c: gcref) -> i64 {
b0:
  %z = iconst.i64 0
  %r = coro_resume.i64 %c, %z
  ret %r
}
func @abody(%n: i64) -> i64 {
b0:
  %c = coro_create @bthrow(%n)
  %v1 = invoke.i64 @res(%c), ok1, bad
ok1:
  %v2 = invoke.i64 @res(%c), ok2, bad
ok2:
  ret %v2
bad:
  %e = landing_pad.i64
  %y = coro_suspend.i64 %e, 1
  %k = add.i64 %e, %y
  ret %k
}
func @drive(%n: i64) -> i64 {
b0:
  %a = coro_create @abody(%n)
  %z = iconst.i64 0
  %r0 = coro_resume.i64 %a, %z
  %th = iconst.i64 1000
  %r1 = coro_resume.i64 %a, %th
  coro_destroy %a
  %m = iconst.i64 10000
  %t = mul.i64 %r0, %m
  %s = add.i64 %t, %r1
  ret %s
}
func @abandon(%n: i64) -> i64 {
b0:
  %c = coro_create @bthrow(%n)
  %z = iconst.i64 0
  %r = coro_resume.i64 %c, %z
  ret %r
}
func @churn(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 64
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %n
}
)";

std::unique_ptr<Module> lowered_coro_module() {
    auto mod = parse_ok(kCoroSrc);
    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(*mod));
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

// Runs @first in one Interpreter, then collects in a fresh one: the first
// heap's frames must not be among the second's roots.
int64_t run_then_collect_elsewhere(const char* first) {
    auto mod = lowered_coro_module();
    int64_t r = 0;
    {
        Interpreter in;
        ActiveGc a(&in.gc());
        in.set_module(mod.get());
        r = in.run(*mod->get_function(first), {RuntimeValue::from_i64(1500)}).as_i64();
    }
    {
        Interpreter in;
        ActiveGc a(&in.gc());
        in.set_module(mod.get());
        CHECK_EQ(in.run(*mod->get_function("churn"), {RuntimeValue::from_i64(100000)}).as_i64(), 100000);
        brass_gc_collect();
    }
    return r;
}

} // namespace

TEST_CASE("Sweep34 - a Tier-0 throw through the bridge reaches a tier-2 pad (host)") {
    const auto r = run_bridge(BridgeMode::host);
    CHECK_EQ(r[0], 11);
    CHECK_EQ(r[1], 2500);
}

TEST_CASE("Sweep34 - a Tier-0 throw through the bridge reaches the inner tier-2 pad, not the Tier-0 one (interp)") {
    const auto r = run_bridge(BridgeMode::interp);
    CHECK_EQ(r[0], 11);
    CHECK_EQ(r[1], 2500);
}

TEST_CASE("Sweep34 - a Tier-0 throw through the bridge after a collection reaches the pad with its gcref live (gc)") {
    const auto r = run_bridge(BridgeMode::gc);
    CHECK_EQ(r[0], 20);
    CHECK_EQ(r[1], 2500);
}

TEST_CASE("Sweep34 - a coroutine whose body threw is not a root of a later heap (drive)") {
    CHECK_EQ(run_then_collect_elsewhere("drive"), 1500LL * 10000 + 2500);
}

TEST_CASE("Sweep34 - an abandoned suspended coroutine is not a root of a later heap (abandon)") {
    CHECK_EQ(run_then_collect_elsewhere("abandon"), 1500);
}

TEST_CASE("Sweep34 - a coroutine whose body threw is finished and unregistered") {
    auto mod = lowered_coro_module();
    for (bool fast : {false, true}) {
        uintptr_t h = 0;
        bool threw = false;
        if (fast) {
            FastInterpreter fi(1 << 20);
            ActiveGc a(&fi.gc());
            fi.set_module(mod.get());
            h = fi.coro_create(*mod, "bthrow", {RuntimeValue::from_i64(7)});
            CHECK_EQ(fi.coro_resume(h, 0), 7u);
            CHECK(is_active_coro_frame(h));
            try {
                fi.coro_resume(h, 0);
            } catch (const InterpreterThrownException& e) {
                threw = true;
                CHECK_EQ(e.value().as_i64(), 7);
            }
            CHECK(fi.coro_is_done(h));
            CHECK(!is_active_coro_frame(h));
            // Finished: a resume does not run the body again.
            CHECK_EQ(fi.coro_resume(h, 0), 0u);
        } else {
            MiniCheneyGC heap(1 << 20);
            ActiveGc a(&heap);
            h = create_mir_coro_frame(*mod->get_function("bthrow"), 2, 0);
            REQUIRE(h != 0);
            reinterpret_cast<BrassCoroFrame*>(h)->slots[0] = 7;
            CHECK_EQ(brass_coro_resume(h, 0), 7u);
            try {
                brass_coro_resume(h, 0);
            } catch (const InterpreterThrownException& e) {
                threw = true;
                CHECK_EQ(e.value().as_i64(), 7);
            }
            CHECK(brass_coro_is_done(h) != 0);
            CHECK(!is_active_coro_frame(h));
            CHECK_EQ(brass_coro_resume(h, 0), 0u);
        }
        CHECK(threw);
    }
}
