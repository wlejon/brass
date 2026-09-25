// Regressions from bug sweep 32:
// - A native brass throw's pad search (brass_seh_raise, and a deopt
//   continuation's through brass_seh_raise_above) walked past C++ frames (a
//   host callback, Tier 0) between the throwing native code and an outer
//   tier-2 frame's pad, and raised the SEH exception to that pad. The C++
//   frames in between are /EHs: their catch blocks and destructors never
//   ran, the inner Tier-0 `invoke` never saw the throw, and the stale
//   GeneratedCodeEntryScope aborted the process. The search now stops at the
//   innermost generated-code entry; with no pad below it the throw leaves as
//   a C++ exception, which the C++ frame that entered the native code sees.
// Shape: tier-2 @fhost (pad) invokes @hop, which calls a host callback
// through call_indirect; the callback runs @mid in Tier 0, which invokes
// @g. @g throws 1500 (natively, or from its guard's Tier-0 continuation);
// @mid's pad catches it and returns 2500.
#include "test_framework.hpp"
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
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

const char* kSrc = R"(module @s32
func @gstub(%x: i64) -> i64 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  throw %x
ok:
  %s = iconst.i64 7
  %r = mul.i64 %x, %s
  ret %r
}
func @g(%x: i64) -> i64 {
b0:
  %lim = iconst.i64 1000
  %ok = slt.i64 %x, %lim
  guard %ok, @gstub, [%x]
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
func @gt(%x: i64) -> i64 {
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
func @mid(%x: i64) -> i64 {
b0:
  %v = invoke.i64 @g(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad
  %k = iconst.i64 1000
  %r = add.i64 %e, %k
  ret %r
}
func @midt(%x: i64) -> i64 {
b0:
  %v = invoke.i64 @gt(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad
  %k = iconst.i64 1000
  %r = add.i64 %e, %k
  ret %r
}
func @hop(%x: i64, %p: ptr) -> i64 {
b0:
  %r = call_indirect.i64 %p(%x)
  ret %r
}
func @fhost(%x: i64, %p: ptr) -> i64 {
b0:
  %v = invoke.i64 @hop(%x, %p), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad
  %k = iconst.i64 1000000000
  %r = add.i64 %e, %k
  ret %r
}
func @fhostnp(%x: i64, %p: ptr) -> i64 {
b0:
  %v = call.i64 @hop(%x, %p)
  ret %v
}
)";

enum class Mode { hnest, hnestctl, hnestnp };
enum class Entry { interpreter, fast, host };

FunctionDispatchTable* g_prog = nullptr;
Module* g_mod = nullptr;
bool g_fast = false;
const char* g_mid = "mid";

// Called from tier-2 @hop: runs @mid (or @midt) in a fresh Tier 0.
int64_t host_cb(int64_t x) {
    const Function& f = *g_mod->get_function(g_mid);
    if (g_fast) {
        FastInterpreter fi;
        fi.set_dispatch_table(g_prog);
        fi.set_module(g_mod);
        return fi.run(f, {RuntimeValue::from_i64(x)}).as_i64();
    }
    Interpreter in;
    in.set_dispatch_table(g_prog);
    in.set_module(g_mod);
    return in.run(f, {RuntimeValue::from_i64(x)}).as_i64();
}

void run_nested(Mode mode, Entry entry_kind) {
    auto mod = [] {
        DiagnosticReporter diag;
        auto m = parse_module(kSrc, &diag);
        REQUIRE(m != nullptr);
        REQUIRE(verify_module(*m, &diag));
        return m;
    }();
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(entry_kind == Entry::fast);
    prog.pipeline().initialize(cfg);
    CodeInstaller installer(prog);
    auto handle = [&](const char* name) {
        prog.tiering().get_feedback(name).set_deopt_threshold(1000000);
        return prog.get_or_create(name, mod->get_function(name));
    };
    auto install = [&](const char* name) {
        FunctionHandle* h = handle(name);
        REQUIRE(installer.install_tier2(*h, *mod, name).success);
        REQUIRE(h->has_native_entry());
    };
    const char* entry = "fhost";
    g_mid = "mid";
    switch (mode) {
        case Mode::hnest:  // @g: a callee handle with a resumer, deopts
            handle("g");
            install("fhost");
            break;
        case Mode::hnestctl:  // @gt: tier-2 code that throws natively
            install("gt");
            install("fhost");
            g_mid = "midt";
            break;
        case Mode::hnestnp:  // as hnest, no pad in the outer tier-2 frame
            handle("g");
            install("fhostnp");
            entry = "fhostnp";
            break;
    }
    g_prog = &prog;
    g_mod = mod.get();
    g_fast = entry_kind == Entry::fast;

    gc::Heap heap;
    auto run = [&](int64_t x) -> int64_t {
        std::vector<RuntimeValue> args{RuntimeValue::from_i64(x),
                                       RuntimeValue::from_ptr(reinterpret_cast<const void*>(&host_cb))};
        switch (entry_kind) {
            case Entry::host: {
                gc::HeapScope a(heap);
                ProgramScope scope(prog);
                FunctionHandle* h = prog.find(entry);
                REQUIRE(h != nullptr);
                return h->call_native(args).as_i64();
            }
            case Entry::fast: {
                FastInterpreter fi;
                gc::HeapScope a(fi.heap());
                fi.set_dispatch_table(&prog);
                fi.set_module(mod.get());
                return fi.run(*mod->get_function(entry), args).as_i64();
            }
            case Entry::interpreter:
                break;
        }
        Interpreter in;
        in.set_dispatch_table(&prog);
        in.set_module(mod.get());
        return in.run(*mod->get_function(entry), args).as_i64();
    };
    CHECK_EQ(run(10), 11);
    if (mode == Mode::hnest || mode == Mode::hnestnp) CHECK_EQ(run(1200), 8400);
    CHECK_EQ(run(1500), 2500);
    CHECK_EQ(run(20), 21);  // nothing stale is left behind
    g_prog = nullptr;
    g_mod = nullptr;
}

} // namespace

TEST_CASE("Sweep32 - a deopt continuation's throw under a host callback reaches the Tier-0 invoke (hnest)") {
    run_nested(Mode::hnest, Entry::interpreter);
    run_nested(Mode::hnest, Entry::fast);
    run_nested(Mode::hnest, Entry::host);
}

TEST_CASE("Sweep32 - a native throw under a host callback reaches the Tier-0 invoke (hnestctl)") {
    run_nested(Mode::hnestctl, Entry::interpreter);
    run_nested(Mode::hnestctl, Entry::fast);
    run_nested(Mode::hnestctl, Entry::host);
}

TEST_CASE("Sweep32 - control: the same shape with no outer pad (hnestnp)") {
    run_nested(Mode::hnestnp, Entry::interpreter);
    run_nested(Mode::hnestnp, Entry::fast);
    run_nested(Mode::hnestnp, Entry::host);
}
