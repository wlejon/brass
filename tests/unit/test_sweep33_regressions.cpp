// Regressions from bug sweep 33:
// - JitExecutionEngine::invoke's 256-bit-vector path called generated code
//   with no GeneratedCodeEntryScope: a native throw under it searched past
//   the C++ frames above (a host catch, Tier 0's invoke) to an outer tier-2
//   pad (v256, v256h).
// - brass_coro_resume entered a native coroutine body with no entry scope: a
//   throw in the body skipped a host's catch (coroh), and when a generated
//   caller's pad caught it the SEH unwind skipped brass_coro_resume's
//   NativeFramesScope/ThreadRootsScope destructors, which a later collection
//   then read from dead stack (corog). The body's throw now leaves the
//   resume as a C++ exception; the entry generated code calls raises it again
//   natively once its scopes are gone.
// - The Interpreter's catch of a native callee's throw kept the value as
//   i64: a gcref was not rooted (and moved under a collection in the pad),
//   i32 values compared and returned wrongly, an f64 was read as an integer.
//   The landing pad now retypes it, and the in-flight value is a root.
// - x64 `throw` of an f64 passed the value through a GPR move from an XMM
//   vreg (garbage), and a float landing_pad read RAX into a GPR move.
#include "test_framework.hpp"
#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/interpreter.hpp>
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

const char* kSrc = R"(module @s33
func @gf(%x: i64) -> f64 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  %d = fconst.f64 2.5
  throw %d
ok:
  %r = fconst.f64 0.5
  ret %r
}
func @midf(%x: i64) -> f64 {
b0:
  %v = invoke.f64 @gf(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.f64
  %k = fconst.f64 1.0
  %r = add.f64 %e, %k
  ret %r
}
func @gff(%x: i64) -> f32 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  %d = fconst.f32 2.5
  throw %d
ok:
  %r = fconst.f32 0.5
  ret %r
}
func @midff(%x: i64) -> f32 {
b0:
  %v = invoke.f32 @gff(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.f32
  %k = fconst.f32 1.0
  %r = add.f32 %e, %k
  ret %r
}
func @gi(%x: i64) -> i32 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  %d = iconst.i32 -5
  throw %d
ok:
  %r = iconst.i32 3
  ret %r
}
func @midi(%x: i64) -> i32 {
b0:
  %v = invoke.i32 @gi(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.i32
  ret %e
}
func @midid(%x: i64) -> i64 {
b0:
  %v = invoke.i32 @gi(%x), ok, bad
ok:
  %w = sext.i64 %v
  ret %w
bad:
  %e = landing_pad.i32
  %z = iconst.i32 0
  %c = slt.i32 %e, %z
  %s = zext.i64 %c
  ret %s
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
func @gg(%x: i64) -> i64 {
b0:
  %t = iconst.i64 1500
  %bad = eq.i64 %x, %t
  br_if %bad, thr, ok
thr:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  throw %o
ok:
  ret %x
}
func @midg(%x: i64) -> i64 {
b0:
  %v = invoke.i64 @gg(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.gcref
  %n = iconst.i64 200000
  %c = call.i64 @churn(%n)
  %a = load.i64 %e, 16
  ret %a
}
func @gv(%v: f64x4) -> f64x4 {
b0:
  %t = iconst.i64 1500
  throw %t
}
func @midv(%x: i64) -> i64 {
b0:
  %t = iconst.i64 1500
  %c = eq.i64 %x, %t
  br_if %c, go, ok
go:
  %z = vzero.f64x4
  %v = invoke.f64x4 @gv(%z), ok, bad
ok:
  %r = iconst.i64 7
  ret %r
bad:
  %e = landing_pad.i64
  %k = iconst.i64 1000
  %r2 = add.i64 %e, %k
  ret %r2
}
func @gbody(%f: ptr) -> i64 {
b0:
  %t = iconst.i64 1500
  throw %t
}
func @gbody2(%f: ptr) -> i64 {
b0:
  %t = iconst.i64 2500
  ret %t
}
func @fcr(%x: i64, %f: ptr) -> i64 {
b0:
  %z = iconst.i64 0
  %v = invoke.i64 @brass_coro_resume(%f, %z), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad.i64
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
  %e = landing_pad.i64
  %k = iconst.i64 1000000000
  %r = add.i64 %e, %k
  ret %r
}
)";

struct ActiveGc {
    explicit ActiveGc(MiniCheneyGC* gc) noexcept : prev(brass_get_active_gc()) { brass_set_active_gc(gc); }
    ~ActiveGc() { brass_set_active_gc(prev); }
    ActiveGc(const ActiveGc&) = delete;
    ActiveGc& operator=(const ActiveGc&) = delete;
    MiniCheneyGC* prev;
};

std::unique_ptr<Module> parse_src() {
    DiagnosticReporter diag;
    auto m = parse_module(kSrc, &diag);
    REQUIRE(m != nullptr);
    REQUIRE(verify_module(*m, &diag));
    return m;
}

void init_prog(FunctionDispatchTable& prog, bool fast) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    prog.pipeline().initialize(cfg);
}

void install(FunctionDispatchTable& prog, Module& mod, const char* name) {
    CodeInstaller installer(prog);
    FunctionHandle* h = prog.get_or_create(name, mod.get_function(name));
    REQUIRE(installer.install_tier2(*h, mod, name).success);
    REQUIRE(h->has_native_entry());
}

// Tier-2 @callee throws natively; @entry runs in Tier 0 (or natively) and
// invokes it.
RuntimeValue run_t0(const char* callee, const char* entry, bool fast, int64_t x, bool native_entry = false) {
    auto mod = parse_src();
    FunctionDispatchTable prog;
    init_prog(prog, fast);
    install(prog, *mod, callee);
    std::vector<RuntimeValue> args{RuntimeValue::from_i64(x)};
    if (native_entry) {
        install(prog, *mod, entry);
        MiniCheneyGC heap(1 << 20);
        ActiveGc a(&heap);
        ProgramScope scope(prog);
        return prog.find(entry)->call_native(args);
    }
    if (fast) {
        FastInterpreter fi(1 << 20);
        ActiveGc a(&fi.gc());
        fi.set_dispatch_table(&prog);
        fi.set_module(mod.get());
        return fi.run(*mod->get_function(entry), args);
    }
    Interpreter in;
    ActiveGc a(&in.gc());
    in.set_dispatch_table(&prog);
    in.set_module(mod.get());
    return in.run(*mod->get_function(entry), args);
}

FunctionDispatchTable* g_prog = nullptr;
Module* g_mod = nullptr;
enum class HostMode { v256, v256h, coroh };
HostMode g_mode = HostMode::v256;
int g_guards = 0;

struct Guard {  // an RAII object in the host frame: its destructor must run
    ~Guard() { ++g_guards; }
};

// Called from tier-2 @hop under @fhost's pad: enters generated code again,
// which throws 1500; this frame's catch (or Tier 0's invoke) must see it.
int64_t host_cb(int64_t x) {
    Guard g;
    if (g_mode == HostMode::v256) {
        Interpreter in;
        in.set_dispatch_table(g_prog);
        in.set_module(g_mod);
        return in.run(*g_mod->get_function("midv"), {RuntimeValue::from_i64(x)}).as_i64();
    }
    if (x != 1500) return 7;
    try {
        if (g_mode == HostMode::coroh) {
            uintptr_t f = brass_coro_create_at(g_prog->find("gbody")->native_entry(), 2, 0, 0, 0);
            return static_cast<int64_t>(brass_coro_resume(f, 0));
        }
        alignas(32) uint8_t zero[32] = {};
        (void)g_prog->find("gv")->call_native({RuntimeValue::from_v256(Type::f64x4(), zero)});
        return 7;
    } catch (const BrassException& e) {
        return static_cast<int64_t>(e.value().raw()) + 1000;
    }
}

void run_host_nested(HostMode mode) {
    auto mod = parse_src();
    FunctionDispatchTable prog;
    init_prog(prog, false);
    install(prog, *mod, "gv");
    install(prog, *mod, "gbody");
    install(prog, *mod, "fhost");
    prog.get_or_create("midv", mod->get_function("midv"));
    g_prog = &prog;
    g_mod = mod.get();
    g_mode = mode;
    MiniCheneyGC heap(1 << 20);
    ActiveGc a(&heap);
    auto run = [&](int64_t x) {
        ProgramScope scope(prog);
        return prog.find("fhost")
            ->call_native({RuntimeValue::from_i64(x), RuntimeValue::from_ptr(reinterpret_cast<const void*>(&host_cb))})
            .as_i64();
    };
    g_guards = 0;
    CHECK_EQ(run(10), 7);
    CHECK_EQ(run(1500), 2500);
    CHECK_EQ(g_guards, 2);
    CHECK(brass_innermost_entry_scope() == nullptr);
    brass_gc_collect();
    g_prog = nullptr;
    g_mod = nullptr;
}

// Generated @fcr invokes brass_coro_resume, whose native body throws (or
// returns): @fcr's pad sees the throw, and the resume's scopes are gone.
int64_t run_corog(const char* body) {
    auto mod = parse_src();
    FunctionDispatchTable prog;
    init_prog(prog, false);
    install(prog, *mod, body);
    install(prog, *mod, "fcr");
    MiniCheneyGC heap(1 << 20);
    ActiveGc a(&heap);
    uintptr_t f = brass_coro_create_at(prog.find(body)->native_entry(), 2, 0, 0, 0);
    int64_t r = 0;
    {
        ProgramScope scope(prog);
        r = prog.find("fcr")
                ->call_native({RuntimeValue::from_i64(1500), RuntimeValue::from_ptr(reinterpret_cast<const void*>(f))})
                .as_i64();
    }
    CHECK(brass_innermost_entry_scope() == nullptr);
    // (The frame lives in `heap`: a later test's collection must not visit it.)
    brass_coro_destroy(f);
    // A stale native-frame or thread-root scope would point into the stack
    // this reuses.
    {
        volatile unsigned char buf[65536];
        for (size_t i = 0; i < sizeof(buf); ++i) buf[i] = 0xCC;
    }
    brass_gc_collect();
    return r;
}

} // namespace

TEST_CASE("Sweep33 - a native throw of a gcref caught in Tier 0 stays rooted through a collection in the pad") {
    CHECK_EQ(run_t0("gg", "midg", false, 1500).as_i64(), 1500);
    CHECK_EQ(run_t0("gg", "midg", true, 1500).as_i64(), 1500);
    CHECK_EQ(run_t0("gg", "midg", false, 10).as_i64(), 10);
}

TEST_CASE("Sweep33 - a native throw of an i32 caught in Tier 0 keeps its type") {
    for (bool fast : {false, true}) {
        CHECK_EQ(run_t0("gi", "midid", fast, 1500).as_i64(), 1);
        RuntimeValue r = run_t0("gi", "midi", fast, 1500);
        CHECK_EQ(r.as_i32(), -5);
        CHECK_EQ(r.as_i64(), -5);
    }
}

TEST_CASE("Sweep33 - a native throw of an f64 caught in Tier 0 keeps its type") {
    CHECK_EQ(run_t0("gf", "midf", false, 1500).as_f64(), 3.5);
    CHECK_EQ(run_t0("gf", "midf", true, 1500).as_f64(), 3.5);
}

TEST_CASE("Sweep33 - a native throw of a float reaches a native pad with its bits (f64nat)") {
    CHECK_EQ(run_t0("gf", "midf", false, 1500, true).as_f64(), 3.5);
    CHECK_EQ(run_t0("gf", "midf", false, 10, true).as_f64(), 0.5);
    CHECK_EQ(run_t0("gff", "midff", false, 1500, true).as_f32(), 3.5f);
    CHECK_EQ(run_t0("gff", "midff", false, 10, true).as_f32(), 0.5f);
}

TEST_CASE("Sweep33 - a vector-result invoke under a host catch keeps its throw below the entry (v256h)") {
    run_host_nested(HostMode::v256h);
}

TEST_CASE("Sweep33 - a vector-result invoke from Tier 0 under a host callback reaches its pad (v256)") {
    run_host_nested(HostMode::v256);
}

TEST_CASE("Sweep33 - a native coroutine body's throw reaches the host catch around the resume (coroh)") {
    run_host_nested(HostMode::coroh);
}

TEST_CASE("Sweep33 - a native coroutine body's throw reaches a generated caller's pad and unwinds the resume (corog)") {
    CHECK_EQ(run_corog("gbody"), 2500);
    CHECK_EQ(run_corog("gbody2"), 2500);
}
