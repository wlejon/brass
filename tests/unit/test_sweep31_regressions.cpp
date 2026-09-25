// Regressions from bug sweep 31:
// - A guard exit's chained unwind info said RSP was the record's allocation
//   below the prologue's frame, but a guard whose exit stub takes stack
//   arguments (Win64: 5 or more state values) lowered RSP again before
//   calling the stub. Anything unwinding through .pdata from inside the stub
//   (a C++ exception, a debugger, RtlVirtualUnwind) misread the frame: the
//   walk went into garbage and a C++ exception killed the process. The
//   outgoing area is now part of the one allocation the region describes;
//   a large record's pages are probed without moving RSP, and the region
//   ends at the `add rsp` that releases it.
// - A tier-2 function's callee, deoptimized through its resumer, whose
//   Tier-0 continuation throws: the throw must reach the caller's `invoke`
//   pad as a native throw does, not leave as a C++ exception.
#include "test_framework.hpp"
#include <brass/embedding/embedding.hpp>
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
#include <stdexcept>
#include <string>
// The walk is made with the platform's unwinder: .pdata on Windows x64, the
// DWARF unwind info generated code registers elsewhere (AArch64 exit stubs
// take stack arguments past x7 / d7).
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#define S31_WIN64_WALK 1
#define S31_WALK 1
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(_WIN32) && (defined(__x86_64__) || defined(__aarch64__))
#define S31_WALK 1
#if defined(__APPLE__)
#include <libunwind.h>
#else
#include <unwind.h>
#endif
#endif

#if defined(_MSC_VER)
#define S31_NOINLINE __declspec(noinline)
#else
#define S31_NOINLINE __attribute__((noinline))
#endif

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src.c_str(), &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

#if defined(S31_WALK)

// @g's guard fails for x >= 1000 and exits to @gs with `nstate` state values
// (nstate - 1 copies of %x, then %p). @gs calls the host function %p(x).
// @g2 has two such guards; the second one fails. @main / @main2 invoke them.
std::string guard_exit_src(int nstate) {
    std::string params, state;
    for (int i = 0; i < nstate - 1; ++i) {
        params += "%a" + std::to_string(i) + ": i64, ";
        state += "%x, ";
    }
    params += "%p: ptr";
    state += "%p";
    std::string s = "module @s31\n";
    s += "func @gs(" + params + ") -> i64 {\nb0:\n  %r = call_indirect.i64 %p(%a0)\n  ret %r\n}\n";
    s += "func @g(%x: i64, %p: ptr) -> i64 {\nb0:\n  %lim = iconst.i64 1000\n  %ok = slt.i64 %x, %lim\n"
         "  guard %ok, @gs, [" + state + "]\n  %one = iconst.i64 1\n  %r = add.i64 %x, %one\n  ret %r\n}\n";
    s += "func @g2(%x: i64, %p: ptr) -> i64 {\nb0:\n  %lim0 = iconst.i64 5000\n  %ok0 = slt.i64 %x, %lim0\n"
         "  guard %ok0, @gs, [" + state + "]\n  %lim = iconst.i64 1000\n  %ok = slt.i64 %x, %lim\n"
         "  guard %ok, @gs, [" + state + "]\n  %one = iconst.i64 1\n  %r = add.i64 %x, %one\n  ret %r\n}\n";
    for (const char* m : {"main", "main2"}) {
        const char* callee = std::string(m) == "main" ? "g" : "g2";
        s += std::string("func @") + m + "(%x: i64, %p: ptr) -> i64 {\nb0:\n  %v = invoke.i64 @" + callee +
             "(%x, %p), ok, bad\nok:\n  ret %v\nbad:\n  %e = landing_pad\n  %k = iconst.i64 1000000\n"
             "  %r = add.i64 %e, %k\n  ret %r\n}\n";
    }
    return s;
}

uintptr_t g_limit_rsp = 0;  // an address in s31_call_main's frame
uintptr_t g_main_lo = 0, g_main_hi = 0;
bool g_walk_ok = false;
bool g_cpp_throw = false;

#if defined(S31_WIN64_WALK)
// Unwinds from here through .pdata and checks it reaches s31_call_main with
// an RSP below that frame's locals.
extern "C" __declspec(noinline) int64_t s31_host_walk(int64_t x) {
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    g_walk_ok = false;
    for (int depth = 0; depth < 64; ++depth) {
        const DWORD64 pc = ctx.Rip;
        if (pc >= g_main_lo && pc < g_main_hi) {
            g_walk_ok = ctx.Rsp < g_limit_rsp;
            break;
        }
        DWORD64 base = 0;
        PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(pc, &base, nullptr);
        if (!fe) break;  // every frame between here and the caller is described
        PVOID hd = nullptr;
        DWORD64 est = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, pc, fe, &ctx, &hd, &est, nullptr);
    }
    if (g_cpp_throw) throw std::runtime_error("s31 host exception");
    return x * 7;
}
#elif defined(__APPLE__)
// Unwinds from here with the DWARF unwind info and checks it reaches
// s31_call_main with an SP below that frame's locals.
extern "C" S31_NOINLINE int64_t s31_host_walk(int64_t x) {
    g_walk_ok = false;
    unw_context_t uc;
    unw_cursor_t cur;
    if (unw_getcontext(&uc) == 0 && unw_init_local(&cur, &uc) == 0) {
        for (int depth = 0; depth < 64 && unw_step(&cur) > 0; ++depth) {
            unw_proc_info_t info;
            if (unw_get_proc_info(&cur, &info) != 0) break;  // every frame here is described
            if (info.start_ip == g_main_lo) {
                unw_word_t sp = 0;
                g_walk_ok = unw_get_reg(&cur, UNW_REG_SP, &sp) == 0 && sp < g_limit_rsp;
                break;
            }
        }
    }
    if (g_cpp_throw) throw std::runtime_error("s31 host exception");
    return x * 7;
}
#else
struct S31Walk {
    int depth = 0;
    uintptr_t callee_cfa = 0;  // the callee frame's CFA: this frame's SP
};

_Unwind_Reason_Code s31_walk_frame(struct _Unwind_Context* ctx, void* arg) {
    auto* w = static_cast<S31Walk*>(arg);
    const uintptr_t start = reinterpret_cast<uintptr_t>(
        _Unwind_FindEnclosingFunction(reinterpret_cast<void*>(_Unwind_GetIP(ctx))));
    if (start == g_main_lo) {
        g_walk_ok = w->callee_cfa != 0 && w->callee_cfa < g_limit_rsp;
        return _URC_END_OF_STACK;
    }
    w->callee_cfa = static_cast<uintptr_t>(_Unwind_GetCFA(ctx));
    return ++w->depth < 64 ? _URC_NO_REASON : _URC_END_OF_STACK;
}

// Unwinds from here with the DWARF unwind info and checks it reaches
// s31_call_main with an SP below that frame's locals.
extern "C" S31_NOINLINE int64_t s31_host_walk(int64_t x) {
    g_walk_ok = false;
    S31Walk w;
    _Unwind_Backtrace(&s31_walk_frame, &w);
    if (g_cpp_throw) throw std::runtime_error("s31 host exception");
    return x * 7;
}
#endif

using MainFn = int64_t (*)(int64_t, const void*);

S31_NOINLINE int64_t s31_call_main(MainFn f, int64_t x) {
    volatile int marker = 0;
    g_limit_rsp = reinterpret_cast<uintptr_t>(&marker);
    int64_t r = f(x, reinterpret_cast<const void*>(&s31_host_walk));
    return r + marker;
}

// Compiles the module natively (no tiering, no deopt handler: the failing
// guard calls its exit stub) and runs `entry`(1500).
void run_guard_exit(int nstate, const char* entry, bool cpp_throw) {
#if defined(S31_WIN64_WALK)
    DWORD64 base = 0;
    const DWORD64 addr = reinterpret_cast<DWORD64>(&s31_call_main);
    PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(addr, &base, nullptr);
    REQUIRE(fe != nullptr);
    g_main_lo = base + fe->BeginAddress;
    g_main_hi = base + fe->EndAddress;
#else
    g_main_lo = reinterpret_cast<uintptr_t>(&s31_call_main);
#endif
    g_cpp_throw = cpp_throw;

    auto mod = parse_ok(guard_exit_src(nstate));
    HostEngine eng;
    auto cm = eng.compile(*mod);
    REQUIRE(cm != nullptr);
    auto f = reinterpret_cast<MainFn>(cm->get_symbol_address(entry));
    REQUIRE(f != nullptr);
    if (cpp_throw) {
        bool caught = false;
        try {
            s31_call_main(f, 1500);
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "s31 host exception";
        }
        CHECK(caught);
    } else {
        CHECK_EQ(s31_call_main(f, 1500), 1500 * 7);
    }
    CHECK(g_walk_ok);
    g_cpp_throw = false;
}

#endif

} // namespace

#if defined(S31_WALK)

TEST_CASE("Sweep31 - an exit stub with stack arguments unwinds to its native caller (walk5)") {
    run_guard_exit(4, "main", false);
    run_guard_exit(5, "main", false);
    run_guard_exit(7, "main", false);
    run_guard_exit(10, "main", false);  // AArch64: x0-x7, then the stack
}

TEST_CASE("Sweep31 - a C++ exception from an exit stub with stack arguments reaches the host (cpp5)") {
    run_guard_exit(5, "main", true);
    run_guard_exit(7, "main", true);
    run_guard_exit(10, "main", true);
}

TEST_CASE("Sweep31 - the second guard of a function unwinds with stack arguments too (two5)") {
    run_guard_exit(5, "main2", false);
    run_guard_exit(5, "main2", true);
}

TEST_CASE("Sweep31 - a guard exit larger than a page unwinds while its exit stub runs") {
    run_guard_exit(700, "main", false);
    run_guard_exit(700, "main2", true);
}

#endif

namespace {

// @fmain invokes @g(x). @g's guard fails for x >= 1000; its exit stub
// @gstub returns 7x but throws x at x == 1500. @fmain's pad adds 1e9.
const char* kTier2ThrowSrc = R"(module @s31_t2
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
func @fmain(%x: i64) -> i64 {
b0:
  %v = invoke.i64 @g(%x), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad
  %k = iconst.i64 1000000000
  %r = add.i64 %e, %k
  ret %r
}
)";

// Installs tier-2 code for @fmain (and, with `install_g`, for @g first),
// then runs @fmain in Tier 0, which dispatches to the tier-2 code.
void run_tier2_throw(bool install_g, bool fast) {
    auto mod = parse_ok(kTier2ThrowSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    prog.pipeline().initialize(cfg);
    prog.tiering().get_feedback("g").set_deopt_threshold(1000000);
    prog.tiering().get_feedback("fmain").set_deopt_threshold(1000000);
    FunctionHandle* hg = prog.get_or_create("g", mod->get_function("g"));
    FunctionHandle* hf = prog.get_or_create("fmain", mod->get_function("fmain"));
    CodeInstaller installer(prog);
    if (install_g) REQUIRE(installer.install_tier2(*hg, *mod, "g").success);
    REQUIRE(installer.install_tier2(*hf, *mod, "fmain").success);
    REQUIRE(hg->has_native_entry());
    REQUIRE(hf->has_native_entry());
    auto run = [&](int64_t x) -> int64_t {
        if (fast) {
            FastInterpreter fi;
            fi.set_dispatch_table(&prog);
            fi.set_module(mod.get());
            return fi.run(*mod->get_function("fmain"), {RuntimeValue::from_i64(x)}).as_i64();
        }
        Interpreter in;
        in.set_dispatch_table(&prog);
        in.set_module(mod.get());
        return in.run(*mod->get_function("fmain"), {RuntimeValue::from_i64(x)}).as_i64();
    };
    CHECK_EQ(run(10), 11);
    CHECK_EQ(run(1200), 8400);
    CHECK_EQ(run(1500), 1000000000 + 1500);
}

} // namespace

TEST_CASE("Sweep31 - a tier-2 callee's Tier-0 throw after a deopt reaches the tier-2 caller's invoke pad") {
    for (bool fast : {false, true}) {
        run_tier2_throw(false, fast);
        run_tier2_throw(true, fast);
    }
}
