// A brass value raised by compiled (C++) code: a host function that
// generated code called throws a C++ BrassException, and the generated
// caller's landing pad catches it; the C++ frames between unwind as they do
// for any C++ exception, destructors included. The same must hold whichever
// tier compiled the catching frame.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/mir/parser.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <exception>

using namespace brass;
using namespace brass::runtime;

namespace {

int64_t host_throw_brass(int64_t v) {
    if (v > 100) throw BrassException(HostValue::from_raw(static_cast<uint64_t>(v)));
    return v * 2;
}

int g_guard_destroyed = 0;
struct Guard {
    ~Guard() { ++g_guard_destroyed; }
};

using ThrowerFn = int64_t (*)(int64_t);
ThrowerFn g_thrower = nullptr;

// Generated code entered from C++: its throw finds no pad below this frame
// and leaves as a C++ BrassException, which runs ~Guard on its way out.
int64_t host_reenter(int64_t v) {
    Guard guard;
    GeneratedCodeEntryScope entry;
    return g_thrower(v) + 1;
}

const char* kBoundaryModule = R"(
module @native_boundary

extern @host_throw_brass
extern @host_reenter

func @thrower(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 50
  %2 = sgt.i64 %0, %1
  br_if %2, bb1, bb2
bb1:
  throw %0
bb2:
  ret %0
}

func @catch_host(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 7
  %2 = mul.i64 %0, %1
  %3 = invoke.i64 @host_throw_brass(%0), bb1, bb2
bb1:
  %4 = add.i64 %3, %2
  ret %4
bb2:
  %5 = landing_pad
  %6 = add.i64 %5, %2
  ret %6
}

func @catch_reentry(%0: i64) -> i64 {
bb0:
  %1 = invoke.i64 @host_reenter(%0), bb1, bb2
bb1:
  ret %1
bb2:
  %2 = landing_pad
  %3 = iconst.i64 1000
  %4 = add.i64 %2, %3
  ret %4
}
)";

std::unique_ptr<Module> parse_boundary_module() {
    DiagnosticReporter diag;
    auto mod = parse_module(kBoundaryModule, &diag);
    REQUIRE(mod != nullptr);
    return mod;
}

} // namespace

TEST_CASE("Native boundary - a host function's C++ BrassException lands at the JIT caller's pad") {
    auto mod = parse_boundary_module();
    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("host_throw_brass", reinterpret_cast<void*>(&host_throw_brass));
    jit.register_external_symbol("host_reenter", reinterpret_cast<void*>(&host_reenter));
    REQUIRE(jit.compile_and_load(*mod));

    auto catch_host = jit.get_function_ptr<int64_t (*)(int64_t)>("catch_host");
    REQUIRE(catch_host != nullptr);
    CHECK_EQ(catch_host(5), 5 * 2 + 5 * 7);
    CHECK_EQ(catch_host(300), 300 + 300 * 7);
    CHECK_EQ(std::uncaught_exceptions(), 0);
    // And again: nothing of the first landing is left behind.
    CHECK_EQ(catch_host(301), 301 + 301 * 7);
    CHECK_EQ(catch_host(6), 6 * 2 + 6 * 7);
}

TEST_CASE("Native boundary - a generated throw leaves re-entered C++ as a C++ exception, destructors run") {
    auto mod = parse_boundary_module();
    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("host_throw_brass", reinterpret_cast<void*>(&host_throw_brass));
    jit.register_external_symbol("host_reenter", reinterpret_cast<void*>(&host_reenter));
    REQUIRE(jit.compile_and_load(*mod));

    g_thrower = jit.get_function_ptr<ThrowerFn>("thrower");
    auto catch_reentry = jit.get_function_ptr<int64_t (*)(int64_t)>("catch_reentry");
    REQUIRE(g_thrower != nullptr);
    REQUIRE(catch_reentry != nullptr);

    g_guard_destroyed = 0;
    CHECK_EQ(catch_reentry(10), 11);
    CHECK_EQ(g_guard_destroyed, 1);
    CHECK_EQ(catch_reentry(70), 1070);
    CHECK_EQ(g_guard_destroyed, 2);
    CHECK_EQ(std::uncaught_exceptions(), 0);
    g_thrower = nullptr;
}

#if defined(__x86_64__) || defined(_M_X64)

namespace {

// Baseline frames that throw and catch: a native throw from a baseline
// callee (the frame walker), a host's C++ BrassException (the OS unwinder
// and the personality), and a pad in a loop whose unwind edge passes
// arguments. A pad that did not reset RSP would leak the call's stack
// adjustment per catch; the loop runs long enough to exhaust the stack then.
const char* kBaselineModule = R"(
module @bl_eh

extern @host_throw_brass

func @bl_thrower(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 50
  %2 = sgt.i64 %0, %1
  br_if %2, bb1, bb2
bb1:
  throw %0
bb2:
  ret %0
}

func @bl_catch(%0: i64) -> i64 {
bb0:
  %1 = invoke.i64 @bl_thrower(%0), bb1, bb2
bb1:
  %m = iconst.i64 3
  %p = mul.i64 %1, %m
  %2 = invoke.i64 @host_throw_brass(%p), bb3, bb4
bb2:
  %3 = landing_pad
  ret %3
bb3:
  ret %2
bb4:
  %4 = landing_pad
  %5 = iconst.i64 100000
  %6 = add.i64 %4, %5
  ret %6
}

func @bl_loop(%0: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br head(%z, %z)
head(%i: i64, %acc: i64):
  %done = sge.i64 %i, %0
  br_if %done, exit(%acc), body
body:
  %big = iconst.i64 60
  %v = add.i64 %i, %big
  %r = invoke.i64 @bl_thrower(%v), cont, pad(%i, %acc)
cont:
  %acc2 = add.i64 %acc, %r
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br head(%i2, %acc2)
pad(%pi: i64, %pacc: i64):
  %e = landing_pad
  %acc3 = add.i64 %pacc, %e
  %one2 = iconst.i64 1
  %i3 = add.i64 %pi, %one2
  br head(%i3, %acc3)
exit(%res: i64):
  ret %res
}
)";

} // namespace

TEST_CASE("Native boundary - baseline code throws and catches at its pads") {
    for (bool pinned_tls : {false, true}) {
        DiagnosticReporter diag;
        auto mod = parse_module(kBaselineModule, &diag);
        REQUIRE(mod != nullptr);
        mod->set_pinned_tls_register(pinned_tls);
        codegen::BaselineJitCompiler compiler;
        compiler.register_external_symbol("host_throw_brass", reinterpret_cast<void*>(&host_throw_brass));
        auto fns = compiler.compile_module(*mod);
        auto find = [&](std::string_view name) -> int64_t (*)(int64_t) {
            for (const auto& f : fns) {
                if (f.name() == name) return f.get_function_ptr<int64_t (*)(int64_t)>();
            }
            return nullptr;
        };
        auto bl_catch = find("bl_catch");
        auto bl_loop = find("bl_loop");
        REQUIRE(bl_catch != nullptr);
        REQUIRE(bl_loop != nullptr);

        CHECK_EQ(bl_catch(20), 120);         // nothing throws: 20 * 3 * 2
        CHECK_EQ(bl_catch(51), 51);          // the baseline callee throws
        CHECK_EQ(bl_catch(40), 100120);      // the host throws 120
        CHECK_EQ(std::uncaught_exceptions(), 0);
        CHECK_EQ(bl_catch(41), 100123);

        const int64_t n = 200000;
        // Every iteration throws i + 60 and the pad adds it.
        CHECK_EQ(bl_loop(n), n * 60 + n * (n - 1) / 2);
        runtime::FunctionDispatchTable::instance().forget_module(*mod);
    }
}

#endif

TEST_CASE("Native boundary - the fast interpreter's invoke catches a host function's BrassException") {
    auto mod = parse_boundary_module();
    FastInterpreter interp;
    interp.register_external_function("host_throw_brass", [](const std::vector<RuntimeValue>& args) {
        return RuntimeValue::from_i64(host_throw_brass(args[0].as_i64()));
    });
    interp.set_module(mod.get());
    const Function* fn = mod->get_function("catch_host");
    REQUIRE(fn != nullptr);
    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(4)}).as_i64(), 4 * 2 + 4 * 7);
    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(200)}).as_i64(), 200 + 200 * 7);
}
