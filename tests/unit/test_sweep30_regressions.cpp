// Regressions from bug sweep 30:
// - A callee's guard failing in OSR code is finished in Tier 0. When that
//   Tier-0 continuation threw, the throw left the deopt handler as a C++
//   exception and unwound into the OSR code, whose `invoke` landing pad only
//   takes native (brass_throw) exceptions: the process crashed. The deopt
//   entry now raises it as a brass exception to the OSR code's pad.
// - While an OSR call ran, its thread-wide deopt handler also took guard
//   failures of native code outside the OSR module (a module the host
//   compiled, reached through a function pointer) and threw "not a function
//   of its OSR module". It now declines them: the outer handler, else the
//   code's own exit stub, takes them, as with no OSR call.
// - The verifier accepted a guard with neither an exit-stub function nor a
//   resume target for its id; Tier 0 then threw DeoptException when it failed.
#include "test_framework.hpp"
#include <brass/embedding/embedding.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

// @fe's loop invokes @g(i). @g's guard fails for i >= 1000 and its exit stub
// @gstub returns 7i, but throws i at i == 1500; @fe's pad adds 1e9 + i.
// @fo's own guard fails at i == 1500 and its exit stub throws, caught by a
// Tier-0 invoke in @fomain. @f4 calls through a host pointer.
const char* kOsrSrc = R"(module @s30
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
func @fe(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done(%acc)
body:
  %v = invoke.i64 @g(%i), cont, catch
cont:
  %acc2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %in = add.i64 %i, %one
  br loop(%in, %acc2)
catch:
  %e = landing_pad
  %k = iconst.i64 1000000000
  %ek = add.i64 %e, %k
  %acc3 = add.i64 %acc, %ek
  %one2 = iconst.i64 1
  %in2 = add.i64 %i, %one2
  br loop(%in2, %acc3)
done(%r: i64):
  ret %r
}
func @fostub(%acc: i64, %i: i64) -> i64 {
b0:
  throw %i
}
func @fo(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done(%acc)
body:
  %lim = iconst.i64 1500
  %ok = slt.i64 %i, %lim
  guard %ok, @fostub, [%acc, %i]
  %acc2 = add.i64 %acc, %i
  %one = iconst.i64 1
  %in = add.i64 %i, %one
  br loop(%in, %acc2)
done(%r: i64):
  ret %r
}
func @fomain(%n: i64) -> i64 {
b0:
  %v = invoke.i64 @fo(%n), ok, bad
ok:
  ret %v
bad:
  %e = landing_pad
  %k = iconst.i64 1000000000
  %r = add.i64 %e, %k
  ret %r
}
func @f4(%n: i64, %p: ptr) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done(%acc)
body:
  %v = call_indirect.i64 %p(%i)
  %acc2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %in = add.i64 %i, %one
  br loop(%in, %acc2)
done(%r: i64):
  ret %r
}
)";

// A separately compiled module: @h's guard fails for x >= 1000 and its exit
// stub @hs returns 7x.
const char* kForeignSrc = R"(module @s30_foreign
func @hs(%x: i64) -> i64 {
b0:
  %s = iconst.i64 7
  %r = mul.i64 %x, %s
  ret %r
}
func @h(%x: i64) -> i64 {
b0:
  %lim = iconst.i64 1000
  %ok = slt.i64 %x, %lim
  guard %ok, @hs, [%x]
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
)";

int64_t (*g_foreign_h)(int64_t) = nullptr;
extern "C" int64_t s30_hostfn(int64_t x) { return g_foreign_h(x); }

// Runs `fname` in Tier 0 with OSR (threshold 50) on or off.
int64_t run_case(Module& mod, const char* fname, std::vector<RuntimeValue> args, bool osr, bool fast,
                 uint64_t* osr_count) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(osr);
    prog.osr().set_threshold(50);
    int64_t got = 0;
    if (fast) {
        FastInterpreter fi;
        fi.set_dispatch_table(&prog);
        fi.set_module(&mod);
        fi.register_function_pointer(reinterpret_cast<uintptr_t>(&s30_hostfn),
            std::function<RuntimeValue(const std::vector<RuntimeValue>&)>(
                [](const std::vector<RuntimeValue>& a) { return RuntimeValue::from_i64(s30_hostfn(a[0].as_i64())); }));
        got = fi.run(*mod.get_function(fname), args).as_i64();
    } else {
        Interpreter in;
        in.set_dispatch_table(&prog);
        in.set_module(&mod);
        in.register_function_pointer(reinterpret_cast<uintptr_t>(&s30_hostfn),
            HostFn([](Interpreter&, const std::vector<RuntimeValue>& a) {
                return RuntimeValue::from_i64(s30_hostfn(a[0].as_i64()));
            }));
        got = in.run(*mod.get_function(fname), args).as_i64();
    }
    // An OSR call a throw ends is not counted as a migration; its deopt is.
    if (osr_count) *osr_count = prog.osr().total_osr_migrations() + prog.osr().total_native_deopts();
    return got;
}

// sum_{i<1000} (i+1) + sum_{1000<=i<2000, i!=1500} 7i + 1e9 + 1500
constexpr int64_t kExcWant = 1010988000;

void check_osr_matches_tier0(Module& mod, const char* fname, std::vector<RuntimeValue> args, bool fast,
                             int64_t want) {
    CHECK_EQ(run_case(mod, fname, args, false, fast, nullptr), want);
    uint64_t osr = 0;
    CHECK_EQ(run_case(mod, fname, args, true, fast, &osr), want);
    CHECK(osr > 0);
}

} // namespace

TEST_CASE("Sweep30 - a callee's Tier-0 throw after a deopt reaches the OSR code's invoke pad (Interpreter)") {
    auto mod = parse_ok(kOsrSrc);
    check_osr_matches_tier0(*mod, "fe", {RuntimeValue::from_i64(2000)}, false, kExcWant);
}

TEST_CASE("Sweep30 - a callee's Tier-0 throw after a deopt reaches the OSR code's invoke pad (FastInterpreter)") {
    auto mod = parse_ok(kOsrSrc);
    check_osr_matches_tier0(*mod, "fe", {RuntimeValue::from_i64(2000)}, true, kExcWant);
}

TEST_CASE("Sweep30 - the OSR'd function's own exit stub throwing still leaves the OSR call") {
    auto mod = parse_ok(kOsrSrc);
    for (bool fast : {false, true}) {
        check_osr_matches_tier0(*mod, "fomain", {RuntimeValue::from_i64(2000)}, fast, 1000000000 + 1500);
    }
}

TEST_CASE("Sweep30 - an OSR call leaves guard failures of foreign native code to their exit stubs") {
    auto fm = parse_ok(kForeignSrc);
    HostEngine eng;
    auto foreign = eng.compile(*fm);
    REQUIRE(foreign != nullptr);
    g_foreign_h = reinterpret_cast<int64_t (*)(int64_t)>(foreign->get_symbol_address("h"));
    REQUIRE(g_foreign_h != nullptr);
    CHECK_EQ(g_foreign_h(1500), 10500);
    auto mod = parse_ok(kOsrSrc);
    // sum_{i<1000} (i+1) + sum_{1000<=i<2000} 7i
    const int64_t want = 10997000;
    for (bool fast : {false, true}) {
        check_osr_matches_tier0(*mod, "f4",
                                {RuntimeValue::from_i64(2000),
                                 RuntimeValue::from_ptr(reinterpret_cast<const void*>(&s30_hostfn))},
                                fast, want);
    }
    g_foreign_h = nullptr;
}

TEST_CASE("Sweep30 - the verifier rejects a guard with no exit stub and no resume target for its id") {
    const char* src = R"(module @s30_rt
func @two(%x: i64) -> i64 {
b0:
  %a = iconst.i64 100
  %ok1 = slt.i64 %x, %a
  guard %ok1, @nostub, [%x]
  %b = iconst.i64 50
  %ok2 = slt.i64 %x, %b
  guard %ok2, @nostub, [%x]
  ret %x
bres(%v: i64):
  %k = iconst.i64 1000
  %r = add.i64 %v, %k
  ret %r
resume_table {
  entry 0 -> bres
}
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    CHECK(!verify_module(*mod, &diag));
    CHECK(diag.format_all().find("neither an exit stub") != std::string::npos);
}

TEST_CASE("Sweep30 - guards sharing a resume block through one entry per id verify and resume there") {
    const char* src = R"(module @s30_rt2
func @two(%x: i64) -> i64 {
b0:
  %a = iconst.i64 100
  %ok1 = slt.i64 %x, %a
  guard %ok1, @nostub, [%x]
  %b = iconst.i64 50
  %ok2 = slt.i64 %x, %b
  guard %ok2, @nostub, [%x]
  ret %x
bres(%v: i64):
  %k = iconst.i64 1000
  %r = add.i64 %v, %k
  ret %r
resume_table {
  entry 0 -> bres
  entry 1 -> bres
}
}
)";
    auto mod = parse_ok(src);
    for (int64_t x : {10, 70, 200}) {
        Interpreter in;
        in.set_module(mod.get());
        const int64_t want = x < 50 ? x : x + 1000;
        CHECK_EQ(in.run(*mod->get_function("two"), {RuntimeValue::from_i64(x)}).as_i64(), want);
    }
}
