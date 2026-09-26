// OSR entry and resume paths.
//
// The entry: plan_osr_entry's live-in set (the header's parameters, then
// the other live values, constants rematerialized rather than carried) and
// the entry function build_osr_entry_function makes from it, compiled and
// entered with a hand-filled buffer part way through the loop.
//
// The resumes: a guard failing inside OSR code (forced by deopt stress, so
// every run takes it) finishes the OSR call in Tier 0 through its exit
// stub; a callee's resume-only guard failing inside OSR code resumes the
// callee in Tier 0 at its resume block and returns into the OSR code; a
// function whose resume points are its guards' is entered through OSR, one
// with a resume point no guard names is not. Each answer is the
// interpreter's.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/osr_entry.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/deopt_stress.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_or_fail(std::string_view src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

const BasicBlock* block_named(const Function& fn, std::string_view name) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() == name) return bb;
    }
    return nullptr;
}

// sum over i in [0, n) of (step(i) + k + 5), k = n + 7 computed before the
// loop and 5 a constant defined there too.
constexpr std::string_view kSumLoop = R"(module @osr_resume_sum
func @osr_step(%x: i64) -> i64 {
entry:
  %c3 = iconst.i64 3
  %r = mul %x, %c3
  ret %r
}
func @osr_sum(%n: i64) -> i64 {
entry:
  %c7 = iconst.i64 7
  %k = add %n, %c7
  %c5 = iconst.i64 5
  %c0 = iconst.i64 0
  br loop(%c0, %c0)
loop(%i: i64, %acc: i64):
  %s = call.i64 @osr_step(%i)
  %t = add %s, %k
  %t5 = add %t, %c5
  %a = add %acc, %t5
  %c1 = iconst.i64 1
  %j = add %i, %c1
  %more = slt %j, %n
  br_if %more, loop(%j, %a), done(%a)
done(%r: i64):
  ret %r
}
)";

// The rest of osr_sum from the header with (i, acc).
int64_t sum_from(int64_t n, int64_t i, int64_t acc) {
    do {
        acc += i * 3 + n + 7 + 5;
        ++i;
    } while (i < n);
    return acc;
}

// A loop speculating on its accumulator, with an exit stub that finishes
// the loop generically: exits equivalent to the fast path.
constexpr std::string_view kStubLoop = R"(module @osr_resume_stub
func @sum_rest(%i: i64, %acc: i64, %n: i64) -> i64 {
entry:
  br loop(%i, %acc)
loop(%j: i64, %a: i64):
  %c = slt %j, %n
  br_if %c, body, done(%a)
body:
  %a2 = add %a, %j
  %one = iconst.i64 1
  %j2 = add %j, %one
  br loop(%j2, %a2)
done(%r: i64):
  ret %r
}
func @sum_spec(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, body, done(%acc)
body:
  %lim = iconst.i64 1000000000000
  %ok = slt %acc, %lim
  guard %ok, @sum_rest, [%i, %acc, %n]
  %a2 = add %acc, %i
  %one = iconst.i64 1
  %i2 = add %i, %one
  br loop(%i2, %a2)
done(%r: i64):
  ret %r
}
)";

// A loop calling a callee whose guard exits only to a resume block.
constexpr std::string_view kCalleeResume = R"(module @osr_resume_callee
func @clampr(%x: i64, %hi: i64) -> i64 {
bb0:
  %ok = slt %x, %hi
  guard %ok, @slow, [%x, %hi]
  ret %x
bres(%a: i64, %h: i64):
  %c = slt %a, %h
  %r = select %c, %a, %h
  ret %r

resume_table {
  entry 0 -> bres
}
}
func @run(%n: i64) -> i64 {
bb0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, body, done(%acc)
body:
  %seven = iconst.i64 7
  %m = mul %i, %seven
  %thirteen = iconst.i64 13
  %v = smod %m, %thirteen
  %hi = iconst.i64 9
  %cl = call.i64 @clampr(%v, %hi)
  %acc2 = add %acc, %cl
  %one = iconst.i64 1
  %i2 = add %i, %one
  br loop(%i2, %acc2)
done(%r: i64):
  ret %r
}
)";

int64_t callee_expected(int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = (i * 7) % 13;
        acc += v < 9 ? v : 9;
    }
    return acc;
}

void init_program(FunctionDispatchTable& prog, uint64_t threshold) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(threshold);
}

struct OsrRun {
    int64_t result = 0;
    uint64_t migrations = 0;
    uint64_t tier2_deopts = 0;
};

// Runs `fn(arg)` on a program's fast interpreter until a run enters OSR
// code (at most 8 runs; the first asks for the code, which compiles in the
// background). Each run must give `expected`.
OsrRun run_through_osr(const Function& fn, int64_t arg, int64_t expected) {
    FunctionDispatchTable prog;
    init_program(prog, 20);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    OsrRun out;
    for (int run = 0; run < 8; ++run) {
        out.result = interp.run(fn, {RuntimeValue::from_i64(arg)}).as_i64();
        CHECK_EQ(out.result, expected);
        CompilePool::shared().wait_owner(&prog.osr());
        if (prog.osr().total_osr_migrations() > 0 && run > 0) break;
    }
    out.migrations = prog.osr().total_osr_migrations();
    out.tier2_deopts = prog.pipeline().tier2_deopts();
    return out;
}

} // namespace

TEST_CASE("OSR resume - the entry plan carries the header's parameters first and rematerializes constants") {
    auto mod = parse_or_fail(kSumLoop);
    const Function& fn = *mod->get_function("osr_sum");
    const BasicBlock* loop = block_named(fn, "loop");
    REQUIRE(loop != nullptr);
    std::string why;
    auto plan = plan_osr_entry(fn, *loop, &why);
    if (!plan) std::cerr << why << "\n";
    REQUIRE(plan.has_value());
    CHECK(plan->function == &fn);
    CHECK(plan->block == loop);
    // %i, %acc, then %n, %k and the constant %c5.
    REQUIRE_EQ(plan->live_ins.size(), 5u);
    CHECK(plan->live_ins[0].value == loop->param(0));
    CHECK(plan->live_ins[1].value == loop->param(1));
    size_t remat = 0;
    bool has_n = false;
    for (const auto& li : plan->live_ins) {
        remat += li.rematerialize ? 1 : 0;
        has_n = has_n || li.value == fn.entry_block()->param(0);
    }
    CHECK_EQ(remat, 1u);
    CHECK(has_n);
    // The region: the header and what it reaches, not the entry block.
    CHECK_EQ(plan->region.size(), 2u);
    for (const BasicBlock* bb : plan->region) CHECK(bb != fn.entry_block());
}

TEST_CASE("OSR resume - no entry at the entry block, into a function with an unguarded resume point, or carrying a gcref") {
    std::string why;
    {
        auto mod = parse_or_fail(kSumLoop);
        const Function& fn = *mod->get_function("osr_sum");
        CHECK(!plan_osr_entry(fn, *fn.entry_block(), &why).has_value());
    }
    {
        // Resume point 5 belongs to no guard: nothing in the OSR code could
        // name it, so nothing says what a resume there would carry.
        auto mod = parse_or_fail(R"(module @osr_resume_points
func @f(%n: i64) -> i64 {
bb0:
  %z = iconst.i64 0
  br bb1(%z)
bb1(%i: i64):
  %big = iconst.i64 1000000
  %ok = slt %i, %big
  guard %ok, @slow, [%i], id 0
  %one = iconst.i64 1
  %j = add %i, %one
  %c = slt %j, %n
  br_if %c, bb1(%j), bb2(%j)
bb2(%r: i64):
  ret %r
bres(%x: i64):
  ret %x

resume_table {
  entry 5 -> bres
}
}
)");
        const Function& fn = *mod->get_function("f");
        why.clear();
        CHECK(!plan_osr_entry(fn, *block_named(fn, "bb1"), &why).has_value());
        CHECK(!why.empty());
    }
    {
        auto mod = parse_or_fail(R"(module @osr_resume_gcref
extern @brass_gc_alloc
func @f(%n: i64) -> i64 {
bb0:
  %sz = iconst.i64 16
  %m = iconst.i64 0
  %t = iconst.i32 1
  %o = call.gcref @brass_gc_alloc(%sz, %m, %t)
  %z = iconst.i64 0
  br bb1(%z)
bb1(%i: i64):
  store.i64 %o, 0, %i
  %one = iconst.i64 1
  %j = add %i, %one
  %c = slt %j, %n
  br_if %c, bb1(%j), bb2
bb2:
  %v = load.i64 %o, 0
  ret %v
}
)");
        const Function& fn = *mod->get_function("f");
        why.clear();
        CHECK(!plan_osr_entry(fn, *block_named(fn, "bb1"), &why).has_value());
        CHECK(!why.empty());
    }
}

TEST_CASE("OSR resume - the entry function, compiled, continues the loop from a filled buffer") {
    auto mod = parse_or_fail(kSumLoop);
    const Function& fn = *mod->get_function("osr_sum");
    const BasicBlock* loop = block_named(fn, "loop");
    auto plan = plan_osr_entry(fn, *loop);
    REQUIRE(plan.has_value());

    Module dst("osr_resume_entry");
    clone_function(*mod->get_function("osr_step"), dst);
    std::string why;
    Function* entry = build_osr_entry_function(*plan, dst, "osr_sum.osr", &why);
    if (!entry) std::cerr << why << "\n";
    REQUIRE(entry != nullptr);
    REQUIRE(verify_module(dst));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(dst));
    const Value* n_param = fn.entry_block()->param(0);
    for (int64_t start : {int64_t{0}, int64_t{1}, int64_t{37}, int64_t{99}}) {
        const int64_t n = 100;
        const int64_t acc0 = start * 1000 + 11;
        std::vector<uint64_t> buf(plan->live_ins.size(), 0xDEADBEEFDEADBEEFull);
        for (size_t i = 0; i < plan->live_ins.size(); ++i) {
            const auto& li = plan->live_ins[i];
            if (li.rematerialize) continue;
            if (li.value == loop->param(0)) buf[i] = static_cast<uint64_t>(start);
            else if (li.value == loop->param(1)) buf[i] = static_cast<uint64_t>(acc0);
            else if (li.value == n_param) buf[i] = static_cast<uint64_t>(n);
            else buf[i] = static_cast<uint64_t>(n + 7); // %k
        }
        const RuntimeValue r = jit.invoke("osr_sum.osr", {RuntimeValue::from_ptr(buf.data())});
        CHECK_EQ(r.as_i64(), sum_from(n, start, acc0));
    }
}

TEST_CASE("OSR resume - a guard failing inside OSR code finishes the call through its exit stub") {
    auto mod = parse_or_fail(kStubLoop);
    const Function& fn = *mod->get_function("sum_spec");
    const int64_t n = 20000;
    const int64_t expected = (n - 1) * n / 2;
    OsrRun plain;
    {
        DeoptStressScope off(0);
        plain = run_through_osr(fn, n, expected);
        CHECK(plain.migrations > 0);
    }
    // Every 5000th evaluation fails: in Tier 0 before the OSR code exists,
    // then inside it once it runs, where the failure leaves the OSR call.
    DeoptStressScope stress(5000);
    const OsrRun forced = run_through_osr(fn, n, expected);
    CHECK(forced.migrations > 0);
    CHECK(forced.tier2_deopts > 0);
    CHECK(deopt_stress_forced() > 0);
}

TEST_CASE("OSR resume - a callee's resume-only guard failing under OSR resumes the callee and returns") {
    auto mod = parse_or_fail(kCalleeResume);
    const Function& fn = *mod->get_function("run");
    const int64_t n = 5000;
    {
        DeoptStressScope off(0);
        const OsrRun plain = run_through_osr(fn, n, callee_expected(n));
        CHECK(plain.migrations > 0);
    }
    // Every 7th evaluation of the callee's guard fails, whichever tier runs it.
    DeoptStressScope stress(7);
    const OsrRun forced = run_through_osr(fn, n, callee_expected(n));
    CHECK(forced.migrations > 0);
    CHECK(deopt_stress_forced() > 0);
}

TEST_CASE("OSR resume - a loop in a function whose resume points are its guards' enters OSR") {
    auto mod = parse_or_fail(R"(module @osr_resume_stays
func @f(%n: i64) -> i64 {
bb0:
  %z = iconst.i64 0
  br bb1(%z, %z)
bb1(%i: i64, %acc: i64):
  %big = iconst.i64 1000000000
  %ok = slt %i, %big
  guard %ok, @slow, [%i, %acc]
  %acc2 = add %acc, %i
  %one = iconst.i64 1
  %j = add %i, %one
  %c = slt %j, %n
  br_if %c, bb1(%j, %acc2), bb2(%acc2)
bb2(%r: i64):
  ret %r
bres(%x: i64, %a: i64):
  ret %a

resume_table {
  entry 0 -> bres
}
}
)");
    const Function& fn = *mod->get_function("f");
    // Its exit is not the fast path's answer: no stress from the environment.
    DeoptStressScope off(0);
    FunctionDispatchTable prog;
    init_program(prog, 20);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    for (int run = 0; run < 3; ++run) {
        CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(3000)}).as_i64(), 2999 * 3000 / 2);
        CompilePool::shared().wait_owner(&prog.osr());
    }
    // The guard's resume point is one the OSR code's guard names, so a
    // failure there would resume at bres in Tier 0 with the guard's state.
    CHECK(prog.osr().total_osr_migrations() > 0u);
}

#endif
