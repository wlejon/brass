// Deopt stress (runtime/deopt_stress.hpp) and the cross-tier differential
// (diff_harness.hpp, assert_diff_tiers): each program runs on the reference
// interpreter, the fast interpreter, Tier 1, pipeline Tier 2 and the
// standalone JIT, then again with its guards forced to fail everywhere and
// at each site alone. The programs keep speculation's contract (a guard's
// exits compute what its fast path computes), so every stressed answer is
// also the unstressed one: each state map the stress walks through, typed
// values, spilled ones, a gcref across a collection, is checked by value.

#include "diff_harness.hpp"
#include "gc_test_heap.hpp"
#include <brass/mir/deopt_stress.hpp>
#include <brass/mir/parser.hpp>
#include <memory>
#include <string_view>

using namespace brass;
using namespace brass::test;

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__)

namespace {

std::unique_ptr<Module> parse_or_fail(std::string_view src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

std::vector<RuntimeValue> i64s(std::initializer_list<int64_t> v) {
    std::vector<RuntimeValue> out;
    for (int64_t x : v) out.push_back(RuntimeValue::from_i64(x));
    return out;
}

TierDiffOptions contract() {
    TierDiffOptions o;
    o.exits_equivalent = true;
    return o;
}

// A loop speculating that its accumulator stays small; the exit stub
// finishes the loop generically from the guard's state (an f64 among it).
constexpr std::string_view kLoopStub = R"(module @ds_loop_stub
func @sum_rest(%i: i64, %acc: i64, %n: i64, %w: f64) -> i64 {
entry:
  br loop(%i, %acc)
loop(%j: i64, %a: i64):
  %c = slt %j, %n
  br_if %c, body, done(%a)
body:
  %wi = fptosi.i64 %w
  %t = mul %j, %wi
  %a2 = add %a, %t
  %one = iconst.i64 1
  %j2 = add %j, %one
  br loop(%j2, %a2)
done(%r: i64):
  ret %r
}

func @sum_spec(%n: i64, %lim: i64) -> i64 {
entry:
  %w = fconst.f64 3.0
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, body, done(%acc)
body:
  %ok = slt %acc, %lim
  guard %ok, @sum_rest, [%i, %acc, %n, %w]
  %wi = fptosi.i64 %w
  %t = mul %i, %wi
  %a2 = add %acc, %t
  %one = iconst.i64 1
  %i2 = add %i, %one
  br loop(%i2, %a2)
done(%r: i64):
  ret %r
}
)";

// A guard whose only exit is a resume block of its own function that does
// the iteration the slow way and rejoins the loop. Its state carries every
// value the code from the resume block on reads (%n and %lim besides the
// block's parameters): the verifier requires it, since a lower tier
// resuming there has nothing else.
constexpr std::string_view kLoopResume = R"(module @ds_loop_resume
func @f(%n: i64, %lim: i64) -> i64 {
bb0:
  %z = iconst.i64 0
  br bb1(%z, %z)
bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)
bb2:
  %ok = slt %i, %lim
  guard %ok, @slow, [%i, %acc, %n, %lim]
  %sq = mul %i, %i
  %acc2 = add %acc, %sq
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)
bb3(%r: i64):
  ret %r
bres(%j: i64, %a: i64):
  %z2 = iconst.i64 0
  br slow_loop(%z2, %a)
slow_loop(%k: i64, %s: i64):
  %more = slt %k, %j
  br_if %more, slow_body, slow_done(%s)
slow_body:
  %s2 = add %s, %j
  %one2 = iconst.i64 1
  %k2 = add %k, %one2
  br slow_loop(%k2, %s2)
slow_done(%res: i64):
  %one3 = iconst.i64 1
  %jn = add %j, %one3
  br bb1(%jn, %res)

resume_table {
  entry 0 -> bres
}
}
)";

// Twelve state values of four types, more than both x64 conventions pass
// in registers, most of them live across the guard: the fast path and the
// exit stub compute one formula from them.
constexpr std::string_view kManyValues = R"(module @ds_many
func @formula(%a: i64, %b: f64, %k: i32, %d: i64, %e: f64, %f: i64, %g: i64, %m: i32, %n: i64, %h: f32, %p: i64, %q: i64) -> i64 {
bb0:
  %c10 = iconst.i64 10
  %c100 = iconst.i64 100
  %c1k = iconst.i64 1000
  %c10k = iconst.i64 10000
  %d10 = mul %d, %c10
  %f100 = mul %f, %c100
  %g1k = mul %g, %c1k
  %n10k = mul %n, %c10k
  %kx = sext.i64 %k
  %mx = sext.i64 %m
  %km = mul %kx, %mx
  %be = add %b, %e
  %hd = fpext.f64.f32 %h
  %beh = add %be, %hd
  %four = fconst.f64 4.0
  %be4 = mul %beh, %four
  %bei = fptosi.i64 %be4
  %pq = mul %p, %q
  %s1 = add %a, %d10
  %s2 = add %s1, %f100
  %s3 = add %s2, %g1k
  %s4 = add %s3, %n10k
  %s5 = add %s4, %km
  %s6 = add %s5, %bei
  %s7 = add %s6, %pq
  ret %s7
}

func @spec(%x: i64, %c: i32) -> i64 {
bb0:
  %one = iconst.i64 1
  %a = add %x, %one
  %xf = sitofp.f64 %x
  %half = fconst.f64 0.5
  %b = mul %xf, %half
  %k = iconst.i32 7
  %three = iconst.i64 3
  %d = mul %x, %three
  %e = fconst.f64 2.25
  %f = sub %x, %one
  %i5 = iconst.i64 5
  %g = add %x, %i5
  %m = iconst.i32 -2
  %i9 = iconst.i64 9
  %n = mul %x, %i9
  %h = fconst.f32 1.5
  %p = add %x, %three
  %q = sub %x, %three
  guard %c, @formula, [%a, %b, %k, %d, %e, %f, %g, %m, %n, %h, %p, %q]
  %r = call.i64 @formula(%a, %b, %k, %d, %e, %f, %g, %m, %n, %h, %p, %q)
  ret %r
}
)";

// A guarded callee in a loop: its stub returns to the caller's loop.
constexpr std::string_view kCallee = R"(module @ds_callee
func @clamp_slow(%x: i64, %hi: i64) -> i64 {
bb0:
  %c = slt %x, %hi
  %r = select %c, %x, %hi
  ret %r
}

func @clamp(%x: i64, %hi: i64) -> i64 {
bb0:
  %ok = slt %x, %hi
  guard %ok, @clamp_slow, [%x, %hi]
  ret %x
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
  %cl = call.i64 @clamp(%v, %hi)
  %acc2 = add %acc, %cl
  %one = iconst.i64 1
  %i2 = add %i, %one
  br loop(%i2, %acc2)
done(%r: i64):
  ret %r
}
)";

// A gcref in the state map, with allocation between guards: a collection
// may move the object before the exit reads it.
constexpr std::string_view kGcRef = R"(module @ds_gcref
extern @brass_gc_alloc

func @box_rest(%o: gcref, %acc: i64, %i: i64, %n: i64) -> i64 {
bb0:
  %v = load.i64 %o, 0
  %r = add %v, %acc
  %one = iconst.i64 1
  %i1 = add %i, %one
  br loop(%i1, %r)
loop(%j: i64, %a: i64):
  %c = slt %j, %n
  br_if %c, body, done(%a)
body:
  %three = iconst.i64 3
  %t = mul %j, %three
  %a2 = add %a, %t
  %one2 = iconst.i64 1
  %j2 = add %j, %one2
  br loop(%j2, %a2)
done(%res: i64):
  ret %res
}

func @boxed(%n: i64) -> i64 {
bb0:
  %sz = iconst.i64 16
  %mask = iconst.i64 0
  %tag = iconst.i32 1
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, body, done(%acc)
body:
  %o = call.gcref @brass_gc_alloc(%sz, %mask, %tag)
  %three = iconst.i64 3
  %v = mul %i, %three
  store.i64 %o, 0, %v
  %big = iconst.i64 1000000000
  %ok = slt %i, %big
  guard %ok, @box_rest, [%o, %acc, %i, %n]
  %x = load.i64 %o, 0
  %r = add %x, %acc
  %one = iconst.i64 1
  %i2 = add %i, %one
  br loop(%i2, %r)
done(%res: i64):
  ret %res
}
)";

} // namespace

TEST_CASE("Deopt stress - configuration, counting and site selection") {
    runtime::DeoptStressScope off(0);
    CHECK(!runtime::deopt_stress_active());
    CHECK(!runtime::deopt_stress_should_fail(0));
    CHECK_EQ(runtime::deopt_stress_evaluations(), 0u);
    {
        runtime::DeoptStressScope every3(3);
        CHECK(runtime::deopt_stress_active());
        int fails = 0;
        for (int i = 0; i < 9; ++i) fails += runtime::deopt_stress_should_fail(static_cast<uint32_t>(i)) ? 1 : 0;
        CHECK_EQ(fails, 3);
        CHECK_EQ(runtime::deopt_stress_evaluations(), 9u);
        CHECK_EQ(runtime::deopt_stress_forced(), 3u);
    }
    {
        runtime::DeoptStressScope site(1, 2);
        CHECK(!runtime::deopt_stress_should_fail(1));
        CHECK(runtime::deopt_stress_should_fail(2));
        CHECK_EQ(runtime::deopt_stress_evaluations(), 1u);
    }
    CHECK(!runtime::deopt_stress_active());
    CHECK_EQ(runtime::deopt_stress_forced(), 0u);
}

TEST_CASE("Deopt stress - tier 2 gets one stress guard per eligible guard, with the real guard's exits") {
    auto mod = parse_or_fail(kLoopResume);
    Function& fn = *mod->get_function("f");
    uint64_t evals = 0, forced = 0;
    DeoptStressPlan plan;
    plan.period = 1;
    plan.evaluations = &evals;
    plan.forced = &forced;
    // Standalone: a resume-only guard has nowhere to go.
    plan.resume_targets = false;
    CHECK(!has_deopt_stress_guards(fn, mod.get(), plan));
    plan.resume_targets = true;
    CHECK_EQ(insert_deopt_stress_guards(fn, mod.get(), plan), 1u);
    int guards = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() != Opcode::guard) continue;
            ++guards;
            CHECK_EQ(inst->resume_id(), 0u);
            CHECK_EQ(inst->state_map().size(), 4u);
        }
    }
    CHECK_EQ(guards, 2);
}

TEST_CASE("Deopt stress - a loop's exit stub finishes it on every tier") {
    auto mod = parse_or_fail(kLoopStub);
    assert_diff_tiers(*mod, "sum_spec", i64s({200, 1000000000}), contract());
    // The guard failing for real, part way.
    assert_diff_tiers(*mod, "sum_spec", i64s({200, 5000}), contract());
}

TEST_CASE("Deopt stress - a resume block rejoins the loop on every tier") {
    auto mod = parse_or_fail(kLoopResume);
    assert_diff_tiers(*mod, "f", i64s({60, 1000}), contract());
    // Failing for real: standalone tier 2 has no lower tier to resume in.
    TierDiffOptions o = contract();
    o.jit = false;
    assert_diff_tiers(*mod, "f", i64s({60, 25}), o);
}

TEST_CASE("Deopt stress - the verifier rejects a resume block reading a value the guard does not carry") {
    // kLoopResume with %n and %lim left out of the state: Tier 0 resuming
    // at bres after a tier-2 deopt read them as 0 and ended the loop early.
    std::string src(kLoopResume);
    const std::string full = "[%i, %acc, %n, %lim]";
    src.replace(src.find(full), full.size(), "[%i, %acc]");
    auto mod = parse_or_fail(src);
    DiagnosticReporter diag;
    CHECK(!verify_module(*mod, &diag));
    CHECK(diag.format_all().find("resumes at 'bres'") != std::string::npos);
}

TEST_CASE("Deopt stress - twelve state values of four types reach the exit stub intact") {
    auto mod = parse_or_fail(kManyValues);
    for (int64_t x : {int64_t{0}, int64_t{17}, int64_t{-40}, int64_t{123456}}) {
        std::vector<RuntimeValue> args{RuntimeValue::from_i64(x), RuntimeValue::from_i32(1)};
        assert_diff_tiers(*mod, "spec", args, contract());
    }
}

TEST_CASE("Deopt stress - a guarded callee's exit returns into its caller's loop") {
    auto mod = parse_or_fail(kCallee);
    assert_diff_tiers(*mod, "run", i64s({100}), contract());
}

TEST_CASE("Deopt stress - a gcref state value survives collections into its exit") {
    auto mod = parse_or_fail(kGcRef);
    BoundHeap heap;
    heap->set_stress(gc::StressMode::Minor);
    // The standalone JIT has no heap of its program: the bound one serves.
    assert_diff_tiers(*mod, "boxed", i64s({300}), contract());
    CHECK(heap.minor_collections() > 0);
}

TEST_CASE("Deopt stress - the fuzz speculation shape exits where the interpreter does on every tier") {
    // Exit stubs that do not compute the fast path's answer: stress at one
    // site or all of them still agrees across tiers, not with the
    // unstressed answer.
    auto mod = parse_or_fail(R"(module @ds_nonequiv
func @exit0(%a: i64, %b: i64) -> i64 {
bb0:
  %r = add %a, %b
  ret %r
}
func @exit1(%a: i64, %b: i64) -> i64 {
bb0:
  %ten = iconst.i64 10
  %x = add %a, %ten
  %r = mul %x, %b
  ret %r
}
func @spec(%a: i64, %b: i64, %t: i32) -> i64 {
bb0:
  guard %t, @exit0, [%a, %b]
  %lim = iconst.i64 1000
  %ok = slt %a, %lim
  guard %ok, @exit1, [%a, %b]
  %r = mul %a, %b
  ret %r
}
)");
    std::vector<RuntimeValue> args{RuntimeValue::from_i64(12), RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)};
    assert_diff_tiers(*mod, "spec", args);
}

#endif
