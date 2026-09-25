// Regressions from bug sweep 23:
// - The reference interpreter's (and so the FastInterpreter's) float-vector
//   add/sub/mul/div used C++ operators per lane, which do not pin the NaN a
//   two-NaN lane returns; the JIT tiers return the lhs NaN (float_arith.hpp).
// - FastInterpreter's own coro_create put a lowered coroutine body's
//   arguments into its registers, so the body's frame parameter held the
//   first argument; coro_resume ran the body as an unlowered one and
//   coro_destroy left the frame registered. Lowered bodies now get the frame
//   layout every other tier uses.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/interpreter/float_arith.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include "gc_test_heap.hpp"
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <brass/gc/native_frames.hpp>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

std::unique_ptr<Module> lower(const char* src) {
    auto mod = parse_ok(src);
    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(*mod));
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::printf("%s", vdiag.format_all().c_str());
    REQUIRE(ok);
    return mod;
}

// ---------------------------------------------------------------------------
// Float-vector NaN lanes
// ---------------------------------------------------------------------------

// @f_<T>(%p): a = [p], b = [p+32]; [p+64] = a+b, [p+96] = a-b,
// [p+128] = a*b, [p+160] = a/b, [p+192] = b*a (the swapped product).
const char* kVecLanes = R"(module @vl
func @f_f32x4(%p: ptr) -> i64 {
b0:
  %a: f32x4 = vload.f32x4 %p, 0
  %b: f32x4 = vload.f32x4 %p, 32
  %s: f32x4 = vadd %a, %b
  %d: f32x4 = vsub %a, %b
  %m: f32x4 = vmul %a, %b
  %q: f32x4 = vdiv %a, %b
  %w: f32x4 = vmul %b, %a
  vstore.f32x4 %p, 64, %s
  vstore.f32x4 %p, 96, %d
  vstore.f32x4 %p, 128, %m
  vstore.f32x4 %p, 160, %q
  vstore.f32x4 %p, 192, %w
  %z = iconst.i64 0
  ret %z
}
func @f_f64x2(%p: ptr) -> i64 {
b0:
  %a: f64x2 = vload.f64x2 %p, 0
  %b: f64x2 = vload.f64x2 %p, 32
  %s: f64x2 = vadd %a, %b
  %d: f64x2 = vsub %a, %b
  %m: f64x2 = vmul %a, %b
  %q: f64x2 = vdiv %a, %b
  %w: f64x2 = vmul %b, %a
  vstore.f64x2 %p, 64, %s
  vstore.f64x2 %p, 96, %d
  vstore.f64x2 %p, 128, %m
  vstore.f64x2 %p, 160, %q
  vstore.f64x2 %p, 192, %w
  %z = iconst.i64 0
  ret %z
}
func @f_f32x8(%p: ptr) -> i64 {
b0:
  %a: f32x8 = vload.f32x8 %p, 0
  %b: f32x8 = vload.f32x8 %p, 32
  %s: f32x8 = vadd %a, %b
  %d: f32x8 = vsub %a, %b
  %m: f32x8 = vmul %a, %b
  %q: f32x8 = vdiv %a, %b
  %w: f32x8 = vmul %b, %a
  vstore.f32x8 %p, 64, %s
  vstore.f32x8 %p, 96, %d
  vstore.f32x8 %p, 128, %m
  vstore.f32x8 %p, 160, %q
  vstore.f32x8 %p, 192, %w
  %z = iconst.i64 0
  ret %z
}
func @f_f64x4(%p: ptr) -> i64 {
b0:
  %a: f64x4 = vload.f64x4 %p, 0
  %b: f64x4 = vload.f64x4 %p, 32
  %s: f64x4 = vadd %a, %b
  %d: f64x4 = vsub %a, %b
  %m: f64x4 = vmul %a, %b
  %q: f64x4 = vdiv %a, %b
  %w: f64x4 = vmul %b, %a
  vstore.f64x4 %p, 64, %s
  vstore.f64x4 %p, 96, %d
  vstore.f64x4 %p, 128, %m
  vstore.f64x4 %p, 160, %q
  vstore.f64x4 %p, 192, %w
  %z = iconst.i64 0
  ret %z
}
)";

enum class VTier { Interp, Fast, Baseline, Jit };

// Lane patterns: two distinct NaNs (both signs, signalling and quiet), a
// NaN against a number each way round, and two numbers.
const uint64_t kLhs64[4] = {0x7FF0000000000005ull, 0xFFF8000000000001ull, 0x4000000000000000ull, 0x7FFC00007FC00003ull};
const uint64_t kRhs64[4] = {0xFFF4000000000009ull, 0x7FF8000000000002ull, 0xFFF0000000000077ull, 0x3FF8000000000000ull};
const uint32_t kLhs32[8] = {0x7F800005u, 0xFFC00001u, 0x40000000u, 0x7FE00003u,
                            0xFF800011u, 0x3F800000u, 0x7FC00000u, 0x40400000u};
const uint32_t kRhs32[8] = {0xFFA00009u, 0x7FC00002u, 0xFF800077u, 0x3FC00000u,
                            0x7F800021u, 0xFFC00005u, 0x7FC00001u, 0x40800000u};

template <typename T, typename Bits>
void expect_lanes(const uint8_t* buf, const Bits* lhs, const Bits* rhs, size_t lanes, const char* what) {
    int bad = 0;
    for (size_t i = 0; i < lanes; ++i) {
        const T a = std::bit_cast<T>(lhs[i]);
        const T b = std::bit_cast<T>(rhs[i]);
        const T want[5] = {fparith::add(a, b), fparith::sub(a, b), fparith::mul(a, b), fparith::div(a, b),
                           fparith::mul(b, a)};
        for (int op = 0; op < 5; ++op) {
            Bits got;
            std::memcpy(&got, buf + 64 + op * 32 + i * sizeof(T), sizeof(T));
            if (got != std::bit_cast<Bits>(want[op])) {
                ++bad;
                std::printf("  %s op %d lane %zu: got %llx want %llx\n", what, op, i,
                            static_cast<unsigned long long>(got),
                            static_cast<unsigned long long>(std::bit_cast<Bits>(want[op])));
            }
        }
    }
    CHECK_EQ(bad, 0);
}

void run_vec_lanes(Module& mod, const char* name, VTier tier, uint8_t* buf) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    const std::vector<RuntimeValue> args = {RuntimeValue::from_ptr(buf)};
    switch (tier) {
        case VTier::Interp: {
            Interpreter interp;
            interp.set_module(&mod);
            interp.run(*fn, args);
            break;
        }
        case VTier::Fast: {
            FastInterpreter fast;
            fast.run(*fn, args);
            break;
        }
        case VTier::Baseline: {
            codegen::BaselineJitCompiler compiler;
            auto compiled = compiler.compile(*fn);
            REQUIRE(compiled.is_valid());
            compiled.get_function_ptr<int64_t (*)(uint8_t*)>()(buf);
            break;
        }
        case VTier::Jit: {
            codegen::JitExecutionEngine jit;
            REQUIRE(jit.compile_and_load(mod));
            auto f = reinterpret_cast<int64_t (*)(uint8_t*)>(jit.get_symbol_address(name));
            REQUIRE(f != nullptr);
            f(buf);
            break;
        }
    }
}

const char* vtier_name(VTier t) {
    switch (t) {
        case VTier::Interp: return "interpreter";
        case VTier::Fast: return "fast interpreter";
        case VTier::Baseline: return "baseline JIT";
        case VTier::Jit: return "optimizing JIT";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// FastInterpreter coroutines
// ---------------------------------------------------------------------------

// Each body with a caller that creates it from MIR and resumes it four
// times (0..3; a done coroutine returns its result again), folding the
// results into one number.
const char* kCoroShapes = R"(module @cs
func @pc(%p: i64) -> i64 {
b0:
  %k = iconst.i64 100
  br mid
mid:
  %y = coro_suspend.i64 %p, 1
  %s = add.i64 %p, %k
  %t = add.i64 %s, %y
  ret %t
}
func @lp(%p: i64) -> i64 {
b0:
  %k = iconst.i64 100
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %a1 = add.i64 %acc, %p
  %a2 = add.i64 %a1, %k
  %y = coro_suspend.i64 %a2, 1
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %lim = iconst.i64 3
  %more = slt.i64 %ni, %lim
  br_if %more, loop(%ni, %a2), done(%a2, %k)
done(%r: i64, %kk: i64):
  %s = add.i64 %r, %kk
  ret %s
}
func @ts(%p: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %p)
loop(%i: i64, %acc: i64):
  %y1 = coro_suspend.i64 %acc, 1
  %b = add.i64 %acc, %y1
  %y2 = coro_suspend.i64 %b, 2
  %c = add.i64 %b, %y2
  %c2 = add.i64 %c, %p
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %lim = iconst.i64 2
  %more = slt.i64 %ni, %lim
  br_if %more, loop(%ni, %c2), done
done:
  ret %c2
}
func @vl(%p: i64) -> i64 {
b0:
  %t = trunc_i32 %p
  %a: i32x4 = vbroadcast.i32x4 %t
  %k = iconst.i64 1000
  %y = coro_suspend.i64 %p, 1
  %y32 = trunc_i32 %y
  %c: i32x4 = vbroadcast.i32x4 %y32
  %s: i32x4 = vadd %a, %c
  %l0: i32 = vextract_lane %s, 0
  %l3: i32 = vextract_lane %s, 3
  %m = add.i32 %l0, %l3
  %me = sext_i64 %m
  %r = add.i64 %me, %k
  ret %r
}
func @va(%v: i32x4, %n: i64) -> i64 {
b0:
  %y = coro_suspend.i64 %n, 1
  %l3: i32 = vextract_lane %v, 3
  %e = sext_i64 %l3
  %r = add.i64 %e, %n
  ret %r
}
func @drive_pc(%p: i64) -> i64 {
b0:
  %c = coro_create @pc(%p)
  %r = call.i64 @fold(%c)
  ret %r
}
func @drive_lp(%p: i64) -> i64 {
b0:
  %c = coro_create @lp(%p)
  %r = call.i64 @fold(%c)
  ret %r
}
func @drive_ts(%p: i64) -> i64 {
b0:
  %c = coro_create @ts(%p)
  %r = call.i64 @fold(%c)
  ret %r
}
func @drive_vl(%p: i64) -> i64 {
b0:
  %c = coro_create @vl(%p)
  %r = call.i64 @fold(%c)
  ret %r
}
func @drive_va(%p: i64) -> i64 {
b0:
  %t = trunc_i32 %p
  %v: i32x4 = vbroadcast.i32x4 %t
  %n = iconst.i64 100
  %c = coro_create @va(%v, %n)
  %r = call.i64 @fold(%c)
  ret %r
}
func @fold(%c: ptr) -> i64 {
b0:
  %z = iconst.i64 0
  %one = iconst.i64 1
  %two = iconst.i64 2
  %three = iconst.i64 3
  %m = iconst.i64 10007
  %r0 = coro_resume.i64 %c, %z
  %r1 = coro_resume.i64 %c, %one
  %r2 = coro_resume.i64 %c, %two
  %r3 = coro_resume.i64 %c, %three
  coro_destroy %c
  %h0 = mul.i64 %r0, %m
  %h1 = add.i64 %h0, %r1
  %h2 = mul.i64 %h1, %m
  %h3 = add.i64 %h2, %r2
  %h4 = mul.i64 %h3, %m
  %h5 = add.i64 %h4, %r3
  ret %h5
}
)";

// A gcref argument and a loop-carried gcref, both live across suspends
// (pointer-mask slots), created from MIR.
const char* kGc = R"(module @gc
func @gcb(%o: gcref, %n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %o)
loop(%i: i64, %cur: gcref):
  %v = load.i64 %cur, 16
  %y = coro_suspend.i64 %v, 1
  %one = iconst.i64 1
  %v2 = add.i64 %v, %one
  store.i64 %cur, 16, %v2
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %cur), done
done:
  %w = load.i64 %o, 16
  ret %w
}
func @make(%o: gcref) -> i64 {
b0:
  %n = iconst.i64 3
  %c = coro_create @gcb(%o, %n)
  %z = iconst.i64 0
  %r0 = coro_resume.i64 %c, %z
  %r1 = coro_resume.i64 %c, %z
  %r2 = coro_resume.i64 %c, %z
  %r3 = coro_resume.i64 %c, %z
  coro_destroy %c
  %m = iconst.i64 100
  %t0 = mul.i64 %r0, %m
  %t1 = add.i64 %t0, %r1
  %t2 = mul.i64 %t1, %m
  %t3 = add.i64 %t2, %r2
  %t4 = mul.i64 %t3, %m
  %t5 = add.i64 %t4, %r3
  ret %t5
}
)";

using GenHeap = test::BoundHeap;

// Resumes a FastInterpreter-created coroutine with 0, 1, ... until done.
std::vector<uint64_t> fast_yields(FastInterpreter& fast, uintptr_t h) {
    std::vector<uint64_t> out;
    for (uint64_t step = 0; step < 16 && !fast.coro_is_done(h); ++step) out.push_back(fast.coro_resume(h, step));
    return out;
}

} // namespace

TEST_CASE("Sweep23 - float-vector lanes follow the lhs-NaN rule on every tier") {
    auto mod = parse_ok(kVecLanes);
    struct Case { const char* fn; bool f32; size_t lanes; bool v256; };
    const Case cases[] = {
        {"f_f32x4", true, 4, false}, {"f_f64x2", false, 2, false},
        {"f_f32x8", true, 8, true},  {"f_f64x4", false, 4, true},
    };
    const VTier tiers[] = {VTier::Interp, VTier::Fast, VTier::Baseline, VTier::Jit};
    for (const Case& c : cases) {
        for (VTier t : tiers) {
            // The baseline tier compiles v128 code only.
            if (t == VTier::Baseline && c.v256) continue;
            alignas(32) uint8_t buf[256] = {};
            if (c.f32) {
                std::memcpy(buf, kLhs32, c.lanes * 4);
                std::memcpy(buf + 32, kRhs32, c.lanes * 4);
            } else {
                std::memcpy(buf, kLhs64, c.lanes * 8);
                std::memcpy(buf + 32, kRhs64, c.lanes * 8);
            }
            run_vec_lanes(*mod, c.fn, t, buf);
            if (c.f32) {
                expect_lanes<float, uint32_t>(buf, kLhs32, kRhs32, c.lanes, vtier_name(t));
            } else {
                expect_lanes<double, uint64_t>(buf, kLhs64, kRhs64, c.lanes, vtier_name(t));
            }
        }
    }
}

// n5 from sweep 22: vmul a,b and vmul b,a of two NaN f64x2 lanes, through
// every fuzzer tier including the optimized ones.
TEST_CASE("Sweep23 - swapped f64x2 vector ops agree across tiers under two NaNs") {
    auto mod = parse_ok(R"(module @n5
func @fuzz_fn(%a: i64, %b: i64) -> i64 {
b0:
  %fa = bitcast.f64 %a
  %fb = bitcast.f64 %b
  %va: f64x2 = vbroadcast.f64x2 %fa
  %vb: f64x2 = vbroadcast.f64x2 %fb
  %vu: f64x2 = vmul %va, %vb
  %vw: f64x2 = vmul %vb, %va
  %vs: f64x2 = vadd %vb, %va
  %vd: f64x2 = vdiv %vb, %va
  %ve: f64x2 = vsub %va, %vb
  %eu: f64 = vextract_lane %vu, 0
  %ew: f64 = vextract_lane %vw, 1
  %es: f64 = vextract_lane %vs, 0
  %ed: f64 = vextract_lane %vd, 1
  %ee: f64 = vextract_lane %ve, 0
  %eui = bitcast.i64 %eu
  %ewi = bitcast.i64 %ew
  %esi = bitcast.i64 %es
  %edi = bitcast.i64 %ed
  %eei = bitcast.i64 %ee
  %one = iconst.i64 1
  %fs = shl.i64 %eui, %one
  %r0 = xor.i64 %fs, %ewi
  %r1 = shl.i64 %r0, %one
  %r2 = xor.i64 %r1, %esi
  %r3 = shl.i64 %r2, %one
  %r4 = xor.i64 %r3, %edi
  %r5 = shl.i64 %r4, %one
  %r6 = xor.i64 %r5, %eei
  ret %r6
}
)");
    const int64_t nans[] = {
        static_cast<int64_t>(0x7FF8000000000000ull),
        static_cast<int64_t>(0xFFF8000000000001ull),
        static_cast<int64_t>(0x7FF0000000000005ull),
        static_cast<int64_t>(0x7FFC00007FC00003ull),
        static_cast<int64_t>(0x4000000000000000ull),
    };
    fuzz::DiffFuzzerOptions opt;
    opt.tier6_baseline = true;
    opt.save_reproducers = false;
    fuzz::DiffFuzzer fz(opt);
    int failures = 0;
    for (int64_t a : nans) {
        for (int64_t b : nans) {
            std::vector<RuntimeValue> args{RuntimeValue::from_i64(a), RuntimeValue::from_i64(b)};
            auto r = fz.run_test(*mod, "fuzz_fn", args, 0);
            if (!r.passed) {
                ++failures;
                std::printf("a=%llx b=%llx: %s (%s)\n", static_cast<unsigned long long>(a),
                            static_cast<unsigned long long>(b), r.mismatch_reason.c_str(),
                            r.failing_pass.c_str());
            }
        }
    }
    CHECK_EQ(failures, 0);
}

TEST_CASE("Sweep23 - FastInterpreter's coro_create runs lowered bodies from their frame") {
    auto mod = lower(kCoroShapes);
    struct Want { const char* fn; std::vector<RuntimeValue> args; std::vector<uint64_t> yields; };
    const std::vector<Want> wants = {
        {"pc", {RuntimeValue::from_i64(5)}, {5, 106}},
        {"lp", {RuntimeValue::from_i64(5)}, {105, 210, 315, 415}},
        {"vl", {RuntimeValue::from_i64(5)}, {5, 0x3f4}},
        // A whole v128 argument, then a scalar in the slot after it.
        {"va", {RuntimeValue::from_i32x4(7, 8, 9, 10), RuntimeValue::from_i64(100)}, {100, 110}},
    };
    for (const Want& w : wants) {
        FastInterpreter fast;
        const uintptr_t h = fast.coro_create(*mod, w.fn, w.args);
        REQUIRE(h != 0);
        CHECK(runtime::is_active_coro_frame(h));
        const auto got = fast_yields(fast, h);
        if (got != w.yields) std::printf("  @%s\n", w.fn);
        CHECK(got == w.yields);
        CHECK(fast.coro_is_done(h));
        // A done coroutine returns its result again.
        CHECK_EQ(fast.coro_resume(h, 9), w.yields.back());
        fast.coro_destroy(h);
        CHECK(!runtime::is_active_coro_frame(h));
    }
    // Destroyed while suspended: the frame is no longer a root.
    {
        FastInterpreter fast;
        const uintptr_t h = fast.coro_create(*mod, "lp", {RuntimeValue::from_i64(1)});
        CHECK_EQ(fast.coro_resume(h, 0), 101u);
        CHECK(runtime::is_active_coro_frame(h));
        fast.coro_destroy(h);
        CHECK(!runtime::is_active_coro_frame(h));
        CHECK(fast.coro_is_done(h));
    }
}

TEST_CASE("Sweep23 - MIR coro_create/resume/destroy agree on the interpreter, FastInterpreter and JIT") {
    auto mod = lower(kCoroShapes);
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    const char* drivers[] = {"drive_pc", "drive_lp", "drive_ts", "drive_vl", "drive_va"};
    for (const char* d : drivers) {
        for (int64_t p : {5, -3}) {
            Interpreter interp;
            interp.set_module(mod.get());
            const int64_t ref = interp.run(*mod->get_function(d), {RuntimeValue::from_i64(p)}).as_i64();
            FastInterpreter fast;
            const int64_t fr = fast.run(*mod->get_function(d), {RuntimeValue::from_i64(p)}).as_i64();
            auto jf = reinterpret_cast<int64_t (*)(int64_t)>(jit.get_symbol_address(d));
            REQUIRE(jf != nullptr);
            const int64_t jr = jf(p);
            if (fr != ref || jr != ref) {
                std::printf("  @%s(%lld): interp %lld fast %lld jit %lld\n", d, static_cast<long long>(p),
                            static_cast<long long>(ref), static_cast<long long>(fr), static_cast<long long>(jr));
            }
            CHECK_EQ(fr, ref);
            CHECK_EQ(jr, ref);
        }
    }
}

TEST_CASE("Sweep23 - FastInterpreter coroutine with gcref slots created from MIR") {
    auto mod = lower(kGc);
    GenHeap heap;
    uintptr_t obj = heap->allocate_masked(48, 0, 2);
    REQUIRE(obj != 0);
    *reinterpret_cast<int64_t*>(obj + 16) = 40;
    // A root: the frames' allocations may move it (BRASS_GC_STRESS).
    ThreadRootsScope keep_obj([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &obj);
    FastInterpreter fast;
    CHECK_EQ(fast.run(*mod->get_function("make"), {RuntimeValue::from_gcref(obj)}).as_i64(), 40414243);
    // Through the host API: the gcref argument lands in the frame's pointer slot.
    *reinterpret_cast<int64_t*>(obj + 16) = 40;
    const uintptr_t h = fast.coro_create(*mod, "gcb", {RuntimeValue::from_gcref(obj), RuntimeValue::from_i64(3)});
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("gcb"));
    CHECK((layout.pointer_mask & 1u) != 0);
    CHECK_EQ(reinterpret_cast<runtime::BrassCoroFrame*>(h)->slots[0], obj);
    CHECK((fast_yields(fast, h) == std::vector<uint64_t>{40, 41, 42, 43}));
    fast.coro_destroy(h);
}
