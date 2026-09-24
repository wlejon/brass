// Regressions from bug sweep 22:
// - CoroTransformPass gave every value live across a suspend one 8-byte
//   frame slot, so a v128 clobbered the next slot (or wrote past the frame),
//   coro_create copied a v128 argument into one slot (dropping lanes 2-3),
//   and nothing rejected a v128 yield, which overwrote the resume field.
// - GVN/CSE treated float add/mul as commutative, but when both operands
//   are NaN the lhs NaN wins, so `a + b` and `b + a` differ.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <cstdio>
#include <memory>
#include <stdexcept>
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

enum class Tier { Interp, Fast, Baseline, Jit };
constexpr Tier kTiers[] = {Tier::Interp, Tier::Fast, Tier::Baseline, Tier::Jit};

// Resumes the lowered body @name (scalar `args` in its first slots) with
// 0, 1, ... until done; returns every resume's result.
std::vector<uint64_t> drive(Module& mod, const char* name, Tier tier, const std::vector<uint64_t>& args) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    const CoroFrameLayout layout = compute_coro_frame_layout(*fn);
    const uint32_t slots = std::max<uint32_t>(layout.slot_count, static_cast<uint32_t>(args.size()));

    std::unique_ptr<codegen::JitExecutionEngine> jit;
    codegen::BaselineCompiledFunction baseline;
    void* code = nullptr;
    if (tier == Tier::Jit) {
        jit = std::make_unique<codegen::JitExecutionEngine>();
        REQUIRE(jit->compile_and_load(mod));
        code = jit->get_symbol_address(name);
    } else if (tier == Tier::Baseline) {
        codegen::BaselineJitCompiler compiler;
        baseline = compiler.compile(*fn);
        REQUIRE(baseline.is_valid());
        code = reinterpret_cast<void*>(baseline.get_function_ptr<uint64_t (*)(uintptr_t)>());
    }
    const bool native = tier == Tier::Jit || tier == Tier::Baseline;
    if (native) REQUIRE(code != nullptr);

    uintptr_t frame = brass_coro_create_at(native ? code : nullptr, slots, layout.pointer_mask, 0, 0);
    REQUIRE(frame != 0);
    ThreadRootsScope keep([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &frame);
    auto* cf = reinterpret_cast<runtime::BrassCoroFrame*>(frame);
    for (size_t i = 0; i < args.size(); ++i) cf->slots[i] = args[i];

    Interpreter interp;
    FastInterpreter fast;
    std::vector<uint64_t> out;
    for (uint64_t step = 0; step < 8; ++step) {
        uint64_t y = 0;
        if (native) {
            y = brass_coro_resume(frame, step);
        } else {
            reinterpret_cast<runtime::BrassCoroFrame*>(frame)->resume_arg = step;
            const std::vector<RuntimeValue> a = {RuntimeValue::from_ptr(frame)};
            const RuntimeValue r = tier == Tier::Interp ? interp.run(*fn, a) : fast.run(*fn, a);
            y = static_cast<uint64_t>(r.as_i64());
        }
        out.push_back(y);
        if (reinterpret_cast<runtime::BrassCoroFrame*>(frame)->is_done) break;
    }
    brass_coro_destroy(frame);
    return out;
}

// A v128 live across a suspend, spilled next to a scalar (%k): with one
// 8-byte slot the vector's upper half overwrote %k.
const char* kV128Live = R"(module @v
func @vb(%p: i64) -> i64 {
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
)";

// A v128 coroutine argument followed by a scalar one: coro_create copied
// only the vector's low 8 bytes, so lane 3 read as 0.
const char* kV128Arg = R"(module @va
func @vb(%v: i32x4, %n: i64) -> i64 {
b0:
  %y = coro_suspend.i64 %n, 1
  %l3: i32 = vextract_lane %v, 3
  %e = sext_i64 %l3
  %r = add.i64 %e, %n
  ret %r
}
func @main(%p: i64, %q: i64) -> i64 {
b0:
  %t = trunc_i32 %p
  %v: i32x4 = vbroadcast.i32x4 %t
  %c = coro_create @vb(%v, %q)
  %z = iconst.i64 0
  %a = coro_resume.i64 %c, %z
  %b = coro_resume.i64 %c, %z
  coro_destroy %c
  ret %b
}
)";

const char* kV128Yield = R"(module @vy
func @vy(%p: i64) -> i64 {
b0:
  %t = trunc_i32 %p
  %a: i32x4 = vbroadcast.i32x4 %t
  %y = coro_suspend.i64 %a, 1
  ret %y
}
)";

// Swapped-operand float add/mul: CSE must not merge them. (Float-vector
// lanes: test_sweep23_regressions.cpp.)
const char* kSwapNan = R"(module @n1
func @fuzz_fn(%a: i64, %b: i64) -> i64 {
b0:
  %fa = bitcast.f64 %a
  %fb = bitcast.f64 %b
  %x = add.f64 %fa, %fb
  %y = add.f64 %fb, %fa
  %u = mul.f64 %fa, %fb
  %w = mul.f64 %fb, %fa
  %xi = bitcast.i64 %x
  %yi = bitcast.i64 %y
  %ui = bitcast.i64 %u
  %wi = bitcast.i64 %w
  %one = iconst.i64 1
  %two = iconst.i64 2
  %xs = shl.i64 %xi, %one
  %r0 = xor.i64 %xs, %yi
  %us = shl.i64 %ui, %two
  %r1 = xor.i64 %r0, %us
  %r2 = xor.i64 %r1, %wi
  ret %r2
}
)";

const int64_t kNans[] = {
    static_cast<int64_t>(0x7FF8000000000000ull),
    static_cast<int64_t>(0xFFF8000000000001ull),
    static_cast<int64_t>(0x7FF0000000000005ull),
    static_cast<int64_t>(0x7FFC00007FC00003ull),
    static_cast<int64_t>(0x4000000000000000ull),
};

} // namespace

TEST_CASE("Sweep22 - a v128 live across a suspend gets two frame slots") {
    auto mod = lower(kV128Live);
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("vb"));
    // %p (argument slot 0), %a (slots 1-2), %k (slot 3).
    CHECK_EQ(layout.slot_count, 4u);
    const std::vector<uint64_t> want = {5, 0x3f4};  // yields %p, then lanes (5+1) + (5+1) + 1000
    for (Tier t : kTiers) {
        const auto got = drive(*mod, "vb", t, {5});
        if (got != want) std::printf("  tier %d\n", static_cast<int>(t));
        CHECK(got == want);
    }
}

TEST_CASE("Sweep22 - coro_create copies a whole v128 argument") {
    {
        auto mod = lower(kV128Arg);
        Interpreter interp;
        interp.set_module(mod.get());
        const auto r = interp.run(*mod->get_function("main"),
                                  {RuntimeValue::from_i64(7), RuntimeValue::from_i64(100)});
        CHECK_EQ(r.as_i64(), 107);
    }
    {
        auto mod = lower(kV128Arg);
        // The vector takes slots 0-1, %n slot 2.
        const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("vb"));
        CHECK_EQ(layout.slot_count, 3u);
        codegen::JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(*mod));
        auto* main_fn = reinterpret_cast<int64_t (*)(int64_t, int64_t)>(jit.get_symbol_address("main"));
        REQUIRE(main_fn != nullptr);
        CHECK_EQ(main_fn(7, 100), 107);
        CHECK_EQ(main_fn(-3, 5), 2);
    }
}

TEST_CASE("Sweep22 - a v128 yield is rejected, not truncated") {
    auto mod = parse_module(kV128Yield);
    REQUIRE(mod != nullptr);
    DiagnosticReporter diag;
    CHECK(!verify_module(*mod, &diag));
    CoroTransformPass pass;
    bool threw = false;
    try {
        pass.run_on_module(*mod);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Sweep22 - CSE keeps swapped float add/mul apart under two NaNs") {
    auto mod = parse_ok(kSwapNan);
    fuzz::DiffFuzzerOptions opt;
    opt.tier6_baseline = true;
    opt.save_reproducers = false;
    fuzz::DiffFuzzer fz(opt);
    int failures = 0;
    for (int64_t a : kNans) {
        for (int64_t b : kNans) {
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
