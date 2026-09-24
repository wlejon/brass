// Regressions from bug sweep 21: every collection unwound the host's whole
// C++ stack beneath the outermost generated frame, the reference
// interpreter returned the rhs NaN of a two-NaN add where every other tier
// returns the lhs one, and brass_coro_resume's writes after a body that
// moved its frame had no test that saw them.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/gc/stack_walker.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/interpreter/float_arith.hpp>
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#if defined(_MSC_VER)
#define SWEEP21_NOINLINE __declspec(noinline)
#else
#define SWEEP21_NOINLINE __attribute__((noinline))
#endif

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

using I64Fn = int64_t (*)(int64_t);

struct GenHeap {
    GenerationalGC gc{32 * 1024, 16 * 1024, 1 << 20, 2};
    GenHeap() { brass_set_active_generational_gc(&gc); }
    ~GenHeap() { brass_set_active_generational_gc(nullptr); }
};

// ---------------------------------------------------------------------------
// 1. Walk cost against host stack depth.

// @g holds %o across n allocations; @f holds one across a call to the host
// function host_cb, which calls @g back.
const char* kWalk = R"(module @w
func @g(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  br loop(%z)
loop(%i: i64):
  %t = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %v = load.i64 %o, 16
  ret %v
}
func @f(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %r = call.i64 @host_cb(%n)
  %v = load.i64 %o, 16
  %s = add.i64 %v, %r
  ret %s
}
)";

enum class Entry { RawWithScope, EngineInvoke };

I64Fn g_g = nullptr;
codegen::JitExecutionEngine* g_jit = nullptr;

extern "C" SWEEP21_NOINLINE int64_t sweep21_host_cb(int64_t n) {
    // The generated @f lies beneath this frame: the scope's memo must find it.
    GeneratedCodeEntryScope entry;
    return g_g(n);
}

extern "C" SWEEP21_NOINLINE int64_t sweep21_host_rec(int depth, int64_t n, Entry how) {
    volatile char pad[64];
    pad[0] = static_cast<char>(depth);
    if (depth <= 0) {
        if (how == Entry::EngineInvoke) {
            return g_jit->invoke("g", {RuntimeValue::from_i64(n)}).as_i64();
        }
        GeneratedCodeEntryScope entry;
        return g_g(n);
    }
    const int64_t r = sweep21_host_rec(depth - 1, n, how);
    return r + pad[0] - static_cast<char>(depth);
}

struct WalkCost {
    int64_t result = 0;
    size_t minors = 0;
    size_t steps = 0;
};

WalkCost run_walk(int depth, Entry how) {
    auto mod = parse_ok(kWalk);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    jit.register_external_symbol("host_cb", reinterpret_cast<void*>(&sweep21_host_cb));
    REQUIRE(jit.compile_and_load(*mod));
    g_g = reinterpret_cast<I64Fn>(jit.get_symbol_address("g"));
    REQUIRE(g_g != nullptr);
    g_jit = &jit;
    const size_t before = brass_stack_walk_unwind_steps();
    WalkCost c;
    c.result = sweep21_host_rec(depth, 20000, how);
    c.steps = brass_stack_walk_unwind_steps() - before;
    c.minors = heap.gc.minor_collection_count();
    g_jit = nullptr;
    return c;
}

} // namespace

#if defined(_WIN32) && defined(_M_X64)
// Only the first walk under an entry scope unwinds the host's stack; the
// rest stop at the scope. Unwind steps, not time, so the test cannot flake.
TEST_CASE("Sweep21 - a collection's unwind does not grow with the host's stack depth") {
    for (Entry how : {Entry::RawWithScope, Entry::EngineInvoke}) {
        const WalkCost shallow = run_walk(0, how);
        const WalkCost deep = run_walk(3000, how);
        CHECK_EQ(shallow.result, 42);
        CHECK_EQ(deep.result, 42);
        REQUIRE(shallow.minors > 20);
        CHECK_EQ(deep.minors, shallow.minors);
        // One unwind of the 3000 frames, not one per collection (before the
        // fix: over 3000 * minors); each later walk takes a few steps.
        CHECK(deep.steps >= 3000);
        CHECK(deep.steps <= shallow.steps + 3000 + 64);
        CHECK(shallow.steps <= 4 * shallow.minors + 256);
    }
}

// The scope in host_cb learns that @f lies beneath it; every later walk
// takes that answer and must still update @f's gcref.
TEST_CASE("Sweep21 - an entry scope under a generated caller answers with that caller") {
    auto mod = parse_ok(kWalk);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    jit.register_external_symbol("host_cb", reinterpret_cast<void*>(&sweep21_host_cb));
    REQUIRE(jit.compile_and_load(*mod));
    g_g = reinterpret_cast<I64Fn>(jit.get_symbol_address("g"));
    auto f = reinterpret_cast<I64Fn>(jit.get_symbol_address("f"));
    REQUIRE(g_g != nullptr);
    REQUIRE(f != nullptr);
    CHECK_EQ(f(20000), 84);
    CHECK(heap.gc.minor_collection_count() > 20);
}
#endif

// ---------------------------------------------------------------------------
// 2. NaN operands of add/sub/mul/div: lhs NaN if the lhs is one, else the
// rhs NaN, quieted, in every tier and in the constant folder.

namespace {

const char* kNan = R"(module @nan
func @fuzz_fn(%a: i64, %b: i64) -> i64 {
b0:
  %fa = bitcast.f64 %a
  %fb = bitcast.f64 %b
  %s = add.f64 %fa, %fb
  %d = sub.f64 %fa, %fb
  %m = mul.f64 %fa, %fb
  %q = sdiv.f64 %fa, %fb
  %is = bitcast.i64 %s
  %id = bitcast.i64 %d
  %im = bitcast.i64 %m
  %iq = bitcast.i64 %q
  %x1 = xor.i64 %is, %id
  %c3 = iconst.i64 3
  %r1 = shl.i64 %im, %c3
  %x2 = xor.i64 %x1, %r1
  %c7 = iconst.i64 7
  %r2 = shl.i64 %iq, %c7
  %x3 = xor.i64 %x2, %r2
  ret %x3
}
)";

// f64 bit patterns: NaNs of each sign, quiet and signalling, with distinct
// payloads, and ordinary numbers. (MIR has no f32 bitcast; f32 is checked
// on fparith directly.)
const int64_t kNanArgs[] = {
    static_cast<int64_t>(0x7FF8000000000000ull),  // default quiet NaN
    static_cast<int64_t>(0xFFF8000000000001ull),  // negative quiet, payload 1
    static_cast<int64_t>(0x7FF0000000000005ull),  // signalling, payload 5
    static_cast<int64_t>(0x7FFC00007FC00003ull),  // quiet, a wide payload
    static_cast<int64_t>(0x7FF400007F800009ull),  // signalling, a wide payload
    static_cast<int64_t>(0x4000000000000000ull),  // 2.0
    static_cast<int64_t>(0x3FF0000000000000ull),  // 1.0
    -1,
    0,
};

} // namespace

TEST_CASE("Sweep21 - fparith returns the lhs NaN, else the rhs NaN, quieted") {
    const double qa = std::bit_cast<double>(0xFFF8000000000001ull);
    const double sb = std::bit_cast<double>(0x7FF0000000000005ull);
    const double one = 1.0;
#if defined(__aarch64__) || defined(_M_ARM64)
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::add(qa, sb)), 0x7FF8000000000005ull);
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::add(sb, qa)), 0x7FF8000000000005ull);
#else
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::add(qa, sb)), 0xFFF8000000000001ull);
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::add(sb, qa)), 0x7FF8000000000005ull);
#endif
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::mul(one, sb)), 0x7FF8000000000005ull);
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::div(sb, one)), 0x7FF8000000000005ull);
    CHECK_EQ(std::bit_cast<uint64_t>(fparith::sub(one, qa)), 0xFFF8000000000001ull);
    const float fa = std::bit_cast<float>(0xFF800009u);
    const float fb = std::bit_cast<float>(0x7FC00003u);
    CHECK_EQ(std::bit_cast<uint32_t>(fparith::add(fa, fb)), 0xFFC00009u);
#if defined(__aarch64__) || defined(_M_ARM64)
    CHECK_EQ(std::bit_cast<uint32_t>(fparith::sub(fb, fa)), 0xFFC00009u);
#else
    CHECK_EQ(std::bit_cast<uint32_t>(fparith::sub(fb, fa)), 0x7FC00003u);
#endif
    CHECK_EQ(fparith::add(1.5, 2.25), 3.75);
}

TEST_CASE("Sweep21 - every tier returns the same NaN payload for add/sub/mul/div") {
    auto mod = parse_ok(kNan);
    fuzz::DiffFuzzerOptions opt;
    opt.tier6_baseline = true;
    opt.save_reproducers = false;
    opt.bisect = false;
    fuzz::DiffFuzzer fz(opt);
    int failures = 0;
    for (int64_t a : kNanArgs) {
        for (int64_t b : kNanArgs) {
            std::vector<RuntimeValue> args{RuntimeValue::from_i64(a), RuntimeValue::from_i64(b)};
            auto r = fz.run_test(*mod, "fuzz_fn", args, 0);
            if (!r.passed) {
                ++failures;
                std::printf("a=%lld b=%lld: %s\n", static_cast<long long>(a), static_cast<long long>(b),
                            r.mismatch_reason.c_str());
            }
        }
    }
    CHECK_EQ(failures, 0);
}

TEST_CASE("Sweep21 - the JIT's NaN result is the rule's") {
    auto mod = parse_ok(kNan);
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    for (int64_t a : kNanArgs) {
        for (int64_t b : kNanArgs) {
            const double fa = std::bit_cast<double>(a), fb = std::bit_cast<double>(b);
            auto bits = [](double d) { return std::bit_cast<uint64_t>(d); };
            const uint64_t x3 = (bits(fparith::add(fa, fb)) ^ bits(fparith::sub(fa, fb))) ^
                                (bits(fparith::mul(fa, fb)) << 3) ^ (bits(fparith::div(fa, fb)) << 7);
            const auto got = jit.invoke("fuzz_fn", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b)});
            CHECK_EQ(static_cast<uint64_t>(got.as_i64()), x3);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. brass_coro_resume after a body that moved the coroutine frame.
//
// @body allocates 2000 objects in a 32 KB nursery, so its collections move
// its own frame, then finishes with 7. brass_coro_resume, after the body
// returns, reads is_done and unregisters a finished frame; through its
// stale pointer it would read the old copy (poisoned or reused nursery
// memory) and leave the live frame registered.

namespace {

const char* kMoved = R"(module @cm
func @body() -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 48
  %zm = iconst.i64 0
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %zm, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %n = iconst.i64 2000
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %seven = iconst.i64 7
  ret %seven
}
func @make() -> i64 {
b0:
  %c = coro_create @body()
  coro_destroy %c
  %z = iconst.i64 0
  ret %z
}
)";

} // namespace

TEST_CASE("Sweep21 - coro_resume reads and unregisters the moved frame's live copy") {
    auto mod = parse_ok(kMoved);
    CoroTransformPass pass;
    pass.run_on_module(*mod);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    REQUIRE(jit.compile_and_load(*mod));
    void* body = jit.get_symbol_address("body");
    REQUIRE(body != nullptr);
    uintptr_t frame = brass_coro_create_at(body, 16, 0, 0, 0);
    REQUIRE(frame != 0);
    // The test's own reference to the frame, updated by every collection.
    ThreadRootsScope keep([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &frame);
    const uintptr_t original = frame;
    CHECK_EQ(brass_coro_resume(frame, 0), 7u);
    REQUIRE(heap.gc.minor_collection_count() > 0);
    CHECK(frame != original);  // the body's collections moved the frame
    CHECK_EQ(brass_coro_is_done(frame), 1u);
    // A finished body's resume unregisters the frame: through a stale
    // pointer it would read is_done from the old copy and keep the live one.
    CHECK(!runtime::is_active_coro_frame(frame));
    CHECK_EQ(brass_coro_resume(frame, 0), 7u);
    brass_coro_destroy(frame);
}
