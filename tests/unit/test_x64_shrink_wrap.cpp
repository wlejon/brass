// Shrink-wrapping (codegen/shrink_wrap.hpp): an x64 function whose fast path
// needs no frame returns from it with a bare `ret`, and builds the frame only
// in the blocks that need it. These run such functions through both paths, an
// exception and a collection from the framed part, a loop around it, a
// recursion the stack-limit check (in the frameless part) has to stop, a
// guard's deopt from it, and check the frame states the unwind information
// describes: against the code's bytes, and by stepping out of samples taken
// at every point of the code from a signal handler.

#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/codegen/shrink_wrap.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/native_unwind.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) && !defined(_WIN32)

#if defined(__linux__)
#include <csignal>
#include <ctime>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#endif

using namespace brass;
using namespace brass::runtime;
using namespace brass::object;

namespace {

struct HostError : std::runtime_error {
    int64_t value;
    explicit HostError(int64_t v) : std::runtime_error("host error"), value(v) {}
};

int64_t sw_host(int64_t v) {
    if (v > 1000) throw HostError(v);
    return v * 2;
}

int64_t sw_overflow(int64_t n) { throw HostError(n); }

// -4222124650659840 is 0xFFF1 << 48 (the reference tag), 281474976710655 the
// 48-bit address mask.
const char* kSrc = R"(module @sw
extern @sw_host
extern @sw_overflow
func @sw_fast(%x: i64) -> i64 {
entry:
  %c = iconst.i64 100
  %small = slt.i64 %x, %c
  br_if %small, fast, slow
fast:
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
slow:
  %k = iconst.i64 3
  %a = mul.i64 %x, %k
  %h = call.i64 @sw_host(%x)
  %s = add.i64 %h, %a
  %s2 = add.i64 %s, %x
  ret %s2
}
func @sw_outer(%x: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %neg = slt.i64 %x, %z
  br_if %neg, fast, slow
fast:
  ret %z
slow:
  %k = iconst.i64 7
  %m = mul.i64 %x, %k
  %r = call.i64 @sw_fast(%x)
  %s = add.i64 %r, %m
  ret %s
}
func @sw_sp(%x: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %c = eq.i64 %x, %z
  br_if %c, fast, slow
fast:
  %sp = read_sp
  ret %sp
slow:
  %h = call.i64 @sw_host(%x)
  %sp2 = read_sp
  %r = add.i64 %sp2, %h
  ret %r
}
func @sw_loop(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %none = sle.i64 %n, %z
  br_if %none, fast, pre
fast:
  ret %z
pre:
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %h = call.i64 @sw_host(%i)
  %a2 = add.i64 %acc, %h
  %one = iconst.i64 1
  %j = add.i64 %i, %one
  %more = slt.i64 %j, %n
  br_if %more, loop(%j, %a2), done(%a2)
done(%r: i64):
  ret %r
}
func @sw_depth(%n: i64, %limit: i64) -> i64 {
entry:
  %sp = read_sp
  %low = ult.i64 %sp, %limit
  br_if %low, over, body
over:
  %o = call.i64 @sw_overflow(%n)
  ret %o
body:
  %z = iconst.i64 0
  %done = eq.i64 %n, %z
  br_if %done, base, rec
base:
  ret %z
rec:
  %one = iconst.i64 1
  %m = sub.i64 %n, %one
  %d = call.i64 @sw_depth(%m, %limit)
  %r = add.i64 %d, %one
  ret %r
}
func @sw_churn(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 64
  %kind = iconst.i32 2
  %t = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %n
}
func @sw_gc(%t: tagged, %n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %none = eq.i64 %n, %z
  br_if %none, fast, slow
fast:
  ret %z
slow:
  %r = call.i64 @sw_churn(%n)
  %mask = iconst.i64 281474976710655
  %u = bitcast.i64.tagged %t
  %p = and.i64 %u, %mask
  %v = load.i64 %p, 16
  ret %v
}
func @sw_gc_run(%n: i64) -> i64 {
entry:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %r = call.i64 @sw_gc(%ta, %n)
  ret %r
}
func @sw_exit(%n: i64, %g: i32) -> i64 {
b0:
  %k = iconst.i64 5
  %r = mul.i64 %n, %k
  ret %r
}
func @sw_spec(%n: i64, %g: i32) -> i64 {
b0:
  %z = iconst.i64 0
  %zero = eq.i64 %n, %z
  br_if %zero, fast, slow
fast:
  ret %z
slow:
  %one = iconst.i32 1
  %c = eq.i32 %g, %one
  guard %c, @sw_exit, [%n, %g]
  %k = iconst.i64 5
  %r = mul.i64 %n, %k
  ret %r
}
)";

// Shrink-wrapped by this module's compile: a fast path without a frame.
const char* const kWrapped[] = {"sw_fast", "sw_outer", "sw_sp", "sw_loop", "sw_depth", "sw_gc", "sw_spec"};
// A call in the entry, or a loop through the entry's region: the prologue at
// entry.
const char* const kNotWrapped[] = {"sw_churn", "sw_gc_run"};

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
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

std::unique_ptr<codegen::JitExecutionEngine> load(const Module& mod) {
    auto jit = std::make_unique<codegen::JitExecutionEngine>(Target::host());
    jit->register_external_symbol("sw_host", reinterpret_cast<void*>(&sw_host));
    jit->register_external_symbol("sw_overflow", reinterpret_cast<void*>(&sw_overflow));
    REQUIRE(jit->compile_and_load(mod));
    return jit;
}

const CompiledFunctionInfo* find_fn(const ObjectFile& obj, const std::string& name) {
    for (const auto& f : obj.functions) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

using I64Fn = int64_t (*)(int64_t);
using DepthFn = int64_t (*)(int64_t, int64_t);
using GcFn = int64_t (*)(int64_t);

// Kept out of line so that its locals live in callee-saved registers across
// the call that throws: the unwinder restores them from the frames' CFI.
__attribute__((noinline)) int64_t call_and_catch(I64Fn fn, int64_t x, int64_t& caught) {
    volatile int64_t seed = x;
    int64_t a = seed * 17 + 3;
    int64_t b = seed ^ 0x5a5a;
    int64_t c = seed - 99;
    caught = -1;
    try {
        (void)fn(x);
    } catch (const HostError& e) {
        caught = e.value;
    }
    return a + b + c;
}

__attribute__((noinline)) int64_t depth_and_catch(DepthFn fn, int64_t n, int64_t limit, int64_t& caught) {
    caught = -1;
    try {
        return fn(n, limit);
    } catch (const HostError& e) {
        caught = e.value;
    }
    return -1;
}

// Both calls from one frame at one stack pointer: read_sp without a frame
// (rsp lowered for it) and with the frame read the same value.
__attribute__((noinline)) void read_sp_twice(I64Fn fn, int64_t& frameless, int64_t& framed) {
    frameless = fn(0);
    framed = fn(1) - 2;
}

} // namespace

TEST_CASE("Shrink-wrap - regions: the save point leaves loops, the entry and loops through it stay framed") {
    using codegen::FrameGraph;
    using codegen::FrameRegions;
    auto graph = [](std::vector<std::vector<uint32_t>> succs, std::vector<uint8_t> needs, std::vector<uint8_t> rets) {
        FrameGraph g;
        g.succs = std::move(succs);
        g.cold.assign(g.succs.size(), 0);
        g.needs_frame = std::move(needs);
        g.returns = std::move(rets);
        return g;
    };
    // A diamond: 1 returns without a frame, 2 needs one.
    {
        const FrameRegions r = codegen::choose_frame_regions(graph({{1, 2}, {}, {}}, {0, 0, 1}, {0, 1, 1}));
        REQUIRE(r.wrapped);
        REQUIRE_EQ(r.saves.size(), size_t{1});
        CHECK_EQ(r.saves[0], 2u);
        CHECK_EQ(r.region[1], FrameRegions::kNone);
        CHECK_EQ(r.region[2], 2u);
    }
    // A loop needing the frame (3) after a preheader (2): the region starts
    // at the preheader, outside the loop, and holds the exit.
    {
        const FrameRegions r = codegen::choose_frame_regions(
            graph({{1, 2}, {}, {3}, {3, 4}, {}}, {0, 0, 0, 1, 0}, {0, 1, 0, 0, 1}));
        REQUIRE(r.wrapped);
        REQUIRE_EQ(r.saves.size(), size_t{1});
        CHECK_EQ(r.saves[0], 2u);
        CHECK_EQ(r.region[3], 2u);
        CHECK_EQ(r.region[4], 2u);
        CHECK_EQ(r.region[1], FrameRegions::kNone);
    }
    // Two regions on either side of a frameless return.
    {
        const FrameRegions r = codegen::choose_frame_regions(
            graph({{1, 2, 3}, {}, {}, {}}, {0, 0, 1, 1}, {0, 1, 1, 1}));
        REQUIRE(r.wrapped);
        CHECK_EQ(r.saves.size(), size_t{2});
    }
    // The entry needs the frame.
    CHECK(!codegen::choose_frame_regions(graph({{1, 2}, {}, {}}, {1, 0, 0}, {0, 1, 1})).wrapped);
    // No return without the frame.
    CHECK(!codegen::choose_frame_regions(graph({{1}, {}}, {0, 1}, {0, 1})).wrapped);
    // A loop headed by 1 holding the block that needs the frame, with no
    // preheader below the entry: the region would start at the entry.
    CHECK(!codegen::choose_frame_regions(
        graph({{1}, {2, 3}, {1}, {}}, {0, 0, 1, 0}, {0, 0, 0, 1})).wrapped);
}

TEST_CASE("Shrink-wrap - which functions are wrapped, and their frame states against the code") {
    auto mod = parse_or_fail(kSrc);
    const ObjectFile obj = compile_module_to_object(*mod, Target::host());
    const Section* text = obj.get_section(".text");
    if (text == nullptr) text = obj.get_section("__text");
    REQUIRE(text != nullptr);
    for (const char* name : kWrapped) {
        const CompiledFunctionInfo* f = find_fn(obj, name);
        REQUIRE(f != nullptr);
        if (f->frame_rows.empty()) std::cerr << "not shrink-wrapped: " << name << "\n";
        CHECK(!f->frame_rows.empty());
    }
    for (const char* name : kNotWrapped) {
        const CompiledFunctionInfo* f = find_fn(obj, name);
        REQUIRE(f != nullptr);
        CHECK(f->frame_rows.empty());
    }
    for (const CompiledFunctionInfo& f : obj.functions) {
        if (f.frame_rows.empty()) continue;
        const uint8_t* code = text->data.data() + f.text_offset;
        // Without a frame at entry: the first byte is no push rbp.
        CHECK(code[0] != 0x55);
        uint32_t prev = 0;
        size_t prologues = 0;
        for (size_t i = 0; i < f.frame_rows.size(); ++i) {
            const codegen::FrameRow& row = f.frame_rows[i];
            const uint32_t o = row.offset;
            CHECK(o > 0 && o < f.text_size);
            CHECK(i == 0 || o > prev);
            prev = o;
            switch (row.state) {
            case codegen::FrameState::Pushed:
                // After push rbp.
                CHECK_EQ(code[o - 1], uint8_t{0x55});
                ++prologues;
                break;
            case codegen::FrameState::Framed:
                // After mov rbp, rsp; or after a ret that ended framed code
                // with more of it following.
                if (i > 0 && f.frame_rows[i - 1].state == codegen::FrameState::Pushed) {
                    CHECK_EQ(f.frame_rows[i - 1].offset + 3, o);
                    CHECK(code[o - 3] == 0x48 && code[o - 2] == 0x89 && code[o - 1] == 0xE5);
                } else {
                    CHECK(code[o - 1] == 0xC3 || code[o - 5] == 0xE9 || code[o - 2] == 0xEB);
                }
                break;
            case codegen::FrameState::Frameless:
                // After pop rbp (a teardown, or the epilogue before its ret);
                // after the raise of rsp a read of it lowered; or at a block
                // without the frame, where only padding follows the ret or
                // jmp that ended the framed code before it.
                if (i > 0 && f.frame_rows[i - 1].state == codegen::FrameState::Lowered) {
                    CHECK(code[o - 5] == 0x48 || code[o - 8] == 0x48);
                } else if (code[o - 1] != 0x5D) {
                    REQUIRE(i > 0);
                    const uint32_t p = f.frame_rows[i - 1].offset;
                    CHECK(f.frame_rows[i - 1].state == codegen::FrameState::Framed);
                    CHECK(code[p - 1] == 0xC3 || code[p - 5] == 0xE9 || code[p - 2] == 0xEB);
                    // Nothing between but nops (0x90, 0F 1F .., 66 ..).
                    for (uint32_t q = p; q < o; ++q) {
                        const uint8_t c = code[q];
                        CHECK(c == 0x90 || c == 0x0F || c == 0x1F || c == 0x66 || c == 0x2E ||
                              c == 0x00 || c == 0x40 || c == 0x44 || c == 0x80 || c == 0x84);
                    }
                }
                break;
            case codegen::FrameState::Lowered:
                // After lea rsp, [rsp - K].
                CHECK(row.lowered > 0);
                CHECK((code[o - 5] == 0x48 && code[o - 4] == 0x8D && code[o - 3] == 0x64) ||
                      (code[o - 8] == 0x48 && code[o - 7] == 0x8D && code[o - 6] == 0xA4));
                break;
            }
        }
        CHECK(prologues > 0);
    }
}

TEST_CASE("Shrink-wrap - fast and slow paths, and read_sp with and without the frame") {
    auto mod = parse_or_fail(kSrc);
    auto jit = load(*mod);
    auto fast = jit->get_function_ptr<I64Fn>("sw_fast");
    auto outer = jit->get_function_ptr<I64Fn>("sw_outer");
    auto loop = jit->get_function_ptr<I64Fn>("sw_loop");
    auto sp = jit->get_function_ptr<I64Fn>("sw_sp");
    REQUIRE(fast && outer && loop && sp);
    CHECK(reinterpret_cast<const uint8_t*>(fast)[0] != 0x55);
    for (int64_t x : {-5, 0, 7, 99, 100, 101, 500, 1000}) {
        const int64_t want = x < 100 ? x + 1 : 6 * x;
        CHECK_EQ(fast(x), want);
        CHECK_EQ(outer(x), x < 0 ? 0 : want + 7 * x);
    }
    // A loop around the framed part: the region is entered once.
    for (int64_t n : {-3, 0, 1, 2, 10, 500}) CHECK_EQ(loop(n), n <= 0 ? 0 : n * (n - 1));
    int64_t frameless = 0, framed = 0;
    read_sp_twice(sp, frameless, framed);
    CHECK(frameless != 0);
    CHECK_EQ(frameless, framed);
}

TEST_CASE("Shrink-wrap - an exception from the framed part unwinds through two wrapped frames") {
    auto mod = parse_or_fail(kSrc);
    auto jit = load(*mod);
    auto outer = jit->get_function_ptr<I64Fn>("sw_outer");
    REQUIRE(outer != nullptr);
    for (int64_t x : {1001, 5000, 77777}) {
        int64_t caught = 0;
        const int64_t locals = call_and_catch(outer, x, caught);
        CHECK_EQ(caught, x);
        CHECK_EQ(locals, (x * 17 + 3) + (x ^ 0x5a5a) + (x - 99));
    }
    CHECK_EQ(outer(5), 6 + 35);
    CHECK_EQ(outer(200), 1200 + 1400);
}

TEST_CASE("Shrink-wrap - the stack-limit check without a frame stops a deep recursion") {
    auto mod = parse_or_fail(kSrc);
    auto jit = load(*mod);
    auto depth = jit->get_function_ptr<DepthFn>("sw_depth");
    REQUIRE(depth != nullptr);
    int64_t caught = 0;
    CHECK_EQ(depth_and_catch(depth, 1000, 0, caught), 1000);
    CHECK_EQ(caught, -1);
    // A limit 256 KiB below here: the check fires at a depth the frames'
    // size bounds, and the throw unwinds through every one of them.
    volatile char here = 0;
    const int64_t budget = 256 * 1024;
    const int64_t limit = static_cast<int64_t>(reinterpret_cast<uintptr_t>(&here)) - budget;
    const int64_t n = 10'000'000;
    CHECK_EQ(depth_and_catch(depth, n, limit, caught), -1);
    REQUIRE(caught > 0);
    const int64_t frames = n - caught;
    CHECK(frames * 16 <= budget);          // a frame is at least rbp and the return address
    CHECK(frames * 512 >= budget - 4096);  // and nowhere near 512 bytes
    // And it runs normally after.
    CHECK_EQ(depth_and_catch(depth, 50, limit, caught), 50);
}

TEST_CASE("Shrink-wrap - a collection in the framed part updates a root held across it") {
    for (gc::StressMode stress : {gc::StressMode::None, gc::StressMode::Minor, gc::StressMode::Alternate}) {
        auto mod = parse_or_fail(kSrc);
        gc::HeapConfig config = test::small_heap_config();
        config.reference_tags = {0xFFF1};
        config.poison = true;
        gc::Heap heap(config);
        heap.set_stress(stress);
        gc::HeapScope bind(heap);
        auto jit = load(*mod);
        auto run = jit->get_function_ptr<GcFn>("sw_gc_run");
        REQUIRE(run != nullptr);
        for (int64_t n : {0, 1, 100, 1000}) {
            const size_t before = heap.collection_count();
            CHECK_EQ(run(n), n == 0 ? 0 : 42);
            if (n == 1000) CHECK(heap.collection_count() > before);
        }
    }
}

TEST_CASE("Shrink-wrap - a guard in the framed part deopts") {
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    prog.pipeline().initialize(cfg);
    prog.tiering().get_feedback("sw_spec").set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create("sw_spec", mod->get_function("sw_spec"));
    CodeInstaller installer(prog);
    REQUIRE(installer.install_tier2(*h, *mod, "sw_spec").success);
    CHECK_EQ(h->call_native({RuntimeValue::from_i64(0), RuntimeValue::from_i32(1)}).as_i64(), 0);
    for (int32_t g : {1, 0}) {
        CHECK_EQ(h->call_native({RuntimeValue::from_i64(9), RuntimeValue::from_i32(g)}).as_i64(), 45);
    }
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
}

#if defined(__linux__)

namespace {

// A profiler's view: a signal at any instruction of the generated code steps
// out of it by the unwind information alone, to the C++ frame that called in.
struct SampleState {
    uintptr_t driver_lo = 0, driver_hi = 0;
    volatile int samples = 0;
    volatile int in_driver = 0;
    volatile int failures = 0;
    uintptr_t failed_ip[32] = {};
    int failed_steps[32] = {};
};
SampleState g_sample;

bool in_driver(uintptr_t ip) { return ip >= g_sample.driver_lo && ip < g_sample.driver_hi; }

void on_sample(int, siginfo_t*, void* ctx) {
    const auto* uc = static_cast<const ucontext_t*>(ctx);
    NativeUnwindFrame f;
    f.ip = static_cast<uintptr_t>(uc->uc_mcontext.gregs[REG_RIP]);
    f.sp = static_cast<uintptr_t>(uc->uc_mcontext.gregs[REG_RSP]);
    f.fp = static_cast<uintptr_t>(uc->uc_mcontext.gregs[REG_RBP]);
    g_sample.samples = g_sample.samples + 1;
    if (in_driver(f.ip)) {
        g_sample.in_driver = g_sample.in_driver + 1;
        return;
    }
    const uintptr_t ip0 = f.ip;
    bool first = true;
    int step = 0;
    for (; step < 16; ++step) {
        if (!brass_unwind_step(f, !first)) break;
        first = false;
        if (in_driver(f.ip)) return;
    }
    const int k = g_sample.failures;
    if (k < 32) {
        g_sample.failed_ip[k] = ip0;
        g_sample.failed_steps[k] = step;
    }
    g_sample.failures = k + 1;
}

// No clone of it (a constant-propagated copy) runs in its place: the samples
// look for its own code.
#if defined(__clang__)
#define SW_NO_IPA __attribute__((noinline))
#else
#define SW_NO_IPA __attribute__((noipa))
#endif
SW_NO_IPA int64_t sample_driver(I64Fn fast, I64Fn outer, I64Fn sp, DepthFn depth, int64_t iters) {
    // Until 3000 samples landed in the callees (or 10 M rounds).
    int64_t acc = 0;
    for (int64_t i = 0; i < iters && g_sample.samples - g_sample.in_driver < 3000; ++i) {
        acc += fast(i & 127);   // both paths: x < 100 and not
        acc += outer((i & 255) - 20);
        acc += sp(i & 1) & 1;
        acc += depth(i & 7, 0);
    }
    return acc;
}
SW_NO_IPA void sample_driver_end() { asm volatile(""); }

} // namespace

TEST_CASE("Shrink-wrap - a signal anywhere in wrapped code unwinds to its caller") {
    auto mod = parse_or_fail(kSrc);
    auto jit = load(*mod);
    auto fast = jit->get_function_ptr<I64Fn>("sw_fast");
    auto outer = jit->get_function_ptr<I64Fn>("sw_outer");
    auto sp = jit->get_function_ptr<I64Fn>("sw_sp");
    auto depth = jit->get_function_ptr<DepthFn>("sw_depth");
    REQUIRE(fast && outer && sp && depth);
    const auto lo = reinterpret_cast<uintptr_t>(&sample_driver);
    const auto hi = reinterpret_cast<uintptr_t>(&sample_driver_end);
    // The driver's code, as far as the next function; a bound if the linker
    // placed them apart.
    g_sample.driver_lo = lo;
    g_sample.driver_hi = hi > lo && hi - lo < 4096 ? hi : lo + 1024;

    // Warm the unwinder's tables outside the handler.
    NativeUnwindFrame warm;
    if (brass_capture_frame(warm, 0)) (void)brass_unwind_step(warm);
    (void)sample_driver(fast, outer, sp, depth, 16);

    struct sigaction sa = {}, old = {};
    sa.sa_sigaction = &on_sample;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    REQUIRE(sigaction(SIGPROF, &sa, &old) == 0);
    sigevent sev = {};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGPROF;
    sev._sigev_un._tid = static_cast<pid_t>(syscall(SYS_gettid));
    timer_t timer{};
    REQUIRE(timer_create(CLOCK_MONOTONIC, &sev, &timer) == 0);
    itimerspec its = {};
    its.it_value.tv_nsec = 20'000;
    its.it_interval.tv_nsec = 20'000;
    REQUIRE(timer_settime(timer, 0, &its, nullptr) == 0);
    const int64_t acc = sample_driver(fast, outer, sp, depth, 10'000'000);
    timer_delete(timer);
    sigaction(SIGPROF, &old, nullptr);
    CHECK(acc != 0);
    const int outside = g_sample.samples - g_sample.in_driver;
    std::cerr << "shrink-wrap samples: " << g_sample.samples << " (" << outside << " in callees), "
              << g_sample.failures << " failed to unwind\n";
    for (int k = 0; k < g_sample.failures && k < 32; ++k) {
        std::cerr << "  failed at " << std::hex << g_sample.failed_ip[k] << std::dec << " after "
                  << g_sample.failed_steps[k] << " steps\n";
    }
    if (g_sample.failures > 0) {
        std::cerr << std::hex << "  fast " << reinterpret_cast<uintptr_t>(fast) << " outer "
                  << reinterpret_cast<uintptr_t>(outer) << " sp " << reinterpret_cast<uintptr_t>(sp) << " depth "
                  << reinterpret_cast<uintptr_t>(depth) << std::dec << "\n";
    }
    CHECK(outside > 200);
    CHECK_EQ(g_sample.failures, 0);
}

#endif // __linux__

#endif // x86_64 && !_WIN32
