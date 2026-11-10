// The tagged allocator against a real heap, at every tier. The module's TLS
// block holds the heap's allocation window (bind_allocation_buffer) where
// the allocator's attribute says, so tier 2 bumps it inline and calls the
// allocator only when the window is full, and tiers 0 and 1 call it every
// time. @ta_list builds a list whose nodes point at each other through
// tagged words, collecting between nodes; @ta_hold keeps a reference in an
// object that does not escape across a collection, which scalar replacement
// turns into a tagged value, a root. The heap poisons what it frees, so a
// reference a collection did not update reads poison.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/object.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/tagged_alloc_lowering.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

#if defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

constexpr uint64_t kTag = 0xFFF1000000000000ULL;

// -4222124650659840 is 0xFFF1 << 48, -2814749767106560 0xFFF6 << 48 (the
// list's end), -281474976710656 the 16 tag bits and 281474976710655 the 48
// address bits.
const char* kSrc = R"(module @tag
attributes pinned_tls_register tagged_allocator @ta_heap_alloc 16 24 0xfff1000000000000

func @ta_churn(%n: i64) -> i64 {
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
func @ta_sum(%head: tagged) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%head, %z, %z)
loop(%cur: tagged, %acc: i64, %cnt: i64):
  %cb = bitcast.i64.tagged %cur
  %tm = iconst.i64 -281474976710656
  %tbits = and.i64 %cb, %tm
  %tag = iconst.i64 -4222124650659840
  %isobj = eq.i64 %tbits, %tag
  br_if %isobj, body, done
body:
  %mask = iconst.i64 281474976710655
  %p = and.i64 %cb, %mask
  %v = load.i64 %p, 8
  %nx = load.i64 %p, 0
  %nt = bitcast.tagged.i64 %nx
  %acc2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %cnt2 = add.i64 %cnt, %one
  br loop(%nt, %acc2, %cnt2)
done:
  %m = iconst.i64 1000000
  %k = mul.i64 %cnt, %m
  %r = add.i64 %acc, %k
  ret %r
}
func @ta_list(%tls: i64, %hdr: i64, %n: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %end = iconst.i64 -2814749767106560
  %first = bitcast.tagged.i64 %end
  %z = iconst.i64 0
  br loop(%z, %first)
loop(%i: i64, %prev: tagged):
  %bytes = iconst.i64 16
  %o = call.tagged @ta_heap_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %mask = iconst.i64 281474976710655
  %p = and.i64 %ob, %mask
  %pb = bitcast.i64.tagged %prev
  store.i64 %p, 0, %pb
  store.i64 %p, 8, %i
  write_barrier %ob, %pb
  %one = iconst.i64 1
  %c = call.i64 @ta_churn(%one)
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %o), done
done:
  %s = call.i64 @ta_sum(%o)
  ret %s
}
func @ta_pairs(%tls: i64, %hdr: i64, %n: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %end = iconst.i64 -2814749767106560
  %first = bitcast.tagged.i64 %end
  %z = iconst.i64 0
  br loop(%z, %first)
loop(%i: i64, %prev: tagged):
  %bytes = iconst.i64 16
  %mask = iconst.i64 281474976710655
  %in = call.tagged @ta_heap_alloc(%bytes, %hdr)
  %inb = bitcast.i64.tagged %in
  %ip = and.i64 %inb, %mask
  store.i64 %ip, 0, %end
  store.i64 %ip, 8, %i
  %obytes = iconst.i64 24
  %out = call.tagged @ta_heap_alloc(%obytes, %hdr)
  %outb = bitcast.i64.tagged %out
  %op = and.i64 %outb, %mask
  %pb = bitcast.i64.tagged %prev
  %inb2 = bitcast.i64.tagged %in
  store.i64 %op, 0, %pb
  store.i64 %op, 8, %inb2
  store.i64 %op, 16, %i
  %one = iconst.i64 1
  %c = call.i64 @ta_churn(%one)
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %out), done
done:
  br walk(%out, %z)
walk(%cur: tagged, %acc: i64):
  %cb = bitcast.i64.tagged %cur
  %tm = iconst.i64 -281474976710656
  %tbits = and.i64 %cb, %tm
  %tag = iconst.i64 -4222124650659840
  %isobj = eq.i64 %tbits, %tag
  br_if %isobj, body, out
body:
  %m2 = iconst.i64 281474976710655
  %p = and.i64 %cb, %m2
  %inner = load.i64 %p, 8
  %innerp = and.i64 %inner, %m2
  %v = load.i64 %innerp, 8
  %w = load.i64 %p, 16
  %vw = add.i64 %v, %w
  %acc2 = add.i64 %acc, %vw
  %nx = load.i64 %p, 0
  %nt = bitcast.tagged.i64 %nx
  br walk(%nt, %acc2)
out:
  ret %acc
}
func @ta_hold(%tls: i64, %hdr: i64, %n: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %mask = iconst.i64 281474976710655
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %bytes = iconst.i64 16
  %o = call.tagged @ta_heap_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %p = and.i64 %ob, %mask
  %tb = bitcast.i64.tagged %ta
  store.i64 %p, 0, %tb
  %k7 = iconst.i64 7
  store.i64 %p, 8, %k7
  %c = call.i64 @ta_churn(%n)
  %ob2 = bitcast.i64.tagged %o
  %p2 = and.i64 %ob2, %mask
  %w = load.i64 %p2, 0
  %wp = and.i64 %w, %mask
  %v = load.i64 %wp, 16
  %hi = xor.i64 %w, %wp
  %ok = eq.i64 %hi, %tag
  %hundred = iconst.i64 100
  %sel = select.i64 %ok, %hundred, %z
  %k = load.i64 %p2, 8
  %r0 = add.i64 %v, %k
  %r = add.i64 %r0, %sel
  ret %r
}
)";

constexpr int64_t kHoldExpected = 42 + 7 + 100;

int64_t list_expected(int64_t n) { return n * (n - 1) / 2 + n * 1000000; }

int g_slow_calls = 0;

// The allocator's slow path: a young object (or an old one, when it does not
// fit eden) of the header's layout, NaN-boxed.
extern "C" uint64_t ta_heap_alloc(uint64_t bytes, uint64_t header) {
    ++g_slow_calls;
    gc::Heap* heap = gc::Heap::current();
    const auto layout = static_cast<gc::LayoutId>((header >> 32) & 0xFFFF);
    const auto host_bits = static_cast<uint8_t>(header >> 56);
    return static_cast<uint64_t>(heap->allocate(static_cast<size_t>(bytes), layout, 0, host_bits)) | kTag;
}

// A payload of tagged words: each a reference under 0xFFF1, else data.
uint64_t words_header() {
    static const gc::LayoutId id = [] {
        gc::LayoutDescriptor d;
        d.kind = gc::LayoutKind::Words;
        return gc::register_layout(d);
    }();
    return static_cast<uint64_t>(id) << 32;
}

gc::HeapConfig heap_config() {
    gc::HeapConfig config = test::small_heap_config();
    config.reference_tags = {0xFFF1};
    config.poison = true;
    config.walk_stack_on_host_collection = true;
    return config;
}

// The TLS block: the allocation window at 16 / 24, bound while it lives.
struct Tls {
    alignas(16) uint64_t words[4] = {};
    gc::Heap& heap;
    explicit Tls(gc::Heap& h) : heap(h) {
        static_assert(sizeof(gc::Heap::AllocationBuffer) == 16, "top, end");
        heap.bind_allocation_buffer(reinterpret_cast<gc::Heap::AllocationBuffer*>(&words[2]));
    }
    ~Tls() { heap.bind_allocation_buffer(nullptr); }
    Tls(const Tls&) = delete;
    Tls& operator=(const Tls&) = delete;
    int64_t addr() const { return static_cast<int64_t>(reinterpret_cast<uintptr_t>(&words[0])); }
};

std::unique_ptr<Module> parse_ok() {
    DiagnosticReporter diag;
    auto mod = parse_module(kSrc, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

TieringConfig no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

constexpr gc::StressMode kModes[] = {gc::StressMode::None, gc::StressMode::Minor, gc::StressMode::Full,
                                     gc::StressMode::Incremental};
const char* mode_name(gc::StressMode m) {
    switch (m) {
        case gc::StressMode::None: return "none";
        case gc::StressMode::Minor: return "minor";
        case gc::StressMode::Full: return "full";
        case gc::StressMode::Incremental: return "incremental";
        default: return "other";
    }
}

// Runs both functions through `call(name, tls, n)`. `inline_window`: the
// list's nodes come from the window, the allocator is called for few.
template <typename Call>
void check_runs(const char* tier, gc::Heap& heap, gc::StressMode mode, bool inline_window, Call&& call) {
    Tls tls(heap);
    for (int64_t n : {1, 50, 2000}) {
        g_slow_calls = 0;
        const size_t before = heap.collection_count();
        const int64_t got = call("ta_list", tls.addr(), n);
        if (got != list_expected(n)) std::cerr << tier << "/" << mode_name(mode) << " list n=" << n << ": " << got << "\n";
        CHECK_EQ(got, list_expected(n));
        if (inline_window && mode == gc::StressMode::None) CHECK(g_slow_calls < n || n == 1);
        if (!inline_window) CHECK_EQ(g_slow_calls, static_cast<int>(n));
        if (n == 2000 || mode == gc::StressMode::Minor || mode == gc::StressMode::Full) CHECK(heap.collection_count() > before);
        g_slow_calls = 0;
        const int64_t pairs = call("ta_pairs", tls.addr(), n);
        if (pairs != n * (n - 1)) std::cerr << tier << "/" << mode_name(mode) << " pairs n=" << n << ": " << pairs << "\n";
        CHECK_EQ(pairs, n * (n - 1));
        if (inline_window && mode == gc::StressMode::None) CHECK(g_slow_calls < n || n == 1);
        if (!inline_window) CHECK_EQ(g_slow_calls, static_cast<int>(2 * n));
        const int64_t held = call("ta_hold", tls.addr(), n);
        if (held != kHoldExpected) std::cerr << tier << "/" << mode_name(mode) << " hold n=" << n << ": " << held << "\n";
        CHECK_EQ(held, kHoldExpected);
    }
}

} // namespace

TEST_CASE("Tagged allocation - tier 2 bumps the window inline and calls the allocator from a cold block") {
    auto mod = parse_ok();
    Function& fn = *mod->get_function("ta_list");
    REQUIRE(needs_tagged_allocation_expansion(fn));
    const size_t before = fn.blocks().size();
    CHECK_EQ(expand_tagged_allocations(fn), size_t{1});
    CHECK_EQ(fn.blocks().size(), before + 3);
    size_t cold = 0, calls = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        cold += bb->is_cold();
        for (const Instruction* inst : *bb) calls += inst->opcode() == Opcode::call && inst->symbol() == "ta_heap_alloc";
    }
    CHECK_EQ(cold, size_t{1});
    CHECK_EQ(calls, size_t{1});
    DiagnosticReporter diag;
    const bool ok = verify_function(fn, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    CHECK(ok);
    // Two allocations with no GC point between them take the window once:
    // one test of it, the second's own test only after a cold first.
    Function& pairs = *mod->get_function("ta_pairs");
    size_t tests_before = 0;
    CHECK_EQ(expand_tagged_allocations(pairs), size_t{2});
    for (const BasicBlock* bb : pairs.blocks()) {
        for (const Instruction* inst : *bb) tests_before += inst->opcode() == Opcode::ule;
    }
    CHECK_EQ(tests_before, size_t{2});
    DiagnosticReporter pdiag;
    const bool pok = verify_function(pairs, &pdiag);
    if (!pok) std::cerr << pdiag.format_all() << "\n";
    CHECK(pok);
    // Without a window nothing is expanded.
    mod->set_tagged_allocator("ta_heap_alloc", -1, -1, kTag);
    CHECK(!needs_tagged_allocation_expansion(*mod->get_function("ta_hold")));
}

TEST_CASE("Tagged allocation - the fast interpreter calls the allocator and its objects survive collections") {
    for (gc::StressMode mode : kModes) {
        auto mod = parse_ok();
        FastInterpreter fi(heap_config());
        fi.heap().set_stress(mode);
        gc::HeapScope bind(fi.heap());
        fi.set_module(mod.get());
        fi.register_external_function(
            "ta_heap_alloc", std::function<RuntimeValue(const std::vector<RuntimeValue>&)>(
                                 [](const std::vector<RuntimeValue>& a) {
                                     return RuntimeValue::from_tagged(ta_heap_alloc(a[0].raw_bits(), a[1].raw_bits()));
                                 }));
        check_runs("tier 0", fi.heap(), mode, false, [&](const char* name, int64_t tls, int64_t n) {
            return fi.run(*mod->get_function(name), {RuntimeValue::from_i64(tls),
                                                     RuntimeValue::from_i64(static_cast<int64_t>(words_header())),
                                                     RuntimeValue::from_i64(n)})
                .as_i64();
        });
    }
}

TEST_CASE("Tagged allocation - baseline code calls the allocator and its objects survive collections") {
    for (gc::StressMode mode : kModes) {
        auto mod = parse_ok();
        FunctionDispatchTable prog;
        prog.pipeline().initialize(no_tierup());
        prog.pipeline().register_external_symbol("ta_heap_alloc", reinterpret_cast<void*>(&ta_heap_alloc));
        Interpreter interp(heap_config());
        interp.heap().set_stress(mode);
        gc::HeapScope bind(interp.heap());
        interp.set_dispatch_table(&prog);
        interp.set_module(mod.get());
        for (const char* name : {"ta_churn", "ta_sum", "ta_list", "ta_pairs", "ta_hold"}) {
            REQUIRE(prog.pipeline().compile_and_install_tier1(name, mod->get_function(name)));
        }
        check_runs("tier 1", interp.heap(), mode, false, [&](const char* name, int64_t tls, int64_t n) {
            return interp.run(*mod->get_function(name), {RuntimeValue::from_i64(tls),
                                                         RuntimeValue::from_i64(static_cast<int64_t>(words_header())),
                                                         RuntimeValue::from_i64(n)})
                .as_i64();
        });
    }
}

TEST_CASE("Tagged allocation - tier 2's inline window and a scalar-replaced holder survive collections") {
    for (gc::StressMode mode : kModes) {
        auto mod = parse_ok();
        // The holder does not escape: its words become tagged values.
        CHECK_EQ(scalar_replace_tagged_allocations(*mod->get_function("ta_hold"), SroaOptions{}), size_t{1});
        gc::Heap heap(heap_config());
        heap.set_stress(mode);
        gc::HeapScope bind(heap);
        codegen::JitExecutionEngine jit(Target::host());
        jit.register_external_symbol("ta_heap_alloc", reinterpret_cast<void*>(&ta_heap_alloc));
        REQUIRE(jit.compile_and_load(*mod));
        check_runs("tier 2", heap, mode, true, [&](const char* name, int64_t tls, int64_t n) {
            return jit.invoke(name, {RuntimeValue::from_i64(tls),
                                     RuntimeValue::from_i64(static_cast<int64_t>(words_header())),
                                     RuntimeValue::from_i64(n)})
                .as_i64();
        });
        runtime::FunctionDispatchTable::instance().forget_module(*mod);
    }
}

#endif
