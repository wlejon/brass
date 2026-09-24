// Regressions from bug sweep 19: a GenerationalGC major collection given the
// same root slot twice, and brass_coro_create from generated code dropping
// the calling frame's roots.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cstring>
#include <memory>
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

TieringConfig no_tierup(bool fast) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    return cfg;
}

// @specd deopts to @exitd, which runs in a fresh Tier-0 interpreter: it
// allocates y (42), stores it in `old`, runs a collection (a major one: the
// tenured space is nearly full) and returns old->y->16 + old->24.
const char* kMajor = R"(module @gd
func @exitd(%old: gcref) -> i64 {
entry:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %y = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %y, 16, %k
  store.gcref %old, 16, %y
  write_barrier %old, %y
  call @brass_gc_collect()
  %m = load.i64 %old, 24
  %p = load.gcref %old, 16
  %v = load.i64 %p, 16
  %s = add.i64 %v, %m
  ret %s
}
func @specd(%old: gcref, %t: i32) -> i64 {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @exitd, [%old]
  %v = iconst.i64 1
  ret %v
}
)";

void run_major_after_deopt(bool fast) {
    auto mod = parse_ok(kMajor);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    GenerationalGC g(64 * 1024, 16 * 1024, 4096, 1);
    brass_set_active_generational_gc(&g);
    prog.tiering().get_feedback("specd").set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create("specd", mod->get_function("specd"));
    CodeInstaller installer(prog);
    REQUIRE(installer.install_tier2(*h, *mod, "specd").success);

    // 45 fillers, then `old`, promoted; the fillers die, leaving the tenured
    // space 3680/4096 full, so brass_gc_collect runs a major collection
    // whose final addresses overlap the dead fillers.
    std::vector<uintptr_t> fill(45);
    for (auto& f : fill) {
        f = g.allocate(48, 0, 2);
        reinterpret_cast<int64_t*>(f)[3] = 1000;
        g.register_root(&f);
    }
    uintptr_t old = g.allocate(48, 1ULL << 2, 2);
    reinterpret_cast<int64_t*>(old)[3] = 7;
    g.register_root(&old);
    g.minor_collect();
    for (auto& f : fill) g.unregister_root(&f);
    REQUIRE(g.is_old(old));

    const size_t majors = g.major_collection_count();
    RuntimeValue r = h->call_native({RuntimeValue::from_bits(Type::gcref(), old), RuntimeValue::from_i32(0)});
    CHECK_EQ(r.as_i64(), 49);
    CHECK_EQ(g.major_collection_count(), majors + 1);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
    CHECK_EQ(reinterpret_cast<const int64_t*>(old)[3], 7);
    g.unregister_root(&old);
    brass_set_active_generational_gc(nullptr);
}

// Semispace copying host heap collecting on every 4th allocate_at, with the
// roots brass_enumerate_thread_roots reports for the frame it is given.
class CopyHeap final : public HostHeap {
public:
    static constexpr size_t kCap = 1 << 20;
    CopyHeap() : a_(kCap), b_(kCap) { from_ = a_.data(); to_ = b_.data(); }
    uintptr_t allocate(size_t size, uint64_t mask, uint32_t tag) override {
        return allocate_at(size, mask, tag, 0, 0);
    }
    uintptr_t allocate_at(size_t size, uint64_t mask, uint32_t, uintptr_t fp, uintptr_t ip) override {
        if (fp != 0 && ip != 0) ++framed;
        if (++allocs_ % 4 == 0) collect_now(fp, ip);
        size_t need = 24 + ((size + 7) & ~size_t(7));
        if (top_ + need > kCap) collect_now(fp, ip);
        if (top_ + need > kCap) return 0;
        uint64_t* hd = reinterpret_cast<uint64_t*>(from_ + top_);
        hd[0] = size; hd[1] = mask; hd[2] = 0;
        std::memset(hd + 3, 0, need - 24);
        top_ += need;
        return reinterpret_cast<uintptr_t>(hd + 3);
    }
    void collect() override { collect_now(0, 0); }
    void safepoint_at(uintptr_t fp, uintptr_t ip) override { collect_now(fp, ip); }
    int framed = 0;
    int collections = 0;

private:
    bool in_from(uintptr_t p) const { return p >= reinterpret_cast<uintptr_t>(from_) && p < reinterpret_cast<uintptr_t>(from_) + kCap; }
    uintptr_t copy(uintptr_t p) {
        uint64_t* hd = reinterpret_cast<uint64_t*>(p) - 3;
        if (hd[2]) return hd[2];
        size_t need = 24 + ((hd[0] + 7) & ~size_t(7));
        uint64_t* n = reinterpret_cast<uint64_t*>(to_ + ttop_);
        std::memcpy(n, hd, need);
        n[2] = 0;
        ttop_ += need;
        hd[2] = reinterpret_cast<uintptr_t>(n + 3);
        return hd[2];
    }
    void fix(uintptr_t* slot) {
        uintptr_t v = *slot;
        uintptr_t p = v & 0x0000FFFFFFFFFFFFULL;
        if (!in_from(p)) return;
        *slot = (v & ~0x0000FFFFFFFFFFFFULL) | copy(p);
    }
    void collect_now(uintptr_t fp, uintptr_t ip) {
        std::vector<uintptr_t*> roots;
        brass_enumerate_thread_roots(fp, ip, roots);
        ttop_ = 0;
        for (uintptr_t* s : roots) fix(s);
        size_t scan = 0;
        while (scan < ttop_) {
            uint64_t* hd = reinterpret_cast<uint64_t*>(to_ + scan);
            size_t n = (hd[0] + 7) / 8;
            for (size_t i = 0; i < n; ++i) {
                if ((hd[1] >> (i < 63 ? i : 63)) & 1) fix(reinterpret_cast<uintptr_t*>(hd + 3 + i));
            }
            scan += 24 + n * 8;
        }
        uint64_t* fw = reinterpret_cast<uint64_t*>(from_);
        for (size_t i = 0; i < kCap / 8; ++i) fw[i] = 0xDEADBEEFDEADBEEFULL;
        std::swap(from_, to_);
        top_ = ttop_;
        ++collections;
    }
    std::vector<uint8_t> a_, b_;
    uint8_t* from_;
    uint8_t* to_;
    size_t top_ = 0, ttop_ = 0;
    int allocs_ = 0;
};

// @ck holds %o across coro_create calls that fill the heap.
const char* kCoro = R"(module @co
func @gen(%a: i64) -> i64 {
bb0:
  %1 = iconst.i64 1
  %2 = coro_suspend.i64 %1, 1
  %3 = add.i64 %2, %a
  ret %3
}
func @ck(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  br loop(%z)
loop(%i: i64):
  %c = coro_create @gen(%i)
  coro_destroy %c
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %v = load.i64 %o, 16
  ret %v
}
)";

using CkFn = int64_t (*)(int64_t);

CkFn compile_ck(codegen::JitExecutionEngine& jit, Module& mod) {
    CoroTransformPass pass;
    pass.run_on_module(mod);
    REQUIRE(jit.compile_and_load(mod));
    auto fn = reinterpret_cast<CkFn>(jit.get_symbol_address("ck"));
    REQUIRE(fn != nullptr);
    return fn;
}

} // namespace

TEST_CASE("Sweep19 - GenerationalGC major collection tolerates a root slot reported twice") {
    GenerationalGC g(64 * 1024, 16 * 1024, 64 * 1024, 1);
    uintptr_t dead = g.allocate(48, 0, 2);
    uintptr_t live = g.allocate(48, 0, 2);
    reinterpret_cast<int64_t*>(dead)[2] = 1234;
    reinterpret_cast<int64_t*>(live)[2] = 42;
    g.register_root(&dead);
    g.register_root(&live);
    g.minor_collect();
    REQUIRE(g.is_old(dead));
    REQUIRE(g.is_old(live));
    g.unregister_root(&dead);
    g.unregister_root(&live);
    std::vector<uintptr_t*> extra{&live, &live};
    g.major_collect(extra);
    CHECK_EQ(reinterpret_cast<const int64_t*>(live)[2], 42);
    CHECK_EQ(g.tenured_used(), size_t{48 + sizeof(GenGcHeader)});
}

TEST_CASE("Sweep19 - GenerationalGC minor collection tolerates a root slot reported twice") {
    GenerationalGC g(64 * 1024, 16 * 1024, 64 * 1024, 3);
    uintptr_t live = g.allocate(48, 0, 2);
    reinterpret_cast<int64_t*>(live)[2] = 42;
    std::vector<uintptr_t*> extra{&live, &live};
    g.minor_collect(extra);
    g.minor_collect(extra);
    CHECK_EQ(reinterpret_cast<const int64_t*>(live)[2], 42);
}

TEST_CASE("Sweep19 - a major collection from a deopt's Tier-0 interpreter keeps its frames (Interpreter)") {
    run_major_after_deopt(false);
}

TEST_CASE("Sweep19 - a major collection from a deopt's Tier-0 interpreter keeps its frames (FastInterpreter)") {
    run_major_after_deopt(true);
}

TEST_CASE("Sweep19 - coro_create from generated code keeps the JIT frame's roots (GenerationalGC)") {
    auto mod = parse_ok(kCoro);
    codegen::JitExecutionEngine jit;
    GenerationalGC g(32 * 1024, 16 * 1024, 1 << 20, 2);
    brass_set_active_generational_gc(&g);
    CkFn ck = compile_ck(jit, *mod);
    const size_t minors = g.minor_collection_count();
    CHECK_EQ(ck(2000), 42);
    CHECK(g.minor_collection_count() > minors);
    brass_set_active_generational_gc(nullptr);
}

TEST_CASE("Sweep19 - coro_create from generated code keeps the JIT frame's roots (MiniCheneyGC)") {
    auto mod = parse_ok(kCoro);
    codegen::JitExecutionEngine jit;
    MiniCheneyGC gc(16 * 1024);
    brass_set_active_gc(&gc);
    CkFn ck = compile_ck(jit, *mod);
    const size_t collections = gc.collection_count();
    CHECK_EQ(ck(2000), 42);
    CHECK(gc.collection_count() > collections);
    brass_set_active_gc(nullptr);
}

TEST_CASE("Sweep19 - coro_create from generated code gives a host heap the JIT frame") {
    auto mod = parse_ok(kCoro);
    codegen::JitExecutionEngine jit;
    CopyHeap heap;
    set_host_heap(&heap);
    CkFn ck = compile_ck(jit, *mod);
    CHECK_EQ(ck(20), 42);
    CHECK(heap.collections > 0);
    CHECK_EQ(heap.framed, 21); // brass_gc_alloc and each coro_create
    set_host_heap(nullptr);
}
