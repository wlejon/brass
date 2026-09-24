// Regressions from bug sweep 17: interpreter moves, the heap a fresh Tier-0
// interpreter (deoptimization, native-to-Tier-0 call) allocates from, and
// the roots a host heap can see.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <memory>
#include <type_traits>
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

// @specb deopts to @exitb (t != 1), which returns a new object; @specc
// deopts to @exitc, which collects and then reads the object it was given.
const char* kDeopt = R"(module @dg
func @exitb(%x: i64) -> gcref {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
func @specb(%x: i64, %t: i32) -> gcref {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @exitb, [%x]
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
func @exitc(%o: gcref, %x: i64) -> i64 {
b0:
  call @brass_gc_collect()
  %v = load.i64 %o, 16
  ret %v
}
func @specc(%o: gcref, %x: i64, %t: i32) -> i64 {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @exitc, [%o, %x]
  %v = load.i64 %o, 16
  ret %v
}
func @keepj() -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  call @brass_gc_collect()
  %v = load.i64 %o, 16
  ret %v
}
func @keepa() -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %p = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %p, 16, %k
  %v = load.i64 %o, 16
  ret %v
}
)";

FunctionHandle* install2(FunctionDispatchTable& prog, Module& mod, const char* name) {
    prog.tiering().get_feedback(name).set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create(name, mod.get_function(name));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, mod, name);
    REQUIRE(res.success);
    return h;
}

// A host heap that does not move or free anything, but on every
// collection asks brass for the thread's roots and counts the slots naming
// the watched object.
class RootCountHeap final : public HostHeap {
public:
    uintptr_t allocate(size_t size, uint64_t, uint32_t) override {
        blocks_.push_back(std::make_unique<uint64_t[]>((size + 7) / 8 + 1));
        return reinterpret_cast<uintptr_t>(blocks_.back().get());
    }
    void collect() override { count(0, 0); }
    void safepoint_at(uintptr_t fp, uintptr_t ip) override {
        if (fp != 0 && ip != 0) ++framed_safepoints;
        count(fp, ip);
    }

    void count(uintptr_t fp, uintptr_t ip) {
        ++collects;
        std::vector<uintptr_t*> roots;
        brass_append_native_frame_roots(roots);
        thread_roots_naming = 0;
        for (uintptr_t* s : roots) if (s && *s == watched) ++thread_roots_naming;
        roots.clear();
        brass_enumerate_thread_roots(fp, ip, roots);
        enumerated_naming = 0;
        for (uintptr_t* s : roots) if (s && *s == watched) ++enumerated_naming;
    }

    uintptr_t watched = 0;
    size_t thread_roots_naming = 0;
    size_t enumerated_naming = 0;
    int collects = 0;
    int framed_safepoints = 0;

private:
    std::vector<std::unique_ptr<uint64_t[]>> blocks_;
};

struct HeapScope {
    explicit HeapScope(HostHeap* heap) { set_host_heap(heap); }
    ~HeapScope() { set_host_heap(nullptr); }
};

struct GenScope {
    explicit GenScope(GenerationalGC* gc) { brass_set_active_generational_gc(gc); }
    ~GenScope() { brass_set_active_generational_gc(nullptr); }
};

void host_roots_case(bool fast) {
    auto mod = parse_ok(kDeopt);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    RootCountHeap heap;
    HeapScope scope(&heap);
    FunctionHandle* h = install2(prog, *mod, "specc");
    const uintptr_t o = brass_gc_alloc(48, 0, 2);
    reinterpret_cast<int64_t*>(o)[2] = 42;
    heap.watched = o;
    RuntimeValue r = h->call_native({RuntimeValue::from_bits(Type::gcref(), o), RuntimeValue::from_i64(7),
                                     RuntimeValue::from_i32(0)});
    CHECK_EQ(r.as_i64(), 42);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
    REQUIRE_EQ(heap.collects, 1);
    // The fresh interpreter's frame holding %o is a thread root although
    // no MiniCheneyGC is active.
    CHECK(heap.thread_roots_naming >= 1u);
    CHECK(heap.enumerated_naming >= 1u);
}

void generational_case(bool fast) {
    auto mod = parse_ok(kDeopt);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    GenerationalGC g;
    GenScope scope(&g);
    FunctionHandle* h = install2(prog, *mod, "specb");
    for (int t : {1, 0}) {
        RuntimeValue r = h->call_native({RuntimeValue::from_i64(7), RuntimeValue::from_i32(t)});
        const uintptr_t p = r.raw_bits();
        CHECK(g.is_valid_object(p));
        CHECK_EQ(reinterpret_cast<const int64_t*>(p)[2], 7);
    }
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
}

} // namespace

TEST_CASE("Sweep17 - interpreters are not movable (their heap's root provider captures this)") {
    CHECK(!std::is_move_constructible_v<Interpreter>);
    CHECK(!std::is_move_assignable_v<Interpreter>);
    CHECK(!std::is_move_constructible_v<FastInterpreter>);
    CHECK(!std::is_move_assignable_v<FastInterpreter>);
}

TEST_CASE("Sweep17 - deopt from host-called native code allocates in the active GenerationalGC (Interpreter)") {
    generational_case(false);
}

TEST_CASE("Sweep17 - deopt from host-called native code allocates in the active GenerationalGC (FastInterpreter)") {
    generational_case(true);
}

TEST_CASE("Sweep17 - a host heap sees the fresh Tier-0 interpreter's roots (Interpreter)") {
    host_roots_case(false);
}

TEST_CASE("Sweep17 - a host heap sees the fresh Tier-0 interpreter's roots (FastInterpreter)") {
    host_roots_case(true);
}

TEST_CASE("Sweep17 - a host heap's safepoint gets the JIT frame and can enumerate its roots") {
    auto mod = parse_ok(kDeopt);
    RootCountHeap heap;
    HeapScope scope(&heap);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)()>("keepj");
    REQUIRE(fn != nullptr);
    // Watches the object @keepj allocates and holds across its collection.
    struct Watch final : HostHeap {
        RootCountHeap& inner;
        explicit Watch(RootCountHeap& h) : inner(h) {}
        uintptr_t allocate(size_t s, uint64_t m, uint32_t t) override {
            const uintptr_t p = inner.allocate(s, m, t);
            inner.watched = p;
            return p;
        }
        void collect() override { inner.collect(); }
        void safepoint_at(uintptr_t fp, uintptr_t ip) override { inner.safepoint_at(fp, ip); }
    } watch(heap);
    HeapScope inner_scope(&watch);
    CHECK_EQ(fn(), 42);
    REQUIRE_EQ(heap.collects, 1);
    CHECK_EQ(heap.framed_safepoints, 1);
    CHECK(heap.enumerated_naming >= 1u);
}

TEST_CASE("Sweep18 - a host heap's allocation from generated code gets the JIT frame and can enumerate its roots") {
    auto mod = parse_ok(kDeopt);
    RootCountHeap heap;
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)()>("keepa");
    REQUIRE(fn != nullptr);
    // Watches @keepa's first object and, at its second allocation (where a
    // host may collect), asks brass for the roots from the frame it gets.
    struct Watch final : HostHeap {
        RootCountHeap& inner;
        int allocations = 0;
        int framed = 0;
        explicit Watch(RootCountHeap& h) : inner(h) {}
        uintptr_t allocate(size_t s, uint64_t m, uint32_t t) override { return inner.allocate(s, m, t); }
        uintptr_t allocate_at(size_t s, uint64_t m, uint32_t t, uintptr_t fp, uintptr_t ip) override {
            if (fp != 0 && ip != 0) ++framed;
            if (++allocations == 2) inner.count(fp, ip);
            const uintptr_t p = inner.allocate(s, m, t);
            if (allocations == 1) inner.watched = p;
            return p;
        }
    } watch(heap);
    HeapScope scope(&watch);
    CHECK_EQ(fn(), 42);
    CHECK_EQ(watch.allocations, 2);
    CHECK_EQ(watch.framed, 2);
    CHECK(heap.enumerated_naming >= 1u);
}
