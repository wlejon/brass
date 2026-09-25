// Regressions from bug sweep 17: interpreter moves, the heap a fresh Tier-0
// interpreter (deoptimization, native-to-Tier-0 call) allocates from, and
// the roots a collection on the thread's heap sees.
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
#include <brass/gc/heap.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <algorithm>
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

// A heap bound to the thread whose freed memory is poisoned (a stale slot
// reads 0xDB bytes), plus a root source that visits nothing but, at every
// collection, asks brass for the thread roots the heap visits and counts the
// slots naming the watched object.
struct RootCountHeap {
    gc::Heap heap;
    gc::HeapScope bind{heap};
    gc::Heap::RootSourceId source = 0;
    uintptr_t watched = 0;
    size_t thread_roots_naming = 0;
    int collects = 0;

    RootCountHeap() {
        heap.set_poison(true);
        source = heap.add_root_source([this](gc::Tracer&) { count(); });
    }
    ~RootCountHeap() { heap.remove_root_source(source); }

    // The most slots any one root pass saw: a verifying heap (BRASS_GC_VERIFY)
    // runs the sources again after the object has moved, naming it no more.
    void count() {
        ++collects;
        std::vector<uintptr_t*> roots;
        brass_append_native_frame_roots(roots);
        size_t naming = 0;
        for (uintptr_t* s : roots) if (s && *s == watched) ++naming;
        thread_roots_naming = std::max(thread_roots_naming, naming);
    }
};

void host_roots_case(bool fast) {
    auto mod = parse_ok(kDeopt);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    RootCountHeap heap;
    FunctionHandle* h = install2(prog, *mod, "specc");
    // The host's reference is not a root: only the frames holding %o are.
    const uintptr_t o = brass_gc_alloc(48, 0, 2);
    reinterpret_cast<int64_t*>(o)[2] = 42;
    heap.watched = o;
    RuntimeValue r = h->call_native({RuntimeValue::from_bits(Type::gcref(), o), RuntimeValue::from_i64(7),
                                     RuntimeValue::from_i32(0)});
    CHECK_EQ(r.as_i64(), 42);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
    REQUIRE(heap.collects >= 1);
    // The fresh interpreter's frame holding %o is a thread root.
    CHECK(heap.thread_roots_naming >= 1u);
}

void generational_case(bool fast) {
    auto mod = parse_ok(kDeopt);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    gc::Heap g;
    gc::HeapScope scope(g);
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

TEST_CASE("Sweep17 - deopt from host-called native code allocates in the thread's heap (Interpreter)") {
    generational_case(false);
}

TEST_CASE("Sweep17 - deopt from host-called native code allocates in the thread's heap (FastInterpreter)") {
    generational_case(true);
}

TEST_CASE("Sweep17 - a collection sees the fresh Tier-0 interpreter's roots (Interpreter)") {
    host_roots_case(false);
}

TEST_CASE("Sweep17 - a collection sees the fresh Tier-0 interpreter's roots (FastInterpreter)") {
    host_roots_case(true);
}

// brass_gc_collect from generated code walks the JIT frame: the object
// @keepj holds across it moves (a full collection promotes it) and the
// frame's slot is updated, else the load reads poison.
TEST_CASE("Sweep17 - a collection requested by generated code walks the JIT frame") {
    auto mod = parse_ok(kDeopt);
    gc::Heap heap;
    heap.set_poison(true);
    gc::HeapScope scope(heap);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)()>("keepj");
    REQUIRE(fn != nullptr);
    const uint64_t before = heap.stats().full_collections;
    CHECK_EQ(fn(), 42);
    CHECK(heap.stats().full_collections > before);
    CHECK(heap.stats().promoted_bytes > 0);
}

// An allocation from generated code that collects walks the JIT frame: in
// stress mode @keepa's second allocation collects while its first object
// is live in the frame.
TEST_CASE("Sweep18 - a collection in an allocation from generated code walks the JIT frame") {
    auto mod = parse_ok(kDeopt);
    gc::Heap heap;
    heap.set_poison(true);
    gc::HeapScope scope(heap);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)()>("keepa");
    REQUIRE(fn != nullptr);
    heap.set_stress(gc::StressMode::Minor);
    const uint64_t before = heap.collection_count();
    const int64_t r = fn();
    heap.set_stress(gc::StressMode::None);
    CHECK_EQ(r, 42);
    CHECK(heap.collection_count() >= before + 2);
}
