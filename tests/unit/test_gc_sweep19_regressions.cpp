// Regressions from bug sweep 19: a collection given the same root slot twice,
// a full collection from a deopt's Tier-0 interpreter, and brass_coro_create
// from generated code dropping the calling frame's roots.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/tracer.hpp>
#include <brass/codegen/jit_exec.hpp>
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
// allocates y (42), stores it in `old`, runs a full collection
// (brass_gc_collect) and returns old->y->16 + old->24.
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

void run_full_after_deopt(bool fast) {
    auto mod = parse_ok(kMajor);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    gc::HeapConfig config = test::small_heap_config(1);
    config.poison = true;
    gc::Heap heap(config);
    gc::HeapScope bind(heap);
    prog.tiering().get_feedback("specd").set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create("specd", mod->get_function("specd"));
    CodeInstaller installer(prog);
    REQUIRE(installer.install_tier2(*h, *mod, "specd").success);

    // 45 fillers, then `old`, promoted by one minor collection; the fillers
    // then die, so the full collection reclaims old-generation space around
    // the live `old`.
    std::vector<uint64_t> fill(45);
    for (auto& f : fill) {
        f = heap.allocate_masked(48, 0, 2);
        reinterpret_cast<int64_t*>(f)[3] = 1000;
        heap.add_root(&f);
    }
    uint64_t old = heap.allocate_masked(48, 1ULL << 2, 2);
    reinterpret_cast<int64_t*>(old)[3] = 7;
    heap.add_root(&old);
    heap.collect(gc::CollectionKind::Minor);
    for (auto& f : fill) heap.remove_root(&f);
    REQUIRE(heap.is_old(old));

    const uint64_t fulls = heap.stats().full_collections;
    RuntimeValue r = h->call_native({RuntimeValue::from_bits(Type::gcref(), old), RuntimeValue::from_i32(0)});
    CHECK_EQ(r.as_i64(), 49);
    CHECK(heap.stats().full_collections >= fulls + 1);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
    CHECK_EQ(reinterpret_cast<const int64_t*>(old)[3], 7);
    const uint64_t y = gc::Heap::load(old, 2);
    CHECK(heap.is_old(y)); // a full collection promotes every young survivor
    CHECK_EQ(reinterpret_cast<const int64_t*>(y)[2], 42);
    heap.remove_root(&old);
}

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

// Runs @ck with every coro_create collecting (`mode`), the object %o held
// only by the JIT frame, poisoned on every move.
void run_coro_create(gc::StressMode mode) {
    auto mod = parse_ok(kCoro);
    codegen::JitExecutionEngine jit;
    gc::HeapConfig config = test::small_heap_config();
    config.stress = mode;
    config.poison = true;
    config.read_environment = false;
    test::BoundHeap heap(config);
    CkFn ck = compile_ck(jit, *mod);
    const uint64_t collections = heap->collection_count();
    CHECK_EQ(ck(200), 42);
    CHECK(heap->collection_count() >= collections + 200);
}

// Visits `slot` twice, as a frame walk reporting the same slot for two
// stack-map entries does.
gc::Heap::RootSourceId add_twice(gc::Heap& heap, uint64_t* slot) {
    return heap.add_root_source([slot](gc::Tracer& t) {
        t.visit(slot);
        t.visit(slot);
    });
}

} // namespace

TEST_CASE("Sweep19 - a full collection tolerates a root slot reported twice") {
    gc::HeapConfig config = test::small_heap_config(1);
    config.poison = true;
    gc::Heap heap(config);
    uint64_t dead = heap.allocate_masked(48, 0, 2);
    uint64_t live = heap.allocate_masked(48, 0, 2);
    reinterpret_cast<int64_t*>(dead)[2] = 1234;
    reinterpret_cast<int64_t*>(live)[2] = 42;
    heap.add_root(&dead);
    heap.add_root(&live);
    heap.collect(gc::CollectionKind::Minor);
    REQUIRE(heap.is_old(dead));
    REQUIRE(heap.is_old(live));
    heap.collect(gc::CollectionKind::Full);
    const uint64_t both_live = heap.stats().old_live_bytes;
    heap.remove_root(&dead);
    heap.remove_root(&live);
    auto id = add_twice(heap, &live);
    heap.collect(gc::CollectionKind::Full);
    heap.collect(gc::CollectionKind::Full);
    CHECK(heap.is_valid_object(live));
    CHECK_EQ(reinterpret_cast<const int64_t*>(live)[2], 42);
    CHECK(heap.stats().old_live_bytes < both_live); // `dead` was reclaimed once, not kept
    heap.remove_root_source(id);
}

TEST_CASE("Sweep19 - a minor collection tolerates a root slot reported twice") {
    gc::HeapConfig config = test::small_heap_config(3);
    config.poison = true;
    gc::Heap heap(config);
    uint64_t live = heap.allocate_masked(48, 0, 2);
    reinterpret_cast<int64_t*>(live)[2] = 42;
    const uint64_t first = live;
    auto id = add_twice(heap, &live);
    heap.collect(gc::CollectionKind::Minor);
    heap.collect(gc::CollectionKind::Minor);
    CHECK(heap.is_young(live)); // copied once per collection, not twice
    CHECK_NE(live, first);
    CHECK(heap.is_valid_object(live));
    CHECK_EQ(reinterpret_cast<const int64_t*>(live)[2], 42);
    heap.remove_root_source(id);
}

TEST_CASE("Sweep19 - a full collection from a deopt's Tier-0 interpreter keeps its frames (Interpreter)") {
    run_full_after_deopt(false);
}

TEST_CASE("Sweep19 - a full collection from a deopt's Tier-0 interpreter keeps its frames (FastInterpreter)") {
    run_full_after_deopt(true);
}

TEST_CASE("Sweep19 - coro_create from generated code keeps the JIT frame's roots (minor stress)") {
    run_coro_create(gc::StressMode::Minor);
}

TEST_CASE("Sweep19 - coro_create from generated code keeps the JIT frame's roots (full stress)") {
    run_coro_create(gc::StressMode::Full);
}

TEST_CASE("Sweep19 - coro_create from generated code keeps the JIT frame's roots (alternate stress)") {
    run_coro_create(gc::StressMode::Alternate);
}
