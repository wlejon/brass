// Regressions: the reference Interpreter's coroutine opcodes.
// - coro_create returned the frame as a ptr and coro_resume passed it to the
//   body as one; the interpreter's roots report only gcrefs, so a collection
//   that moved the frame left both stale (and yielded_val was written through
//   the stale pointer after the body returned). Reached whenever the
//   Interpreter allocates from the frame's heap: a shared heap (the fresh
//   Tier-0 interpreter a deopt starts) or the thread's heap.
// - coro_resume took every frame's fn_ptr for a Function*; a frame generated
//   code created holds a native code address there.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/runtime/coroutine.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> lower(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(*mod));
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::printf("%s", vdiag.format_all().c_str());
    REQUIRE(ok);
    return mod;
}

// @body allocates enough between its suspends to collect (and move its own
// frame); @main drives it with 10, 20, 30, 30.
const char* kMove = R"(module @m1
extern @brass_gc_alloc
func @body(%p: i64) -> i64 {
b0:
  %y0 = coro_suspend.i64 %p, 1
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 48
  %zm = iconst.i64 0
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %zm, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %n = iconst.i64 3000
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %k = add.i64 %y0, %p
  %y1 = coro_suspend.i64 %k, 2
  %r = add.i64 %y1, %k
  ret %r
}
func @main(%p: i64) -> i64 {
b0:
  %c = coro_create @body(%p)
  %a = iconst.i64 10
  %b = iconst.i64 20
  %d = iconst.i64 30
  %r0 = coro_resume.i64 %c, %a
  %r1 = coro_resume.i64 %c, %b
  %r2 = coro_resume.i64 %c, %d
  %r3 = coro_resume.i64 %c, %d
  coro_destroy %c
  %m = iconst.i64 1000
  %h0 = mul.i64 %r0, %m
  %h1 = add.i64 %h0, %r1
  %h2 = mul.i64 %h1, %m
  %h3 = add.i64 %h2, %r2
  %h4 = mul.i64 %h3, %m
  %h5 = add.i64 %h4, %r3
  ret %h5
}
)";

constexpr int64_t kMoveWant = 5025055055;

// As tier_deopt's fresh interpreter: allocate from the thread's heap and
// publish the interpreter's frames as thread roots as well.
int64_t run_borrowing(Module& mod, gc::Heap& heap) {
    Interpreter in(&heap);
    REQUIRE(!in.owns_heap());
    in.set_module(&mod);
    ThreadRootsScope roots([](void* ctx, std::vector<uintptr_t*>& out) {
        static_cast<Interpreter*>(ctx)->collect_all_roots(out);
    }, &in);
    return in.run(*mod.get_function("main"), {RuntimeValue::from_i64(5)}).as_i64();
}

// @drv resumes a frame it is handed; the frame here comes from generated code.
const char* kMix = R"(module @m2
func @body(%p: i64) -> i64 {
b0:
  %y0 = coro_suspend.i64 %p, 1
  %k = add.i64 %y0, %p
  %y1 = coro_suspend.i64 %k, 2
  %r = add.i64 %y1, %k
  ret %r
}
func @drv(%c: gcref) -> i64 {
b0:
  %a = iconst.i64 10
  %b = iconst.i64 20
  %d = iconst.i64 30
  %r0 = coro_resume.i64 %c, %a
  %r1 = coro_resume.i64 %c, %b
  %r2 = coro_resume.i64 %c, %d
  coro_destroy %c
  %m = iconst.i64 1000
  %h0 = mul.i64 %r0, %m
  %h1 = add.i64 %h0, %r1
  %h2 = mul.i64 %h1, %m
  %h3 = add.i64 %h2, %r2
  ret %h3
}
)";

} // namespace

// Promoted by its first minor collection: the frame moves into the old
// generation while the body runs.
TEST_CASE("InterpCoro - a borrowing Interpreter's coroutine frame survives a promoting move") {
    auto mod = lower(kMove);
    test::BoundHeap heap(test::small_heap_config(/*tenure_age=*/1));
    heap->set_poison(true);
    const int64_t r = run_borrowing(*mod, heap.heap);
    CHECK_EQ(r, kMoveWant);
    CHECK(heap.minor_collections() > 0);
}

// Copied between survivor spaces before its promotion.
TEST_CASE("InterpCoro - a borrowing Interpreter's coroutine frame survives a young-generation move") {
    auto mod = lower(kMove);
    test::BoundHeap heap(test::small_heap_config(/*tenure_age=*/2));
    heap->set_poison(true);
    const int64_t r = run_borrowing(*mod, heap.heap);
    CHECK_EQ(r, kMoveWant);
    CHECK(heap.minor_collections() > 0);
}

// A collection at every allocation (minor ones, every eighth full), freed
// memory poisoned: every stale frame pointer reads 0xDB bytes.
TEST_CASE("InterpCoro - an Interpreter's coroutine frame survives a collection at every allocation") {
    auto mod = lower(kMove);
    test::BoundHeap heap;
    heap->set_poison(true);
    heap->set_stress(gc::StressMode::Alternate);
    int64_t r = 0;
    {
        Interpreter in;
        REQUIRE(&in.heap() == &heap.heap);
        in.set_module(mod.get());
        r = in.run(*mod->get_function("main"), {RuntimeValue::from_i64(5)}).as_i64();
    }
    heap->set_stress(gc::StressMode::None);
    CHECK_EQ(r, kMoveWant);
    CHECK(heap.minor_collections() > 1000);
    CHECK(heap.full_collections() > 100);
}

TEST_CASE("InterpCoro - the Interpreter resumes a coroutine frame generated code created") {
    auto mod = lower(kMix);
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    void* code = jit.get_symbol_address("body");
    REQUIRE(code != nullptr);
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("body"));
    uintptr_t frame = brass_coro_create_at(code, std::max<uint32_t>(layout.slot_count, 1), layout.pointer_mask, 0, 0);
    REQUIRE(frame != 0);
    reinterpret_cast<runtime::BrassCoroFrame*>(frame)->slots[0] = 5;
    Interpreter in;
    in.set_module(mod.get());
    const RuntimeValue r = in.run(*mod->get_function("drv"), {RuntimeValue::from_gcref(frame)});
    CHECK_EQ(r.as_i64(), 5025055);
    CHECK(brass_coro_is_done(frame) != 0);
}
