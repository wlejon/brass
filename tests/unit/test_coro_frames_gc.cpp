// Coroutine frames and the collector, at every tier of the body: a frame
// traces any number of reference slots (no 64-bit mask limit), a frame that
// became old keeps the young references its body stores after a resume (the
// transform's barriers), suspended frames survive stress collections, and
// the awaiter link keeps its frame alive, follows it when it moves, and is
// the chain an async stack walks.
#include "coro_tier_harness.hpp"
#include <brass/gc/native_frames.hpp>
#include <brass/runtime/coroutine.hpp>
#include <bit>
#include <string>

using namespace brass;
using namespace brass::coro_test;
using namespace brass::runtime;

namespace {

// @big allocates n objects (sentinel 7i + 1 in each), suspends with all of
// them live, allocates `churn` more objects in a loop, then sums the
// sentinels. @mkbig creates its frame.
std::string big_body_source(int n) {
    std::string s = "module @big\nextern @brass_gc_alloc\nfunc @big(%churn: i64) -> i64 {\nb0:\n"
                    "  %sz = iconst.i64 16\n  %zm = iconst.i64 0\n  %tg = iconst.i32 7\n";
    for (int i = 0; i < n; ++i) {
        const std::string o = "%o" + std::to_string(i);
        s += "  " + o + " = call.gcref @brass_gc_alloc(%sz, %zm, %tg)\n";
        s += "  %s" + std::to_string(i) + " = iconst.i64 " + std::to_string(7 * i + 1) + "\n";
        s += "  store.i64 " + o + ", 0, %s" + std::to_string(i) + "\n";
    }
    s += "  %y = coro_suspend.i64 %zm, 1\n  br loop(%zm)\n"
         "loop(%i: i64):\n"
         "  %t = call.gcref @brass_gc_alloc(%sz, %zm, %tg)\n"
         "  %one = iconst.i64 1\n  %ni = add.i64 %i, %one\n"
         "  %more = slt.i64 %ni, %churn\n  br_if %more, loop(%ni), done\n"
         "done:\n  %a0 = iconst.i64 0\n";
    for (int i = 0; i < n; ++i) {
        s += "  %l" + std::to_string(i) + " = load.i64 %o" + std::to_string(i) + ", 0\n";
        s += "  %a" + std::to_string(i + 1) + " = add.i64 %a" + std::to_string(i) + ", %l" + std::to_string(i) + "\n";
    }
    s += "  ret %a" + std::to_string(n) + "\n}\n";
    s += "func @runbig(%churn: i64) -> i64 {\nb0:\n  %c = coro_create @big(%churn)\n"
         "  %z = iconst.i64 0\n  %y = coro_resume.i64 %c, %z\n  %r = coro_resume.i64 %c, %z\n"
         "  coro_destroy %c\n  ret %r\n}\n";
    return s;
}

int64_t big_sum(int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += 7 * i + 1;
    return sum;
}

// @keep allocates an object after its first resume and keeps it across a
// second suspend; @mk creates its frame.
constexpr std::string_view kKeep = R"(module @keep
extern @brass_gc_alloc
func @keep(%p: i64) -> i64 {
b0:
  %z = iconst.i64 0
  %a = coro_suspend.i64 %z, 1
  %sz = iconst.i64 16
  %tg = iconst.i32 7
  %y = call.gcref @brass_gc_alloc(%sz, %z, %tg)
  store.i64 %y, 0, %p
  %b = coro_suspend.i64 %z, 2
  %v = load.i64 %y, 0
  ret %v
}
func @mk(%p: i64) -> gcref {
b0:
  %c = coro_create @keep(%p)
  ret %c
}
)";

// A host root for one frame handle.
struct RootedFrame {
    uintptr_t frame = 0;
    ThreadRootsScope scope{[](void* ctx, std::vector<uintptr_t*>& roots) {
                               roots.push_back(static_cast<uintptr_t*>(ctx));
                           }, &frame};
};

} // namespace

TEST_CASE("CoroFramesGc - a frame traces more reference slots than a 64-bit mask holds") {
    constexpr int kRefs = 100;
    const std::string src = big_body_source(kRefs);
    auto mod = parse_program(src);
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("big"));
    CHECK(!layout.fits_pointer_mask);
    uint32_t refs = 0;
    for (uint64_t w : layout.ref_bits) refs += static_cast<uint32_t>(std::popcount(w));
    CHECK(refs >= static_cast<uint32_t>(kRefs));
    for (Tier bt : kTiers) {
        for (Tier dt : {Tier::Interp, Tier::Base, Tier::Opt}) {
            test::BoundHeap heap;
            Program prog(*mod);
            prog.place("big", bt);
            prog.place("runbig", dt);
            const uint64_t before = heap->collection_count();
            const int64_t got = prog.call("runbig", dt, {RuntimeValue::from_i64(6000)}).as_i64();
            if (got != big_sum(kRefs)) std::printf("body %s, driver %s\n", tier_name(bt), tier_name(dt));
            CHECK_EQ(got, big_sum(kRefs));
            CHECK(heap->collection_count() > before);
        }
    }
}

TEST_CASE("CoroFramesGc - suspended frames with many references survive stress collections at every tier") {
    constexpr int kRefs = 70;
    const std::string src = big_body_source(kRefs);
    auto mod = parse_program(src);
    for (gc::StressMode mode : {gc::StressMode::Minor, gc::StressMode::Full, gc::StressMode::Alternate}) {
        for (Tier bt : kTiers) {
            gc::HeapConfig cfg = test::small_heap_config();
            cfg.stress = mode;
            cfg.poison = true;
            test::BoundHeap heap(cfg);
            Program prog(*mod);
            prog.place("big", bt);
            prog.place("runbig", Tier::Base);
            CHECK_EQ(prog.call("runbig", Tier::Base, {RuntimeValue::from_i64(50)}).as_i64(), big_sum(kRefs));
        }
    }
}

TEST_CASE("CoroFramesGc - a frame that became old keeps the young objects its body stores after a resume") {
    auto mod = parse_program(kKeep);
    for (Tier bt : kTiers) {
        gc::HeapConfig cfg = test::small_heap_config();
        cfg.poison = true;
        test::BoundHeap heap(cfg);
        Program prog(*mod);
        prog.place("keep", bt);
        RootedFrame f;
        f.frame = static_cast<uintptr_t>(prog.call("mk", Tier::Interp, {RuntimeValue::from_i64(4242)}).raw_bits());
        REQUIRE(f.frame != 0);
        runtime::ProgramScope scope(prog.table);
        CHECK_EQ(brass_coro_resume(f.frame, 0), 0u);
        heap->collect(gc::CollectionKind::Full);
        heap->collect(gc::CollectionKind::Full);
        REQUIRE(heap->is_old(f.frame));
        // The body allocates a young object and keeps it in the old frame.
        CHECK_EQ(brass_coro_resume(f.frame, 0), 0u);
        heap->collect(gc::CollectionKind::Minor);
        for (int i = 0; i < 2000; ++i) (void)heap->allocate_masked(16, 0, 1);
        heap->collect(gc::CollectionKind::Minor);
        const uint64_t got = brass_coro_resume(f.frame, 0);
        if (got != 4242u) std::printf("body %s\n", tier_name(bt));
        CHECK_EQ(got, 4242u);
        CHECK(brass_coro_is_done(f.frame) != 0);
    }
}

TEST_CASE("CoroFramesGc - the awaiter link keeps its frame alive, moves with it, and is the async stack") {
    auto mod = parse_program(kKeep);
    gc::HeapConfig cfg = test::small_heap_config();
    cfg.poison = true;
    test::BoundHeap heap(cfg);
    Program prog(*mod);
    RootedFrame inner, outer;
    inner.frame = static_cast<uintptr_t>(prog.call("mk", Tier::Interp, {RuntimeValue::from_i64(1)}).raw_bits());
    outer.frame = static_cast<uintptr_t>(prog.call("mk", Tier::Interp, {RuntimeValue::from_i64(2)}).raw_bits());
    brass_coro_set_awaiter(inner.frame, outer.frame);
    CHECK_EQ(brass_coro_awaiter(inner.frame), outer.frame);
    // Finished, the outer frame is no longer a root of its own: only the
    // link reaches it.
    brass_coro_destroy(outer.frame);
    outer.frame = 0;
    const uintptr_t before = brass_coro_awaiter(inner.frame);
    for (int i = 0; i < 3; ++i) heap->collect(gc::CollectionKind::Full);
    const uintptr_t after = brass_coro_awaiter(inner.frame);
    CHECK(after != 0);
    CHECK(heap->contains(after));
    (void)before;
    const auto* o = reinterpret_cast<const BrassCoroFrame*>(after);
    CHECK_EQ(o->is_done, 1u);
    CHECK(coro_body(o) != nullptr);
    const std::vector<CoroStackEntry> stack = coro_async_stack(inner.frame);
    REQUIRE_EQ(stack.size(), size_t{2});
    CHECK_EQ(stack[0].frame, inner.frame);
    CHECK(stack[0].name == "keep");
    CHECK_EQ(stack[1].frame, after);
    // A cycle ends the walk.
    brass_coro_set_awaiter(after, inner.frame);
    CHECK_EQ(coro_async_stack(inner.frame).size(), size_t{2});
    brass_coro_set_awaiter(after, 0);
}

namespace {
int64_t g_seen_depth = -1;
uintptr_t g_seen_frame = 0;
extern "C" int64_t coro_test_probe() {
    const std::vector<CoroStackEntry> s = current_async_stack();
    g_seen_depth = static_cast<int64_t>(s.size());
    g_seen_frame = current_coro_frame();
    return g_seen_depth;
}
} // namespace

TEST_CASE("CoroFramesGc - while a body runs its frame is the current one, and its awaiters the async stack") {
    constexpr std::string_view src = R"(module @probe
extern @coro_test_probe
extern @brass_coro_set_awaiter
func @pb(%f: gcref, %aw: gcref) -> i64 {
b0:
  call @brass_coro_set_awaiter(%f, %aw)
  %z = iconst.i64 0
  %y = coro_suspend.i64 %z, 1
  %d = call.i64 @coro_test_probe()
  ret %d
}
func @idle() -> i64 {
b0:
  %z = iconst.i64 0
  %y = coro_suspend.i64 %z, 1
  ret %z
}
func @pdrv(%x: i64) -> i64 {
b0:
  %outer = coro_create @idle()
  %c = coro_create @pb(%outer)
  %z = iconst.i64 0
  %y0 = coro_resume.i64 %c, %z
  %d = coro_resume.i64 %c, %z
  ret %d
}
)";
    auto mod = parse_program(src);
    for (Tier bt : {Tier::Base, Tier::Opt}) {
        test::BoundHeap heap;
        Program prog(*mod);
        prog.table.pipeline().register_external_symbol("coro_test_probe", reinterpret_cast<void*>(&coro_test_probe));
        prog.place("pb", bt);
        g_seen_depth = -1;
        CHECK_EQ(prog.call("pdrv", Tier::Interp, {RuntimeValue::from_i64(0)}).as_i64(), 2);
        CHECK_EQ(g_seen_depth, 2);
        CHECK(g_seen_frame != 0);
        CHECK_EQ(current_coro_frame(), uintptr_t{0});
    }
}
