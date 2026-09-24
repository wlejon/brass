// Regressions: the reference Interpreter's coroutine opcodes.
// - coro_create returned the frame as a ptr and coro_resume passed it to the
//   body as one; the interpreter's roots report only gcrefs, so a collection
//   that moved the frame left both stale (and yielded_val was written through
//   the stale pointer after the body returned). Reached whenever the
//   Interpreter allocates from the frame's heap: a borrowed heap (the fresh
//   Tier-0 interpreter a deopt starts) or a host heap.
// - coro_resume took every frame's fn_ptr for a Function*; a frame generated
//   code created holds a native code address there.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/host_heap.hpp>
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

// As tier_deopt's fresh interpreter: borrow the thread's heap and publish
// the interpreter's frames as thread roots.
int64_t run_borrowing(Module& mod, GenerationalGC* gen, MiniCheneyGC* ch) {
    Interpreter in;
    in.set_module(&mod);
    if (gen) in.borrow_generational_gc(gen);
    if (ch) in.borrow_gc(ch);
    ThreadRootsScope roots([](void* ctx, std::vector<uintptr_t*>& out) {
        static_cast<Interpreter*>(ctx)->collect_all_roots(out);
    }, &in);
    return in.run(*mod.get_function("main"), {RuntimeValue::from_i64(5)}).as_i64();
}

// Semispace copying host heap collecting on every 4th allocation, with the
// roots brass_enumerate_thread_roots reports.
class CopyHeap final : public HostHeap {
public:
    static constexpr size_t kCap = 1 << 20;
    CopyHeap() : a_(kCap), b_(kCap) { from_ = a_.data(); to_ = b_.data(); }
    uintptr_t allocate(size_t size, uint64_t mask, uint32_t tag) override {
        return allocate_at(size, mask, tag, 0, 0);
    }
    uintptr_t allocate_at(size_t size, uint64_t mask, uint32_t, uintptr_t fp, uintptr_t ip) override {
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
    int collections = 0;

private:
    bool in_from(uintptr_t p) const {
        return p >= reinterpret_cast<uintptr_t>(from_) && p < reinterpret_cast<uintptr_t>(from_) + kCap;
    }
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
        // Copies start at a different offset each time, so an object that
        // survives alone never lands where a stale pointer to it points.
        ttop_ = 64 * static_cast<size_t>(collections % 3 + 1);
        const size_t start = ttop_;
        for (uintptr_t* s : roots) fix(s);
        size_t scan = start;
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

TEST_CASE("InterpCoro - a borrowing Interpreter's coroutine frame survives a MiniCheneyGC move") {
    auto mod = lower(kMove);
    MiniCheneyGC ch(64 * 1024);
    brass_set_active_gc(&ch);
    int64_t r = 0;
    try {
        r = run_borrowing(*mod, nullptr, &ch);
    } catch (...) {
        brass_set_active_gc(nullptr);
        throw;
    }
    brass_set_active_gc(nullptr);
    CHECK_EQ(r, kMoveWant);
}

TEST_CASE("InterpCoro - a borrowing Interpreter's coroutine frame survives a GenerationalGC move") {
    auto mod = lower(kMove);
    GenerationalGC gen(32 * 1024, 16 * 1024, 1 << 20, 2);
    brass_set_active_generational_gc(&gen);
    int64_t r = 0;
    try {
        r = run_borrowing(*mod, &gen, nullptr);
    } catch (...) {
        brass_set_active_generational_gc(nullptr);
        throw;
    }
    brass_set_active_generational_gc(nullptr);
    CHECK_EQ(r, kMoveWant);
    CHECK(gen.minor_collection_count() > 0);
}

TEST_CASE("InterpCoro - an Interpreter's coroutine frame survives a copying host heap") {
    auto mod = lower(kMove);
    CopyHeap heap;
    set_host_heap(&heap);
    int64_t r = 0;
    try {
        Interpreter in;
        in.set_module(mod.get());
        r = in.run(*mod->get_function("main"), {RuntimeValue::from_i64(5)}).as_i64();
    } catch (...) {
        set_host_heap(nullptr);
        throw;
    }
    set_host_heap(nullptr);
    CHECK_EQ(r, kMoveWant);
    CHECK(heap.collections > 0);
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
