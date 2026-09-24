// Regressions from bug sweep 35:
// - MicrotaskQueue tasks and a Promise's waiters held raw coroutine frame
//   pointers: after the frame's heap was torn down the queue resumed freed
//   memory, and after a collection moved the frame it resumed the old copy.
//   They now hold a CoroFrameRef, which follows the frame and turns a torn
//   down heap into a hard error.
// - Unregistering a frame read other heaps' registries while their
//   collections wrote them. Each registry now has its own lock, held by its
//   heap's collection (CoroRootsLock) and taken by every reader.
// - Generated code's coro_resume linked "brass_coro_resume" (the C++-throwing
//   entry) in AOT objects; the JIT alone bound it to the natively re-raising
//   brass_coro_resume_from_generated. Both targets now name the latter.
// - The parallel runtime called generated kernels from C++ with no
//   GeneratedCodeEntryScope.
// - retype_exception_value reinterpreted a thrown value whose type a landing
//   pad could not hold; that is now an InterpreterException.
#include "test_framework.hpp"
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/target/target.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

struct ActiveGc {
    explicit ActiveGc(MiniCheneyGC* gc) noexcept : prev(brass_get_active_gc()) { brass_set_active_gc(gc); }
    ~ActiveGc() { brass_set_active_gc(prev); }
    MiniCheneyGC* prev;
};

std::unique_ptr<Module> lower(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    CoroTransformPass pass;
    pass.run_on_module(*mod);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::printf("%s", vdiag.format_all().c_str());
    REQUIRE(ok);
    return mod;
}

// @body yields p, then y0 + p, then returns y1 + y0 + p. Resumed with 10,
// 20, 30 from p = 5: 5, 25, 55.
const char* kBodyMod = R"(module @s35body
func @body(%p: i64) -> i64 {
b0:
  %y0 = coro_suspend.i64 %p, 1
  %k = add.i64 %y0, %p
  %y1 = coro_suspend.i64 %k, 2
  %r = add.i64 %y1, %k
  ret %r
}
func @mk(%p: i64) -> gcref {
b0:
  %c = coro_create @body(%p)
  ret %c
}
)";

// A frame for @body with p = 5, allocated from the thread's active heap.
uintptr_t make_frame(Module& mod) {
    Interpreter in;
    in.set_module(&mod);
    const RuntimeValue c = in.run(*mod.get_function("mk"), {RuntimeValue::from_i64(5)});
    const uintptr_t frame = static_cast<uintptr_t>(c.raw_bits());
    REQUIRE(frame != 0);
    return frame;
}

template <typename F>
bool throws_logic_error(F&& f) {
    try {
        f();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

} // namespace

TEST_CASE("Sweep35 - a queued coroutine whose heap was torn down is a hard error, not a resume of freed memory") {
    auto mod = lower(kBodyMod);
    MicrotaskQueue q;
    Promise p;
    uintptr_t f = 0;
    {
        MiniCheneyGC heap(1 << 20);
        ActiveGc a(&heap);
        f = make_frame(*mod);
        REQUIRE(is_active_coro_frame(f));
        q.enqueue_coro(reinterpret_cast<BrassCoroFrame*>(f), 10);
        p.await_in(reinterpret_cast<BrassCoroFrame*>(f));
    }
    CHECK(!is_active_coro_frame(f));
    CHECK(throws_logic_error([&] { q.run_all(); }));
    q.clear();

    // The Promise's waiter is resolved when the global queue runs it.
    auto& global = get_global_microtask_queue();
    global.clear();
    p.fulfill(1);
    CHECK(throws_logic_error([&] { global.run_all(); }));
    global.clear();

    // A reference taken while the heap lived stays a hard error.
    CHECK(!is_active_coro_frame(f));
}

TEST_CASE("Sweep35 - brass_coro_destroy writes only to a frame that is still registered") {
    auto mod = lower(kBodyMod);
    // Stand-in memory for a frame whose heap is gone: an unregistered handle
    // is never written through.
    alignas(8) unsigned char fake[sizeof(BrassCoroFrame)];
    std::memset(fake, 0xAB, sizeof fake);
    brass_coro_destroy(reinterpret_cast<uintptr_t>(fake));
    for (unsigned char c : fake) CHECK_EQ(static_cast<unsigned>(c), 0xABu);

    // A live frame is finished and unregistered.
    MiniCheneyGC heap(1 << 20);
    ActiveGc a(&heap);
    const uintptr_t f = make_frame(*mod);
    CoroFrameRef ref = coro_frame_ref(reinterpret_cast<BrassCoroFrame*>(f));
    REQUIRE(is_active_coro_frame(f));
    brass_coro_destroy(f);
    CHECK(brass_coro_is_done(f) != 0);
    CHECK(!is_active_coro_frame(f));
    CHECK(resolve_coro_frame_ref(ref) == nullptr);
}

TEST_CASE("Sweep35 - a queued coroutine is resumed where a collection moved it") {
    auto mod = lower(kBodyMod);
    MiniCheneyGC heap(1 << 20);
    ActiveGc a(&heap);
    const uintptr_t f = make_frame(*mod);
    CoroFrameRef ref = coro_frame_ref(reinterpret_cast<BrassCoroFrame*>(f));
    MicrotaskQueue q;
    q.enqueue_coro(reinterpret_cast<BrassCoroFrame*>(f), 10);
    q.enqueue_coro(reinterpret_cast<BrassCoroFrame*>(f), 20);
    heap.collect();  // a semispace copy: the frame moves
    BrassCoroFrame* moved = resolve_coro_frame_ref(ref);
    REQUIRE(moved != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(moved) != f);
    q.run_all();
    moved = resolve_coro_frame_ref(ref);
    REQUIRE(moved != nullptr);
    CHECK_EQ(moved->yielded_val, 25u);
    q.enqueue_coro(ref, 30);
    q.run_all();
    // Finished: the reference resolves to nothing and a further task is a
    // no-op.
    CHECK(resolve_coro_frame_ref(ref) == nullptr);
    q.enqueue_coro(ref, 40);
    q.run_all();
    CHECK(resolve_coro_frame_ref(ref) == nullptr);
}

TEST_CASE("Sweep35 - unregistering waits for another heap's collection to finish with its coroutine roots") {
    auto mod = lower(kBodyMod);
    MiniCheneyGC heap(1 << 20);
    uintptr_t f = 0;
    {
        ActiveGc a(&heap);
        f = make_frame(*mod);
    }
    REQUIRE(is_active_coro_frame(f));
    std::atomic<bool> done{false};
    std::thread t;
    {
        // As the heap's collection holds it while it updates root slots.
        CoroRootsLock collecting(heap.coro_frames());
        // No heap is active on that thread: the frame is in another heap's
        // registry, which it may read only once the collection is done.
        t = std::thread([&] {
            unregister_active_coro_frame(reinterpret_cast<BrassCoroFrame*>(f));
            done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(!done.load());
    }
    t.join();
    CHECK(done.load());
    CHECK(!is_active_coro_frame(f));
}

TEST_CASE("Sweep35 - AOT code resumes coroutines through brass_coro_resume_from_generated") {
    const char* src = R"(module @s35aot
func @drv(%c: gcref) -> i64 {
b0:
  %a = iconst.i64 10
  %r = coro_resume.i64 %c, %a
  ret %r
}
)";
    for (const Target& target : {Target::x64_windows(), Target::x64_linux(), Target::aarch64_windows(),
                                 Target::aarch64_linux()}) {
        auto mod = lower(src);
        object::ObjectFile obj = object::compile_module_to_object(*mod, target);
        bool from_generated = false;
        bool cxx_entry = false;
        for (const auto& sym : obj.symbols) {
            if (sym.name == "brass_coro_resume_from_generated") from_generated = true;
            if (sym.name == "brass_coro_resume") cxx_entry = true;
        }
        CHECK(from_generated);
        CHECK(!cxx_entry);
    }
}

namespace {

std::atomic<int> g_unbounded_chunks{0};
std::atomic<int> g_chunks{0};

void entry_scope_kernel(uint64_t, uint64_t, void*) {
    g_chunks.fetch_add(1);
    if (brass_innermost_entry_scope() == nullptr) g_unbounded_chunks.fetch_add(1);
}

} // namespace

TEST_CASE("Sweep35 - the parallel runtime enters its kernels under a generated-code entry scope") {
    for (uint64_t n : {uint64_t(8), uint64_t(100000)}) {
        g_unbounded_chunks = 0;
        g_chunks = 0;
        brass_parallel_for(n, 64, entry_scope_kernel, nullptr, ReductionKind::None, nullptr);
        CHECK(g_chunks.load() > 0);
        CHECK_EQ(g_unbounded_chunks.load(), 0);
    }
    CHECK(brass_innermost_entry_scope() == nullptr);
}

namespace {

const char* kRetypeSrc = R"(module @s35eh
func @thr32(%x: i32) -> i32 {
b0:
  throw %x
}
func @thrf(%x: f64) -> f64 {
b0:
  throw %x
}
func @thr64(%x: i64) -> i64 {
b0:
  throw %x
}
func @i32_as_f64(%x: i32) -> f64 {
b0:
  %v = invoke.i32 @thr32(%x), ok, bad
ok:
  %z = fconst.f64 0.0
  ret %z
bad:
  %e = landing_pad.f64
  ret %e
}
func @f64_as_i64(%x: f64) -> i64 {
b0:
  %v = invoke.f64 @thrf(%x), ok, bad
ok:
  %z = iconst.i64 0
  ret %z
bad:
  %e = landing_pad.i64
  ret %e
}
func @i64_as_i32(%x: i64) -> i32 {
b0:
  %v = invoke.i64 @thr64(%x), ok, bad
ok:
  %z = iconst.i32 0
  ret %z
bad:
  %e = landing_pad.i32
  ret %e
}
)";

} // namespace

TEST_CASE("Sweep35 - a landing pad whose type cannot hold the thrown value is a hard error") {
    auto mod = lower(kRetypeSrc);
    Interpreter in;
    in.set_module(mod.get());
    bool threw = false;
    try {
        (void)in.run(*mod->get_function("i32_as_f64"), {RuntimeValue::from_i32(7)});
    } catch (const InterpreterThrownException&) {
        // The mismatch must not pass as the thrown value either.
    } catch (const InterpreterException& e) {
        threw = std::string(e.what()).find("landing_pad") != std::string::npos;
    }
    CHECK(threw);

    // The same width is a legal reinterpretation (f64 bits in an i64 pad),
    // and an untyped/i64 value narrows to the pad's integer width.
    Interpreter in2;
    in2.set_module(mod.get());
    const double d = 2.5;
    uint64_t dbits = 0;
    std::memcpy(&dbits, &d, sizeof d);
    CHECK_EQ(static_cast<uint64_t>(in2.run(*mod->get_function("f64_as_i64"), {RuntimeValue::from_f64(d)}).as_i64()),
             dbits);
    Interpreter in3;
    in3.set_module(mod.get());
    CHECK_EQ(in3.run(*mod->get_function("i64_as_i32"), {RuntimeValue::from_i64(0x123456789LL)}).as_i64(),
             int64_t(0x23456789));

    // Directly: width mismatches and vector/scalar mixes throw.
    auto rejects = [](RuntimeValue v, Type t) {
        try {
            (void)retype_exception_value(v, t);
        } catch (const InterpreterException&) {
            return true;
        }
        return false;
    };
    CHECK(rejects(RuntimeValue::from_i32(1), Type::f64()));
    CHECK(rejects(RuntimeValue::from_f32(1.0f), Type::f64()));
    CHECK(rejects(RuntimeValue::from_f64(1.0), Type::f32()));
    CHECK(rejects(RuntimeValue::from_f32(1.0f), Type::ptr()));
    CHECK(rejects(RuntimeValue::from_i32(1), Type::i32x4()));
    CHECK(!rejects(RuntimeValue::from_i32(1), Type::f32()));
    CHECK(!rejects(RuntimeValue::from_i64(1), Type::i16()));
    CHECK(!rejects(RuntimeValue::from_i32(1), Type::i64()));  // i64 carries any scalar's bits
    CHECK(!rejects(RuntimeValue::from_f64(1.0), Type::gcref()));
}
