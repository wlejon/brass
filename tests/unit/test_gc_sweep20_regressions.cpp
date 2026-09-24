// Regressions from bug sweep 20: the stack walk lost the generated frame
// under a C++ frame (brass_coro_resume, a host function calling generated
// code back), brass_coro_resume wrote to a coroutine frame a collection had
// moved, and C API entry points that created a function they then failed
// to return.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/brass_c_api.h>
#include <cstring>
#include <memory>
#include <string>

#if defined(_MSC_VER)
#include <malloc.h>
#define SWEEP20_NOINLINE __declspec(noinline)
#else
#define SWEEP20_NOINLINE __attribute__((noinline))
#endif

using namespace brass;

namespace {

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

using I64Fn = int64_t (*)(int64_t);

// A generational heap whose 32 KB nursery the callee fills many times.
struct GenHeap {
    GenerationalGC gc{32 * 1024, 16 * 1024, 1 << 20, 2};
    GenHeap() { brass_set_active_generational_gc(&gc); }
    ~GenHeap() { brass_set_active_generational_gc(nullptr); }
};

// @cr holds %o across a coro_resume whose body allocates n objects.
const char* kCr = R"(module @cr
func @body(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 48
  %zm = iconst.i64 0
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %zm, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %s = coro_suspend.i64 %ni, 1
  ret %s
}
func @cr(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %c = coro_create @body(%n)
  %r = coro_resume.i64 %c, %z
  coro_destroy %c
  %v = load.i64 %o, 16
  ret %v
}
)";

int64_t run_cr(int64_t n, size_t* minors) {
    auto mod = parse_ok(kCr);
    CoroTransformPass pass;
    pass.run_on_module(*mod);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    REQUIRE(jit.compile_and_load(*mod));
    auto cr = reinterpret_cast<I64Fn>(jit.get_symbol_address("cr"));
    REQUIRE(cr != nullptr);
    const int64_t r = cr(n);
    *minors = heap.gc.minor_collection_count();
    return r;
}

// @runs resumes a coroutine whose body allocates n objects and finishes
// with 7 without suspending, then resumes it again: a finished coroutine's
// resume returns the value its last run recorded in the frame.
const char* kDone = R"(module @cd
func @body(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 48
  %zm = iconst.i64 0
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %zm, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  %seven = iconst.i64 7
  ret %seven
}
func @runs(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  %c = coro_create @body(%n)
  %r1 = coro_resume.i64 %c, %z
  %r2 = coro_resume.i64 %c, %z
  coro_destroy %c
  %h = iconst.i64 100
  %a = mul.i64 %r1, %h
  %v = add.i64 %a, %r2
  ret %v
}
)";

// @f holds %o across a call to the host function host_cb, which calls the
// generated @g back; @g allocates n objects.
const char* kCb = R"(module @cb
func @g(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 48
  %zm = iconst.i64 0
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %zm, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %ni
}
func @f(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %r = call.i64 @host_cb(%n)
  %v = load.i64 %o, 16
  ret %v
}
)";

enum class CbMode { Plain, Scope, FramePointer };

I64Fn g_callback = nullptr;
CbMode g_cb_mode = CbMode::Plain;

extern "C" SWEEP20_NOINLINE int64_t sweep20_host_cb(int64_t n) {
    switch (g_cb_mode) {
    case CbMode::Scope: {
        uintptr_t rbp = 0, ip = 0;
        REQUIRE(brass_capture_caller_frame(rbp, ip));
        NativeFramesScope scope(rbp, ip);
        return g_callback(n);
    }
    case CbMode::FramePointer: {
#if defined(_MSC_VER)
        // _alloca gives the function an rbp-based frame, so rbp points at
        // this function's locals while it calls back.
        volatile char* p = static_cast<volatile char*>(_alloca(static_cast<size_t>(n & 0xff) + 64));
        p[0] = 1;
        return g_callback(n) + p[0] - 1;
#else
        return g_callback(n);
#endif
    }
    case CbMode::Plain:
        break;
    }
    return g_callback(n);
}

int64_t run_cb(CbMode mode, size_t* minors) {
    g_cb_mode = mode;
    auto mod = parse_ok(kCb);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    jit.register_external_symbol("host_cb", reinterpret_cast<void*>(&sweep20_host_cb));
    REQUIRE(jit.compile_and_load(*mod));
    g_callback = reinterpret_cast<I64Fn>(jit.get_symbol_address("g"));
    auto f = reinterpret_cast<I64Fn>(jit.get_symbol_address("f"));
    REQUIRE(g_callback != nullptr);
    REQUIRE(f != nullptr);
    const int64_t r = f(2000);
    *minors = heap.gc.minor_collection_count();
    g_callback = nullptr;
    return r;
}

} // namespace

TEST_CASE("Sweep20 - coro_resume keeps its generated caller's gcrefs across a collection in the body") {
    size_t minors = 0;
    CHECK_EQ(run_cr(10, &minors), 42);
    CHECK_EQ(minors, 0u);
    CHECK_EQ(run_cr(2000, &minors), 42);
    CHECK(minors > 0);
}

// The body's collections move the frame; both resumes must see the live copy.
TEST_CASE("Sweep20 - coro_resume of a body that collects and finishes, then resumed again") {
    auto mod = parse_ok(kDone);
    CoroTransformPass pass;
    pass.run_on_module(*mod);
    codegen::JitExecutionEngine jit;
    GenHeap heap;
    REQUIRE(jit.compile_and_load(*mod));
    auto runs = reinterpret_cast<I64Fn>(jit.get_symbol_address("runs"));
    REQUIRE(runs != nullptr);
    CHECK_EQ(runs(2000), 707);
    CHECK(heap.gc.minor_collection_count() > 0);
}

TEST_CASE("Sweep20 - host callback with a NativeFramesScope keeps the generated caller's gcrefs") {
    size_t minors = 0;
    CHECK_EQ(run_cb(CbMode::Scope, &minors), 42);
    CHECK(minors > 0);
}

#if defined(_WIN32) && defined(_M_X64)
// Without a scope, the walk must unwind through the host function's frame
// (Windows x64 only; elsewhere the host must record the transition).
TEST_CASE("Sweep20 - host callback without a scope keeps the generated caller's gcrefs") {
    size_t minors = 0;
    CHECK_EQ(run_cb(CbMode::Plain, &minors), 42);
    CHECK(minors > 0);
}

TEST_CASE("Sweep20 - host callback with an rbp frame keeps the generated caller's gcrefs") {
    size_t minors = 0;
    CHECK_EQ(run_cb(CbMode::FramePointer, &minors), 42);
    CHECK(minors > 0);
}
#endif

TEST_CASE("Sweep20 - C API function_create on a context-less module fails before creating") {
    BrassModule mod = brass_module_create(nullptr, "noctx");
    REQUIRE(mod != nullptr);
    BrassType params[1] = {brass_type_i64()};
    CHECK(brass_function_create(mod, "should_not_exist", brass_type_i64(), params, 1) == nullptr);
    char* mir = nullptr;
    REQUIRE(brass_module_print_mir(mod, &mir) == BRASS_OK);
    REQUIRE(mir != nullptr);
    CHECK(std::strstr(mir, "should_not_exist") == nullptr);
    brass_free_string(mir);
    brass_module_destroy(mod);
}

TEST_CASE("Sweep20 - C API function_create with a context returns the function") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);
    BrassModule mod = brass_module_create(ctx, "withctx");
    REQUIRE(mod != nullptr);
    BrassType params[1] = {brass_type_i64()};
    CHECK(brass_function_create(mod, "made", brass_type_i64(), params, 1) != nullptr);
    char* mir = nullptr;
    REQUIRE(brass_module_print_mir(mod, &mir) == BRASS_OK);
    CHECK(std::strstr(mir, "made") != nullptr);
    brass_free_string(mir);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}
