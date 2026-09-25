// Regressions: a coroutine frame an interpreter creates has a MIR body
// (CORO_FLAG_MIR_BODY: fn_ptr is the lowered body's Function). Every tier
// must resume it:
// - brass_coro_resume took any fn_ptr for machine code and called the
//   Function object: a frame the Interpreter or FastInterpreter created and
//   handed to tier-2 code (or to the host C API, or MicrotaskQueue) crashed.
// - the interpreters recognized only bodies of their current module, so a
//   body from another module went to brass_coro_resume the same way.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/coroutine.hpp>
#include "gc_test_heap.hpp"
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> lower(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    // (It reports no change for a module without coroutine bodies.)
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
#define BODY_SRC                                   \
    "func @body(%p: i64) -> i64 {\n"               \
    "b0:\n"                                        \
    "  %y0 = coro_suspend.i64 %p, 1\n"             \
    "  %k = add.i64 %y0, %p\n"                     \
    "  %y1 = coro_suspend.i64 %k, 2\n"             \
    "  %r = add.i64 %y1, %k\n"                     \
    "  ret %r\n"                                   \
    "}\n"

#define DRV_SRC                                    \
    "func @drv(%c: gcref) -> i64 {\n"              \
    "b0:\n"                                        \
    "  %a = iconst.i64 10\n"                       \
    "  %b = iconst.i64 20\n"                       \
    "  %d = iconst.i64 30\n"                       \
    "  %r0 = coro_resume.i64 %c, %a\n"             \
    "  %r1 = coro_resume.i64 %c, %b\n"             \
    "  %r2 = coro_resume.i64 %c, %d\n"             \
    "  coro_destroy %c\n"                          \
    "  %m = iconst.i64 1000\n"                     \
    "  %h0 = mul.i64 %r0, %m\n"                    \
    "  %h1 = add.i64 %h0, %r1\n"                   \
    "  %h2 = mul.i64 %h1, %m\n"                    \
    "  %h3 = add.i64 %h2, %r2\n"                   \
    "  ret %h3\n"                                  \
    "}\n"

constexpr int64_t kWant = 5025055;

// @main creates the frame (interpreted body) and hands it to @drv.
const char* kI2N = "module @i2n\n" BODY_SRC DRV_SRC
                   "func @main(%p: i64) -> i64 {\n"
                   "b0:\n"
                   "  %c = coro_create @body(%p)\n"
                   "  %r = call.i64 @drv(%c)\n"
                   "  ret %r\n"
                   "}\n";

// The body alone, and a maker returning its frame.
const char* kBodyMod = "module @bodies\n" BODY_SRC
                       "func @mk(%p: i64) -> gcref {\n"
                       "b0:\n"
                       "  %c = coro_create @body(%p)\n"
                       "  ret %c\n"
                       "}\n";

// Another module's driver.
const char* kDrvMod = "module @drivers\n" DRV_SRC;

TieringConfig no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

// Installs @drv natively at tier 2 in `prog`.
void install_drv_tier2(FunctionDispatchTable& prog, Module& mod) {
    prog.pipeline().initialize(no_tierup());
    prog.tiering().get_feedback("drv").set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create("drv", mod.get_function("drv"));
    REQUIRE(h != nullptr);
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, mod, "drv");
    if (!res.success) std::printf("install_tier2: %s\n", res.error_message.c_str());
    REQUIRE(res.success);
    REQUIRE(h->native_entry() != nullptr);
}

// A frame the Interpreter made for @body of kBodyMod, with p = 5. The caller
// binds a heap first: the frame is an object of the thread's heap, which must
// outlive the interpreter (one with no heap bound would take the frame down
// with its own).
uintptr_t make_frame(Module& bodies) {
    REQUIRE(gc::Heap::current() != nullptr);
    Interpreter in;
    in.set_module(&bodies);
    const RuntimeValue c = in.run(*bodies.get_function("mk"), {RuntimeValue::from_i64(5)});
    const uintptr_t frame = static_cast<uintptr_t>(c.raw_bits());
    REQUIRE(frame != 0);
    REQUIRE((reinterpret_cast<BrassCoroFrame*>(frame)->flags & CORO_FLAG_MIR_BODY) != 0);
    return frame;
}

} // namespace

TEST_CASE("CoroMirFrames - tier-2 code resumes a frame the Interpreter created") {
    auto mod = lower(kI2N);
    FunctionDispatchTable prog;
    install_drv_tier2(prog, *mod);
    Interpreter in;
    in.set_dispatch_table(&prog);
    in.set_module(mod.get());
    CHECK_EQ(in.run(*mod->get_function("main"), {RuntimeValue::from_i64(5)}).as_i64(), kWant);
}

TEST_CASE("CoroMirFrames - tier-2 code resumes a frame the FastInterpreter created") {
    auto mod = lower(kI2N);
    FunctionDispatchTable prog;
    install_drv_tier2(prog, *mod);
    FastInterpreter fi;
    fi.set_dispatch_table(&prog);
    fi.set_module(mod.get());
    CHECK_EQ(fi.run(*mod->get_function("main"), {RuntimeValue::from_i64(5)}).as_i64(), kWant);
}

TEST_CASE("CoroMirFrames - the Interpreter resumes a body of another module") {
    auto bodies = lower(kBodyMod);
    auto drivers = lower(kDrvMod);
    test::BoundHeap heap;
    const uintptr_t frame = make_frame(*bodies);
    Interpreter in;
    in.set_module(drivers.get());
    const RuntimeValue r = in.run(*drivers->get_function("drv"), {RuntimeValue::from_gcref(frame)});
    CHECK_EQ(r.as_i64(), kWant);
    CHECK(brass_coro_is_done(frame) != 0);
}

TEST_CASE("CoroMirFrames - the FastInterpreter resumes a body of another module") {
    auto bodies = lower(kBodyMod);
    auto drivers = lower(kDrvMod);
    test::BoundHeap heap;
    const uintptr_t frame = make_frame(*bodies);
    FastInterpreter fi;
    fi.set_module(drivers.get());
    const RuntimeValue r = fi.run(*drivers->get_function("drv"), {RuntimeValue::from_gcref(frame)});
    CHECK_EQ(r.as_i64(), kWant);
    CHECK(brass_coro_is_done(frame) != 0);
}

TEST_CASE("CoroMirFrames - the host resumes an interpreter-created frame through brass_coro_resume") {
    auto bodies = lower(kBodyMod);
    test::BoundHeap heap;
    const uintptr_t frame = make_frame(*bodies);
    CHECK_EQ(brass_coro_resume(frame, 10), 5u);
    CHECK_EQ(brass_coro_resume(frame, 20), 25u);
    CHECK(brass_coro_is_done(frame) == 0);
    CHECK_EQ(brass_coro_resume(frame, 30), 55u);
    CHECK(brass_coro_is_done(frame) != 0);
    CHECK(!is_active_coro_frame(frame));
    // Done: further resumes return the last value.
    CHECK_EQ(brass_coro_resume(frame, 40), 55u);
}

TEST_CASE("CoroMirFrames - MicrotaskQueue resumes an interpreter-created frame") {
    auto bodies = lower(kBodyMod);
    test::BoundHeap heap;
    const uintptr_t frame = make_frame(*bodies);
    auto* cf = reinterpret_cast<BrassCoroFrame*>(frame);
    MicrotaskQueue q;
    q.enqueue_coro(cf, 10);
    q.enqueue_coro(cf, 20);
    q.run_all();
    CHECK_EQ(cf->yielded_val, 25u);
    q.enqueue_coro(cf, 30);
    q.run_all();
    CHECK_EQ(cf->yielded_val, 55u);
    CHECK(cf->is_done != 0);
}

// The host resumes, with a moving heap active, a frame whose body collects
// between its suspends: the fresh Tier-0 interpreter allocates from that
// heap, and the frame (moved) is reached through the host's root.
TEST_CASE("CoroMirFrames - a host-resumed MIR frame survives a moving collection") {
    const char* src = R"(module @mv
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
func @mk(%p: i64) -> gcref {
b0:
  %c = coro_create @body(%p)
  ret %c
}
)";
    auto mod = lower(src);
    test::BoundHeap ch;  // 64 KB eden: the body's 3000 objects collect it
    uint64_t y[3] = {};
    size_t collections = 0;
    {
        uintptr_t frame = 0;
        ThreadRootsScope root([](void* ctx, std::vector<uintptr_t*>& roots) {
            roots.push_back(static_cast<uintptr_t*>(ctx));
        }, &frame);
        {
            Interpreter in;
            in.set_module(mod.get());
            REQUIRE(&in.heap() == &ch.heap);
            frame = static_cast<uintptr_t>(in.run(*mod->get_function("mk"), {RuntimeValue::from_i64(5)}).raw_bits());
        }
        REQUIRE(frame != 0);
        y[0] = brass_coro_resume(frame, 10);
        y[1] = brass_coro_resume(frame, 20);
        y[2] = brass_coro_resume(frame, 30);
        collections = ch.heap.collection_count();
    }
    CHECK_EQ(y[0], 5u);
    CHECK_EQ(y[1], 25u);
    CHECK_EQ(y[2], 55u);
    CHECK(collections > 0);
}
