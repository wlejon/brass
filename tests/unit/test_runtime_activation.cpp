// Runtime state that belongs to the code on a thread's stack: the stack maps
// a collection walks with, and the deopt handler a failed guard resumes
// through. Each must be the one of the code actually running on the thread,
// whichever thread compiled it and whatever was compiled last.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// @ra_work(n) keeps one object alive across a loop that allocates and polls
// a safepoint every iteration: sum over i < n of (3i + 1) + 7 + i. With a
// small heap every call collects many times, so a native frame whose gcref
// slots are not rooted reads a moved object. @ra_spin loops long enough to
// be OSR-compiled.
const char* kWorkSrc = R"(module @ra
func @ra_leaf(%x: i64) -> i64 {
b0:
  %c = iconst.i64 3
  %r = mul.i64 %x, %c
  ret %r
}
func @ra_helper(%x: i64) -> i64 {
b0:
  %a = call.i64 @ra_leaf(%x)
  %one = iconst.i64 1
  %r = add.i64 %a, %one
  ret %r
}
func @ra_spin(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %s = add.i64 %acc, %i
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %s), done(%s)
done(%r: i64):
  ret %r
}
func @ra_work(%n: i64) -> i64 {
entry:
  %sz = iconst.i64 64
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %keep = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %seven = iconst.i64 7
  store.i64 %keep, 16, %seven
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %tmp = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %tmp, 16, %i
  %h = call.i64 @ra_helper(%i)
  safepoint
  %kv = load.i64 %keep, 16
  %tv = load.i64 %tmp, 16
  %s1 = add.i64 %acc, %h
  %s2 = add.i64 %s1, %kv
  %s3 = add.i64 %s2, %tv
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %s3), done(%s3)
done(%r: i64):
  ret %r
}
)";

int64_t work_expected(int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) acc += (3 * i + 1) + 7 + i;
    return acc;
}

std::unique_ptr<Module> parse_work() {
    DiagnosticReporter diag;
    auto mod = parse_module(kWorkSrc, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

TieringConfig work_config() {
    TieringConfig c;
    c.invocation_tier1_threshold = 2;
    c.invocation_tier2_threshold = 1000000;
    c.backedge_osr_threshold = 50;
    c.enable_osr = true;
    c.enable_background_compile = false;
    return c;
}

constexpr size_t kSmallHeap = 4096;

// Restores the calling thread's active collector and stack maps.
struct GcStateGuard {
    MiniCheneyGC* gc = brass_get_active_gc();
    const ModuleStackMap* maps = brass_get_active_stack_maps();
    ~GcStateGuard() {
        brass_set_active_gc(gc);
        brass_set_active_stack_maps(maps);
    }
};

// @ra_guard_loop(n, deopt_at): 10 per iteration, n iterations. Its guard
// fails at iteration deopt_at, where the interpreter resumes the loop.
Function* build_guard_loop(Module& mod) {
    Function* fn = mod.create_function("ra_guard_loop", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(*fn);
    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b_resume = b.append_block("b_resume");
    BasicBlock* b3 = b.append_block("b3");
    fn->add_resume_point(1, b_resume);

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    Value* deopt_at = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* acc_val = b.add_param(Type::i64());
    b.build_br_if(b.build_slt(i_val, n), b2, {}, b3, {});

    b.position_at_end(b2);
    Instruction* g = b.build_guard(b.build_slt(i_val, deopt_at), "", {i_val, acc_val});
    g->set_resume_id(1);
    b.build_br(b1, {b.build_add(i_val, b.build_iconst_i64(1)), b.build_add(acc_val, b.build_iconst_i64(10))});

    b.position_at_end(b_resume);
    b.build_br(b1, {b.build_add(i_val, b.build_iconst_i64(1)), b.build_add(acc_val, b.build_iconst_i64(10))});

    b.position_at_end(b3);
    b.build_ret(acc_val);
    return fn;
}

} // namespace

TEST_CASE("Runtime activation - tier-1 code keeps its GC roots after an OSR compile on its thread") {
    GcStateGuard restore;
    auto mod = parse_work();
    FunctionDispatchTable prog;
    prog.pipeline().initialize(work_config());
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(50);
    Interpreter interp(kSmallHeap);
    brass_set_active_gc(&interp.gc());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    Function* work = mod->get_function("ra_work");
    Function* spin = mod->get_function("ra_spin");
    const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(150)};

    for (int i = 0; i < 3; ++i) CHECK_EQ(interp.run(*work, args).as_i64(), work_expected(150));
    REQUIRE(prog.find("ra_work")->tier() == TierLevel::Tier1_Baseline);

    // A different function OSRs: compiling and running its OSR module must
    // leave the maps tier-1 code on this thread is found with.
    const ModuleStackMap* maps_before = brass_get_active_stack_maps();
    const uint64_t osr0 = prog.osr().total_osr_migrations();
    CHECK_EQ(interp.run(*spin, {RuntimeValue::from_i64(1000)}).as_i64(), 499500);
    REQUIRE(prog.osr().total_osr_migrations() > osr0);
    CHECK(brass_get_active_stack_maps() == maps_before);

    for (int i = 0; i < 3; ++i) CHECK_EQ(interp.run(*work, args).as_i64(), work_expected(150));
    prog.pipeline().shutdown();
}

TEST_CASE("Runtime activation - a worker thread runs and collects in tier-1 code another thread compiled") {
    GcStateGuard restore;
    auto mod = parse_work();
    FunctionDispatchTable prog;
    prog.pipeline().initialize(work_config());
    prog.osr().set_enabled(false);
    Function* work = mod->get_function("ra_work");
    const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(150)};
    {
        Interpreter interp(kSmallHeap);
        brass_set_active_gc(&interp.gc());
        interp.set_dispatch_table(&prog);
        interp.set_module(mod.get());
        for (int i = 0; i < 4; ++i) CHECK_EQ(interp.run(*work, args).as_i64(), work_expected(150));
        brass_set_active_gc(restore.gc);
    }
    FunctionHandle* h = prog.find("ra_work");
    REQUIRE(h != nullptr);
    REQUIRE(h->tier() == TierLevel::Tier1_Baseline);

    // The worker installs only its interpreter's collector: no stack maps.
    std::atomic<int> bad{0};
    std::atomic<int64_t> direct{0};
    std::thread worker([&] {
        Interpreter interp(kSmallHeap);
        brass_set_active_gc(&interp.gc());
        interp.set_dispatch_table(&prog);
        interp.set_module(mod.get());
        for (int it = 0; it < 5; ++it) {
            const int64_t n = 100 + it * 37;
            if (interp.run(*work, {RuntimeValue::from_i64(n)}).as_i64() != work_expected(n)) bad.fetch_add(1);
        }
        auto fn = reinterpret_cast<int64_t (*)(int64_t)>(h->native_entry());
        direct.store(fn(150));
        brass_set_active_gc(nullptr);
    });
    worker.join();
    CHECK_EQ(bad.load(), 0);
    CHECK_EQ(direct.load(), work_expected(150));
    prog.pipeline().shutdown();
}

TEST_CASE("Runtime activation - concurrent OSR calls on two threads each resume their own deopt") {
    // Each thread runs its own program with a different trip count, and both
    // are inside their OSR code at once for most of it. A guard failure must
    // resume the interpreter frame of its own thread's OSR call.
    constexpr int kThreads = 2;
    constexpr int kRounds = 8;
    std::atomic<int> bad{0};
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            Module mod("ra_deopt_" + std::to_string(t));
            Function* fn = build_guard_loop(mod);
            FunctionDispatchTable prog;
            prog.osr().set_enabled(true);
            prog.osr().set_threshold(30);
            Interpreter interp;
            interp.set_dispatch_table(&prog);
            interp.set_module(&mod);
            for (int round = 0; round < kRounds; ++round) {
                const int64_t n = 3000000 + 1000000 * t;
                ready.fetch_add(1);
                while (ready.load() < kThreads * (round + 1)) std::this_thread::yield();
                const int64_t got =
                    interp.run(*fn, {RuntimeValue::from_i64(n), RuntimeValue::from_i64(n - 40)}).as_i64();
                if (got != 10 * n) bad.fetch_add(1);
            }
            if (prog.osr().total_osr_migrations() == 0 || prog.osr().total_native_deopts() == 0) bad.fetch_add(1);
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(bad.load(), 0);
}

#endif
