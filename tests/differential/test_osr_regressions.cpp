// OSR regressions: loops whose OSR entry carries constants defined before
// the loop (S3-1: they once clobbered loop-carried values on entry) and
// loops that allocate or hold gcrefs (S3-3: native allocations once had no
// heap). Each entry runs in a program on its fast interpreter, first to ask
// for the loops' OSR code and then, once it is compiled, entering it; both
// runs answer what the interpreter answers without OSR.

#include "test_framework.hpp"
#include <brass/core/diagnostics.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/mir/parser.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <initializer_list>
#include <memory>
#include <string_view>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_or_fail(std::string_view src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    return mod;
}

// Native code allocates through brass_gc_alloc: on the interpreter's heap
// while the scope lasts.
using ActiveHeap = gc::HeapScope;

// Runs `entry` in a program on its fast interpreter with OSR at `threshold`
// (0 = OSR off), twice: the second run enters the OSR code the first asked
// for. Both runs must agree.
int64_t run_entry(const Module& mod, std::string_view entry, uint64_t threshold) {
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(threshold != 0);
    if (threshold != 0) prog.osr().set_threshold(threshold);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    ActiveHeap heap(interp.heap());
    const int64_t first = interp.run(mod, entry).as_i64();
    CompilePool::shared().wait_owner(&prog.osr());
    const int64_t second = interp.run(mod, entry).as_i64();
    CHECK_EQ(first, second);
    return second;
}

constexpr std::string_view kHoistedConstants = R"(
module @osr_consts

func @dense() -> i64 {
b0:
  %0 = iconst.i64 7
  %1 = iconst.i64 300
  %2 = iconst.i64 0
  %3 = iconst.i64 1
  br b1(%2, %2)

b1(%4: i64, %5: i64):
  %6 = slt.i64 %5, %1
  br_if %6, b2, b3

b2:
  %7 = add.i64 %4, %0
  %8 = add.i64 %5, %3
  br b1(%7, %8)

b3:
  ret %4
}

func @distinct() -> i64 {
b0:
  %0 = iconst.i64 7
  %1 = iconst.i64 300
  %2 = iconst.i64 0
  %3 = iconst.i64 1
  %4 = iconst.i64 0
  br b1(%2, %4)

b1(%5: i64, %6: i64):
  %7 = slt.i64 %6, %1
  br_if %7, b2, b3

b2:
  %8 = add.i64 %5, %0
  %9 = add.i64 %6, %3
  br b1(%8, %9)

b3:
  ret %5
}

func @spin(%0: i64, %1: i64) -> i64 {
b0:
  %2 = iconst.i64 0
  %3 = iconst.i64 1
  %4 = iconst.i64 1000000000000
  br b1(%2, %2)

b1(%5: i64, %6: i64):
  %7 = slt.i64 %6, %1
  br_if %7, b2, b3

b2:
  %8 = add.i64 %5, %0
  %9 = add.i64 %8, %4
  %10 = sub.i64 %9, %4
  %11 = add.i64 %6, %3
  br b1(%10, %11)

b3:
  ret %5
}

func @main_spin() -> i64 {
b0:
  %0 = iconst.i64 7
  %1 = iconst.i64 300
  %2 = call.i64 @spin(%0, %1)
  ret %2
}
)";

constexpr std::string_view kAllocatingLoops = R"(
module @osr_alloc

extern @brass_gc_alloc

func @spin(%0: gcref, %1: i64) -> i64 {
b0:
  %2 = iconst.i64 0
  %3 = iconst.i64 1
  br b1(%2, %2)

b1(%5: i64, %6: i64):
  %7 = slt.i64 %6, %1
  br_if %7, b2, b3

b2:
  %8 = load.i64 %0, 0
  %9 = add.i64 %5, %8
  %11 = add.i64 %6, %3
  safepoint
  br b1(%9, %11)

b3:
  ret %5
}

func @spin_alloc(%0: i64) -> i64 {
b0:
  %1 = iconst.i64 16
  %2 = iconst.i64 0
  %3 = iconst.i32 1
  %4 = iconst.i64 1
  br b1(%2, %2)

b1(%5: i64, %6: i64):
  %7 = slt.i64 %6, %0
  br_if %7, b2, b3

b2:
  %8 = call.gcref @brass_gc_alloc(%1, %2, %3)
  store.i64 %8, 0, %6
  %9 = load.i64 %8, 0
  %10 = add.i64 %5, %9
  %11 = add.i64 %6, %4
  br b1(%10, %11)

b3:
  ret %5
}

func @main() -> i64 {
b0:
  %0 = iconst.i64 16
  %1 = iconst.i64 0
  %2 = iconst.i32 1
  %3 = call.gcref @brass_gc_alloc(%0, %1, %2)
  %4 = iconst.i64 7
  store.i64 %3, 0, %4
  %5 = iconst.i64 300
  %6 = call.i64 @spin(%3, %5)
  ret %6
}

func @main2() -> i64 {
b0:
  %0 = iconst.i64 300
  %1 = call.i64 @spin_alloc(%0)
  ret %1
}

func @build(%0: i64) -> gcref {
b0:
  %1 = iconst.i64 24
  %2 = iconst.i64 2
  %3 = iconst.i32 1
  %4 = iconst.i64 0
  %5 = iconst.i64 1
  %6 = call.gcref @brass_gc_alloc(%1, %4, %3)
  store.i64 %6, 0, %4
  br b1(%6, %5)

b1(%7: gcref, %8: i64):
  %9 = sle.i64 %8, %0
  br_if %9, b2, b3

b2:
  %10 = call.gcref @brass_gc_alloc(%1, %2, %3)
  %11 = mul.i64 %8, %8
  store.i64 %10, 0, %11
  store.gcref %10, 8, %7
  safepoint
  %13 = add.i64 %8, %5
  br b1(%10, %13)

b3:
  ret %7
}

func @hold() -> i64 {
b0:
  %0 = iconst.i64 300
  %1 = call.gcref @build(%0)
  %2 = call.gcref @build(%0)
  %3 = load.i64 %1, 0
  %4 = load.i64 %2, 0
  %5 = add.i64 %3, %4
  ret %5
}
)";

} // namespace

TEST_CASE("OSR regression S3-1 - constants hoisted out of the loop do not clobber loop values") {
    auto mod = parse_or_fail(kHoistedConstants);
    for (std::string_view entry : {"dense", "distinct"}) {
        CHECK_EQ(run_entry(*mod, entry, 0), 2100);
        for (uint64_t t : {1u, 2u, 5u, 10u, 100u, 299u}) {
            CHECK_EQ(run_entry(*mod, entry, t), 2100);
        }
    }
}

TEST_CASE("OSR regression S3-1 - live-in parameter and a non-immediate constant") {
    auto mod = parse_or_fail(kHoistedConstants);
    CHECK_EQ(run_entry(*mod, "main_spin", 0), 2100);
    for (uint64_t t : {1u, 3u, 10u, 150u, 299u}) {
        CHECK_EQ(run_entry(*mod, "main_spin", t), 2100);
    }
}

TEST_CASE("OSR regression S3-3 - an allocating loop allocates on the host's heap in its OSR code") {
    auto mod = parse_or_fail(kAllocatingLoops);
    CHECK_EQ(run_entry(*mod, "main2", 0), 44850);
    for (uint64_t t : {1u, 5u, 100u}) {
        CHECK_EQ(run_entry(*mod, "main2", t), 44850);
    }
}

TEST_CASE("OSR regression S3-3 - loops carrying gcrefs answer the interpreter's result") {
    // A gcref live into a loop header is no OSR entry's (the entry's buffer
    // carries no gcref): these loops stay interpreted.
    auto mod = parse_or_fail(kAllocatingLoops);
    CHECK_EQ(run_entry(*mod, "main", 0), 2100);
    CHECK_EQ(run_entry(*mod, "hold", 0), 180000);
    for (uint64_t t : {1u, 5u, 100u}) {
        CHECK_EQ(run_entry(*mod, "main", t), 2100);
        CHECK_EQ(run_entry(*mod, "hold", t), 180000);
    }
}

#endif
