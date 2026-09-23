// A physical register written by one instruction and read by a later one
// (a call's result in RAX until the copy out of it, a call argument in RCX,
// a dividend in RAX) holds a value across every instruction between them.
// The pre-RA scheduler, on by default in brass-opt, moves independent work
// into that gap; the register allocator once gave such work the held
// register, so a call's pointer result was overwritten with a float
// constant's bits and then written through.
//
// Each program runs on the tier-2 JIT with and without scheduling, and
// every answer must match the expected value.

#include "test_framework.hpp"
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/parser.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

namespace {

alignas(16) uint8_t g_buffer[128];

extern "C" void* held_preg_ext_p(int32_t) {
    return g_buffer;
}

int64_t run_jit(const std::string& source, const char* fn, const std::vector<RuntimeValue>& args, bool schedule) {
    DiagnosticReporter diag;
    auto mod = parse_module(source, &diag);
    if (!mod) {
        std::cerr << "parse failed: " << diag.format_all() << "\n";
        return INT64_MIN;
    }
    JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("ext_p", reinterpret_cast<void*>(&held_preg_ext_p));
    SchedOptions sched;
    sched.enable_pre_ra = schedule;
    sched.enable_post_ra = schedule;
    if (!jit.compile_and_load(*mod, 0, sched)) {
        std::cerr << "compile failed\n";
        return INT64_MIN;
    }
    std::memset(g_buffer, 0, sizeof(g_buffer));
    return static_cast<int64_t>(jit.invoke(fn, args).raw_bits());
}

void check_both(const std::string& source, const char* fn, const std::vector<RuntimeValue>& args, int64_t expected) {
    CHECK_EQ(run_jit(source, fn, args, false), expected);
    CHECK_EQ(run_jit(source, fn, args, true), expected);
}

constexpr int64_t kFive = 4617315517961601024; // bits of 5.0

} // namespace

// The sweep repro (cd3.mir @f2, @f4, @f5): a float constant, materialized
// through a GPR temporary, stored through the call's pointer result.
TEST_CASE("Regalloc held pregs - float constant stored through a call result") {
    const std::string src = R"(
module @m
extern @ext_p

func @via_bitcast() -> i64 {
b0:
  %0 = iconst.i32 7
  %1 = call.ptr @ext_p(%0)
  %2 = fconst.f64 5.0
  %3 = bitcast.i64.f64 %2
  store.i64 %1, 32, %3
  %4 = load.i64 %1, 32
  ret %4
}

func @via_store_f64() -> i64 {
b0:
  %0 = iconst.i32 7
  %1 = call.ptr @ext_p(%0)
  %2 = fconst.f64 5.0
  store.f64 %1, 32, %2
  %3 = load.i64 %1, 32
  ret %3
}

func @via_iconst() -> i64 {
b0:
  %0 = iconst.i32 7
  %1 = call.ptr @ext_p(%0)
  %2 = iconst.i64 4617315517961601024
  %3 = bitcast.f64.i64 %2
  store.f64 %1, 32, %3
  %4 = load.i64 %1, 32
  ret %4
}

func @returns_ptr() -> ptr {
b0:
  %0 = iconst.i32 7
  %1 = call.ptr @ext_p(%0)
  %2 = fconst.f64 5.0
  %3 = bitcast.i64.f64 %2
  store.i64 %1, 32, %3
  ret %1
}
)";
    check_both(src, "via_bitcast", {}, kFive);
    check_both(src, "via_store_f64", {}, kFive);
    check_both(src, "via_iconst", {}, kFive);
    const int64_t buf = static_cast<int64_t>(reinterpret_cast<uintptr_t>(g_buffer));
    check_both(src, "returns_ptr", {}, buf);
    int64_t stored = 0;
    std::memcpy(&stored, g_buffer + 32, 8);
    CHECK_EQ(stored, kFive);
}

// cd2.mir: the GC frame push/pop shape bronze emits, with a host function
// standing in for the runtime, twice in one block.
TEST_CASE("Regalloc held pregs - frame push result written between calls") {
    const std::string src = R"(
module @m
extern @ext_p

func @f() -> i64 {
b0:
  %0 = iconst.i32 7
  %1 = call.ptr @ext_p(%0)
  %2 = fconst.f64 5.0
  %3 = bitcast.i64.f64 %2
  store.i64 %1, 32, %3
  %4 = load.i64 %1, 32
  %5 = call.ptr @ext_p(%0)
  %6 = fconst.f64 2.5
  %7 = bitcast.i64.f64 %6
  store.i64 %5, 40, %7
  %8 = load.i64 %5, 40
  %9 = add.i64 %4, %8
  ret %9
}
)";
    constexpr int64_t kTwoAndHalf = 4612811918334230528; // bits of 2.5
    check_both(src, "f", {}, static_cast<int64_t>(static_cast<uint64_t>(kFive) + static_cast<uint64_t>(kTwoAndHalf)));
}

// Siblings: fixed registers around divisions and shifts, with constants
// the scheduler can move into the gap.
TEST_CASE("Regalloc held pregs - division and shift operands with constants") {
    const std::string src = R"(
module @m

func @div(%0: i64, %1: i64) -> i64 {
b0:
  %2 = sdiv.i64 %0, %1
  %3 = iconst.i64 1099511627781
  %4 = fconst.f64 5.0
  %5 = bitcast.i64.f64 %4
  %6 = smod.i64 %0, %1
  %7 = add.i64 %2, %3
  %8 = add.i64 %7, %5
  %9 = add.i64 %8, %6
  ret %9
}

func @shl(%0: i64, %1: i64) -> i64 {
b0:
  %2 = shl.i64 %0, %1
  %3 = fconst.f64 5.0
  %4 = bitcast.i64.f64 %3
  %5 = lshr.i64 %0, %1
  %6 = add.i64 %2, %4
  %7 = add.i64 %6, %5
  ret %7
}
)";
    const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(1000003), RuntimeValue::from_i64(7)};
    const uint64_t div_expected = static_cast<uint64_t>(1000003 / 7) + 1099511627781ull +
                                  static_cast<uint64_t>(kFive) + static_cast<uint64_t>(1000003 % 7);
    check_both(src, "div", args, static_cast<int64_t>(div_expected));
    const uint64_t shl_expected = (1000003ull << 7) + static_cast<uint64_t>(kFive) + (1000003ull >> 7);
    check_both(src, "shl", args, static_cast<int64_t>(shl_expected));
}
