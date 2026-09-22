// Small MIR programs that each pinned one JIT/interpreter disagreement found
// by the differential fuzzer. Every program runs through the reference
// interpreter, the unoptimized JIT and the Bronze pipeline (interpreter and
// JIT) over a handful of arguments, and all answers must agree.

#include "test_framework.hpp"
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/mir/parser.hpp>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace brass;
using namespace brass::fuzz;

namespace {

using ArgPair = std::pair<int64_t, int64_t>;

const std::vector<ArgPair> kIntArgs = {
    {0, 0}, {1, 0}, {3, 1}, {7, 2}, {26, 2}, {28, 0}, {-5, 3},
    {-1099511627781, 9}, {1099511627781, -9},
    {std::numeric_limits<int64_t>::min(), -1}, {std::numeric_limits<int64_t>::max(), 5},
};

// Runs `fuzz_fn` of `source` on every argument pair; true when all tiers agree.
bool all_tiers_agree(const std::string& source, const std::vector<ArgPair>& arg_sets) {
    DiagnosticReporter diag;
    auto mod = parse_module(source, &diag);
    if (!mod) {
        std::cerr << "parse failed: " << diag.format_all() << "\n";
        return false;
    }
    DiffFuzzerOptions opts;
    opts.pipeline = FuzzPipeline::Bronze;
    opts.save_reproducers = false;
    opts.bisect = false;
    opts.timeout_ms = 3000;
    DiffFuzzer fuzzer(opts);
    bool ok = true;
    for (const auto& [a, b] : arg_sets) {
        const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b)};
        DiffResult r = fuzzer.run_test(*mod, "fuzz_fn", args, 0);
        if (!r.passed) {
            std::cerr << "args (" << a << ", " << b << "): [" << r.failure_class << "] "
                      << r.mismatch_reason << "\n";
            ok = false;
        }
    }
    return ok;
}

} // namespace

// A constant used only as a switch case argument got no virtual register, so
// the edge copy read [rbp+0].
TEST_CASE("JIT regression - switch case edge arguments that are constants") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 0
  %3 = iconst.i64 7
  %4 = iconst.i64 3
  %5 = iconst.i64 5
  %6 = iconst.i64 -4
  %7 = iconst.i64 7
  %8 = and.i64 %0, %7
  switch.i64 %8, default: dflt, [3: c5, 1: merge(%2, %4), 6: merge(%2, %5), 0: c5, 4: merge(%2, %6)]
c5:
  br merge(%2, %3)
dflt:
  br merge(%3, %3)
merge(%9: i64, %10: i64):
  %11 = iconst.i64 1099511628211
  %12 = mul.i64 %9, %11
  %13 = xor.i64 %12, %10
  ret %13
}
)";
    std::vector<ArgPair> args;
    for (int64_t i = 0; i < 8; ++i) args.push_back({i, 0});
    REQUIRE(all_tiers_agree(src, args));
}

// Division and remainder by constants: a non-power-of-two divisor was folded
// but never materialized, and a signed remainder by 2^k (k >= 32) truncated
// its mask to 32 bits.
TEST_CASE("JIT regression - division and remainder by constants") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 7
  %3 = udiv.i64 %0, %2
  %4 = umod.i64 %0, %2
  %5 = trunc.i32 %0
  %6 = iconst.i32 7
  %7 = udiv.i32 %5, %6
  %8 = zext.i64 %7
  %9 = iconst.i64 1099511627776
  %10 = smod.i64 %0, %9
  %11 = sdiv.i64 %0, %9
  %12 = umod.i64 %0, %9
  %13 = xor.i64 %3, %4
  %14 = add.i64 %13, %8
  %15 = iconst.i64 31
  %16 = mul.i64 %14, %15
  %17 = xor.i64 %16, %10
  %18 = mul.i64 %17, %15
  %19 = xor.i64 %18, %11
  %20 = mul.i64 %19, %15
  %21 = xor.i64 %20, %12
  ret %21
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// A load was fused into a mul whose constant factor is strength-reduced, and
// into sub/compare with a constant first operand, although the lowering never
// used the fused form: the load's value had no register.
TEST_CASE("JIT regression - loads feeding strength-reduced and constant-first ALU ops") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = alloca 32, 8
  %3 = iconst.i64 0
  %4 = iconst.i64 1
  store_indexed.i64 %2, %3, 8, %0
  store_indexed.i64 %2, %4, 8, %1
  %5 = load_indexed.i64 %2, %3, 8
  %6 = iconst.i64 1099511627776
  %7 = mul.i64 %5, %6
  %8 = load_indexed.i64 %2, %4, 8
  %9 = iconst.i64 100
  %10 = sub.i64 %9, %8
  %11 = load_indexed.i64 %2, %3, 8
  %12 = slt.i64 %9, %11
  %13 = zext.i64 %12
  %14 = xor.i64 %7, %10
  %15 = add.i64 %14, %13
  ret %15
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// Reduced fuzz reproducers: 32-bit compares, selects and shifts whose operand
// fusion disagreed with the lowering.
TEST_CASE("JIT regression - 32-bit compare, select and shift fusion") {
    const char* a = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = trunc.i32 %0
  %3 = trunc.i32 %1
  %4 = not.i32 %2
  %5 = ule.i32 %4, %3
  %6 = zext.i64 %5
  %7 = iconst.i32 1
  %8 = or.i32 %3, %7
  %9 = iconst.i32 -706421614
  %10 = or.i32 %9, %5
  %11 = iconst.i32 5
  %12 = add.i32 %11, %10
  %13 = iconst.i32 31
  %14 = and.i32 %12, %13
  %15 = shl.i32 %8, %14
  %16 = zext.i64 %15
  %17 = iconst.i64 1099511628211
  %18 = mul.i64 %6, %17
  %19 = xor.i64 %18, %16
  ret %19
}
)";
    const char* b = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = trunc.i32 %0
  %3 = iconst.i32 -719182246
  %4 = or.i32 %3, %2
  %5 = slt.i32 %4, %2
  %6 = iconst.i32 -3
  %7 = mul.i32 %2, %6
  %8 = iconst.i32 -2147483648
  %9 = select.i32 %5, %8, %7
  %10 = ssub_overflow.i64 %0, %0
  %11 = iconst.i32 -165191666
  %12 = sub.i32 %2, %11
  %13 = iconst.i32 1
  %14 = or.i32 %12, %13
  %15 = iconst.i32 -1
  %16 = select.i32 %10, %15, %14
  %17 = smod.i32 %9, %16
  %18 = sext.i64 %17
  ret %18
}
)";
    const char* c = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = trunc.i32 %0
  %3 = iconst.i32 5
  %4 = iconst.i32 31
  %5 = and.i32 %3, %4
  %6 = shl.i32 %2, %5
  %7 = uge.i32 %6, %3
  %8 = and.i32 %2, %4
  %9 = lshr.i32 %2, %8
  %10 = iconst.i32 1
  %11 = or.i32 %9, %10
  %12 = smod.i32 %7, %11
  %13 = sext.i64 %12
  ret %13
}
)";
    REQUIRE(all_tiers_agree(a, kIntArgs));
    REQUIRE(all_tiers_agree(b, kIntArgs));
    REQUIRE(all_tiers_agree(c, kIntArgs));
}

// Store-to-load forwarding in the peephole pass reused a stored register
// after the address register had been redefined (fuzz seed 707, reduced).
TEST_CASE("JIT regression - store-to-load forwarding across an address redefinition") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = trunc.i32 %0
  %3 = trunc.i32 %1
  %4 = alloca 32, 8
  %5 = iconst.i64 8
  %6 = iconst.i64 0
  br hdr(%6)
hdr(%7: i64):
  %8 = slt.i64 %7, %5
  br_if %8, body, exit
body:
  %9 = iconst.i64 254459
  %10 = mul.i64 %7, %9
  %11 = xor.i64 %10, %0
  %12 = trunc.i32 %11
  store_indexed.i32 %4, %7, 4, %12
  %13 = iconst.i64 1
  %14 = add.i64 %7, %13
  br hdr(%14)
exit:
  %15 = sext.i64 %3
  %16 = iconst.i64 7
  %17 = and.i64 %15, %16
  %18 = iconst.i32 3
  %19 = add.i32 %3, %18
  store_indexed.i32 %4, %17, 4, %19
  %20 = and.i64 %0, %16
  %21 = load_indexed.i32 %4, %20, 4
  %22 = iconst.i32 31
  %23 = and.i32 %21, %22
  %24 = shl.i32 %2, %23
  %25 = uge.i32 %24, %21
  %26 = and.i32 %2, %22
  %27 = lshr.i32 %2, %26
  %28 = iconst.i32 1
  %29 = or.i32 %27, %28
  %30 = smod.i32 %25, %29
  %31 = sext.i64 %30
  ret %31
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// A conditional branch whose targets both take arguments goes through an
// edge trampoline; the trampoline was not in the CFG, so liveness missed the
// values it copies and the loop-carried values were clobbered.
TEST_CASE("JIT regression - br_if with arguments on both edges inside a loop") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 11
  %3 = iconst.i64 -4294967296
  %4 = iconst.i64 0
  br hdr(%4, %0, %3, %1)
hdr(%5: i64, %6: i64, %7: i64, %8: i64):
  %9 = slt.i64 %5, %2
  br_if %9, body, exit(%6, %7, %8)
body:
  %10 = iconst.i64 1099511628211
  %11 = mul.i64 %6, %10
  %12 = xor.i64 %11, %8
  %13 = iconst.i64 63
  %14 = lshr.i64 %7, %13
  %15 = add.i64 %7, %14
  %16 = and.i64 %8, %13
  %17 = shl.i64 %12, %16
  %18 = add.i64 %8, %17
  %19 = iconst.i64 7
  %20 = and.i64 %12, %19
  %21 = iconst.i64 0
  %22 = eq.i64 %20, %21
  br_if %22, exit(%12, %15, %18), cont(%18, %12)
cont(%23: i64, %24: i64):
  %25 = mul.i64 %24, %10
  %26 = xor.i64 %25, %5
  %27 = iconst.i64 1
  %28 = add.i64 %5, %27
  br hdr(%28, %26, %15, %23)
exit(%29: i64, %30: i64, %31: i64):
  %32 = iconst.i64 1099511628211
  %33 = mul.i64 %29, %32
  %34 = xor.i64 %33, %30
  %35 = mul.i64 %34, %32
  %36 = xor.i64 %35, %31
  ret %36
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// Float compares fused into branches, selects and guards ignored unordered
// results (NaN), and f32 branches compared with ucomisd.
TEST_CASE("JIT regression - fused float compares with NaN operands") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = bitcast.f64.i64 %0
  %3 = bitcast.f64.i64 %1
  %4 = fptrunc.f32.f64 %2
  %5 = fptrunc.f32.f64 %3
  %6 = iconst.i64 0
  %7 = iconst.i64 1
  %8 = slt.f64 %2, %3
  br_if %8, a1, a2
a1:
  br a3(%7)
a2:
  br a3(%6)
a3(%9: i64):
  %10 = eq.f64 %2, %3
  br_if %10, b1, b2
b1:
  br b3(%7)
b2:
  br b3(%6)
b3(%11: i64):
  %12 = sge.f32 %4, %5
  br_if %12, c1, c2
c1:
  br c3(%7)
c2:
  br c3(%6)
c3(%13: i64):
  %14 = ne.f32 %4, %5
  br_if %14, d1, d2
d1:
  br d3(%7)
d2:
  br d3(%6)
d3(%15: i64):
  %16 = sle.f64 %2, %3
  %17 = iconst.i64 16
  %18 = select.i64 %16, %17, %6
  %19 = sgt.f32 %4, %5
  %20 = iconst.i64 32
  %21 = select.i64 %19, %20, %6
  %22 = iconst.i64 2
  %23 = mul.i64 %11, %22
  %24 = iconst.i64 4
  %25 = mul.i64 %13, %24
  %26 = iconst.i64 8
  %27 = mul.i64 %15, %26
  %28 = or.i64 %9, %23
  %29 = or.i64 %28, %25
  %30 = or.i64 %29, %27
  %31 = or.i64 %30, %18
  %32 = or.i64 %31, %21
  ret %32
}
)";
    const int64_t nan = 0x7ff8000000000000LL;
    const int64_t one = 0x3ff0000000000000LL;
    const int64_t two = 0x4000000000000000LL;
    const int64_t neg = static_cast<int64_t>(0xbff0000000000000ULL);
    REQUIRE(all_tiers_agree(src, {
        {nan, nan}, {nan, one}, {one, nan}, {one, one}, {one, two}, {two, one}, {neg, one}, {0, 0},
    }));
}

// A gcref held across a safepoint in a loop whose body is laid out after the
// loop exit: the stack map only listed gcrefs whose last use came later in
// block order, so the collector never updated the slot and the exit read
// from-space memory (fuzz seed 626).
TEST_CASE("JIT regression - gcref live across a safepoint in a later-placed loop body") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 16
  %3 = iconst.i64 0
  %4 = iconst.i32 1
  %5 = call.gcref @brass_gc_alloc(%2, %3, %4)
  store_indexed.i64 %5, %3, 8, %0
  %6 = iconst.i64 1
  store_indexed.i64 %5, %6, 8, %1
  %7 = iconst.i64 7
  %8 = and.i64 %1, %7
  br hdr(%3)
hdr(%9: i64):
  %10 = slt.i64 %9, %8
  br_if %10, body, exit
exit:
  %11 = load_indexed.i64 %5, %3, 8
  %12 = load_indexed.i64 %5, %6, 8
  %13 = iconst.i64 1099511628211
  %14 = mul.i64 %11, %13
  %15 = xor.i64 %14, %12
  ret %15
body:
  safepoint
  %16 = add.i64 %9, %6
  br hdr(%16)
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// The index of a load_indexed is `(row - n + j) + 1`: the constant folds into
// the displacement, leaving `row - n + j` as the index register. That add had
// also been fused into a lea computing the `+ 1` add, which itself was folded
// away, so nothing computed the index and the load read a[1].
TEST_CASE("JIT regression - address index that an absorbed lea would have computed") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = alloca 12800, 8
  %3 = iconst.i64 0
  %4 = iconst.i64 1
  %5 = iconst.i64 7
  %6 = iconst.i64 15
  %7 = and.i64 %0, %6
  %n = add.i64 %7, %4
  %m = sub.i64 %n, %4
  %tot = mul.i64 %n, %n
  br init(%3)
init(%q: i64):
  %ci = slt.i64 %q, %tot
  br_if %ci, init_body, rows(%4)
init_body:
  %qv = mul.i64 %q, %5
  %qx = xor.i64 %qv, %1
  store_indexed.i64 %2, %q, 8, %qx
  %qn = add.i64 %q, %4
  br init(%qn)
rows(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, row_start, sum(%3, %3)
row_start:
  br cols(%3)
cols(%j: i64):
  %c2 = slt.i64 %j, %m
  br_if %c2, cell, row_end
cell:
  %row = mul.i64 %i, %n
  %dst = add.i64 %row, %j
  %prow = sub.i64 %row, %n
  %src0 = add.i64 %prow, %j
  %src = add.i64 %src0, %4
  %v = load_indexed.i64 %2, %src, 8
  %v1 = add.i64 %v, %4
  store_indexed.i64 %2, %dst, 8, %v1
  %jn = add.i64 %j, %4
  br cols(%jn)
row_end:
  %in = add.i64 %i, %4
  br rows(%in)
sum(%k: i64, %s: i64):
  %c3 = slt.i64 %k, %tot
  br_if %c3, sum_body, done
sum_body:
  %x = load_indexed.i64 %2, %k, 8
  %w = mul.i64 %x, %k
  %s2 = add.i64 %s, %w
  %kn = add.i64 %k, %4
  br sum(%kn, %s2)
done:
  ret %s
}
)";
    REQUIRE(all_tiers_agree(src, {{3, 0}, {3, 5}, {5, 1}, {10, 2}, {15, -7}, {0, 0}}));
}

// GVN-PRE: a must-alias store on one path through a loop made the load after
// it available but left the block "transparent", so the load was hoisted out
// of the loop and read the value from before the stores (fuzz seed 300485).
TEST_CASE("GVN-PRE regression - a load is not hoisted past a must-alias store in the loop") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 32
  %3 = iconst.i64 0
  %4 = iconst.i32 1
  %5 = call.gcref @brass_gc_alloc(%2, %3, %4)
  store.i64 %5, 16, %1
  %6 = iconst.i64 6
  br hdr(%3, %3)
hdr(%7: i64, %8: i64):
  %9 = slt.i64 %7, %6
  br_if %9, body, exit(%8)
body:
  %10 = and.i64 %7, %0
  %11 = iconst.i64 1
  %12 = and.i64 %10, %11
  %13 = ne.i64 %12, %3
  br_if %13, then, els
then:
  br merge(%8)
els:
  store.i64 %5, 16, %7
  br merge(%8)
merge(%14: i64):
  %15 = load.i64 %5, 16
  %16 = iconst.i64 31
  %17 = mul.i64 %14, %16
  %18 = add.i64 %17, %15
  %19 = add.i64 %7, %11
  br hdr(%19, %18)
exit(%20: i64):
  ret %20
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// GVN-PRE: commutative operands were ordered by value id even for ptr + i64,
// so a hoisted pointer add was rebuilt as `add i64, ptr` (fuzz seed 3002960).
TEST_CASE("GVN-PRE regression - a hoisted pointer add keeps its pointer operand first") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 7
  %3 = and.i64 %1, %2
  %4 = iconst.i64 8
  %5 = mul.i64 %3, %4
  %6 = alloca 64, 8
  %7 = iconst.i64 0
  %8 = iconst.i64 5
  br hdr(%7, %7)
hdr(%9: i64, %10: i64):
  %11 = slt.i64 %9, %8
  br_if %11, body, exit(%10)
body:
  %12 = add.ptr %6, %5
  %13 = add.i64 %9, %0
  store.i64 %12, 0, %13
  %14 = load.i64 %12, 0
  %15 = add.i64 %10, %14
  %16 = iconst.i64 1
  %17 = add.i64 %9, %16
  br hdr(%17, %15)
exit(%18: i64):
  ret %18
}
)";
    REQUIRE(all_tiers_agree(src, kIntArgs));
}

// trunc.i32 then zext.i64 in one register lowers to mov32 r, r, which the
// peephole pass and the emitter dropped as a self-move, leaving the upper half.
TEST_CASE("JIT regression - zext of a trunc in the same register") {
    const char* src = R"(
module @m
func @fuzz_fn(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 2654435761
  %3 = add.i64 %1, %2
  %4 = trunc.i32 %3
  %5 = zext.i64 %4
  ret %5
}
)";
    REQUIRE(all_tiers_agree(src, {{0, 0}, {0, 4294967296LL}, {0, -1}, {0, 1LL << 62}}));
}

// Enough simultaneously-live zero-extended values to force spills: a 32-bit
// spill store of a zext.i64 left the slot's upper half stale.
TEST_CASE("JIT regression - spilled zero-extended values under register pressure") {
    constexpr int kValues = 24;
    std::ostringstream src;
    src << "module @m\nfunc @fuzz_fn(%0: i64, %1: i64) -> i64 {\nentry:\n";
    int next = 2;
    std::vector<int> wide;
    // First a set of live 64-bit values with high bits set, so their spill
    // slots hold non-zero upper halves when they are reused.
    for (int i = 0; i < kValues; ++i) {
        const int k = next++;
        src << "  %" << k << " = iconst.i64 " << (-1 - static_cast<int64_t>(i) * 4294967296LL) << "\n";
        const int v = next++;
        src << "  %" << v << " = xor.i64 %0, %" << k << "\n";
        wide.push_back(v);
    }
    int acc = next++;
    src << "  %" << acc << " = iconst.i64 0\n";
    for (int v : wide) {
        const int n = next++;
        src << "  %" << n << " = add.i64 %" << acc << ", %" << v << "\n";
        acc = n;
    }
    std::vector<int> narrow;
    for (int i = 0; i < kValues; ++i) {
        const int k = next++;
        src << "  %" << k << " = iconst.i64 " << (i * 2654435761LL) << "\n";
        const int x = next++;
        src << "  %" << x << " = add.i64 %1, %" << k << "\n";
        const int t = next++;
        src << "  %" << t << " = trunc.i32 %" << x << "\n";
        const int z = next++;
        src << "  %" << z << " = zext.i64 %" << t << "\n";
        narrow.push_back(z);
    }
    for (int z : narrow) {
        const int m = next++;
        src << "  %" << m << " = iconst.i64 1099511628211\n";
        const int p = next++;
        src << "  %" << p << " = mul.i64 %" << acc << ", %" << m << "\n";
        const int q = next++;
        src << "  %" << q << " = xor.i64 %" << p << ", %" << z << "\n";
        acc = q;
    }
    src << "  ret %" << acc << "\n}\n";
    REQUIRE(all_tiers_agree(src.str(), kIntArgs));
}
