// Negative and shape tests for DOALL auto-parallelization and 2-D loop
// tiling. Each case is a shape the old transforms miscompiled (lost
// reduction inits, unmapped exit arguments, ignored body control flow, wrong
// step handling) or must refuse (carried dependences, an accumulator read
// outside its update). Every case also checks the answer on the interpreter
// and the JIT against the untransformed program.

#include "soundness_helpers.hpp"
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/loop_tile.hpp>
#include <numeric>

using namespace soundness;

namespace {

// Runs the parallelizer until it stops finding loops (it takes one per call).
size_t parallelize_all(Function& fn) {
    size_t n = 0;
    for (int round = 0; round < 8; ++round) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        ParallelLoopOptions o;
        o.parallel_threshold = 1;
        if (!auto_parallelize_function(fn, dom, o)) break;
        ++n;
    }
    return n;
}

void parallelize_module(Module& m, std::string_view name) {
    parallelize_all(*find_fn(m, name));
}

bool tile(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    return loop_tile_pass(fn, dom);
}

RuntimeValue ptr_arg(void* p) {
    return RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(p));
}

} // namespace

// ---------------------------------------------------------------------------
// Auto-parallelization
// ---------------------------------------------------------------------------

// Fuzz seed 1's shape: a reduction with a non-zero start, a loop-invariant
// header parameter passed to the exit, and the final IV read after the loop.
TEST_CASE("Soundness - parallel reduction keeps its init, exit arguments and final IV") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64, %k: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %init = iconst.i64 1000
  br hdr(%zero, %init, %k)

hdr(%i: i64, %s: i64, %inv: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit(%s, %inv)

body:
  %v = load_indexed.i64 %a, %i, 8
  %s2 = add.i64 %s, %v
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br hdr(%i2, %s2, %inv)

exit(%r: i64, %q: i64):
  %m = mul.i64 %r, %q
  %t = add.i64 %m, %i
  ret %t
}
)";
    auto mod = parse(mir);
    CHECK_EQ(parallelize_all(*find_fn(*mod, "p")), size_t{1});
    CHECK(verifies(*mod));

    std::vector<int64_t> buf(64);
    std::iota(buf.begin(), buf.end(), int64_t{-20});
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(64), i64(3)},
                               {ptr_arg(buf.data()), i64(7), i64(-5)},
                               {ptr_arg(buf.data()), i64(1), i64(2)},
                               {ptr_arg(buf.data()), i64(0), i64(9)},
                               {ptr_arg(buf.data()), i64(-4), i64(9)}});
}

// An i32 sum that wraps: the partial sums are combined in 64 bits by the
// runtime and must come back as the wrapped 32-bit answer.
TEST_CASE("Soundness - parallel i32 reduction wraps like the sequential loop") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %init = iconst.i32 2147483000
  br hdr(%zero, %init)

hdr(%i: i64, %s: i32):
  %c = slt.i64 %i, %n
  br_if %c, body, exit

body:
  %v = load_indexed.i32 %a, %i, 4
  %s2 = add.i32 %s, %v
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br hdr(%i2, %s2)

exit:
  %r = sext.i64 %s
  ret %r
}
)";
    auto mod = parse(mir);
    CHECK_EQ(parallelize_all(*find_fn(*mod, "p")), size_t{1});
    CHECK(verifies(*mod));

    std::vector<int32_t> buf(100);
    for (size_t k = 0; k < buf.size(); ++k) buf[k] = static_cast<int32_t>(k * 7919 + 1000);
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(100)},
                               {ptr_arg(buf.data()), i64(3)},
                               {ptr_arg(buf.data()), i64(0)}});
}

// A body with its own branches: every block of the body goes into the kernel.
TEST_CASE("Soundness - parallel loop with a branching body runs every path") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %three = iconst.i64 3
  br hdr(%zero)

hdr(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, sum_hdr(%zero, %zero)

body:
  %odd = and.i64 %i, %one
  %isodd = ne.i64 %odd, %zero
  br_if %isodd, odd_bb, even_bb

odd_bb:
  %x = mul.i64 %i, %three
  store_indexed.i64 %a, %i, 8, %x
  br latch

even_bb:
  %y = sub.i64 %zero, %i
  store_indexed.i64 %a, %i, 8, %y
  br latch

latch:
  %i2 = add.i64 %i, %one
  br hdr(%i2)

sum_hdr(%k: i64, %s: i64):
  %c2 = slt.i64 %k, %n
  br_if %c2, sum_body, done

sum_body:
  %v = load_indexed.i64 %a, %k, 8
  %w = mul.i64 %v, %k
  %s2 = add.i64 %s, %w
  %k2 = add.i64 %k, %one
  br sum_hdr(%k2, %s2)

done:
  ret %s
}
)";
    auto mod = parse(mir);
    Function* fn = find_fn(*mod, "p");
    CHECK_EQ(parallelize_all(*fn), size_t{2});
    CHECK_EQ(count_opcode(*fn, Opcode::store_indexed), size_t{0}); // both stores moved into a kernel
    CHECK(verifies(*mod));

    std::vector<int64_t> buf(50, 0);
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(50)},
                               {ptr_arg(buf.data()), i64(9)},
                               {ptr_arg(buf.data()), i64(1)}});
}

// Step 3 against a limit known only at run time: iteration k is IV 3k, and
// the trip count is ceil(n / 3).
TEST_CASE("Soundness - parallel loop with a step above one visits the same indices") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %three = iconst.i64 3
  br hdr(%zero)

hdr(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, sum_hdr(%zero, %zero)

body:
  %x = add.i64 %i, %one
  store_indexed.i64 %a, %i, 8, %x
  %i2 = add.i64 %i, %three
  br hdr(%i2)

sum_hdr(%k: i64, %s: i64):
  %c2 = slt.i64 %k, %n
  br_if %c2, sum_body, done

sum_body:
  %v = load_indexed.i64 %a, %k, 8
  %w = mul.i64 %v, %k
  %s2 = add.i64 %s, %w
  %k2 = add.i64 %k, %one
  br sum_hdr(%k2, %s2)

done:
  ret %s
}
)";
    auto mod = parse(mir);
    Function* fn = find_fn(*mod, "p");
    // The limit is not known, so the parallel dispatch is guarded against IV
    // wrap-around and the sequential loop stays behind as the fallback. That
    // leaves the sum loop with two entries (no preheader); it stays serial.
    CHECK_EQ(parallelize_all(*fn), size_t{1});
    size_t kernel_stores = 0;
    for (Function* f : mod->functions()) {
        if (f != fn && f->name().find("_par_k_") != std::string_view::npos) {
            kernel_stores += count_opcode(*f, Opcode::store_indexed);
        }
    }
    CHECK_EQ(kernel_stores, size_t{1});
    CHECK(verifies(*mod));

    std::vector<int64_t> buf(64, 0);
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(64)},
                               {ptr_arg(buf.data()), i64(10)},
                               {ptr_arg(buf.data()), i64(2)},
                               {ptr_arg(buf.data()), i64(0)}});
}

// a[i + 1] = a[i] + 1 carries a value from each iteration to the next.
TEST_CASE("Soundness - parallelization keeps a loop with a carried dependence") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%zero)

hdr(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done

body:
  %v = load_indexed.i64 %a, %i, 8
  %v1 = add.i64 %v, %one
  %i2 = add.i64 %i, %one
  store_indexed.i64 %a, %i2, 8, %v1
  br hdr(%i2)

done:
  %last = load_indexed.i64 %a, %n, 8
  ret %last
}
)";
    auto mod = parse(mir);
    CHECK_EQ(parallelize_all(*find_fn(*mod, "p")), size_t{0});

    std::vector<int64_t> buf(40, 5);
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(30)}});
}

// The accumulator is stored every iteration (a prefix sum), so its
// per-iteration value matters and the loop is not a reduction.
TEST_CASE("Soundness - parallelization keeps a loop that reads its accumulator elsewhere") {
    const char* mir = R"(
func @p(%a: ptr, %n: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %s: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done

body:
  %sq = mul.i64 %i, %i
  store_indexed.i64 %a, %i, 8, %s
  %s2 = add.i64 %s, %sq
  %i2 = add.i64 %i, %one
  br hdr(%i2, %s2)

done:
  %last = load_indexed.i64 %a, %zero, 8
  %r = add.i64 %s, %last
  ret %r
}
)";
    auto mod = parse(mir);
    CHECK_EQ(parallelize_all(*find_fn(*mod, "p")), size_t{0});

    std::vector<int64_t> buf(40, 0);
    check_transform_preserves(mir, "p", [](Module& m) { parallelize_module(m, "p"); },
                              {{ptr_arg(buf.data()), i64(40)}});
}

// ---------------------------------------------------------------------------
// 2-D tiling
// ---------------------------------------------------------------------------

// Row i reads row i - 1 one column to the right (a[i*n + j - (n - 1)]
// is a[i-1][j+1]): inside a tile, (i, j)
// would run before (i - 1, j + 1) when j + 1 falls in the next column tile.
TEST_CASE("Soundness - tiling keeps a nest that reads the previous row") {
    const char* mir = R"(
func @t(%a: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %m = sub.i64 %n, %one
  br bb1(%one)

bb1(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, bb2, bb6(%zero, %zero)

bb2:
  br bb3(%zero)

bb3(%j: i64):
  %c2 = slt.i64 %j, %m
  br_if %c2, bb4, bb5

bb4:
  %row = mul.i64 %i, %n
  %dst = add.i64 %row, %j
  %src = sub.i64 %dst, %m
  %v = load_indexed.i64 %a, %src, 8
  %v1 = add.i64 %v, %one
  store_indexed.i64 %a, %dst, 8, %v1
  %jn = add.i64 %j, %one
  br bb3(%jn)

bb5:
  %in = add.i64 %i, %one
  br bb1(%in)

bb6(%k: i64, %sum: i64):
  %lim = mul.i64 %n, %n
  %c3 = slt.i64 %k, %lim
  br_if %c3, bb7, bb8

bb7:
  %x = load_indexed.i64 %a, %k, 8
  %w = mul.i64 %x, %k
  %s2 = add.i64 %sum, %w
  %kn = add.i64 %k, %one
  br bb6(%kn, %s2)

bb8:
  ret %sum
}
)";
    auto mod = parse(mir);
    CHECK(!tile(*find_fn(*mod, "t")));

    std::vector<int64_t> buf(40 * 40);
    for (size_t k = 0; k < buf.size(); ++k) buf[k] = static_cast<int64_t>(k * 7);
    check_transform_preserves(mir, "t", [](Module& m) { tile(*find_fn(m, "t")); },
                              {{ptr_arg(buf.data()), i64(40)}});
}

// The outer IV leaves the nest as an exit argument; after tiling it must
// still be the final row count, with partial tiles at both edges.
TEST_CASE("Soundness - tiling keeps the outer IV's final value and partial tiles") {
    const char* mir = R"(
func @t(%a: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br bb1(%zero)

bb1(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, bb2, bb6(%zero, %i)

bb2:
  br bb3(%zero)

bb3(%j: i64):
  %c2 = slt.i64 %j, %n
  br_if %c2, bb4, bb5

bb4:
  %row = mul.i64 %i, %n
  %rm = add.i64 %row, %j
  %x = mul.i64 %i, %j
  %y = add.i64 %x, %i
  store_indexed.i64 %a, %rm, 8, %y
  %jn = add.i64 %j, %one
  br bb3(%jn)

bb5:
  %in = add.i64 %i, %one
  br bb1(%in)

bb6(%k: i64, %sum: i64):
  %lim = mul.i64 %n, %n
  %c3 = slt.i64 %k, %lim
  br_if %c3, bb7, bb8

bb7:
  %v = load_indexed.i64 %a, %k, 8
  %w = mul.i64 %v, %k
  %s2 = add.i64 %sum, %w
  %kn = add.i64 %k, %one
  br bb6(%kn, %s2)

bb8:
  ret %sum
}
)";
    auto mod = parse(mir);
    CHECK(tile(*find_fn(*mod, "t")));
    CHECK(verifies(*mod));

    std::vector<int64_t> buf(40 * 40, 0);
    check_transform_preserves(mir, "t", [](Module& m) { tile(*find_fn(m, "t")); },
                              {{ptr_arg(buf.data()), i64(40)},
                               {ptr_arg(buf.data()), i64(17)},
                               {ptr_arg(buf.data()), i64(1)},
                               {ptr_arg(buf.data()), i64(0)}});
}
