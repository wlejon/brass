// Negative tests for the optimizer soundness fixes in write-barrier
// elimination, jump threading, the derived-gcref rule, SCCP, the allocation
// registry, loop unswitching, dead induction cycles, loop fusion and loop
// distribution. Each case is a shape the pass used to miscompile.

#include "soundness_helpers.hpp"
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <sstream>

using namespace soundness;

namespace {

size_t count_loops(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis la(fn, dom);
    return la.post_order_loops().size();
}

bool fuse(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    return loop_fusion_pass(fn, dom, LoopFusionOptions{});
}

bool distribute(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    return loop_distribution_pass(fn, dom, LoopDistributionOptions{});
}

RuntimeValue ptr_arg(void* p) {
    return RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(p));
}

} // namespace

// ---------------------------------------------------------------------------
// Write-barrier elimination
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - WBE keeps a barrier after an earlier barrier on the same object") {
    // The first barrier covered storing %old; the second store puts a
    // possibly-young value into the same, possibly-old, object.
    auto mod = parse(R"(
func @f(%o: gcref, %old: gcref, %young: gcref) -> void {
bb0:
  store.gcref %o, 8, %old
  write_barrier %o, %old
  store.gcref %o, 16, %young
  write_barrier %o, %young
  ret
}
)");
    WriteBarrierElimination wbe;
    wbe.run_on_module(*mod);
    CHECK(verifies(*mod));
    CHECK_EQ(count_opcode(*find_fn(*mod, "f"), Opcode::write_barrier), 2u);
}

TEST_CASE("Soundness - WBE treats only small constant-size allocations as young") {
    // A non-constant size or one above half the smallest nursery can put the
    // object straight into the old generation.
    auto mod = parse(R"(
func @dyn(%n: i64, %v: gcref) -> gcref {
bb0:
  %o = call.gcref @brass_gc_alloc(%n)
  store.gcref %o, 8, %v
  write_barrier %o, %v
  ret %o
}

func @big(%v: gcref) -> gcref {
bb0:
  %n = iconst.i64 1048576
  %o = call.gcref @brass_gc_alloc(%n)
  store.gcref %o, 8, %v
  write_barrier %o, %v
  ret %o
}

func @small(%v: gcref) -> gcref {
bb0:
  %n = iconst.i64 64
  %o = call.gcref @brass_gc_alloc(%n)
  store.gcref %o, 8, %v
  write_barrier %o, %v
  ret %o
}
)");
    WriteBarrierElimination wbe;
    wbe.run_on_module(*mod);
    CHECK(verifies(*mod));
    CHECK_EQ(count_opcode(*find_fn(*mod, "dyn"), Opcode::write_barrier), 1u);
    CHECK_EQ(count_opcode(*find_fn(*mod, "big"), Opcode::write_barrier), 1u);
    CHECK_EQ(count_opcode(*find_fn(*mod, "small"), Opcode::write_barrier), 0u);
}

// ---------------------------------------------------------------------------
// Jump threading
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - jump threading repairs SSA for values used past the threaded block") {
    // bb3's %p is used in bb4/bb5 after the branch is threaded from bb1 and
    // bb2; the threaded copies must hand their own value on.
    const char* mir = R"(
func @g(%x: i64, %y: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %c = slt %x, %zero
  br_if %c, bb1, bb2

bb1:
  %one = iconst.i64 1
  br bb3(%one)

bb2:
  %two = iconst.i64 2
  br bb3(%two)

bb3(%p: i64):
  %one3 = iconst.i64 1
  %isone = eq %p, %one3
  br_if %isone, bb4, bb5

bb4:
  %r1 = add %y, %p
  ret %r1

bb5:
  %r2 = mul %y, %p
  ret %r2
}
)";
    check_transform_preserves(mir, "g", [](Module& m) { jump_thread_module(m); },
                              {{i64(-3), i64(5)}, {i64(3), i64(5)}, {i64(0), i64(-7)}});
}

// ---------------------------------------------------------------------------
// Derived gcrefs
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - verifier rejects a derived gcref live across a safepoint") {
    DiagnosticReporter diag;
    auto mod = parse_module(R"(
func @f(%o: gcref) -> i64 {
bb0:
  %sixteen = iconst.i64 16
  %d = add %o, %sixteen
  safepoint
  %v = load.i64 %d, 0
  ret %v
}
)", &diag);
    REQUIRE(mod != nullptr);
    CHECK(!verify_module(*mod, &diag));

    auto ok = parse(R"(
func @f(%o: gcref) -> i64 {
bb0:
  safepoint
  %sixteen = iconst.i64 16
  %d = add %o, %sixteen
  %v = load.i64 %d, 0
  ret %v
}
)");
    CHECK(verifies(*ok));
}

TEST_CASE("Soundness - LICM leaves a derived gcref next to its use inside a safepointing loop") {
    auto mod = parse(R"(
func @f(%o: gcref, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)

bb2:
  safepoint
  %sixteen = iconst.i64 16
  %d = add %o, %sixteen
  %v = load.i64 %d, 0
  %a2 = add %acc, %v
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %a2)

bb3(%r: i64):
  ret %r
}
)");
    run_loop_pipeline(*mod);
    CHECK(verifies(*mod));
}

// ---------------------------------------------------------------------------
// SCCP
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - SCCP never replaces pointer arithmetic with an integer constant") {
    auto mod = parse(R"(
func @p(%p: ptr) -> ptr {
bb0:
  %z = iconst.i64 0
  %q = add %p, %z
  ret %q
}
)");
    sccp_module(*mod);
    CHECK(verifies(*mod));
    const Function* fn = find_fn(*mod, "p");
    const Instruction* ret = fn->blocks().back()->terminator();
    REQUIRE(ret != nullptr);
    REQUIRE(ret->operand_count() == 1);
    CHECK(ret->operand(0)->type() == Type::ptr());
}

// ---------------------------------------------------------------------------
// f64 demotion
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - f64 demotion keeps values that outgrow 2^53 in f64") {
    // x = 2x + 1 seventy times: f64 rounds past 2^53, i64 would not.
    const char* grow = R"(
func @grow() -> f64 {
bb0:
  %zero = fconst.f64 0.0
  %one = fconst.f64 1.0
  %lim = fconst.f64 70.0
  br bb1(%zero, %one)

bb1(%i: f64, %x: f64):
  %c = slt %i, %lim
  br_if %c, bb2, bb3(%x)

bb2:
  %x2 = add %x, %x
  %x3 = add %x2, %one
  %in = add %i, %one
  br bb1(%in, %x3)

bb3(%r: f64):
  ret %r
}
)";
    check_transform_preserves(grow, "grow", [](Module& m) { run_loop_pipeline(m); }, {{}});

    // Squares just above 2^53 are not all representable in f64.
    const char* squares = R"(
func @sq() -> f64 {
bb0:
  %zero = fconst.f64 0.0
  %one = fconst.f64 1.0
  %start = fconst.f64 94906260.0
  %lim = fconst.f64 94906272.0
  br bb1(%start, %zero)

bb1(%i: f64, %s: f64):
  %c = slt %i, %lim
  br_if %c, bb2, bb3(%s)

bb2:
  %ii = mul %i, %i
  %s2 = add %s, %ii
  %in = add %i, %one
  br bb1(%in, %s2)

bb3(%r: f64):
  ret %r
}
)";
    check_transform_preserves(squares, "sq", [](Module& m) { run_loop_pipeline(m); }, {{}});
}

TEST_CASE("Soundness - f64 demotion never loses a negative zero") {
    // The loop leaves -0.0 in %acc; 1/%acc tells it apart from +0.0.
    const char* mir = R"(
func @nz() -> f64 {
bb0:
  %zero = fconst.f64 0.0
  %one = fconst.f64 1.0
  br bb1(%zero, %one)

bb1(%i: f64, %acc: f64):
  %c = slt %i, %one
  br_if %c, bb2, bb3(%acc)

bb2:
  %m = neg %i
  %in = add %i, %one
  br bb1(%in, %m)

bb3(%r: f64):
  %q = sdiv %one, %r
  ret %q
}
)";
    check_transform_preserves(mir, "nz", [](Module& m) { run_loop_pipeline(m); }, {{}});
}

TEST_CASE("Soundness - f64 demotion does not truncate f64 arguments") {
    const char* mir = R"(
func @arg(%x: f64) -> f64 {
bb0:
  %zero = fconst.f64 0.0
  %one = fconst.f64 1.0
  %lim = fconst.f64 3.0
  br bb1(%zero, %x)

bb1(%i: f64, %s: f64):
  %c = slt %i, %lim
  br_if %c, bb2, bb3(%s)

bb2:
  %s2 = add %s, %one
  %in = add %i, %one
  br bb1(%in, %s2)

bb3(%r: f64):
  ret %r
}
)";
    check_transform_preserves(mir, "arg", [](Module& m) { run_loop_pipeline(m); },
                              {{RuntimeValue::from_f64(0.5)}, {RuntimeValue::from_f64(-2.25)}});
}

// ---------------------------------------------------------------------------
// SROA
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - SROA gives every execution of an allocation fresh fields") {
    // Each iteration allocates a new zeroed object, so every y is 1; carrying
    // the field around the loop would sum 1..n instead.
    const char* mir = R"(
func @s(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)

bb2:
  %sz = iconst.i64 16
  %o = call.gcref @brass_gc_alloc(%sz)
  %x = load.i64 %o, 8
  %one = iconst.i64 1
  %x1 = add %x, %one
  store.i64 %o, 8, %x1
  %y = load.i64 %o, 8
  %a2 = add %acc, %y
  %in = add %i, %one
  br bb1(%in, %a2)

bb3(%r: i64):
  ret %r
}
)";
    check_transform_preserves(mir, "s", [](Module& m) {
        sroa_module(m);
        CHECK_EQ(count_calls(*find_fn(m, "s"), "brass_gc_alloc"), 0u);
    }, {{i64(0)}, {i64(1)}, {i64(5)}});
}

TEST_CASE("Soundness - SROA keeps an object whose alias parameter may be another pointer") {
    auto mod = parse(R"(
func @m(%p: gcref, %k: i64) -> i64 {
bb0:
  %sz = iconst.i64 16
  %o = call.gcref @brass_gc_alloc(%sz)
  %seven = iconst.i64 7
  store.i64 %o, 8, %seven
  %zero = iconst.i64 0
  %c = slt %k, %zero
  br_if %c, bb1, bb2

bb1:
  br bb3(%o)

bb2:
  br bb3(%p)

bb3(%q: gcref):
  %v = load.i64 %q, 8
  ret %v
}
)");
    sroa_module(*mod);
    CHECK(verifies(*mod));
    const Function* fn = find_fn(*mod, "m");
    CHECK_EQ(count_calls(*fn, "brass_gc_alloc"), 1u);
    CHECK_EQ(count_opcode(*fn, Opcode::load), 1u);
}

// ---------------------------------------------------------------------------
// Allocation registry
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - allocators are declared, not guessed from names") {
    CHECK(is_allocation_callee("brass_gc_alloc"));
    CHECK(!is_allocation_callee("alloc_foo"));
    CHECK(!is_allocation_callee("my_alloc"));

    const char* mir = R"(
extern @my_alloc allocator

func @f(%n: i64) -> ptr {
bb0:
  %a = call.ptr @my_alloc(%n)
  %b = call.ptr @alloc_foo(%n)
  ret %a
}
)";
    auto mod = parse(mir);
    CHECK(mod->is_allocation_function("my_alloc"));
    CHECK(!mod->is_allocation_function("alloc_foo"));

    const Function* fn = find_fn(*mod, "f");
    const Instruction* first = fn->entry_block()->head();
    REQUIRE(first != nullptr);
    CHECK(is_allocation_call(first));
    CHECK(!is_allocation_call(first->next()));

    std::ostringstream printed;
    print_module(*mod, printed);
    CHECK(printed.str().find("extern @my_alloc allocator") != std::string::npos);
    auto again = parse(printed.str());
    CHECK(again->is_allocation_function("my_alloc"));
}

// ---------------------------------------------------------------------------
// Loop unswitching
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - unswitch copies every block that can see loop values") {
    // bb7 reads the header's %i and %acc but is reached through bb5, which
    // does not: both must be copied with the loop.
    const char* mir = R"(
func @u(%n: i64, %k: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %five = iconst.i64 5
  %inv = slt %k, %five
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb5

bb2:
  br_if %inv, bb3, bb4

bb3:
  %a1 = add %acc, %i
  br bb6(%a1)

bb4:
  %two = iconst.i64 2
  %m = mul %i, %two
  %a2 = sub %acc, %m
  br bb6(%a2)

bb6(%accn: i64):
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %accn)

bb5:
  br bb7

bb7:
  %r = add %acc, %i
  ret %r
}
)";
    check_transform_preserves(mir, "u", [](Module& m) {
        LoopUnswitchStats stats;
        for (auto& fn : m.functions()) unswitch_loops_in_function(*fn, LoopUnswitchOptions{}, &stats);
        CHECK(stats.loops_unswitched > 0);
    }, {{i64(6), i64(1)}, {i64(6), i64(9)}, {i64(0), i64(1)}});
}

TEST_CASE("Soundness - unswitch keeps invoke operands and both edges in the copy") {
    const char* mir = R"(
func @callee(%x: i64) -> i64 {
bb0:
  ret %x
}

func @u(%n: i64, %k: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %five = iconst.i64 5
  %inv = slt %k, %five
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb8(%acc)

bb2:
  br_if %inv, bb3, bb4

bb3:
  %v = invoke.i64 @callee(%i), bb6, bb9

bb4:
  %two = iconst.i64 2
  br bb7(%two)

bb6:
  br bb7(%v)

bb7(%w: i64):
  %a = add %acc, %w
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %a)

bb8(%r: i64):
  ret %r

bb9:
  %lp = landing_pad
  ret %zero
}
)";
    check_transform_preserves(mir, "u", [](Module& m) {
        for (auto& fn : m.functions()) unswitch_loops_in_function(*fn);
    }, {{i64(6), i64(1)}, {i64(6), i64(9)}});
}

// ---------------------------------------------------------------------------
// Dead induction cycles
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - removing a dead induction cycle updates every switch edge") {
    const char* mir = R"(
func @sw(%n: i64, %k: i32) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero, %zero)

bb1(%i: i64, %j: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb9(%acc)

bb2:
  %one = iconst.i64 1
  %in = add %i, %one
  %jn = add %j, %one
  %a2 = add %acc, %i
  switch.i32 %k, default: bb1(%in, %jn, %acc), [1: bb1(%in, %jn, %a2), 2: bb1(%in, %jn, %in)]

bb9(%r: i64):
  ret %r
}
)";
    check_transform_preserves(mir, "sw", [](Module& m) { run_loop_pipeline(m); },
                              {{i64(5), i32(0)}, {i64(5), i32(1)}, {i64(5), i32(2)}, {i64(0), i32(1)}});
}

// ---------------------------------------------------------------------------
// Loop fusion
// ---------------------------------------------------------------------------

TEST_CASE("Soundness - fusion keeps loops apart when plain loads and stores conflict") {
    // Loop 2 reads the total loop 1 accumulates in p[0]; fused, it would see
    // partial sums.
    const char* mir = R"(
func @f(%p: ptr, %a: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  store.i64 %p, 0, %zero
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  %old = load.i64 %p, 0
  %nw = add %old, %i
  store.i64 %p, 0, %nw
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  br bb4(%zero)

bb4(%k: i64):
  %c2 = slt %k, %n
  br_if %c2, bb5, bb6

bb5:
  %s = load.i64 %p, 0
  store_indexed.i64 %a, %k, 8, %s
  %one2 = iconst.i64 1
  %kn = add %k, %one2
  br bb4(%kn)

bb6:
  %last = iconst.i64 2
  %r = load_indexed.i64 %a, %last, 8
  ret %r
}
)";
    auto mod = parse(mir);
    CHECK(!fuse(*find_fn(*mod, "f")));

    std::vector<int64_t> p(1, 0);
    std::vector<int64_t> a(8, 0);
    check_transform_preserves(mir, "f", [](Module& m) { fuse(*find_fn(m, "f")); },
                              {{ptr_arg(p.data()), ptr_arg(a.data()), i64(5)}});
}

TEST_CASE("Soundness - fusion does not reorder property accesses") {
    auto mod = parse(R"(
func @f(%o: i64, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  call @bronze_prop_set(%o, %i)
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  br bb4(%zero, %zero)

bb4(%k: i64, %acc: i64):
  %c2 = slt %k, %n
  br_if %c2, bb5, bb6(%acc)

bb5:
  %v = call.i64 @bronze_prop_get(%o)
  %a2 = add %acc, %v
  %one2 = iconst.i64 1
  %kn = add %k, %one2
  br bb4(%kn, %a2)

bb6(%r: i64):
  ret %r
}
)");
    CHECK(!fuse(*find_fn(*mod, "f")));
    CHECK(verifies(*mod));
}

TEST_CASE("Soundness - fusion requires identical float limits") {
    const char* shape = R"(
func @fl(%a: ptr) -> i64 {
bb0:
  %zero = iconst.i64 0
  %l1 = fconst.f64 10.0
  %l2 = fconst.f64 LIMIT2
  br bb1(%zero)

bb1(%i: i64):
  %fi = sitofp_f64_i64 %i
  %c = slt %fi, %l1
  br_if %c, bb2, bb3

bb2:
  store_indexed.i64 %a, %i, 8, %i
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  br bb4(%zero)

bb4(%k: i64):
  %fk = sitofp_f64_i64 %k
  %c2 = slt %fk, %l2
  br_if %c2, bb5, bb6

bb5:
  %two = iconst.i64 2
  %k2 = mul %k, %two
  store_indexed.i64 %a, %k, 8, %k2
  %one2 = iconst.i64 1
  %kn = add %k, %one2
  br bb4(%kn)

bb6:
  ret %zero
}
)";
    auto with_limit = [&](const char* limit) {
        std::string text = shape;
        text.replace(text.find("LIMIT2"), 6, limit);
        return text;
    };
    auto differ = parse(with_limit("10.5"));
    CHECK(!fuse(*find_fn(*differ, "fl")));
    auto same = parse(with_limit("10.0"));
    CHECK(fuse(*find_fn(*same, "fl")));
    CHECK(verifies(*same));
    CHECK_EQ(count_loops(*find_fn(*same, "fl")), 1u);
}

TEST_CASE("Soundness - array contraction fuses loops only when asked") {
    auto mod = parse(R"(
func @f(%a: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  store_indexed.i64 %a, %i, 8, %i
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  br bb4(%zero)

bb4(%k: i64):
  %c2 = slt %k, %n
  br_if %c2, bb5, bb6

bb5:
  store_indexed.i64 %a, %k, 8, %zero
  %one2 = iconst.i64 1
  %kn = add %k, %one2
  br bb4(%kn)

bb6:
  ret %zero
}
)");
    Function* fn = find_fn(*mod, "f");
    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    array_contraction_pass(*fn, dom, ArrayContractionOptions{});
    CHECK_EQ(count_loops(*fn), 2u);
}

// ---------------------------------------------------------------------------
// Loop tiling
// ---------------------------------------------------------------------------

namespace {

// A 2-D nest storing STORE_VALUE at index INDEX of %a, then a checksum.
const char* kTileShape = R"(
CALLEE
func @t(%a: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb6(%zero, %zero)

bb2:
  br bb3(%zero)

bb3(%j: i64):
  %c2 = slt %j, %n
  br_if %c2, bb4, bb5

bb4:
  %row = mul %i, %n
  %rm = add %row, %j
  %diag = add %i, %j
  EFFECT
  store_indexed.i64 %a, INDEX, 8, %j
  %one = iconst.i64 1
  %jn = add %j, %one
  br bb3(%jn)

bb5:
  %one2 = iconst.i64 1
  %in = add %i, %one2
  br bb1(%in)

bb6(%k: i64, %sum: i64):
  %lim = iconst.i64 80
  %c3 = slt %k, %lim
  br_if %c3, bb7, bb8

bb7:
  %v = load_indexed.i64 %a, %k, 8
  %w = mul %v, %k
  %s2 = add %sum, %w
  %one3 = iconst.i64 1
  %kn = add %k, %one3
  br bb6(%kn, %s2)

bb8:
  ret %sum
}
)";

std::string tile_module(const char* index, const char* effect = "", const char* callee = "") {
    std::string text = kTileShape;
    text.replace(text.find("CALLEE"), 6, callee);
    text.replace(text.find("EFFECT"), 6, effect);
    text.replace(text.find("INDEX"), 5, index);
    return text;
}

bool tile(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    return loop_tile_pass(fn, dom);
}

} // namespace

TEST_CASE("Soundness - tiling a row-major store is allowed and preserves results") {
    const std::string mir = tile_module("%rm");
    auto mod = parse(mir);
    CHECK(tile(*find_fn(*mod, "t")));
    CHECK(verifies(*mod));
    std::vector<int64_t> buf(40 * 40, 0);
    check_transform_preserves(mir, "t", [](Module& m) { tile(*find_fn(m, "t")); },
                              {{ptr_arg(buf.data()), i64(40)}});
}

TEST_CASE("Soundness - tiling keeps nests whose stores collide across both loops") {
    // a[i + j] is written by many (i, j); tiling would change the last writer.
    const std::string mir = tile_module("%diag");
    auto mod = parse(mir);
    CHECK(!tile(*find_fn(*mod, "t")));
    std::vector<int64_t> buf(128, 0);
    check_transform_preserves(mir, "t", [](Module& m) { tile(*find_fn(m, "t")); },
                              {{ptr_arg(buf.data()), i64(40)}});
}

TEST_CASE("Soundness - tiling keeps nests with calls it cannot see into") {
    const std::string mir = tile_module("%rm", "call @observe(%i, %j)");
    auto mod = parse(mir);
    CHECK(!tile(*find_fn(*mod, "t")));
}

// ---------------------------------------------------------------------------
// Loop distribution
// ---------------------------------------------------------------------------

namespace {

const char* kDistributionShape = R"(
CALLEE
func @d(%a: ptr, %b: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  %v = load_indexed.i64 %a, %i, 8
  %three = iconst.i64 3
  %m = mul %v, %three
  store_indexed.i64 %b, %i, 8, %m
  %u = call.i64 @helper(%m)
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  ret %zero
}
)";

std::string distribution_module(const char* callee) {
    std::string text = kDistributionShape;
    text.replace(text.find("CALLEE"), 6, callee);
    return text;
}

} // namespace

TEST_CASE("Soundness - distribution never moves an opaque call after later stores") {
    // @helper could read b[i+1], which the split loop would already have written.
    auto opaque = parse(distribution_module(""));
    CHECK(!distribute(*find_fn(*opaque, "d")));

    auto pure = parse(distribution_module("func @helper(%x: i64) -> i64 {\nbb0:\n  ret %x\n}\n"));
    CHECK(distribute(*find_fn(*pure, "d")));
    CHECK(verifies(*pure));
    CHECK_EQ(count_loops(*find_fn(*pure, "d")), 2u);
}

TEST_CASE("Soundness - distribution never buffers values the second loop needs") {
    // The second part needs %v, a load it can neither recompute nor reload
    // from a store; the old transform sized a GC buffer by the loop limit.
    auto mod = parse(R"(
func @d(%a: ptr, %b: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %sz = iconst.i64 256
  %c1 = call.ptr @brass_gc_alloc(%sz)
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  %v = load_indexed.i64 %a, %i, 8
  %three = iconst.i64 3
  %m = mul %v, %three
  store_indexed.i64 %b, %i, 8, %m
  %seven = iconst.i64 7
  %idx = and %v, %seven
  store_indexed.i64 %c1, %idx, 8, %i
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  ret %zero
}
)");
    Function* fn = find_fn(*mod, "d");
    CHECK(!distribute(*fn));
    CHECK_EQ(count_calls(*fn, "brass_gc_alloc"), 1u);
}

TEST_CASE("Soundness - distribution keeps loops whose parts may alias") {
    // The scatter into %a and the store to %b[i] may hit the same memory.
    auto mod = parse(R"(
func @helper(%x: i64) -> i64 {
bb0:
  ret %x
}

func @d(%a: ptr, %b: ptr, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero)

bb1(%i: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3

bb2:
  %three = iconst.i64 3
  %m = mul %i, %three
  store_indexed.i64 %b, %i, 8, %m
  %u = call.i64 @helper(%m)
  %seven = iconst.i64 7
  %idx = and %u, %seven
  store_indexed.i64 %a, %idx, 8, %i
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in)

bb3:
  ret %zero
}
)");
    CHECK(!distribute(*find_fn(*mod, "d")));
}
