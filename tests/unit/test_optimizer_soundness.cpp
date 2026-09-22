// Negative tests for the MIR optimizer: shapes a pass must leave alone (or
// must transform without changing the answer), each checked structurally and,
// where the code is executable, by running it before and after on the
// interpreter and after on the JIT.

#include "soundness_helpers.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace brass;
using namespace soundness;

// ---------------------------------------------------------------------------
// Alias analysis

// Two pointer arguments may name the same bytes whatever types the accesses
// use: bronze reads one header word both as i32 and as i64.
constexpr std::string_view kMixedTypeStores = R"(
func @f64_over_i64(%p: ptr, %q: ptr) -> i64 {
bb0:
  %c = iconst.i64 5
  store.i64 %p, 8, %c
  %d = fconst.f64 1.5
  store.f64 %q, 8, %d
  %v = load.i64 %p, 8
  ret %v
}

func @i32_over_i64(%p: ptr, %q: ptr) -> i64 {
bb0:
  %c = iconst.i64 5
  store.i64 %p, 8, %c
  %d = iconst.i32 7
  store.i32 %q, 8, %d
  %v = load.i64 %p, 8
  ret %v
}

func @ptr_vs_gcref(%p: gcref, %q: ptr) -> i64 {
bb0:
  %c = iconst.i64 5
  store.i64 %p, 8, %c
  %d = iconst.i64 7
  store.i64 %q, 8, %d
  %v = load.i64 %p, 8
  ret %v
}
)";

TEST_CASE("Soundness - GVN keeps a load that a differently typed store may overwrite") {
    auto mod = parse(kMixedTypeStores);
    gvn_module(*mod);
    for (std::string_view name : {"f64_over_i64", "i32_over_i64", "ptr_vs_gcref"}) {
        CHECK_EQ(count_opcode(*find_fn(*mod, name), Opcode::load), size_t(1));
    }

    alignas(8) unsigned char buffer[32] = {};
    const auto ptr = RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buffer));
    check_transform_preserves(kMixedTypeStores, "f64_over_i64", [](Module& m) { gvn_module(m); }, {{ptr, ptr}});
    check_transform_preserves(kMixedTypeStores, "i32_over_i64", [](Module& m) { gvn_module(m); }, {{ptr, ptr}});
}

TEST_CASE("Soundness - GVN keeps a load through a select that may pick the stored-to allocation") {
    auto mod = parse(R"(
func @f(%x: i64) -> i64 {
bb0:
  %sz = iconst.i64 64
  %a = call.ptr @malloc(%sz)
  %b = call.ptr @malloc(%sz)
  %z = iconst.i64 0
  %cond = eq %x, %z
  %q = select.ptr %cond, %a, %b
  %c = iconst.i64 5
  store.i64 %a, 8, %c
  %d = fconst.f64 1.5
  store.f64 %q, 8, %d
  %v = load.i64 %a, 8
  ret %v
}
)");
    gvn_module(*mod);
    CHECK_EQ(count_opcode(*find_fn(*mod, "f"), Opcode::load), size_t(1));
}

TEST_CASE("Soundness - alias analysis: a variable index reaches every constant offset") {
    auto mod = parse(R"(
func @f(%p: ptr, %i: i64) -> i64 {
bb0:
  %a = load_indexed.i64 %p, %i, 8, 16
  %c = iconst.i64 99
  store.i64 %p, 24, %c
  %b = load_indexed.i64 %p, %i, 8, 16
  store_indexed.i64 %p, %i, 8, 0, %c
  %e = load.i64 %p, 40
  %r = sub %b, %a
  %s = add %r, %e
  ret %s
}
)");
    Function* fn = find_fn(*mod, "f");
    std::vector<Instruction*> insts;
    for (Instruction* inst : *fn->entry_block()) insts.push_back(inst);
    Instruction* store_const = nullptr;
    Instruction* load_idx = nullptr;
    Instruction* store_idx = nullptr;
    Instruction* load_const = nullptr;
    for (Instruction* inst : insts) {
        if (inst->opcode() == Opcode::store) store_const = inst;
        if (inst->opcode() == Opcode::load_indexed && !load_idx) load_idx = inst;
        if (inst->opcode() == Opcode::store_indexed) store_idx = inst;
        if (inst->opcode() == Opcode::load) load_const = inst;
    }
    REQUIRE(store_const && load_idx && store_idx && load_const);

    AliasAnalysis aa(*fn);
    // p[i*8 + 16] is p+24 when i == 1, and p[i*8] is p+40 when i == 5.
    CHECK(aa.can_clobber(store_const, load_idx));
    CHECK(aa.can_clobber(store_idx, load_const));
    // Accesses of unknown size off one base at different offsets may overlap.
    CHECK_EQ(aa.alias(fn->entry_block()->param(0), 0, fn->entry_block()->param(0), 4), AliasResult::MayAlias);

    alignas(8) int64_t buffer[8] = {};
    const auto ptr = RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buffer));
    // Every run starts from the same memory: the function clears the word the
    // previous run wrote.
    std::string text = R"(
func @f(%p: ptr, %i: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  store.i64 %p, 24, %zero
  %a = load_indexed.i64 %p, %i, 8, 16
  %c = iconst.i64 99
  store.i64 %p, 24, %c
  %b = load_indexed.i64 %p, %i, 8, 16
  %r = sub %b, %a
  ret %r
}
)";
    check_transform_preserves(text, "f", [](Module& m) { gvn_module(m); }, {{ptr, i64(1)}, {ptr, i64(0)}});
}

// ---------------------------------------------------------------------------
// Constant folding: the loop pipeline, SCCP and the machine must agree.

constexpr std::string_view kI32Folds = R"(
func @lshr32() -> i32 {
bb0:
  %a = iconst.i32 -1
  %b = iconst.i32 1
  %r = lshr %a, %b
  ret %r
}

func @udiv32() -> i32 {
bb0:
  %a = iconst.i32 -1
  %b = iconst.i32 2
  %r = udiv %a, %b
  ret %r
}

func @umod32() -> i32 {
bb0:
  %a = iconst.i32 -1
  %b = iconst.i32 10
  %r = umod %a, %b
  ret %r
}

func @shl32_wide_amount() -> i32 {
bb0:
  %a = iconst.i32 1
  %b = iconst.i32 33
  %r = shl %a, %b
  ret %r
}

func @add32_wraps() -> i32 {
bb0:
  %a = iconst.i32 2147483647
  %b = iconst.i32 1
  %r = add %a, %b
  ret %r
}

func @ult32() -> i32 {
bb0:
  %a = iconst.i32 -1
  %b = iconst.i32 1
  %r = ult %a, %b
  ret %r
}

func @ashr32() -> i32 {
bb0:
  %a = iconst.i32 -8
  %b = iconst.i32 1
  %r = ashr %a, %b
  ret %r
}
)";

TEST_CASE("Soundness - loop-pipeline constant folding works at the operation's width") {
    const std::vector<std::pair<std::string_view, int32_t>> expected = {
        {"lshr32", 0x7FFFFFFF}, {"udiv32", 0x7FFFFFFF}, {"umod32", 5},
        {"shl32_wide_amount", 2}, {"add32_wraps", std::numeric_limits<int32_t>::min()},
        {"ult32", 0}, {"ashr32", -4},
    };

    auto loop_mod = parse(kI32Folds);
    run_loop_pipeline(*loop_mod);
    auto sccp_mod = parse(kI32Folds);
    sccp_module(*sccp_mod);

    for (const auto& [name, value] : expected) {
        Interpreter a;
        CHECK_EQ(a.run(*loop_mod, name).as_i32(), value);
        Interpreter b;
        CHECK_EQ(b.run(*sccp_mod, name).as_i32(), value);
        check_transform_preserves(kI32Folds, name, run_loop_pipeline, {{}});
    }
}

TEST_CASE("Soundness - MIN sdiv -1 folds to the wrapped result at both widths") {
    // docs/semantics.md: MIN / -1 == MIN and MIN % -1 == 0.
    constexpr std::string_view text = R"(
func @d(%x: i64) -> i64 {
bb0:
  %a = iconst.i64 -9223372036854775808
  %b = iconst.i64 -1
  %r = sdiv %a, %b
  %m = smod %a, %b
  %s = add %r, %m
  %a32 = iconst.i32 -2147483648
  %b32 = iconst.i32 -1
  %r32 = sdiv %a32, %b32
  %m32 = smod %a32, %b32
  %s32 = add %r32, %m32
  %w = sext_i64 %s32
  %t = xor %s, %w
  ret %t
}
)";
    auto mod = parse(text);
    run_loop_pipeline(*mod);
    sccp_module(*mod);
    CHECK_EQ(count_opcode(*find_fn(*mod, "d"), Opcode::sdiv), size_t(0));
    CHECK_EQ(count_opcode(*find_fn(*mod, "d"), Opcode::smod), size_t(0));
    Interpreter interp;
    const int64_t expected = std::numeric_limits<int64_t>::min() ^
                             static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    CHECK_EQ(interp.run(*mod, "d", {i64(0)}).as_i64(), expected);
    check_transform_preserves(text, "d", [](Module& m) { sccp_module(m); }, {{i64(0)}});
}

TEST_CASE("Soundness - run-time MIN sdiv -1 wraps on the interpreter and both x64 JITs") {
    constexpr std::string_view text = R"(
func @d64(%a: i64, %b: i64) -> i64 {
bb0:
  %q = sdiv %a, %b
  %r = smod %a, %b
  %s = xor %q, %r
  ret %s
}

func @d32(%a: i32, %b: i32) -> i32 {
bb0:
  %q = sdiv %a, %b
  %r = smod %a, %b
  %s = xor %q, %r
  ret %s
}
)";
    const int64_t min64 = std::numeric_limits<int64_t>::min();
    const int32_t min32 = std::numeric_limits<int32_t>::min();
    auto nothing = [](Module&) {};
    check_transform_preserves(text, "d64", nothing,
                              {{i64(min64), i64(-1)}, {i64(min64), i64(1)}, {i64(-7), i64(-1)}, {i64(7), i64(-2)}});
    check_transform_preserves(text, "d32", nothing,
                              {{i32(min32), i32(-1)}, {i32(min32), i32(1)}, {i32(-7), i32(-1)}, {i32(7), i32(-2)}});

    auto mod = parse(text);
    Interpreter interp;
    CHECK_EQ(interp.run(*mod, "d64", {i64(min64), i64(-1)}).as_i64(), min64);
    CHECK_EQ(interp.run(*mod, "d32", {i32(min32), i32(-1)}).as_i32(), min32);

    if (Target::host().is_x64()) {
        codegen::BaselineJitCompiler baseline;
        auto compiled = baseline.compile_module(*mod);
        for (const auto& cf : compiled) {
            REQUIRE(cf.is_valid());
            if (cf.name() == "d64") {
                CHECK_EQ(cf.invoke({i64(min64), i64(-1)}).as_i64(), min64);
                CHECK_EQ(cf.invoke({i64(-9), i64(-1)}).as_i64(), int64_t(9));
            } else {
                CHECK_EQ(cf.invoke({i32(min32), i32(-1)}).as_i32(), min32);
                CHECK_EQ(cf.invoke({i32(-9), i32(-1)}).as_i32(), int32_t(9));
            }
        }
    }
}

TEST_CASE("Soundness - LICM does not hoist a division by a possibly-zero divisor out of a guarded block") {
    // The loop divides only when d != 0; hoisting the invariant division to
    // the preheader would divide by zero when d == 0.
    constexpr std::string_view text = R"(
func @f(%x: i64, %n: i64, %d: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb5(%acc)

bb2:
  %z = iconst.i64 0
  %safe = ne %d, %z
  br_if %safe, bb3, bb4(%acc)

bb3:
  %q = sdiv %x, %d
  %a2 = add %acc, %q
  br bb4(%a2)

bb4(%accn: i64):
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %accn)

bb5(%r: i64):
  ret %r
}
)";
    auto mod = parse(text);
    run_loop_pipeline(*mod);
    Function* fn = find_fn(*mod, "f");
    for (const Instruction* inst : *fn->entry_block()) {
        CHECK_NE(inst->opcode(), Opcode::sdiv);
    }
    check_transform_preserves(text, "f", run_loop_pipeline,
                              {{i64(std::numeric_limits<int64_t>::min()), i64(3), i64(0)},
                               {i64(7), i64(3), i64(-1)}, {i64(7), i64(3), i64(2)}});
}

// ---------------------------------------------------------------------------
// Bounds-check elimination and the ranges it relies on

TEST_CASE("Soundness - BCE leaves an unrelated in-loop condition conditional") {
    // `%t` compares i+1 against an unrelated %k, and the block's branch tests
    // something else entirely; neither may become unconditional.
    constexpr std::string_view text = R"(
func @f(%n: i64, %k: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb5(%acc)

bb2:
  %one = iconst.i64 1
  %ip1 = add %i, %one
  %t = slt %ip1, %k
  %three = iconst.i64 3
  %isthree = eq %i, %three
  br_if %isthree, bb3, bb4

bb3:
  %hund = iconst.i64 100
  %a1 = add %acc, %hund
  br bb6(%a1)

bb4:
  %tz = zext_i64 %t
  %a2 = add %acc, %tz
  br bb6(%a2)

bb6(%accn: i64):
  %one2 = iconst.i64 1
  %in = add %i, %one2
  br bb1(%in, %accn)

bb5(%r: i64):
  ret %r
}
)";
    auto mod = parse(text);
    run_bce(*mod);
    CHECK_EQ(count_opcode(*find_fn(*mod, "f"), Opcode::guard), size_t(0));
    const std::vector<std::vector<RuntimeValue>> args = {
        {i64(10), i64(0)}, {i64(10), i64(5)}, {i64(10), i64(100)}, {i64(0), i64(3)}};
    check_transform_preserves(text, "f", run_bce, args);
    check_transform_preserves(text, "f", run_loop_pipeline, args);
}

TEST_CASE("Soundness - BCE does not drop a constant offset on the index") {
    // Inside `i < n`, `i + 1 < n` fails on the last iteration.
    constexpr std::string_view text = R"(
func @f(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb5(%acc)

bb2:
  %one = iconst.i64 1
  %ip1 = add %i, %one
  %t = slt %ip1, %n
  br_if %t, bb3, bb4(%acc)

bb3:
  %a1 = add %acc, %one
  br bb4(%a1)

bb4(%accn: i64):
  %one2 = iconst.i64 1
  %in = add %i, %one2
  br bb1(%in, %accn)

bb5(%r: i64):
  ret %r
}
)";
    const std::vector<std::vector<RuntimeValue>> args = {{i64(10)}, {i64(1)}, {i64(0)}};
    check_transform_preserves(text, "f", run_bce, args);
    check_transform_preserves(text, "f", run_loop_pipeline, args);
}

TEST_CASE("Soundness - range analysis does not clip a wrapping i32 add") {
    // a is in [0, INT32_MAX], so a + 1 wraps to INT32_MIN when a is INT32_MAX.
    constexpr std::string_view text = R"(
func @f(%x: i32) -> i32 {
bb0:
  %mask = iconst.i32 2147483647
  %a = and %x, %mask
  %one = iconst.i32 1
  %b = add %a, %one
  %zero = iconst.i32 0
  %neg = slt %b, %zero
  br_if %neg, bb1, bb2

bb1:
  %r1 = iconst.i32 1
  ret %r1

bb2:
  %r2 = iconst.i32 2
  ret %r2
}
)";
    check_transform_preserves(text, "f", run_bce, {{i32(std::numeric_limits<int32_t>::max())}, {i32(5)}});
}

TEST_CASE("Soundness - an unsigned branch bounds nothing when its bound may be negative") {
    // x <u y with y == -1 holds for every x, including negative ones.
    constexpr std::string_view text = R"(
func @f(%x: i64, %y: i64) -> i64 {
bb0:
  %lt = ult %x, %y
  br_if %lt, bb1, bb3

bb1:
  %zero = iconst.i64 0
  %neg = slt %x, %zero
  br_if %neg, bb2, bb3

bb2:
  %r1 = iconst.i64 1
  ret %r1

bb3:
  %r2 = iconst.i64 2
  ret %r2
}
)";
    check_transform_preserves(text, "f", run_bce, {{i64(-5), i64(-1)}, {i64(3), i64(10)}});
}

TEST_CASE("Soundness - induction variable ranges cover the exit value and an unentered loop") {
    // Stepping by 4 below 10 leaves i == 12 at the exit; starting at 100 the
    // loop never runs and i stays 100.
    constexpr std::string_view text = R"(
func @f(%start: i64) -> i64 {
bb0:
  %sel = iconst.i64 0
  %isz = eq %start, %sel
  %hundred = iconst.i64 100
  %zero = iconst.i64 0
  %init = select.i64 %isz, %zero, %hundred
  br bb1(%init)

bb1(%i: i64):
  %ten = iconst.i64 10
  %c = slt %i, %ten
  br_if %c, bb2, bb3

bb2:
  %four = iconst.i64 4
  %in = add %i, %four
  br bb1(%in)

bb3:
  %eleven = iconst.i64 11
  %big = sgt %i, %eleven
  %fifty = iconst.i64 50
  %small = ult %i, %fifty
  %bz = zext_i64 %big
  %sz = zext_i64 %small
  %two = iconst.i64 2
  %sz2 = mul %sz, %two
  %r = add %bz, %sz2
  ret %r
}
)";
    check_transform_preserves(text, "f", run_bce, {{i64(0)}, {i64(1)}});
    check_transform_preserves(text, "f", run_loop_pipeline, {{i64(0)}, {i64(1)}});
}

// ---------------------------------------------------------------------------
// Array contraction

TEST_CASE("Soundness - array contraction keeps an array another call still reads") {
    auto mod = parse(R"(
func @f(%n: i64, %j: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %cap = iconst.i64 16
  %arr = call.i64 @bronze_create_array(%cap)
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)

bb2:
  %x = iconst.i64 11
  call @bronze_elem_set(%arr, %i, %x)
  %t = call.i64 @bronze_elem_get(%arr, %i)
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64):
  %key = iconst.i64 7
  %len = call.i64 @bronze_prop_get(%arr, %key)
  %s = add %r, %len
  ret %s
}
)");
    Function* fn = find_fn(*mod, "f");
    DominatorTree dom(*fn);
    CHECK(!array_contraction_pass(*fn, dom));
    CHECK(verify_module(*mod));
    CHECK_EQ(count_calls(*fn, "bronze_create_array"), size_t(1));
    CHECK_EQ(count_calls(*fn, "bronze_elem_get"), size_t(1));
}

TEST_CASE("Soundness - array contraction does not forward past a write at another index") {
    // a[i] = 11; a[j] = 22; a[i] reads 22 when i == j.
    auto mod = parse(R"(
func @f(%n: i64, %j: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %cap = iconst.i64 16
  %arr = call.i64 @bronze_create_array(%cap)
  %mask = iconst.i64 7
  %jj = and %j, %mask
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %eight = iconst.i64 8
  %c = slt %i, %eight
  br_if %c, bb2, bb3(%acc)

bb2:
  %x = iconst.i64 11
  %y = iconst.i64 22
  call @bronze_elem_set(%arr, %i, %x)
  call @bronze_elem_set(%arr, %jj, %y)
  %t = call.i64 @bronze_elem_get(%arr, %i)
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64):
  ret %r
}
)");
    Function* fn = find_fn(*mod, "f");
    DominatorTree dom(*fn);
    CHECK(!array_contraction_pass(*fn, dom));
    CHECK_EQ(count_calls(*fn, "bronze_elem_get"), size_t(1));
}

TEST_CASE("Soundness - array contraction keeps an array that escapes through a select or block argument") {
    auto mod = parse(R"(
func @sel(%n: i64, %other: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %cap = iconst.i64 16
  %arr = call.i64 @bronze_create_array(%cap)
  %p = eq %n, %zero
  %s = select.i64 %p, %arr, %other
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %eight = iconst.i64 8
  %c = slt %i, %eight
  br_if %c, bb2, bb3(%acc)

bb2:
  %x = iconst.i64 11
  call @bronze_elem_set(%arr, %i, %x)
  %t = call.i64 @bronze_elem_get(%arr, %i)
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64):
  %key = iconst.i64 0
  %l = call.i64 @bronze_elem_get(%s, %key)
  %sum = add %r, %l
  ret %sum
}

func @arg(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %cap = iconst.i64 16
  %arr = call.i64 @bronze_create_array(%cap)
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %eight = iconst.i64 8
  %c = slt %i, %eight
  br_if %c, bb2, bb3(%acc, %arr)

bb2:
  %x = iconst.i64 11
  call @bronze_elem_set(%arr, %i, %x)
  %t = call.i64 @bronze_elem_get(%arr, %i)
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64, %a: i64):
  %key = iconst.i64 0
  %l = call.i64 @bronze_elem_get(%a, %key)
  %sum = add %r, %l
  ret %sum
}
)");
    for (std::string_view name : {"sel", "arg"}) {
        Function* fn = find_fn(*mod, name);
        DominatorTree dom(*fn);
        CHECK(!array_contraction_pass(*fn, dom));
        CHECK_EQ(count_calls(*fn, "bronze_create_array"), size_t(1));
    }
}

TEST_CASE("Soundness - array contraction needs indices proven to be plain elements") {
    // A negative index is not an element: the write is dropped and the read
    // sees undefined, so the stored value must not be forwarded.
    auto mod = parse(R"(
func @f(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %cap = iconst.i64 16
  %arr = call.i64 @bronze_create_array(%cap)
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)

bb2:
  %x = iconst.i64 11
  %neg = sub %zero, %i
  call @bronze_elem_set(%arr, %neg, %x)
  %t = call.i64 @bronze_elem_get(%arr, %neg)
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64):
  ret %r
}
)");
    Function* fn = find_fn(*mod, "f");
    DominatorTree dom(*fn);
    CHECK(!array_contraction_pass(*fn, dom));
    CHECK_EQ(count_calls(*fn, "bronze_elem_get"), size_t(1));
}

TEST_CASE("Soundness - raw buffer contraction only forwards the exact bytes written") {
    // The load reads the slot after the one stored; it sees what the
    // previous iteration stored, or zero.
    constexpr std::string_view text = R"(
func @f(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %sz = iconst.i64 1024
  %buf = call.gcref @brass_gc_alloc(%sz)
  br bb1(%zero, %zero)

bb1(%i: i64, %acc: i64):
  %c = slt %i, %n
  br_if %c, bb2, bb3(%acc)

bb2:
  %three = iconst.i64 3
  %v = mul %i, %three
  store_indexed.i64 %buf, %i, 8, 8, %v
  %t = load_indexed.i64 %buf, %i, 8, 0
  %acc2 = add %acc, %t
  %one = iconst.i64 1
  %in = add %i, %one
  br bb1(%in, %acc2)

bb3(%r: i64):
  ret %r
}
)";
    auto mod = parse(text);
    Function* fn = find_fn(*mod, "f");
    DominatorTree dom(*fn);
    CHECK(!array_contraction_pass(*fn, dom));
    CHECK_EQ(count_opcode(*fn, Opcode::load_indexed), size_t(1));

    Interpreter interp;
    // sum over i of 3*(i-1) for i = 1..9 = 3 * 36
    CHECK_EQ(interp.run(*mod, "f", {i64(10)}).as_i64(), int64_t{108});
}
