// Induction-variable strength reduction (strength_reduce_induction_variables):
// it moves a loop's exit compare onto the scaled variable only when that
// cannot overflow, steps a decrementing variable down, and leaves GC-heap
// bases alone; a variable stride becomes a pointer variable, and the
// unroller's hoisted trip test holds at the edges of the counter's range.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <iostream>
#include <limits>
#include <string>

using namespace brass;

namespace {

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

int64_t run(const Module& mod, int64_t arg) {
    Interpreter interp;
    return interp.run(mod, "f", {RuntimeValue::from_i64(arg)}).as_i64();
}

// The header compare of the loop headed by block `hdr`.
const Instruction* header_compare(const Function& fn) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() != "hdr") continue;
        const Instruction* term = bb->terminator();
        REQUIRE(term && term->opcode() == Opcode::br_if);
        return term->operand(0)->defining_instruction();
    }
    REQUIRE(false);
    return nullptr;
}

const Value* header_param(const Function& fn, size_t i) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() == "hdr") return bb->param(i);
    }
    REQUIRE(false);
    return nullptr;
}

void check_ivsr_keeps_answer(std::string_view text, int64_t arg, bool expect_change) {
    auto before = parse(text);
    auto mod = parse(text);
    Function& fn = *mod->get_function("f");
    CHECK_EQ(strength_reduce_induction_variables(fn), expect_change);
    DiagnosticReporter diag;
    const bool ok = verify_module(*mod, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n" << to_string(*mod) << "\n";
    REQUIRE(ok);
    CHECK_EQ(run(*mod, arg), run(*before, arg));
}

// i runs 0..4 and leaves through the early exit, but the header compares it
// with 2^62: scaled by 8 that limit wraps to 0, so `8*i < 8*limit` would end
// the loop at once.
constexpr std::string_view kHugeLimit = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 64, 8
  %zero = iconst.i64 0
  %big = iconst.i64 4611686018427387904
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %big
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %i, 8, %i
  %v = load_indexed.i64 %buf, %i, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  %stop = sgt.i64 %i1, %n
  br_if %stop, exit(%acc2), hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";

constexpr std::string_view kSmallLimit = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 64, 8
  %zero = iconst.i64 0
  %lim = iconst.i64 6
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %lim
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %i, 8, %n
  %v = load_indexed.i64 %buf, %i, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";

// j counts down from 15 while i counts the trips; j only indexes, so it is
// replaced by a scaled variable, which must count down with it.
constexpr std::string_view kCountDown = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 128, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %top = iconst.i64 15
  br hdr(%zero, %top, %zero)

hdr(%i: i64, %j: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %j, 8, %n
  %v = load_indexed.i64 %buf, %j, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  %j1 = sub.i64 %j, %one
  br hdr(%i1, %j1, %acc2)

exit(%r: i64):
  ret %r
}
)";

} // namespace

TEST_CASE("IVSR - an exit compare whose scaled limit would overflow stays unscaled") {
    // i stays live for the compare, so its accesses keep addressing it
    // scaled by 8 and no second, scaled variable is made.
    check_ivsr_keeps_answer(kHugeLimit, 4, false);
    auto mod = parse(kHugeLimit);
    Function& fn = *mod->get_function("f");
    strength_reduce_induction_variables(fn);
    CHECK(header_compare(fn)->operand(0) == header_param(fn, 0));
}

TEST_CASE("IVSR - an exit compare with small constant bounds moves to the scaled variable") {
    check_ivsr_keeps_answer(kSmallLimit, 7, true);
    auto mod = parse(kSmallLimit);
    Function& fn = *mod->get_function("f");
    strength_reduce_induction_variables(fn);
    CHECK(header_compare(fn)->operand(0) != header_param(fn, 0));
}

TEST_CASE("IVSR - a decrementing induction variable scales downwards") {
    for (int64_t n : {0, 1, 5, 12}) check_ivsr_keeps_answer(kCountDown, n, n >= 0);
}

TEST_CASE("IVSR - GC-heap bases and i32 variables are left alone") {
    auto mod = parse(R"(
func @f(%buf: gcref, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit(%acc)

body:
  %v = load_indexed.i64 %buf, %i, 8, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}

func @g(%buf: ptr, %n: i32) -> i64 {
bb0:
  %zero = iconst.i32 0
  %z64 = iconst.i64 0
  %one = iconst.i32 1
  br hdr(%zero, %z64)

hdr(%i: i32, %acc: i64):
  %c = slt.i32 %i, %n
  br_if %c, body, exit(%acc)

body:
  %w = sext.i64 %i
  %v = load_indexed.i64 %buf, %w, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i32 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)");
    CHECK(!strength_reduce_induction_variables(*mod->get_function("f")));
    CHECK(!strength_reduce_induction_variables(*mod->get_function("g")));
    CHECK(verify_module(*mod));
}

namespace {

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

// A column walk: kS steps by a stride only known at run time (1..4), so its
// accesses become loads and stores through a pointer variable, which the
// unroller then copies. The trip count is n & 15.
constexpr std::string_view kVariableStride = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 1024, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %three = iconst.i64 3
  %fifteen = iconst.i64 15
  %four = iconst.i64 4
  %lim4 = iconst.i64 128
  %lim = and.i64 %n, %fifteen
  %hi = lshr.i64 %n, %four
  %sm = and.i64 %hi, %three
  %s = add.i64 %sm, %one
  br fill(%zero)

fill(%f: i64):
  %fc = slt.i64 %f, %lim4
  br_if %fc, fill_body, pre

fill_body:
  %fv = mul.i64 %f, %f
  %fv2 = add.i64 %fv, %n
  store_indexed.i64 %buf, %f, 8, %fv2
  %f2 = add.i64 %f, %one
  br fill(%f2)

pre:
  br hdr(%zero, %zero, %zero)

hdr(%k: i64, %ks: i64, %acc: i64):
  %c = slt.i64 %k, %lim
  br_if %c, body, exit(%acc)

body:
  %v = load_indexed.i64 %buf, %ks, 8, 8
  %acc2 = add.i64 %acc, %v
  %ks2 = add.i64 %ks, %s
  %k2 = add.i64 %k, %one
  br hdr(%k2, %ks2, %acc2)

exit(%r: i64):
  ret %r
}
)";

// Trip tests at the edges of the counter's range: i runs from `start` up to
// `limit` (signed, then unsigned); the unrolled loop's hoisted, clamped
// limit must neither run a copy past the bound nor wrap.
std::string edge_loop(const char* cmp, int64_t start, int64_t limit) {
    return std::string(R"(
func @f(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %seven = iconst.i64 7
  %start = iconst.i64 )") + std::to_string(start) + R"(
  %limit = iconst.i64 )" + std::to_string(limit) + R"(
  %lim = add.i64 %limit, %n
  br hdr(%start, %zero)

hdr(%i: i64, %acc: i64):
  %c = )" + cmp + R"(.i64 %i, %lim
  br_if %c, body, exit(%acc)

body:
  %b = and.i64 %i, %seven
  %acc2 = add.i64 %acc, %b
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";
}

} // namespace

TEST_CASE("IVSR - a variable stride walks a pointer variable the unroller copies") {
    for (int64_t n : {0, 1, 3, 4, 5, 15, 16 + 7, 32 + 9, 48 + 15}) {
        auto before = parse(kVariableStride);
        auto mod = parse(kVariableStride);
        Function& fn = *mod->get_function("f");
        CHECK(optimize_function_loops(fn));
        DiagnosticReporter diag;
        const bool ok = verify_module(*mod, &diag);
        if (!ok) std::cerr << diag.format_all() << "\n" << to_string(*mod) << "\n";
        REQUIRE(ok);
        CHECK_EQ(count_op(fn, Opcode::load_indexed), size_t{0});
        CHECK(count_op(fn, Opcode::load) >= 5u);  // four unrolled copies and the remainder
        CHECK_EQ(run(*mod, n), run(*before, n));
    }
}

TEST_CASE("Loop unroll - the trip test holds at the edges of the counter's range") {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    struct Case {
        const char* cmp;
        int64_t start;
        int64_t limit;
    };
    const Case cases[] = {
        {"slt", kMax - 10, kMax},  // the last copies sit right below MAX
        {"slt", kMin, kMin + 2},   // limit below MIN + 3: only the remainder runs
        {"slt", kMin, kMin},       // no trips
        {"slt", -5, 6},
        {"ult", -11, -1},          // unsigned, right below 2^64 - 1
        {"ult", 0, 2},             // limit below 3
        {"ult", 0, 0},
    };
    for (const Case& c : cases) {
        for (int64_t n : {0, 1, 2, 3}) {
            const std::string text = edge_loop(c.cmp, c.start, c.limit);
            auto before = parse(text);
            auto mod = parse(text);
            optimize_function_loops(*mod->get_function("f"));
            REQUIRE(verify_module(*mod));
            CHECK_EQ(count_op(*mod->get_function("f"), Opcode::select), size_t{1});  // the clamp
            // (n wraps the first and fifth limits to below the start: no trips.)
            CHECK_EQ(run(*mod, n), run(*before, n));
        }
    }
}
