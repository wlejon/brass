// Induction-variable strength reduction (strength_reduce_induction_variables):
// it moves a loop's exit compare onto the scaled variable only when that
// cannot overflow, steps a decrementing variable down, and leaves GC-heap
// bases alone.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <iostream>
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

// i counts down from n; the scaled variable must count down with it.
constexpr std::string_view kCountDown = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 128, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%n, %zero)

hdr(%i: i64, %acc: i64):
  %c = sgt.i64 %i, %zero
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %i, 8, %i
  %v = load_indexed.i64 %buf, %i, 8
  %w = mul.i64 %v, %i
  %acc2 = add.i64 %acc, %w
  %i1 = sub.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";

} // namespace

TEST_CASE("IVSR - an exit compare whose scaled limit would overflow stays unscaled") {
    check_ivsr_keeps_answer(kHugeLimit, 4, true);
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
