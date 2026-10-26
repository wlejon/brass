#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/branch_split.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <memory>
#include <string_view>

using namespace brass;

namespace {

size_t count_ops(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

int64_t run(Function& fn, int64_t a, int64_t b, int64_t c) {
    Interpreter interp;
    return interp.run(fn, {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b), RuntimeValue::from_i64(c)}).as_i64();
}

// (a < b && (b < c || c == 7)) ? a + 1 : b, through block arguments, and a
// loop whose exit test is an and: counts i from a while i < b and i != c.
constexpr std::string_view kSrc = R"(module @m
func @cond(%a: i64, %b: i64, %c: i64) -> i64 {
entry:
  %lt = slt.i64 %a, %b
  %bc = slt.i64 %b, %c
  %seven = iconst.i64 7
  %c7 = eq.i64 %c, %seven
  %or = or.i32 %bc, %c7
  %and = and.i32 %lt, %or
  %one = iconst.i64 1
  %a1 = add.i64 %a, %one
  br_if %and, done(%a1), done(%b)
done(%r: i64):
  ret %r
}
func @loop(%a: i64, %b: i64, %c: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  br head(%a, %zero)
head(%i: i64, %n: i64):
  %in = slt.i64 %i, %b
  %ne = ne.i64 %i, %c
  %go = and.i32 %in, %ne
  br_if %go, body, out(%n)
body:
  %one = iconst.i64 1
  %i1 = add.i64 %i, %one
  %n1 = add.i64 %n, %one
  br head(%i1, %n1)
out(%r: i64):
  ret %r
}
)";

} // namespace

TEST_CASE("Branch split - an and/or condition becomes a branch per comparison") {
    DiagnosticReporter diag;
    auto ref = parse_module(kSrc, &diag);
    auto mod = parse_module(kSrc, &diag);
    REQUIRE(ref != nullptr);
    REQUIRE(mod != nullptr);
    for (const char* name : {"cond", "loop"}) {
        Function* fn = mod->get_function(name);
        Function* ref_fn = ref->get_function(name);
        REQUIRE(fn != nullptr);
        CHECK(split_branch_conditions(*fn));
        CHECK(verify_function(*fn));
        CHECK_EQ(count_ops(*fn, Opcode::and_), 0u);
        CHECK_EQ(count_ops(*fn, Opcode::or_), 0u);
        CHECK_FALSE(split_branch_conditions(*fn));
        for (int64_t a = -2; a <= 4; ++a) {
            for (int64_t b = -2; b <= 4; ++b) {
                for (int64_t c : {-1, 0, 2, 3, 7}) {
                    CHECK_EQ(run(*fn, a, b, c), run(*ref_fn, a, b, c));
                }
            }
        }
    }
    // The or under the and splits as well: three branches in @cond.
    CHECK_EQ(count_ops(*mod->get_function("cond"), Opcode::br_if), 3u);
}

TEST_CASE("Branch split - a condition read twice stays whole") {
    constexpr std::string_view src = R"(module @m
func @f(%a: i64, %b: i64, %c: i64) -> i64 {
entry:
  %x = slt.i64 %a, %b
  %y = slt.i64 %b, %c
  %and = and.i32 %x, %y
  %w = zext.i64 %and
  br_if %and, t(%w), u
t(%r: i64):
  ret %r
u:
  ret %c
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    CHECK_FALSE(split_branch_conditions(*mod->get_function("f")));
}
