// The shared use-walking and RAUW helpers (brass/mir/uses.hpp): every kind of
// use slot - operands, deopt state, br / br_if / switch default and case
// arguments - is counted, replaced and removed.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include <iostream>

using namespace brass;

namespace {

constexpr std::string_view kText = R"(
func @f(%a: i64, %b: i64) -> i64 {
bb0:
  %two = add.i64 %a, %a
  %c = slt.i64 %a, %b
  guard %c, @deopt, [%a, %two]
  switch.i64 %b, default: bb1(%a), [1: bb2(%a, %a)]

bb1(%x: i64):
  br_if %c, bb2(%x, %a), bb1(%a)

bb2(%p: i64, %q: i64):
  ret %q
}
)";

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

BasicBlock* block(Function& fn, std::string_view name) {
    for (BasicBlock* bb : fn.blocks()) {
        if (bb && bb->name() == name) return bb;
    }
    REQUIRE(false);
    return nullptr;
}

} // namespace

TEST_CASE("MIR uses - every slot kind is counted and replaced") {
    auto mod = parse(kText);
    Function& fn = *mod->get_function("f");
    Value* a = fn.entry_block()->param(0);
    Value* b = fn.entry_block()->param(1);

    // add x2, slt, guard state, switch default, switch case x2, br_if true
    // and false edges.
    CHECK_EQ(count_uses(fn, a), size_t{9});
    CHECK(has_uses(fn, a));
    CHECK_EQ(compute_use_counts(fn)[a], 9u);

    const Instruction* guard = nullptr;
    for (const Instruction* inst : *fn.entry_block()) {
        if (inst->opcode() == Opcode::guard) guard = inst;
    }
    REQUIRE(guard != nullptr);
    CHECK(uses_value(*guard, a));

    // Only the slots `where` accepts.
    const size_t in_bb1 = replace_uses_if(fn, a, b, [&](const Instruction& inst) {
        return inst.parent() == block(fn, "bb1");
    });
    CHECK_EQ(in_bb1, size_t{2});
    CHECK_EQ(count_uses(fn, a), size_t{7});

    CHECK_EQ(replace_all_uses(fn, a, b), size_t{7});
    CHECK(!has_uses(fn, a));
    CHECK_EQ(replace_all_uses(fn, a, b), size_t{0});
    CHECK_EQ(replace_all_uses(fn, b, b), size_t{0});
    CHECK_EQ(replace_all_uses(fn, nullptr, b), size_t{0});
    CHECK(verify_module(*mod));
}

TEST_CASE("MIR uses - remove_block_param drops the argument on every edge kind") {
    auto mod = parse(kText);
    Function& fn = *mod->get_function("f");
    BasicBlock* bb2 = block(fn, "bb2");
    Value* q = bb2->param(1);

    remove_block_param(*bb2, 0);
    REQUIRE(bb2->param_count() == 1u);
    CHECK(bb2->param(0) == q);
    for (BasicBlock* bb : fn.blocks()) {
        for_each_edge(*bb->terminator(), [&](const BranchTarget& bt) {
            if (bt.block == bb2) CHECK_EQ(bt.args.size(), size_t{1});
        });
    }
    fn.rebuild_cfg_predecessors();
    CHECK(verify_module(*mod));

    bool threw = false;
    try {
        remove_block_param(*bb2, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("MIR uses - replace_uses_in touches one instruction or one block") {
    auto mod = parse(kText);
    Function& fn = *mod->get_function("f");
    Value* a = fn.entry_block()->param(0);
    Value* b = fn.entry_block()->param(1);
    Instruction* add = fn.entry_block()->head();
    REQUIRE(add->opcode() == Opcode::add);
    CHECK_EQ(replace_uses_in(*add, a, b), size_t{2});
    CHECK_EQ(replace_uses_in(*block(fn, "bb1"), a, b), size_t{2});
    CHECK_EQ(count_uses(fn, a), size_t{5});
}
