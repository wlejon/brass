#include "test_framework.hpp"
#include <brass/codegen/lir.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/target/target.hpp>

using namespace brass;
using namespace brass::codegen;

TEST_CASE("Block Layout - Cold Block Placed at End") {
    LirFunction fn;
    fn.name = "test_cold_layout";

    LirBlock* b0 = fn.create_block("entry");
    LirBlock* b_cold = fn.create_block("deopt_cold_handler");
    LirBlock* b1 = fn.create_block("hot_header");
    LirBlock* b2 = fn.create_block("hot_body");
    LirBlock* b_exit = fn.create_block("exit");

    b0->loop_depth = 0;
    b1->loop_depth = 1;
    b2->loop_depth = 1;
    b_exit->loop_depth = 0;
    b_cold->loop_depth = 0;

    // b0 jumps to b1
    auto jmp0 = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp0->add_use(LirOperand::label(b1->id));
    b0->append_inst(std::move(jmp0));

    // b1 has conditional branch to b2 (true) or b_cold (false)
    auto jcc1 = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc1->condition = x64::Condition::NE;
    jcc1->add_use(LirOperand::label(b2->id));
    b1->append_inst(std::move(jcc1));
    auto jmp1 = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp1->add_use(LirOperand::label(b_cold->id));
    b1->append_inst(std::move(jmp1));

    // b2 has conditional branch to b1 (true) or b_exit (false)
    auto jcc2 = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc2->condition = x64::Condition::E;
    jcc2->add_use(LirOperand::label(b1->id));
    b2->append_inst(std::move(jcc2));
    auto jmp2 = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp2->add_use(LirOperand::label(b_exit->id));
    b2->append_inst(std::move(jmp2));

    // b_cold ends with guard exit / ret
    auto guard_exit = std::make_unique<LirInst>(LirOpcode::GuardExit);
    guard_exit->exit_symbol = "brass_deopt_exit";
    b_cold->append_inst(std::move(guard_exit));

    // b_exit ends with ret
    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    b_exit->append_inst(std::move(ret_inst));

    // Initially in fn: entry, cold, hot_header, hot_body, exit
    CHECK_EQ(fn.blocks.size(), 5);
    CHECK_EQ(fn.blocks[1]->name, "deopt_cold_handler");

    optimize_block_layout(fn);

    // Entry block must remain first
    CHECK_EQ(fn.blocks[0]->id, b0->id);

    // Cold block must be placed at the very end
    CHECK_EQ(fn.blocks.back()->id, b_cold->id);

    // Hot blocks b1 and b2 should be placed before b_exit and before b_cold
    size_t idx_b1 = 0, idx_b2 = 0, idx_cold = 0;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        if (fn.blocks[i]->id == b1->id) idx_b1 = i;
        if (fn.blocks[i]->id == b2->id) idx_b2 = i;
        if (fn.blocks[i]->id == b_cold->id) idx_cold = i;
    }
    CHECK(idx_b1 < idx_cold);
    CHECK(idx_b2 < idx_cold);
    CHECK_EQ(idx_b2, idx_b1 + 1); // b2 placed immediately after b1 for straight-line execution
}

TEST_CASE("Block Layout - Fall-Through Maximization") {
    LirFunction fn;
    fn.name = "test_fallthrough";

    LirBlock* b0 = fn.create_block("entry");
    LirBlock* b1 = fn.create_block("block1");
    LirBlock* b2 = fn.create_block("block2");
    LirBlock* b3 = fn.create_block("block3");

    // b0 -> b2
    auto j0 = std::make_unique<LirInst>(LirOpcode::Jmp);
    j0->add_use(LirOperand::label(b2->id));
    b0->append_inst(std::move(j0));

    // b2 -> b1
    auto j2 = std::make_unique<LirInst>(LirOpcode::Jmp);
    j2->add_use(LirOperand::label(b1->id));
    b2->append_inst(std::move(j2));

    // b1 -> b3
    auto j1 = std::make_unique<LirInst>(LirOpcode::Jmp);
    j1->add_use(LirOperand::label(b3->id));
    b1->append_inst(std::move(j1));

    // b3 -> ret
    b3->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    // Ordering initially: b0, b1, b2, b3 (each jump taken!)
    optimize_block_layout(fn);

    // Ordering should be traced: b0 -> b2 -> b1 -> b3 (fall-through along whole chain!)
    CHECK_EQ(fn.blocks[0]->id, b0->id);
    CHECK_EQ(fn.blocks[1]->id, b2->id);
    CHECK_EQ(fn.blocks[2]->id, b1->id);
    CHECK_EQ(fn.blocks[3]->id, b3->id);

    // Redundant unconditional jumps to immediate fall-through targets are eliminated
    CHECK(fn.blocks[0]->instructions.empty());
    CHECK(fn.blocks[1]->instructions.empty());
    CHECK(fn.blocks[2]->instructions.empty());
    CHECK_EQ(fn.blocks[3]->instructions.size(), 1);
    CHECK(fn.blocks[3]->instructions[0]->opcode == LirOpcode::Ret);
}

namespace {

void jmp_to(LirBlock* from, LirBlock* to) {
    auto j = std::make_unique<LirInst>(LirOpcode::Jmp);
    j->add_use(LirOperand::label(to->id));
    from->append_inst(std::move(j));
}

void branch(LirBlock* from, LirBlock* taken, LirBlock* other) {
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = x64::Condition::NE;
    jcc->add_use(LirOperand::label(taken->id));
    from->append_inst(std::move(jcc));
    jmp_to(from, other);
}

void call(LirBlock* b, const char* symbol) {
    auto c = std::make_unique<LirInst>(LirOpcode::Call);
    c->callee_symbol = symbol;
    b->append_inst(std::move(c));
}

void copy(LirBlock* b) {
    auto m = std::make_unique<LirInst>(LirOpcode::Mov);
    m->add_def(LirOperand::preg_gpr(x64::GPR::RAX));
    m->add_use(LirOperand::preg_gpr(x64::GPR::RCX));
    b->append_inst(std::move(m));
}

size_t index_of(const LirFunction& fn, const LirBlock* b) {
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        if (fn.blocks[i].get() == b) return i;
    }
    return fn.blocks.size();
}

} // namespace

// A type test whose generic side calls a helper and whose fast side only
// carries the edge's copy (a Number's bits as its double): the fast side
// falls through to the join, the call goes out of line. Before, the side with
// work (the call) fell through and every fast path went out and back.
TEST_CASE("Block Layout - the side that calls is laid out of line") {
    LirFunction fn;
    fn.name = "test_call_side";
    LirBlock* entry = fn.create_block("entry");
    LirBlock* slow = fn.create_block("unbox_slow");
    LirBlock* fast = fn.create_block("unbox_num");
    LirBlock* merge = fn.create_block("unbox_merge");
    branch(entry, slow, fast);
    call(slow, "bronze_unbox_f64");
    jmp_to(slow, merge);
    copy(fast);
    jmp_to(fast, merge);
    merge->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(index_of(fn, fast), 1);
    CHECK_EQ(index_of(fn, merge), 2);
    CHECK(index_of(fn, slow) > index_of(fn, merge));
}

// An if-body that is only a call (a stack-limit or TDZ check): the skip falls
// through and the body is entered by the taken branch, so the check costs no
// taken branch when it passes.
TEST_CASE("Block Layout - an if-body that calls does not fall through") {
    LirFunction fn;
    fn.name = "test_call_body";
    LirBlock* entry = fn.create_block("entry");
    LirBlock* body = fn.create_block("check_fail");
    LirBlock* cont = fn.create_block("cont");
    branch(entry, cont, body);
    call(body, "bronze_stack_overflow");
    jmp_to(body, cont);
    cont->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(index_of(fn, cont), 1);
    CHECK_EQ(index_of(fn, body), 2);
}
