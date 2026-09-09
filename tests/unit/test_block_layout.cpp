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
