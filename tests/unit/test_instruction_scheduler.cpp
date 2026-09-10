#include "test_framework.hpp"
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/x64/x64_registers.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("Scheduler - Load Hoisting and Latency Stall Hiding") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: add rbx, 10 (independent ALU, height 1)
    auto a0 = std::make_unique<LirInst>(LirOpcode::Add);
    a0->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    a0->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    a0->add_use(LirOperand::imm(10, 8));
    bb->append_inst(std::move(a0));

    // Inst 1: add rsi, 20 (independent ALU, height 1)
    auto a1 = std::make_unique<LirInst>(LirOpcode::Add);
    a1->add_def(LirOperand::preg_gpr(GPR::RSI, 8));
    a1->add_use(LirOperand::preg_gpr(GPR::RSI, 8));
    a1->add_use(LirOperand::imm(20, 8));
    bb->append_inst(std::move(a1));

    // Inst 2: mov rax, [slot 0] (load, latency 4, height 4 + 1 = 5)
    auto l2 = std::make_unique<LirInst>(LirOpcode::Mov);
    l2->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    l2->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(l2));

    // Inst 3: add rax, 5 (depends on rax from load l2)
    auto a3 = std::make_unique<LirInst>(LirOpcode::Add);
    a3->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    a3->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    a3->add_use(LirOperand::imm(5, 8));
    bb->append_inst(std::move(a3));

    // Inst 4: ret (terminator)
    auto r4 = std::make_unique<LirInst>(LirOpcode::Ret);
    bb->append_inst(std::move(r4));

    SchedOptions opts;
    opts.enable_post_ra = true;
    SchedStats stats = schedule_block(*bb, opts);

    CHECK_EQ(stats.blocks_scheduled, size_t(1));
    CHECK_EQ(bb->instructions.size(), size_t(5));

    // The load (original Inst 2) must be hoisted to position 0 because it has the longest critical path (height 5)
    CHECK(bb->instructions[0]->uses.size() == 1);
    CHECK(bb->instructions[0]->uses[0].is_spill_slot());
    CHECK(bb->instructions[0]->defs[0].preg_val.as_gpr() == GPR::RAX);

    // Independent operations (rbx or rsi) must be placed between the load and its use to hide latency
    CHECK(bb->instructions[1]->defs[0].preg_val.as_gpr() != GPR::RAX);
    CHECK(bb->instructions[2]->defs[0].preg_val.as_gpr() != GPR::RAX);

    // Ret must remain the final instruction
    CHECK(bb->instructions[4]->opcode == LirOpcode::Ret);
    CHECK(stats.loads_hoisted > 0);
}

TEST_CASE("Scheduler - Division Latency Hiding") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: imul rax, rbx (high latency 3, critical path height 3 + 1 = 4)
    auto mul = std::make_unique<LirInst>(LirOpcode::Imul);
    mul->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    mul->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    bb->append_inst(std::move(mul));

    // Inst 1: add rdi, 100 (independent ALU, height 1)
    auto indep1 = std::make_unique<LirInst>(LirOpcode::Add);
    indep1->add_def(LirOperand::preg_gpr(GPR::RDI, 8));
    indep1->add_use(LirOperand::preg_gpr(GPR::RDI, 8));
    indep1->add_use(LirOperand::imm(100, 8));
    bb->append_inst(std::move(indep1));

    // Inst 2: add rsi, 200 (independent ALU, height 1)
    auto indep2 = std::make_unique<LirInst>(LirOpcode::Add);
    indep2->add_def(LirOperand::preg_gpr(GPR::RSI, 8));
    indep2->add_use(LirOperand::preg_gpr(GPR::RSI, 8));
    indep2->add_use(LirOperand::imm(200, 8));
    bb->append_inst(std::move(indep2));

    // Inst 3: add rcx, rax (uses rax from mul)
    auto use_mul = std::make_unique<LirInst>(LirOpcode::Add);
    use_mul->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    use_mul->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    use_mul->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(use_mul));

    SchedOptions opts;
    opts.enable_post_ra = true;
    schedule_block(*bb, opts);

    // Multiplier (Inst 0) scheduled first
    CHECK(bb->instructions[0]->opcode == LirOpcode::Imul);

    // Independent operations scheduled before the user of rax to hide multiplier latency
    CHECK(bb->instructions[1]->opcode == LirOpcode::Add);
    CHECK(bb->instructions[1]->defs[0].preg_val.as_gpr() != GPR::RCX);
    CHECK(bb->instructions[2]->defs[0].preg_val.as_gpr() != GPR::RCX);

    // User of rax scheduled after the stalls are hidden
    CHECK(bb->instructions[3]->defs[0].preg_val.as_gpr() == GPR::RCX);
}

TEST_CASE("Scheduler - Register Pressure Throttling") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Create 4 independent virtual register definitions and uses
    VReg v0 = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v1 = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v2 = fn.allocate_vreg(RegClass::GPR, 8);

    // Inst 0: def v0
    auto d0 = std::make_unique<LirInst>(LirOpcode::Mov);
    d0->add_def(LirOperand::vreg(v0, 8));
    d0->add_use(LirOperand::imm(1, 8));
    bb->append_inst(std::move(d0));

    // Inst 1: def v1
    auto d1 = std::make_unique<LirInst>(LirOpcode::Mov);
    d1->add_def(LirOperand::vreg(v1, 8));
    d1->add_use(LirOperand::imm(2, 8));
    bb->append_inst(std::move(d1));

    // Inst 2: use v0 (kills v0)
    auto u0 = std::make_unique<LirInst>(LirOpcode::Add);
    u0->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    u0->add_use(LirOperand::vreg(v0, 8));
    bb->append_inst(std::move(u0));

    // Inst 3: def v2
    auto d2 = std::make_unique<LirInst>(LirOpcode::Mov);
    d2->add_def(LirOperand::vreg(v2, 8));
    d2->add_use(LirOperand::imm(3, 8));
    bb->append_inst(std::move(d2));

    SchedOptions opts;
    opts.enable_pre_ra = true;
    opts.gpr_pressure_threshold = 1; // Strict throttle
    SchedStats stats = schedule_block(*bb, opts);

    CHECK_EQ(stats.blocks_scheduled, size_t(1));
    CHECK(stats.pressure_throttles > 0);
}

TEST_CASE("Scheduler - Execution Port Multi-Issue Balancing") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Load 0
    auto l0 = std::make_unique<LirInst>(LirOpcode::Mov);
    l0->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    l0->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(l0));

    // Load 1
    auto l1 = std::make_unique<LirInst>(LirOpcode::Mov);
    l1->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    l1->add_use(LirOperand::slot(1, 8));
    bb->append_inst(std::move(l1));

    // ALU 0
    auto a0 = std::make_unique<LirInst>(LirOpcode::Add);
    a0->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
    a0->add_use(LirOperand::imm(10, 8));
    bb->append_inst(std::move(a0));

    // ALU 1
    auto a1 = std::make_unique<LirInst>(LirOpcode::Sub);
    a1->add_def(LirOperand::preg_gpr(GPR::RSI, 8));
    a1->add_use(LirOperand::imm(20, 8));
    bb->append_inst(std::move(a1));

    SchedOptions opts;
    opts.enable_post_ra = true;
    opts.balance_issue_ports = true;
    schedule_block(*bb, opts);

    // Verify interleaving: consecutive instructions should balance load vs non-load
    bool first_is_load = bb->instructions[0]->uses[0].is_spill_slot();
    bool second_is_load = bb->instructions[1]->uses[0].is_spill_slot();
    CHECK(first_is_load != second_is_load);
}

TEST_CASE("Scheduler - Pre-RA Virtual Registers and Function Scheduling") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    VReg v_load = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v_indep = fn.allocate_vreg(RegClass::GPR, 8);
    VReg v_res = fn.allocate_vreg(RegClass::GPR, 8);

    // Inst 0: v_indep = 5
    auto m0 = std::make_unique<LirInst>(LirOpcode::Mov);
    m0->add_def(LirOperand::vreg(v_indep, 8));
    m0->add_use(LirOperand::imm(5, 8));
    bb->append_inst(std::move(m0));

    // Inst 1: v_load = [slot 0]
    auto m1 = std::make_unique<LirInst>(LirOpcode::Mov);
    m1->add_def(LirOperand::vreg(v_load, 8));
    m1->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(m1));

    // Inst 2: v_res = add v_load, 1
    auto m2 = std::make_unique<LirInst>(LirOpcode::Add);
    m2->add_def(LirOperand::vreg(v_res, 8));
    m2->add_use(LirOperand::vreg(v_load, 8));
    m2->add_use(LirOperand::imm(1, 8));
    bb->append_inst(std::move(m2));

    SchedOptions opts;
    opts.enable_pre_ra = true;
    SchedStats stats = schedule_function(fn, opts);

    CHECK_EQ(stats.blocks_scheduled, size_t(1));
    // Load should be scheduled first due to higher critical path height
    CHECK(bb->instructions[0]->uses[0].is_spill_slot());
}
