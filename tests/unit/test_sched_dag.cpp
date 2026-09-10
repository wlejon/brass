#include "test_framework.hpp"
#include <brass/codegen/sched_dag.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/x64/x64_registers.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("SchedDAG - RAW Data Dependence and Latencies") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: Load rax, [slot 0] (latency 4)
    auto load = std::make_unique<LirInst>(LirOpcode::Mov);
    load->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    load->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(load));

    // Inst 1: Add rax, 10 (latency 1, uses rax, defs rax)
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    add->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    add->add_use(LirOperand::imm(10, 8));
    bb->append_inst(std::move(add));

    // Inst 2: Imul rcx, rax (latency 3, uses rax, defs rcx)
    auto mul = std::make_unique<LirInst>(LirOpcode::Imul);
    mul->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    mul->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(mul));

    // Inst 3: Addps xmm0, xmm1 (latency 3)
    auto addps = std::make_unique<LirInst>(LirOpcode::Addps);
    addps->add_def(LirOperand::preg_xmm(XMM::XMM0, 16));
    addps->add_use(LirOperand::preg_xmm(XMM::XMM0, 16));
    addps->add_use(LirOperand::preg_xmm(XMM::XMM1, 16));
    bb->append_inst(std::move(addps));

    // Inst 4: Idiv rbx (latency 13)
    auto idiv = std::make_unique<LirInst>(LirOpcode::Idiv);
    idiv->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    bb->append_inst(std::move(idiv));

    SchedDAG dag(*bb, false);
    dag.build();

    CHECK_EQ(dag.size(), size_t(5));

    // Check individual latencies
    CHECK_EQ(dag.node(0).latency, 4u);
    CHECK_EQ(dag.node(1).latency, 1u);
    CHECK_EQ(dag.node(2).latency, 3u);
    CHECK_EQ(dag.node(3).latency, 3u);
    CHECK_EQ(dag.node(4).latency, 13u);

    // Edge (0 -> 1) must be RAW with latency 4
    bool found_0_1 = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 1) {
            found_0_1 = true;
            CHECK(s.kind == EdgeKind::RAW);
            CHECK_EQ(s.latency, 4u);
        }
    }
    CHECK(found_0_1);

    // Edge (1 -> 2) must be RAW with latency 1
    bool found_1_2 = false;
    for (const auto& s : dag.node(1).succs) {
        if (s.target_node == 2) {
            found_1_2 = true;
            CHECK(s.kind == EdgeKind::RAW);
            CHECK_EQ(s.latency, 1u);
        }
    }
    CHECK(found_1_2);
}

TEST_CASE("SchedDAG - WAR Anti-dependence and WAW Output dependence") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: mov rbx, rax (reads rax)
    auto m0 = std::make_unique<LirInst>(LirOpcode::Mov);
    m0->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    m0->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(m0));

    // Inst 1: mov rax, 42 (overwrites rax) -> WAR edge from 0 to 1 with latency 0
    auto m1 = std::make_unique<LirInst>(LirOpcode::Mov);
    m1->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    m1->add_use(LirOperand::imm(42, 8));
    bb->append_inst(std::move(m1));

    // Inst 2: mov rax, 100 (overwrites rax) -> WAW edge from 1 to 2 with latency 1
    auto m2 = std::make_unique<LirInst>(LirOpcode::Mov);
    m2->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    m2->add_use(LirOperand::imm(100, 8));
    bb->append_inst(std::move(m2));

    SchedDAG dag(*bb, false);
    dag.build();

    CHECK_EQ(dag.size(), size_t(3));

    // Check WAR (0 -> 1)
    bool found_war = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 1) {
            found_war = true;
            CHECK(s.kind == EdgeKind::WAR);
            CHECK_EQ(s.latency, 0u);
        }
    }
    CHECK(found_war);

    // Check WAW (1 -> 2)
    bool found_waw = false;
    for (const auto& s : dag.node(1).succs) {
        if (s.target_node == 2) {
            found_waw = true;
            CHECK(s.kind == EdgeKind::WAW);
            CHECK_EQ(s.latency, 1u);
        }
    }
    CHECK(found_waw);
}

TEST_CASE("SchedDAG - Memory Hazards and Disambiguation") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: Store [slot 0], rax
    auto s0 = std::make_unique<LirInst>(LirOpcode::Mov);
    s0->add_def(LirOperand::slot(0, 8));
    s0->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(s0));

    // Inst 1: Load rbx, [slot 0] (same slot -> MemRAW from 0 with latency 4)
    auto l1 = std::make_unique<LirInst>(LirOpcode::Mov);
    l1->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    l1->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(l1));

    // Inst 2: Store [slot 1], rdx (different slot -> no edge to/from slot 0)
    auto s2 = std::make_unique<LirInst>(LirOpcode::Mov);
    s2->add_def(LirOperand::slot(1, 8));
    s2->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
    bb->append_inst(std::move(s2));

    // Inst 3: Load rsi, [rcx + 0] (heap -> no edge with spill slot)
    auto l3 = std::make_unique<LirInst>(LirOpcode::Mov);
    l3->add_def(LirOperand::preg_gpr(GPR::RSI, 8));
    l3->add_use(LirOperand::mem(PReg::gpr(GPR::RCX), 0, 8));
    bb->append_inst(std::move(l3));

    // Inst 4: Store [rcx + 16], rdi (heap disjoint displacement -> no edge with l3)
    auto s4 = std::make_unique<LirInst>(LirOpcode::Mov);
    s4->add_def(LirOperand::mem(PReg::gpr(GPR::RCX), 16, 8));
    s4->add_use(LirOperand::preg_gpr(GPR::RDI, 8));
    bb->append_inst(std::move(s4));

    SchedDAG dag(*bb, false);
    dag.build();

    // Check MemRAW hazard between 0 and 1
    bool found_hazard_0_1 = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 1) {
            found_hazard_0_1 = true;
            CHECK(s.kind == EdgeKind::MemRAW);
            CHECK_EQ(s.latency, 4u);
        }
    }
    CHECK(found_hazard_0_1);

    // Verify slot 0 and slot 1 do not alias
    bool found_hazard_0_2 = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 2) found_hazard_0_2 = true;
    }
    CHECK(!found_hazard_0_2);

    // Verify slot 0 and heap l3 do not alias
    bool found_hazard_0_3 = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 3) found_hazard_0_3 = true;
    }
    CHECK(!found_hazard_0_3);

    // Verify disjoint heap accesses [rcx + 0] and [rcx + 16] do not alias
    bool found_hazard_3_4 = false;
    for (const auto& s : dag.node(3).succs) {
        if (s.target_node == 4) found_hazard_3_4 = true;
    }
    CHECK(!found_hazard_3_4);
}

TEST_CASE("SchedDAG - Scheduling Barriers") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: Store [rax], rbx
    auto s0 = std::make_unique<LirInst>(LirOpcode::Mov);
    s0->add_def(LirOperand::mem(PReg::gpr(GPR::RAX), 0, 8));
    s0->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    bb->append_inst(std::move(s0));

    // Inst 1: Call (barrier)
    auto c1 = std::make_unique<LirInst>(LirOpcode::Call);
    c1->callee_symbol = "external_foo";
    bb->append_inst(std::move(c1));

    // Inst 2: Load rcx, [rax]
    auto l2 = std::make_unique<LirInst>(LirOpcode::Mov);
    l2->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    l2->add_use(LirOperand::mem(PReg::gpr(GPR::RAX), 0, 8));
    bb->append_inst(std::move(l2));

    // Inst 3: Ret (terminator)
    auto r3 = std::make_unique<LirInst>(LirOpcode::Ret);
    bb->append_inst(std::move(r3));

    SchedDAG dag(*bb, false);
    dag.build();

    // Check barrier edge from Store to Call
    bool edge_0_1 = false;
    for (const auto& s : dag.node(0).succs) {
        if (s.target_node == 1) {
            edge_0_1 = true;
            CHECK(s.kind == EdgeKind::Barrier);
        }
    }
    CHECK(edge_0_1);

    // Check barrier edge from Call to Load
    bool edge_1_2 = false;
    for (const auto& s : dag.node(1).succs) {
        if (s.target_node == 2) {
            edge_1_2 = true;
            CHECK(s.kind == EdgeKind::Barrier);
        }
    }
    CHECK(edge_1_2);

    // Check terminator edge from Load to Ret
    bool edge_2_3 = false;
    for (const auto& s : dag.node(2).succs) {
        if (s.target_node == 3) {
            edge_2_3 = true;
            CHECK(s.kind == EdgeKind::Barrier);
        }
    }
    CHECK(edge_2_3);
}

TEST_CASE("SchedDAG - Critical Path Height and Depth") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Inst 0: Load rax, [slot 0] (latency 4)
    auto l0 = std::make_unique<LirInst>(LirOpcode::Mov);
    l0->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    l0->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(l0));

    // Inst 1: Add rax, 1 (latency 1, depends on 0)
    auto a1 = std::make_unique<LirInst>(LirOpcode::Add);
    a1->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    a1->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    a1->add_use(LirOperand::imm(1, 8));
    bb->append_inst(std::move(a1));

    // Inst 2: Mov rbx, 2 (latency 1, independent)
    auto m2 = std::make_unique<LirInst>(LirOpcode::Mov);
    m2->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    m2->add_use(LirOperand::imm(2, 8));
    bb->append_inst(std::move(m2));

    // Inst 3: Add rbx, rax (latency 1, depends on 1 and 2)
    auto a3 = std::make_unique<LirInst>(LirOpcode::Add);
    a3->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    a3->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    a3->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(a3));

    SchedDAG dag(*bb, false);
    dag.build();

    // Node 0: depth 0, height 4 + 1 + 1 = 6
    CHECK_EQ(dag.node(0).depth, 0u);
    CHECK_EQ(dag.node(0).height, 6u);

    // Node 1: depth 4, height 1 + 1 = 2
    CHECK_EQ(dag.node(1).depth, 4u);
    CHECK_EQ(dag.node(1).height, 2u);

    // Node 2: depth 0, height 1 + 1 = 2
    CHECK_EQ(dag.node(2).depth, 0u);
    CHECK_EQ(dag.node(2).height, 2u);

    // Node 3: depth 5 (max(4+1, 0+1)), height 1
    CHECK_EQ(dag.node(3).depth, 5u);
    CHECK_EQ(dag.node(3).height, 1u);
}
