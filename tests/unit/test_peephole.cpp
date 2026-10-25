#include "test_framework.hpp"
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/x64/x64_registers.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("Peephole - Redundant Move Elimination") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // mov rax, rax (redundant)
    auto m1 = std::make_unique<LirInst>(LirOpcode::Mov);
    m1->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    m1->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(m1));

    // mov rbx, rax (not redundant)
    auto m2 = std::make_unique<LirInst>(LirOpcode::Mov);
    m2->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    m2->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(m2));

    // movsd xmm0, xmm0 (redundant)
    auto m3 = std::make_unique<LirInst>(LirOpcode::Movsd);
    m3->add_def(LirOperand::preg_xmm(XMM::XMM0, 8));
    m3->add_use(LirOperand::preg_xmm(XMM::XMM0, 8));
    bb->append_inst(std::move(m3));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.redundant_moves_eliminated, size_t(2));
    CHECK_EQ(bb->instructions.size(), size_t(1));
    CHECK(bb->instructions[0]->defs[0].preg_val.as_gpr() == GPR::RBX);
}

TEST_CASE("Peephole - Redundant Load After Store Elimination") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // mov [slot 0], rax
    auto store = std::make_unique<LirInst>(LirOpcode::Mov);
    store->add_def(LirOperand::slot(0, 8));
    store->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(store));

    // mov rax, [slot 0] (exact redundant load)
    auto load1 = std::make_unique<LirInst>(LirOpcode::Mov);
    load1->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    load1->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(load1));

    // mov rcx, [slot 0] (forwarding load -> mov rcx, rax)
    auto load2 = std::make_unique<LirInst>(LirOpcode::Mov);
    load2->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    load2->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(load2));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.load_after_store_eliminated, size_t(2));
    CHECK_EQ(bb->instructions.size(), size_t(2));
    // Instruction 0: store [slot 0], rax
    CHECK(bb->instructions[0]->defs[0].is_spill_slot());
    // Instruction 1: mov rcx, rax
    CHECK(bb->instructions[1]->defs[0].is_preg());
    CHECK(bb->instructions[1]->defs[0].preg_val.as_gpr() == GPR::RCX);
    CHECK(bb->instructions[1]->uses[0].is_preg());
    CHECK(bb->instructions[1]->uses[0].preg_val.as_gpr() == GPR::RAX);
}

TEST_CASE("Peephole - Dead Move Elimination") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // mov rax, 42 (dead move - overwritten by next mov)
    auto dead_m = std::make_unique<LirInst>(LirOpcode::Mov);
    dead_m->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    dead_m->add_use(LirOperand::imm(42, 8));
    bb->append_inst(std::move(dead_m));

    // mov rax, 100
    auto live_m = std::make_unique<LirInst>(LirOpcode::Mov);
    live_m->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    live_m->add_use(LirOperand::imm(100, 8));
    bb->append_inst(std::move(live_m));

    // ret rax
    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.dead_moves_eliminated, size_t(1));
    CHECK_EQ(bb->instructions.size(), size_t(2));
    CHECK_EQ(bb->instructions[0]->uses[0].imm_int, int64_t(100));
}

TEST_CASE("Peephole - Arithmetic Simplifications") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // add rax, 0 (eliminated)
    auto add0 = std::make_unique<LirInst>(LirOpcode::Add);
    add0->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    add0->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    add0->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(add0));

    // sub rbx, 0 (eliminated)
    auto sub0 = std::make_unique<LirInst>(LirOpcode::Sub);
    sub0->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
    sub0->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
    sub0->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(sub0));

    // imul rcx, 1 (eliminated)
    auto mul1 = std::make_unique<LirInst>(LirOpcode::Imul);
    mul1->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    mul1->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    mul1->add_use(LirOperand::imm(1, 8));
    bb->append_inst(std::move(mul1));

    // mov rdx, 0 -> xor32 edx, edx
    auto mov0 = std::make_unique<LirInst>(LirOpcode::Mov);
    mov0->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
    mov0->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(mov0));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.arithmetic_simplified, size_t(4));
    CHECK_EQ(bb->instructions.size(), size_t(1));
    CHECK(bb->instructions[0]->opcode == LirOpcode::Xor32);
    CHECK(bb->instructions[0]->defs[0].preg_val.as_gpr() == GPR::RDX);
}

TEST_CASE("Peephole - Branch Simplifications") {
    LirFunction fn;
    LirBlock* b0 = fn.create_block("bb0");
    LirBlock* b1 = fn.create_block("bb1");
    LirBlock* b2 = fn.create_block("bb2");

    b0->id = 0;
    b1->id = 1;
    b2->id = 2;

    // b0 ends with jmp bb1 (bb1 is immediate next block -> eliminated)
    auto jmp_next = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_next->add_use(LirOperand::label(1));
    b0->append_inst(std::move(jmp_next));

    // b1 ends with jcc E, bb0; jmp bb2 (bb2 is immediate next block -> jmp bb2 eliminated)
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = Condition::E;
    jcc->add_use(LirOperand::label(0));
    b1->append_inst(std::move(jcc));

    auto jmp_b2 = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_b2->add_use(LirOperand::label(2));
    b1->append_inst(std::move(jmp_b2));

    // b2 ends with ret
    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    b2->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.branches_simplified, size_t(2));
    CHECK(b0->instructions.empty());
    CHECK_EQ(b1->instructions.size(), size_t(1));
    CHECK(b1->instructions[0]->opcode == LirOpcode::Jcc);
}

TEST_CASE("Peephole - Preserves Safepoint and Deopt Metadata") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // Patchable const mov rax, 0 (must NOT be turned into xor32)
    auto patch_mov = std::make_unique<LirInst>(LirOpcode::Mov32);
    patch_mov->is_patchable = true;
    patch_mov->patch_symbol = "ic_slot_1";
    patch_mov->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
    patch_mov->add_use(LirOperand::imm(0, 4));
    bb->append_inst(std::move(patch_mov));

    // Safepoint instruction with live gcrefs (must NOT be touched)
    auto sp = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp->safepoint_id = 123;
    sp->live_gcrefs.push_back(VReg{1, RegClass::GPR, 8, true});
    bb->append_inst(std::move(sp));

    // Guard exit (must NOT be touched)
    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = 456;
    exit_inst->exit_symbol = "stub_exit";
    bb->append_inst(std::move(exit_inst));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.total_optimizations(), size_t(0));
    CHECK_EQ(bb->instructions.size(), size_t(3));
    CHECK(bb->instructions[0]->is_patchable);
    CHECK(bb->instructions[0]->opcode == LirOpcode::Mov32);
    CHECK_EQ(bb->instructions[1]->safepoint_id, uint32_t(123));
    CHECK_EQ(bb->instructions[2]->resume_id, uint32_t(456));
}

TEST_CASE("Peephole - Multi-Reload Store Forwarding and Narrowed 32-bit Spill Reload") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // 64-bit spill store: mov [slot 0], rax
    auto store = std::make_unique<LirInst>(LirOpcode::Mov);
    store->add_def(LirOperand::slot(0, 8));
    store->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    bb->append_inst(std::move(store));

    // First 32-bit reload: mov32 ecx, [slot 0, 4] -> mov32 ecx, eax
    auto load1 = std::make_unique<LirInst>(LirOpcode::Mov32);
    load1->add_def(LirOperand::preg_gpr(GPR::RCX, 4));
    load1->add_use(LirOperand::slot(0, 4));
    bb->append_inst(std::move(load1));

    // Second 64-bit reload in the same block: mov rdx, [slot 0, 8] -> mov rdx, rax
    auto load2 = std::make_unique<LirInst>(LirOpcode::Mov);
    load2->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
    load2->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(load2));

    // Keep ecx and rdx live at return
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg_gpr(GPR::RCX, 8));
    add->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    add->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
    bb->append_inst(std::move(add));

    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    bb->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.load_after_store_eliminated, size_t(2));
}

TEST_CASE("Peephole - Vector Move Redundancy and Copy Propagation") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // movaps xmm0, xmm0 (redundant self-move)
    auto self_mov = std::make_unique<LirInst>(LirOpcode::Movaps);
    self_mov->add_def(LirOperand::preg_xmm(XMM::XMM0, 16));
    self_mov->add_use(LirOperand::preg_xmm(XMM::XMM0, 16));
    bb->append_inst(std::move(self_mov));

    // vmovaps ymm1, ymm0 (propagated into vaddps ymm2, ymm1, ymm0 and then eliminated as dead)
    auto copy_mov = std::make_unique<LirInst>(LirOpcode::Vmovaps);
    copy_mov->add_def(LirOperand::preg_xmm(XMM::XMM1, 32));
    copy_mov->add_use(LirOperand::preg_xmm(XMM::XMM0, 32));
    bb->append_inst(std::move(copy_mov));

    auto vadd = std::make_unique<LirInst>(LirOpcode::Vaddps);
    vadd->add_def(LirOperand::preg_xmm(XMM::XMM2, 32));
    vadd->add_use(LirOperand::preg_xmm(XMM::XMM1, 32));
    vadd->add_use(LirOperand::preg_xmm(XMM::XMM0, 32));
    bb->append_inst(std::move(vadd));

    // Overwrite ymm1 so copy_mov becomes dead
    auto kill_xmm1 = std::make_unique<LirInst>(LirOpcode::Vmovaps);
    kill_xmm1->add_def(LirOperand::preg_xmm(XMM::XMM1, 32));
    kill_xmm1->add_use(LirOperand::preg_xmm(XMM::XMM2, 32));
    bb->append_inst(std::move(kill_xmm1));

    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_xmm(XMM::XMM1, 32));
    bb->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK(stats.redundant_moves_eliminated >= size_t(2));
    CHECK_EQ(stats.dead_moves_eliminated, size_t(1));
    REQUIRE_EQ(bb->instructions.size(), size_t(3));
    CHECK(bb->instructions[0]->opcode == LirOpcode::Vaddps);
    CHECK(bb->instructions[0]->uses[0].preg_val.as_xmm() == XMM::XMM0);
}

TEST_CASE("Peephole - 32-bit Identity Ops Preserve Upper-Bit Zero Extension") {
    for (LirOpcode op : {LirOpcode::Add32, LirOpcode::Sub32, LirOpcode::Imul32}) {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");
        const int64_t identity_imm = (op == LirOpcode::Imul32) ? 1 : 0;

        auto inst = std::make_unique<LirInst>(op);
        inst->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        inst->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        inst->add_use(LirOperand::imm(identity_imm, 4));
        bb->append_inst(std::move(inst));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        bb->append_inst(std::move(ret));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.arithmetic_simplified, size_t(1));
        CHECK_EQ(stats.redundant_moves_eliminated, size_t(0));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Mov32);
        REQUIRE_EQ(bb->instructions[0]->defs.size(), size_t(1));
        REQUIRE_EQ(bb->instructions[0]->uses.size(), size_t(1));
        CHECK(bb->instructions[0]->defs[0].preg_val.as_gpr() == GPR::RAX);
        CHECK(bb->instructions[0]->uses[0].preg_val.as_gpr() == GPR::RAX);
    }
}

TEST_CASE("Peephole - Memory and Spill Source Identity Ops Rewrite to Mov") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");

    // imul rax, [rcx + 16], 1 -> mov rax, [rcx + 16]
    auto imul_mem = std::make_unique<LirInst>(LirOpcode::Imul);
    imul_mem->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    imul_mem->add_use(LirOperand::mem(PReg::gpr(GPR::RCX), 16, 8));
    imul_mem->add_use(LirOperand::imm(1, 8));
    bb->append_inst(std::move(imul_mem));

    // imul32 edx, [slot 0], 1 -> mov32 edx, [slot 0]
    auto imul_slot = std::make_unique<LirInst>(LirOpcode::Imul32);
    imul_slot->add_def(LirOperand::preg_gpr(GPR::RDX, 4));
    imul_slot->add_use(LirOperand::slot(0, 4));
    imul_slot->add_use(LirOperand::imm(1, 4));
    bb->append_inst(std::move(imul_slot));

    // add r8, [rcx + 24], 0 -> mov r8, [rcx + 24]
    auto add_mem = std::make_unique<LirInst>(LirOpcode::Add);
    add_mem->add_def(LirOperand::preg_gpr(GPR::R8, 8));
    add_mem->add_use(LirOperand::mem(PReg::gpr(GPR::RCX), 24, 8));
    add_mem->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(add_mem));

    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    ret->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
    ret->add_use(LirOperand::preg_gpr(GPR::R8, 8));
    bb->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.arithmetic_simplified, size_t(3));
    REQUIRE_EQ(bb->instructions.size(), size_t(4));
    CHECK(bb->instructions[0]->opcode == LirOpcode::Mov);
    CHECK(bb->instructions[0]->uses[0].is_mem());
    CHECK(bb->instructions[1]->opcode == LirOpcode::Mov32);
    CHECK(bb->instructions[1]->uses[0].is_spill_slot());
    CHECK(bb->instructions[2]->opcode == LirOpcode::Mov);
    CHECK(bb->instructions[2]->uses[0].is_mem());
}

TEST_CASE("Peephole - Identity Arithmetic Preserved When Flags Are Live") {
    LirFunction fn;
    LirBlock* bb = fn.create_block("entry");
    LirBlock* target = fn.create_block("target");
    bb->id = 0;
    target->id = 1;

    // add rax, rcx, 0 followed by jcc (flags live! cannot rewrite to mov rax, rcx)
    auto add_inst = std::make_unique<LirInst>(LirOpcode::Add);
    add_inst->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    add_inst->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    add_inst->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(add_inst));

    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = Condition::E;
    jcc->add_use(LirOperand::label(1));
    bb->append_inst(std::move(jcc));

    // imul rdx, rdx, 1 followed by setcc (flags live! cannot delete)
    auto imul_inst = std::make_unique<LirInst>(LirOpcode::Imul);
    imul_inst->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
    imul_inst->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
    imul_inst->add_use(LirOperand::imm(1, 8));
    target->append_inst(std::move(imul_inst));

    auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
    setcc->condition = Condition::O;
    setcc->add_def(LirOperand::preg_gpr(GPR::RAX, 1));
    target->append_inst(std::move(setcc));

    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    target->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.arithmetic_simplified, size_t(0));
    CHECK(bb->instructions[0]->opcode == LirOpcode::Add);
    CHECK(target->instructions[0]->opcode == LirOpcode::Imul);
}

TEST_CASE("Peephole - Mov32 Self-Move Elimination and 64-bit Reader Preservation") {
    // Case 1: Prior 32-bit def in the same block already zeroed bits [63:32].
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto xor32 = std::make_unique<LirInst>(LirOpcode::Xor32);
        xor32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RCX, 4));
        bb->append_inst(std::move(xor32));

        auto self32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        self32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        self32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        bb->append_inst(std::move(self32));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        bb->append_inst(std::move(ret));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.redundant_moves_eliminated, size_t(1));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Xor32);
        CHECK(bb->instructions[1]->opcode == LirOpcode::Ret);
    }

    // Case 2: Subsequent uses only read 32 bits before a full (>=4-byte) redefinition.
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto self32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        self32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        self32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        bb->append_inst(std::move(self32));

        auto xor32 = std::make_unique<LirInst>(LirOpcode::Xor32);
        xor32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RCX, 4));
        bb->append_inst(std::move(xor32));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        bb->append_inst(std::move(ret));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.redundant_moves_eliminated, size_t(1));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Xor32);
        CHECK(bb->instructions[1]->opcode == LirOpcode::Ret);
    }

    // Case 3: A 64-bit reader follows before any redefinition -> Mov32 must be kept.
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto self32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        self32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        self32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        bb->append_inst(std::move(self32));

        auto add64 = std::make_unique<LirInst>(LirOpcode::Add);
        add64->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
        add64->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
        add64->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        bb->append_inst(std::move(add64));

        auto xor32 = std::make_unique<LirInst>(LirOpcode::Xor32);
        xor32->add_def(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RAX, 4));
        xor32->add_use(LirOperand::preg_gpr(GPR::RCX, 4));
        bb->append_inst(std::move(xor32));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
        ret->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        bb->append_inst(std::move(ret));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.redundant_moves_eliminated, size_t(0));
        REQUIRE_EQ(bb->instructions.size(), size_t(4));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Mov32);
    }
}



