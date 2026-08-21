#include <brass/target/x64/x64_isel.hpp>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_overflow_check(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    VReg op1 = get_vreg(op1_val);
    uint8_t sz = op0.size;
    uint8_t dst_sz = dst.size;

    Opcode op = inst.opcode();

    if (op == Opcode::umul_overflow) {
        if (sz == 4) {
            VReg t0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            VReg t1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov0 = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov0->add_def(LirOperand::vreg(t0, 4));
            mov0->add_use(LirOperand::vreg(op0, 4));
            lir_bb.append_inst(std::move(mov0));

            auto mov1 = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov1->add_def(LirOperand::vreg(t1, 4));
            mov1->add_use(LirOperand::vreg(op1, 4));
            lir_bb.append_inst(std::move(mov1));

            VReg prod = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_p = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_p->add_def(LirOperand::vreg(prod, 8));
            mov_p->add_use(LirOperand::vreg(t0, 8));
            lir_bb.append_inst(std::move(mov_p));

            auto imul_inst = std::make_unique<LirInst>(LirOpcode::Imul);
            imul_inst->add_def(LirOperand::vreg(prod, 8));
            imul_inst->add_use(LirOperand::vreg(prod, 8));
            imul_inst->add_use(LirOperand::vreg(t1, 8));
            lir_bb.append_inst(std::move(imul_inst));

            VReg high = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_h = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_h->add_def(LirOperand::vreg(high, 8));
            mov_h->add_use(LirOperand::vreg(prod, 8));
            lir_bb.append_inst(std::move(mov_h));

            auto shr_inst = std::make_unique<LirInst>(LirOpcode::Shr);
            shr_inst->add_def(LirOperand::vreg(high, 8));
            shr_inst->add_use(LirOperand::vreg(high, 8));
            shr_inst->add_use(LirOperand::imm(32, 1));
            lir_bb.append_inst(std::move(shr_inst));

            auto test_inst = std::make_unique<LirInst>(LirOpcode::Test32);
            test_inst->add_use(LirOperand::vreg(high, 4));
            test_inst->add_use(LirOperand::vreg(high, 4));
            lir_bb.append_inst(std::move(test_inst));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = Condition::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
            return;
        } else {
            VReg prod = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_p = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_p->add_def(LirOperand::vreg(prod, 8));
            mov_p->add_use(LirOperand::vreg(op0, 8));
            lir_bb.append_inst(std::move(mov_p));

            auto imul_inst = std::make_unique<LirInst>(LirOpcode::Imul);
            imul_inst->add_def(LirOperand::vreg(prod, 8));
            imul_inst->add_use(LirOperand::vreg(prod, 8));
            imul_inst->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(imul_inst));

            VReg safe_div = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_sd = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_sd->add_def(LirOperand::vreg(safe_div, 8));
            mov_sd->add_use(LirOperand::vreg(op0, 8));
            lir_bb.append_inst(std::move(mov_sd));

            auto test_z = std::make_unique<LirInst>(LirOpcode::Test);
            test_z->add_use(LirOperand::vreg(op0, 8));
            test_z->add_use(LirOperand::vreg(op0, 8));
            lir_bb.append_inst(std::move(test_z));

            VReg one_vreg = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_one = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_one->add_def(LirOperand::vreg(one_vreg, 8));
            mov_one->add_use(LirOperand::imm(1, 8));
            lir_bb.append_inst(std::move(mov_one));

            auto cmov_z = std::make_unique<LirInst>(LirOpcode::Cmovcc);
            cmov_z->condition = Condition::E;
            cmov_z->add_def(LirOperand::vreg(safe_div, 8));
            cmov_z->add_use(LirOperand::vreg(safe_div, 8));
            cmov_z->add_use(LirOperand::vreg(one_vreg, 8));
            lir_bb.append_inst(std::move(cmov_z));

            auto mov_rax = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_rax->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            mov_rax->add_use(LirOperand::vreg(prod, 8));
            lir_bb.append_inst(std::move(mov_rax));

            auto zero_rdx = std::make_unique<LirInst>(LirOpcode::Xor);
            zero_rdx->add_def(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
            zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
            zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
            lir_bb.append_inst(std::move(zero_rdx));

            auto div_inst = std::make_unique<LirInst>(LirOpcode::Div);
            div_inst->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            div_inst->add_def(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
            div_inst->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            div_inst->add_use(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
            div_inst->add_use(LirOperand::vreg(safe_div, 8));
            lir_bb.append_inst(std::move(div_inst));

            auto cmp_q = std::make_unique<LirInst>(LirOpcode::Cmp);
            cmp_q->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            cmp_q->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(cmp_q));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = Condition::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(movzx));

            VReg zero_reg = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            auto xor_z = std::make_unique<LirInst>(LirOpcode::Xor32);
            xor_z->add_def(LirOperand::vreg(zero_reg, 4));
            xor_z->add_use(LirOperand::vreg(zero_reg, 4));
            xor_z->add_use(LirOperand::vreg(zero_reg, 4));
            lir_bb.append_inst(std::move(xor_z));

            auto test_op0 = std::make_unique<LirInst>(LirOpcode::Test);
            test_op0->add_use(LirOperand::vreg(op0, 8));
            test_op0->add_use(LirOperand::vreg(op0, 8));
            lir_bb.append_inst(std::move(test_op0));

            auto cmov_zero = std::make_unique<LirInst>(LirOpcode::Cmovcc);
            cmov_zero->condition = Condition::E;
            cmov_zero->add_def(LirOperand::vreg(dst, dst_sz));
            cmov_zero->add_use(LirOperand::vreg(dst, dst_sz));
            cmov_zero->add_use(LirOperand::vreg(zero_reg, 4));
            cmov_zero->mir_origin = &inst;
            lir_bb.append_inst(std::move(cmov_zero));
            return;
        }
    }

    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_inst->add_def(LirOperand::vreg(tmp, sz));
    mov_inst->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_inst));

    Condition cond = Condition::O;
    LirOpcode alu_op = (sz == 4) ? LirOpcode::Add32 : LirOpcode::Add;

    switch (op) {
        case Opcode::sadd_overflow:
            alu_op = (sz == 4) ? LirOpcode::Add32 : LirOpcode::Add;
            cond = Condition::O;
            break;
        case Opcode::ssub_overflow:
            alu_op = (sz == 4) ? LirOpcode::Sub32 : LirOpcode::Sub;
            cond = Condition::O;
            break;
        case Opcode::smul_overflow:
            alu_op = (sz == 4) ? LirOpcode::Imul32 : LirOpcode::Imul;
            cond = Condition::O;
            break;
        case Opcode::uadd_overflow:
            alu_op = (sz == 4) ? LirOpcode::Add32 : LirOpcode::Add;
            cond = Condition::B;
            break;
        case Opcode::usub_overflow:
            alu_op = (sz == 4) ? LirOpcode::Sub32 : LirOpcode::Sub;
            cond = Condition::B;
            break;
        default:
            break;
    }

    auto alu_inst = std::make_unique<LirInst>(alu_op);
    alu_inst->add_def(LirOperand::vreg(tmp, sz));
    alu_inst->add_use(LirOperand::vreg(tmp, sz));
    alu_inst->add_use(LirOperand::vreg(op1, sz));
    lir_bb.append_inst(std::move(alu_inst));

    auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
    setcc->condition = cond;
    setcc->add_def(LirOperand::vreg(dst, 1));
    lir_bb.append_inst(std::move(setcc));

    auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
    movzx->add_def(LirOperand::vreg(dst, dst_sz));
    movzx->add_use(LirOperand::vreg(dst, 1));
    movzx->mir_origin = &inst;
    lir_bb.append_inst(std::move(movzx));
}

} // namespace brass::x64
