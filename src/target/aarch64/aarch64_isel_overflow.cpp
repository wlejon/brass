#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;
using LirCond = brass::x64::Condition;

void AArch64ISel::lower_overflow_check(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    VReg op1 = get_vreg(op1_val);
    uint8_t sz = op0.size;
    uint8_t dst_sz = dst.size;

    Opcode op = inst.opcode();

    if (op == Opcode::smul_overflow) {
        if (sz == 4) {
            VReg t0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            VReg t1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto sxt0 = std::make_unique<LirInst>(LirOpcode::Movsxd);
            sxt0->add_def(LirOperand::vreg(t0, 8));
            sxt0->add_use(LirOperand::vreg(op0, 4));
            lir_bb.append_inst(std::move(sxt0));

            auto sxt1 = std::make_unique<LirInst>(LirOpcode::Movsxd);
            sxt1->add_def(LirOperand::vreg(t1, 8));
            sxt1->add_use(LirOperand::vreg(op1, 4));
            lir_bb.append_inst(std::move(sxt1));

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

            VReg ext = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto sxt_p = std::make_unique<LirInst>(LirOpcode::Movsxd);
            sxt_p->add_def(LirOperand::vreg(ext, 8));
            sxt_p->add_use(LirOperand::vreg(prod, 4));
            lir_bb.append_inst(std::move(sxt_p));

            auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp);
            cmp_inst->add_use(LirOperand::vreg(prod, 8));
            cmp_inst->add_use(LirOperand::vreg(ext, 8));
            lir_bb.append_inst(std::move(cmp_inst));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = LirCond::NE;
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

            auto mul_inst = std::make_unique<LirInst>(LirOpcode::Imul);
            mul_inst->add_def(LirOperand::vreg(prod, 8));
            mul_inst->add_use(LirOperand::vreg(prod, 8));
            mul_inst->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(mul_inst));

            VReg high = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto smulh_inst = std::make_unique<LirInst>(LirOpcode::Smulh);
            smulh_inst->add_def(LirOperand::vreg(high, 8));
            smulh_inst->add_use(LirOperand::vreg(op0, 8));
            smulh_inst->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(smulh_inst));

            VReg sign = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto mov_s = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_s->add_def(LirOperand::vreg(sign, 8));
            mov_s->add_use(LirOperand::vreg(prod, 8));
            lir_bb.append_inst(std::move(mov_s));

            auto sar_inst = std::make_unique<LirInst>(LirOpcode::Sar);
            sar_inst->add_def(LirOperand::vreg(sign, 8));
            sar_inst->add_use(LirOperand::vreg(sign, 8));
            sar_inst->add_use(LirOperand::imm(63, 1));
            lir_bb.append_inst(std::move(sar_inst));

            auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp);
            cmp_inst->add_use(LirOperand::vreg(high, 8));
            cmp_inst->add_use(LirOperand::vreg(sign, 8));
            lir_bb.append_inst(std::move(cmp_inst));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = LirCond::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
            return;
        }
    }

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
            setcc->condition = LirCond::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
            return;
        } else {
            VReg high = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            auto umulh_inst = std::make_unique<LirInst>(LirOpcode::Umulh);
            umulh_inst->add_def(LirOperand::vreg(high, 8));
            umulh_inst->add_use(LirOperand::vreg(op0, 8));
            umulh_inst->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(umulh_inst));

            auto test_inst = std::make_unique<LirInst>(LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(high, 8));
            test_inst->add_use(LirOperand::vreg(high, 8));
            lir_bb.append_inst(std::move(test_inst));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = LirCond::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
            return;
        }
    }

    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_inst->add_def(LirOperand::vreg(tmp, sz));
    mov_inst->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_inst));

    LirCond cond = LirCond::O;
    LirOpcode alu_op = (sz == 4) ? LirOpcode::Adds32 : LirOpcode::Adds;

    switch (op) {
        case Opcode::sadd_overflow:
            alu_op = (sz == 4) ? LirOpcode::Adds32 : LirOpcode::Adds;
            cond = LirCond::O;
            break;
        case Opcode::ssub_overflow:
            alu_op = (sz == 4) ? LirOpcode::Subs32 : LirOpcode::Subs;
            cond = LirCond::O;
            break;
        case Opcode::uadd_overflow:
            alu_op = (sz == 4) ? LirOpcode::Adds32 : LirOpcode::Adds;
            cond = LirCond::AE; // Maps to CS (Carry Set) in to_aarch64_cond
            break;
        case Opcode::usub_overflow:
            alu_op = (sz == 4) ? LirOpcode::Subs32 : LirOpcode::Subs;
            cond = LirCond::B; // Maps to CC (Carry Clear / Borrow) in to_aarch64_cond
            break;
        default:
            codegen::throw_unsupported("aarch64 isel (overflow)", opcode_name(op));
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

} // namespace brass::aarch64
