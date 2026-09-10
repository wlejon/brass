#include <brass/target/x64/x64_isel.hpp>

namespace brass::x64 {

using namespace brass::codegen;

static constexpr Condition swap_relational_condition(Condition cond) noexcept {
    switch (cond) {
        case Condition::E:   return Condition::E;
        case Condition::NE:  return Condition::NE;
        case Condition::L:   return Condition::G;
        case Condition::LE:  return Condition::GE;
        case Condition::G:   return Condition::L;
        case Condition::GE:  return Condition::LE;
        case Condition::B:   return Condition::A;
        case Condition::BE:  return Condition::AE;
        case Condition::A:   return Condition::B;
        case Condition::AE:  return Condition::BE;
        default: return cond;
    }
}

void X64ISel::lower_comparison(
    const Instruction& inst,
    LirBlock& lir_bb,
    Condition cond,
    Condition float_cond
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    VReg op1 = get_vreg(op1_val);
    uint8_t dst_sz = dst.size;

    if (op0.is_xmm()) {
        uint8_t sz = op0.size;
        LirOpcode ucomi_op = (sz == 4) ? LirOpcode::Ucomiss : LirOpcode::Ucomisd;

        auto ucomi = std::make_unique<LirInst>(ucomi_op);
        ucomi->add_use(LirOperand::vreg(op0, sz));
        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            ucomi->add_use(get_load_mem_operand(op1_val->defining_instruction()));
        } else {
            ucomi->add_use(LirOperand::vreg(op1, sz));
        }
        lir_bb.append_inst(std::move(ucomi));

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = float_cond;
        setcc->add_def(LirOperand::vreg(dst, 1));
        lir_bb.append_inst(std::move(setcc));

        auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
        movzx->add_def(LirOperand::vreg(dst, dst_sz));
        movzx->add_use(LirOperand::vreg(dst, 1));
        movzx->mir_origin = &inst;
        lir_bb.append_inst(std::move(movzx));
    } else {
        uint8_t sz = static_cast<uint8_t>(op0_val->type().size_in_bytes());
        if (sz == 0) sz = 8;
        Condition final_cond = cond;

        ImmIntInfo imm1 = get_imm_int_info(op1_val);
        ImmIntInfo imm0 = get_imm_int_info(op0_val);

        if (imm1.is_imm && imm1.val == 0) {
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op0, sz));
            test_inst->add_use(LirOperand::vreg(op0, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm0.is_imm && imm0.val == 0) {
            final_cond = swap_relational_condition(cond);
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op1, sz));
            test_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm1.is_imm && imm1.fits_i32) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::imm(imm1.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (imm0.is_imm && imm0.fits_i32) {
            final_cond = swap_relational_condition(cond);
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            cmp_inst->add_use(LirOperand::imm(imm0.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(get_load_mem_operand(op1_val->defining_instruction()));
            lir_bb.append_inst(std::move(cmp_inst));
        } else {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        }

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = final_cond;
        setcc->add_def(LirOperand::vreg(dst, 1));
        lir_bb.append_inst(std::move(setcc));

        auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
        movzx->add_def(LirOperand::vreg(dst, dst_sz));
        movzx->add_use(LirOperand::vreg(dst, 1));
        movzx->mir_origin = &inst;
        lir_bb.append_inst(std::move(movzx));
    }
}

} // namespace brass::x64
