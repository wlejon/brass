#include <brass/target/x64/x64_isel.hpp>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_div_mod(
    const Instruction& inst,
    LirBlock& lir_bb,
    bool is_signed,
    bool is_mod
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    uint8_t sz = dst.size;

    if (!is_signed) {
        ImmIntInfo imm1 = get_imm_int_info(op1_val);
        if (imm1.is_imm && imm1.val > 0) {
            uint64_t C = static_cast<uint64_t>(imm1.val);
            if ((C & (C - 1)) == 0) {
                int k = 0;
                while ((1ULL << k) < C) k++;
                if (!is_mod) {
                    if (k == 0) {
                        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_inst->add_def(LirOperand::vreg(dst, sz));
                        mov_inst->add_use(LirOperand::vreg(op0, sz));
                        mov_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(mov_inst));
                    } else {
                        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_inst->add_def(LirOperand::vreg(dst, sz));
                        mov_inst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_inst));

                        auto shr_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_inst->add_def(LirOperand::vreg(dst, sz));
                        shr_inst->add_use(LirOperand::vreg(dst, sz));
                        shr_inst->add_use(LirOperand::imm(k, 1));
                        shr_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(shr_inst));
                    }
                    return;
                } else {
                    if (k == 0) {
                        auto zero_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                        zero_inst->add_def(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(zero_inst));
                        return;
                    } else {
                        uint64_t mask = C - 1;
                        if (mask <= INT32_MAX) {
                            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                            mov_inst->add_def(LirOperand::vreg(dst, sz));
                            mov_inst->add_use(LirOperand::vreg(op0, sz));
                            lir_bb.append_inst(std::move(mov_inst));

                            auto and_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::And32 : LirOpcode::And);
                            and_inst->add_def(LirOperand::vreg(dst, sz));
                            and_inst->add_use(LirOperand::vreg(dst, sz));
                            and_inst->add_use(LirOperand::imm(static_cast<int32_t>(mask), sz));
                            and_inst->mir_origin = &inst;
                            lir_bb.append_inst(std::move(and_inst));
                            return;
                        }
                    }
                }
            }
        }
    }

    VReg op1 = get_vreg(op1_val);

    // 1. Move op0 into RAX
    auto mov_rax = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_rax->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    mov_rax->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_rax));

    // 2. Sign-extend RAX into RDX:RAX via cdq/cqo, or zero RDX
    if (is_signed) {
        auto extend_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Cdq : LirOpcode::Cqo);
        extend_inst->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        extend_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
        lir_bb.append_inst(std::move(extend_inst));
    } else {
        auto zero_rdx = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
        zero_rdx->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        lir_bb.append_inst(std::move(zero_rdx));
    }

    // 3. Emit idiv or div
    LirOpcode div_op;
    if (is_signed) {
        div_op = sz == 4 ? LirOpcode::Idiv32 : LirOpcode::Idiv;
    } else {
        div_op = sz == 4 ? LirOpcode::Div32 : LirOpcode::Div;
    }

    auto div_inst = std::make_unique<LirInst>(div_op);
    div_inst->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    div_inst->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
    div_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    div_inst->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
    div_inst->add_use(LirOperand::vreg(op1, sz));
    div_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(div_inst));

    // 4. Result is in RAX (div) or RDX (mod)
    GPR res_reg = is_mod ? GPR::RDX : GPR::RAX;
    auto mov_res = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_res->add_def(LirOperand::vreg(dst, sz));
    mov_res->add_use(LirOperand::preg_gpr(res_reg, sz), FixedConstraint::gpr(res_reg));
    lir_bb.append_inst(std::move(mov_res));
}

void X64ISel::lower_shift(
    const Instruction& inst,
    LirBlock& lir_bb,
    LirOpcode op32,
    LirOpcode op64
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    uint8_t sz = dst.size;

    ImmIntInfo imm1 = get_imm_int_info(op1_val);
    if (imm1.is_imm) {
        uint8_t shift_amt = static_cast<uint8_t>(imm1.val & (sz == 4 ? 31 : 63));
        if (shift_amt == 0) {
            auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_dst->add_def(LirOperand::vreg(dst, sz));
            mov_dst->add_use(LirOperand::vreg(op0, sz));
            mov_dst->mir_origin = &inst;
            lir_bb.append_inst(std::move(mov_dst));
            return;
        }

        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_dst->add_def(LirOperand::vreg(dst, sz));
        mov_dst->add_use(LirOperand::vreg(op0, sz));
        lir_bb.append_inst(std::move(mov_dst));

        auto shift_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
        shift_inst->add_def(LirOperand::vreg(dst, sz));
        shift_inst->add_use(LirOperand::vreg(dst, sz));
        shift_inst->add_use(LirOperand::imm(shift_amt, 1));
        shift_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(shift_inst));
        return;
    }

    VReg op1 = get_vreg(op1_val);
    // 1. Copy op0 to dst
    auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_dst->add_def(LirOperand::vreg(dst, sz));
    mov_dst->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_dst));

    // 2. x86 requires shift amount in CL (RCX)
    auto mov_cl = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_cl->add_def(LirOperand::preg_gpr(GPR::RCX, 8), FixedConstraint::gpr(GPR::RCX));
    mov_cl->add_use(LirOperand::vreg(op1, op1.size));
    lir_bb.append_inst(std::move(mov_cl));

    // 3. Shift dst by CL
    auto shift_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
    shift_inst->add_def(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::preg_gpr(GPR::RCX, 1), FixedConstraint::gpr(GPR::RCX));
    shift_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(shift_inst));
}

} // namespace brass::x64
