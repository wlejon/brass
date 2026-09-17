#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstring>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_binary_alu(
    const Instruction& inst,
    LirBlock& lir_bb,
    LirOpcode op32,
    LirOpcode op64,
    LirOpcode op_f64,
    LirOpcode op_f32
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg src0 = get_vreg(op0_val);
    VReg src1 = get_vreg(op1_val);

    if (dst.is_xmm()) {
        uint8_t sz = dst.size;
        LirOpcode mov_op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
        LirOpcode alu_op = (sz == 4) ? op_f32 : op_f64;

        auto emit_float_alu = [&](VReg first_src, LirOperand second_src) {
            if (dst != first_src) {
                auto mov_inst = std::make_unique<LirInst>(mov_op);
                mov_inst->add_def(LirOperand::vreg(dst, sz));
                mov_inst->add_use(LirOperand::vreg(first_src, sz));
                lir_bb.append_inst(std::move(mov_inst));
            }

            auto alu_inst = std::make_unique<LirInst>(alu_op);
            alu_inst->add_def(LirOperand::vreg(dst, sz));
            alu_inst->add_use(LirOperand::vreg(dst, sz));
            alu_inst->add_use(second_src);
            alu_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(alu_inst));
        };

        bool is_comm = (inst.opcode() == Opcode::add || inst.opcode() == Opcode::mul || inst.opcode() == Opcode::xor_);
        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_float_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (is_comm && op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_float_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (is_comm && dst == src1) {
            emit_float_alu(src1, LirOperand::vreg(src0, sz));
        } else {
            emit_float_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    uint8_t sz = dst.size;
    ImmIntInfo imm0 = get_imm_int_info(op0_val);
    ImmIntInfo imm1 = get_imm_int_info(op1_val);

    auto emit_mov_alu = [&](VReg first_src, LirOperand second_src) {
        if (dst != first_src) {
            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_inst->add_def(LirOperand::vreg(dst, sz));
            mov_inst->add_use(LirOperand::vreg(first_src, sz));
            lir_bb.append_inst(std::move(mov_inst));
        }

        auto alu_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
        alu_inst->add_def(LirOperand::vreg(dst, sz));
        alu_inst->add_use(LirOperand::vreg(dst, sz));
        alu_inst->add_use(second_src);
        alu_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(alu_inst));
    };

    if (inst.opcode() == Opcode::mul) {
        const Value* reg_val = imm1.is_imm ? op0_val : (imm0.is_imm ? op1_val : nullptr);
        ImmIntInfo imm_info = imm1.is_imm ? imm1 : imm0;

        if (reg_val != nullptr) {
            int64_t C = imm_info.val;
            VReg r_vreg = get_vreg(reg_val);

            if (C == 0) {
                auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                lir_inst->add_def(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
                return;
            }
            if (C == 1) {
                if (dst != r_vreg) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(r_vreg, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            if (C > 0 && (static_cast<uint64_t>(C) & static_cast<uint64_t>(C - 1)) == 0) {
                int k = 0;
                while ((1ULL << k) < static_cast<uint64_t>(C)) k++;
                if (dst != r_vreg) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(r_vreg, sz));
                    lir_bb.append_inst(std::move(mov_inst));
                }

                auto shl_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shl32 : LirOpcode::Shl);
                shl_inst->add_def(LirOperand::vreg(dst, sz));
                shl_inst->add_use(LirOperand::vreg(dst, sz));
                shl_inst->add_use(LirOperand::imm(k, 1));
                shl_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(shl_inst));
                return;
            }
            if (imm_info.fits_i32) {
                emit_mov_alu(r_vreg, LirOperand::imm(C, sz));
                return;
            }
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (dst == src1) {
            emit_mov_alu(src1, LirOperand::vreg(src0, sz));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    if (inst.opcode() == Opcode::add) {
        if (imm1.is_imm && imm1.fits_i32) {
            if (imm1.val == 0) {
                if (dst != src0) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(src0, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            emit_mov_alu(src0, LirOperand::imm(imm1.val, sz));
            return;
        }
        if (imm0.is_imm && imm0.fits_i32) {
            if (imm0.val == 0) {
                if (dst != src1) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(src1, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            emit_mov_alu(src1, LirOperand::imm(imm0.val, sz));
            return;
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (dst == src1) {
            emit_mov_alu(src1, LirOperand::vreg(src0, sz));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    if (inst.opcode() == Opcode::sub) {
        if (imm1.is_imm && imm1.fits_i32) {
            if (imm1.val == 0) {
                if (dst != src0) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(src0, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            emit_mov_alu(src0, LirOperand::imm(imm1.val, sz));
            return;
        }
        if (imm0.is_imm && imm0.fits_i32) {
            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_inst->add_def(LirOperand::vreg(dst, sz));
            mov_inst->add_use(LirOperand::imm(imm0.val, sz));
            lir_bb.append_inst(std::move(mov_inst));

            auto sub_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
            sub_inst->add_def(LirOperand::vreg(dst, sz));
            sub_inst->add_use(LirOperand::vreg(dst, sz));
            sub_inst->add_use(LirOperand::vreg(src1, sz));
            sub_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(sub_inst));
            return;
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    // Bitwise: and, or, xor
    if (imm1.is_imm && imm1.fits_i32) {
        emit_mov_alu(src0, LirOperand::imm(imm1.val, sz));
    } else if (imm0.is_imm && imm0.fits_i32) {
        emit_mov_alu(src1, LirOperand::imm(imm0.val, sz));
    } else if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
        emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
    } else if (op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
        emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
    } else if (dst == src1) {
        emit_mov_alu(src1, LirOperand::vreg(src0, sz));
    } else {
        emit_mov_alu(src0, LirOperand::vreg(src1, sz));
    }
}

void AArch64ISel::lower_shift(
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

    // ARM64 shifts are standard 3-register operations: no fixed RCX constraint!
    auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_dst->add_def(LirOperand::vreg(dst, sz));
    mov_dst->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_dst));

    auto shift_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
    shift_inst->add_def(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::vreg(op1, op1.size));
    shift_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(shift_inst));
}

void AArch64ISel::lower_div_mod(
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

    ImmIntInfo imm1 = get_imm_int_info(op1_val);

    // Check for power-of-2 unsigned optimizations
    if (!is_signed && imm1.is_imm && imm1.val > 0) {
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

    VReg op1 = get_vreg(op1_val);
    LirOpcode div_op = is_signed ? (sz == 4 ? LirOpcode::Idiv32 : LirOpcode::Idiv)
                                 : (sz == 4 ? LirOpcode::Div32 : LirOpcode::Div);

    if (!is_mod) {
        // Pure 3-register division: Idiv / Div dst, op0, op1 (no fixed RAX/RDX constraints!)
        auto div_inst = std::make_unique<LirInst>(div_op);
        div_inst->add_def(LirOperand::vreg(dst, sz));
        div_inst->add_use(LirOperand::vreg(op0, sz));
        div_inst->add_use(LirOperand::vreg(op1, sz));
        div_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(div_inst));
        return;
    }

    // Modulo (%): lowered as q = div a, b; rem = sub a, (mul q, b)
    VReg q = lir_fn_->allocate_vreg(RegClass::GPR, sz);
    auto div_inst = std::make_unique<LirInst>(div_op);
    div_inst->add_def(LirOperand::vreg(q, sz));
    div_inst->add_use(LirOperand::vreg(op0, sz));
    div_inst->add_use(LirOperand::vreg(op1, sz));
    lir_bb.append_inst(std::move(div_inst));

    VReg prod = lir_fn_->allocate_vreg(RegClass::GPR, sz);
    auto mul_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Imul32 : LirOpcode::Imul);
    mul_inst->add_def(LirOperand::vreg(prod, sz));
    mul_inst->add_use(LirOperand::vreg(q, sz));
    mul_inst->add_use(LirOperand::vreg(op1, sz));
    lir_bb.append_inst(std::move(mul_inst));

    if (dst != op0) {
        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_inst->add_def(LirOperand::vreg(dst, sz));
        mov_inst->add_use(LirOperand::vreg(op0, sz));
        lir_bb.append_inst(std::move(mov_inst));
    }

    auto sub_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
    sub_inst->add_def(LirOperand::vreg(dst, sz));
    sub_inst->add_use(LirOperand::vreg(dst, sz));
    sub_inst->add_use(LirOperand::vreg(prod, sz));
    sub_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sub_inst));
}

} // namespace brass::aarch64
