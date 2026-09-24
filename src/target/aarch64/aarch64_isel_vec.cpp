#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_vector_instruction(const Instruction& inst, LirBlock& lir_bb) {
    auto emit_movaps = [&](VReg dst, VReg src) {
        if (dst != src) {
            auto mov = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov->add_def(LirOperand::vreg(dst, 16));
            mov->add_use(LirOperand::vreg(src, 16));
            lir_bb.append_inst(std::move(mov));
        }
    };

    switch (inst.opcode()) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vmin:
        case Opcode::vmax: {
            Type t = inst.type();

            auto emit_mul_i64x2 = [&](VReg d, VReg a, VReg b) {
                VReg r0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg r1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg s0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg s1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);

                auto pext0 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext0->add_def(LirOperand::vreg(r0, 8));
                pext0->add_use(LirOperand::vreg(a, 16));
                pext0->add_use(LirOperand::imm(0, 1));
                lir_bb.append_inst(std::move(pext0));

                auto pext1 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext1->add_def(LirOperand::vreg(r1, 8));
                pext1->add_use(LirOperand::vreg(b, 16));
                pext1->add_use(LirOperand::imm(0, 1));
                lir_bb.append_inst(std::move(pext1));

                auto mul0 = std::make_unique<LirInst>(LirOpcode::Imul);
                mul0->add_def(LirOperand::vreg(r0, 8));
                mul0->add_use(LirOperand::vreg(r0, 8));
                mul0->add_use(LirOperand::vreg(r1, 8));
                lir_bb.append_inst(std::move(mul0));

                auto movq0 = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                movq0->add_def(LirOperand::vreg(d, 16));
                movq0->add_use(LirOperand::vreg(r0, 8));
                lir_bb.append_inst(std::move(movq0));

                auto pext2 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext2->add_def(LirOperand::vreg(s0, 8));
                pext2->add_use(LirOperand::vreg(a, 16));
                pext2->add_use(LirOperand::imm(1, 1));
                lir_bb.append_inst(std::move(pext2));

                auto pext3 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext3->add_def(LirOperand::vreg(s1, 8));
                pext3->add_use(LirOperand::vreg(b, 16));
                pext3->add_use(LirOperand::imm(1, 1));
                lir_bb.append_inst(std::move(pext3));

                auto mul1 = std::make_unique<LirInst>(LirOpcode::Imul);
                mul1->add_def(LirOperand::vreg(s0, 8));
                mul1->add_use(LirOperand::vreg(s0, 8));
                mul1->add_use(LirOperand::vreg(s1, 8));
                lir_bb.append_inst(std::move(mul1));

                auto pinsr1 = std::make_unique<LirInst>(LirOpcode::Pinsrq);
                pinsr1->add_def(LirOperand::vreg(d, 16));
                pinsr1->add_use(LirOperand::vreg(d, 16));
                pinsr1->add_use(LirOperand::vreg(s0, 8));
                pinsr1->add_use(LirOperand::imm(1, 1));
                pinsr1->mir_origin = &inst;
                lir_bb.append_inst(std::move(pinsr1));
            };

            if (inst.opcode() == Opcode::vmul && t.kind() == TypeKind::I64x2) {
                VReg dst = get_vreg(inst.result());
                VReg v0 = get_vreg(inst.operand(0));
                VReg v1 = get_vreg(inst.operand(1));
                emit_mul_i64x2(dst, v0, v1);
                break;
            }

            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v0 = get_vreg_pair(inst.operand(0));
                VRegPair v1 = get_vreg_pair(inst.operand(1));

                if (inst.opcode() == Opcode::vmul && t.kind() == TypeKind::I64x4) {
                    emit_mul_i64x2(dst.lo, v0.lo, v1.lo);
                    emit_mul_i64x2(dst.hi, v0.hi, v1.hi);
                    break;
                }

                LirOpcode half_op = LirOpcode::Nop;
                if (inst.opcode() == Opcode::vadd) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Addps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Addpd;
                    else if (t.kind() == TypeKind::I32x8) half_op = LirOpcode::Paddd;
                    else if (t.kind() == TypeKind::I64x4) half_op = LirOpcode::Paddq;
                } else if (inst.opcode() == Opcode::vsub) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Subps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Subpd;
                    else if (t.kind() == TypeKind::I32x8) half_op = LirOpcode::Psubd;
                    else if (t.kind() == TypeKind::I64x4) half_op = LirOpcode::Psubq;
                } else if (inst.opcode() == Opcode::vmul) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Mulps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Mulpd;
                    else if (t.kind() == TypeKind::I32x8) half_op = LirOpcode::Pmulld;
                } else if (inst.opcode() == Opcode::vdiv) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Divps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Divpd;
                } else if (inst.opcode() == Opcode::vmin) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Minps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Minpd;
                    else if (t.kind() == TypeKind::I32x8) half_op = LirOpcode::Pminsd;
                } else if (inst.opcode() == Opcode::vmax) {
                    if (t.kind() == TypeKind::F32x8) half_op = LirOpcode::Maxps;
                    else if (t.kind() == TypeKind::F64x4) half_op = LirOpcode::Maxpd;
                    else if (t.kind() == TypeKind::I32x8) half_op = LirOpcode::Pmaxsd;
                }

                if (half_op != LirOpcode::Nop) {
                    emit_movaps(dst.lo, v0.lo);
                    auto binop_lo = std::make_unique<LirInst>(half_op);
                    binop_lo->add_def(LirOperand::vreg(dst.lo, 16));
                    binop_lo->add_use(LirOperand::vreg(dst.lo, 16));
                    binop_lo->add_use(LirOperand::vreg(v1.lo, 16));
                    binop_lo->mir_origin = &inst;
                    lir_bb.append_inst(std::move(binop_lo));

                    emit_movaps(dst.hi, v0.hi);
                    auto binop_hi = std::make_unique<LirInst>(half_op);
                    binop_hi->add_def(LirOperand::vreg(dst.hi, 16));
                    binop_hi->add_use(LirOperand::vreg(dst.hi, 16));
                    binop_hi->add_use(LirOperand::vreg(v1.hi, 16));
                    binop_hi->mir_origin = &inst;
                    lir_bb.append_inst(std::move(binop_hi));
                    break;
                }
            }

            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            VReg v1 = get_vreg(inst.operand(1));

            LirOpcode op = LirOpcode::Nop;
            if (inst.opcode() == Opcode::vadd) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Addps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Addpd;
                else if (t.kind() == TypeKind::I32x4) op = LirOpcode::Paddd;
                else if (t.kind() == TypeKind::I64x2) op = LirOpcode::Paddq;
            } else if (inst.opcode() == Opcode::vsub) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Subps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Subpd;
                else if (t.kind() == TypeKind::I32x4) op = LirOpcode::Psubd;
                else if (t.kind() == TypeKind::I64x2) op = LirOpcode::Psubq;
            } else if (inst.opcode() == Opcode::vmul) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Mulps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Mulpd;
                else if (t.kind() == TypeKind::I32x4) op = LirOpcode::Pmulld;
            } else if (inst.opcode() == Opcode::vdiv) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Divps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Divpd;
            } else if (inst.opcode() == Opcode::vmin) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Minps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Minpd;
                else if (t.kind() == TypeKind::I32x4) op = LirOpcode::Pminsd;
            } else if (inst.opcode() == Opcode::vmax) {
                if (t.kind() == TypeKind::F32x4) op = LirOpcode::Maxps;
                else if (t.kind() == TypeKind::F64x2) op = LirOpcode::Maxpd;
                else if (t.kind() == TypeKind::I32x4) op = LirOpcode::Pmaxsd;
            }

            emit_movaps(dst, v0);

            auto binop = std::make_unique<LirInst>(op);
            binop->add_def(LirOperand::vreg(dst, 16));
            binop->add_use(LirOperand::vreg(dst, 16));
            binop->add_use(LirOperand::vreg(v1, 16));
            binop->mir_origin = &inst;
            lir_bb.append_inst(std::move(binop));
            break;
        }

        case Opcode::vsqrt: {
            Type t = inst.type();
            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v0 = get_vreg_pair(inst.operand(0));
                LirOpcode op = (t.kind() == TypeKind::F64x4) ? LirOpcode::Sqrtpd : LirOpcode::Sqrtps;

                auto s_lo = std::make_unique<LirInst>(op);
                s_lo->add_def(LirOperand::vreg(dst.lo, 16));
                s_lo->add_use(LirOperand::vreg(v0.lo, 16));
                s_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(s_lo));

                auto s_hi = std::make_unique<LirInst>(op);
                s_hi->add_def(LirOperand::vreg(dst.hi, 16));
                s_hi->add_use(LirOperand::vreg(v0.hi, 16));
                s_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(s_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            LirOpcode op = (t.kind() == TypeKind::F64x2) ? LirOpcode::Sqrtpd : LirOpcode::Sqrtps;

            auto s = std::make_unique<LirInst>(op);
            s->add_def(LirOperand::vreg(dst, 16));
            s->add_use(LirOperand::vreg(v0, 16));
            s->mir_origin = &inst;
            lir_bb.append_inst(std::move(s));
            break;
        }

        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor: {
            if (inst.type().is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v0 = get_vreg_pair(inst.operand(0));
                VRegPair v1 = get_vreg_pair(inst.operand(1));
                LirOpcode op = LirOpcode::Pand;
                if (inst.opcode() == Opcode::vor) op = LirOpcode::Por;
                else if (inst.opcode() == Opcode::vxor) op = LirOpcode::Pxor;

                emit_movaps(dst.lo, v0.lo);
                auto binop_lo = std::make_unique<LirInst>(op);
                binop_lo->add_def(LirOperand::vreg(dst.lo, 16));
                binop_lo->add_use(LirOperand::vreg(dst.lo, 16));
                binop_lo->add_use(LirOperand::vreg(v1.lo, 16));
                binop_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(binop_lo));

                emit_movaps(dst.hi, v0.hi);
                auto binop_hi = std::make_unique<LirInst>(op);
                binop_hi->add_def(LirOperand::vreg(dst.hi, 16));
                binop_hi->add_use(LirOperand::vreg(dst.hi, 16));
                binop_hi->add_use(LirOperand::vreg(v1.hi, 16));
                binop_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(binop_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            VReg v1 = get_vreg(inst.operand(1));

            LirOpcode op = LirOpcode::Pand;
            if (inst.opcode() == Opcode::vor) op = LirOpcode::Por;
            else if (inst.opcode() == Opcode::vxor) op = LirOpcode::Pxor;

            emit_movaps(dst, v0);

            auto binop = std::make_unique<LirInst>(op);
            binop->add_def(LirOperand::vreg(dst, 16));
            binop->add_use(LirOperand::vreg(dst, 16));
            binop->add_use(LirOperand::vreg(v1, 16));
            binop->mir_origin = &inst;
            lir_bb.append_inst(std::move(binop));
            break;
        }

        case Opcode::vnot: {
            if (inst.type().is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v0 = get_vreg_pair(inst.operand(0));

                auto vnot_lo = std::make_unique<LirInst>(LirOpcode::Pnot);
                vnot_lo->add_def(LirOperand::vreg(dst.lo, 16));
                vnot_lo->add_use(LirOperand::vreg(v0.lo, 16));
                vnot_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(vnot_lo));

                auto vnot_hi = std::make_unique<LirInst>(LirOpcode::Pnot);
                vnot_hi->add_def(LirOperand::vreg(dst.hi, 16));
                vnot_hi->add_use(LirOperand::vreg(v0.hi, 16));
                vnot_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(vnot_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            auto vnot = std::make_unique<LirInst>(LirOpcode::Pnot);
            vnot->add_def(LirOperand::vreg(dst, 16));
            vnot->add_use(LirOperand::vreg(v0, 16));
            vnot->mir_origin = &inst;
            lir_bb.append_inst(std::move(vnot));
            break;
        }

        case Opcode::vneg: {
            Type t = inst.type();
            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v0 = get_vreg_pair(inst.operand(0));

                if (t.kind() == TypeKind::F32x8) {
                    auto neg_lo = std::make_unique<LirInst>(LirOpcode::Fneg4s);
                    neg_lo->add_def(LirOperand::vreg(dst.lo, 16));
                    neg_lo->add_use(LirOperand::vreg(v0.lo, 16));
                    neg_lo->mir_origin = &inst;
                    lir_bb.append_inst(std::move(neg_lo));

                    auto neg_hi = std::make_unique<LirInst>(LirOpcode::Fneg4s);
                    neg_hi->add_def(LirOperand::vreg(dst.hi, 16));
                    neg_hi->add_use(LirOperand::vreg(v0.hi, 16));
                    neg_hi->mir_origin = &inst;
                    lir_bb.append_inst(std::move(neg_hi));
                } else if (t.kind() == TypeKind::F64x4) {
                    auto neg_lo = std::make_unique<LirInst>(LirOpcode::Fneg2d);
                    neg_lo->add_def(LirOperand::vreg(dst.lo, 16));
                    neg_lo->add_use(LirOperand::vreg(v0.lo, 16));
                    neg_lo->mir_origin = &inst;
                    lir_bb.append_inst(std::move(neg_lo));

                    auto neg_hi = std::make_unique<LirInst>(LirOpcode::Fneg2d);
                    neg_hi->add_def(LirOperand::vreg(dst.hi, 16));
                    neg_hi->add_use(LirOperand::vreg(v0.hi, 16));
                    neg_hi->mir_origin = &inst;
                    lir_bb.append_inst(std::move(neg_hi));
                } else {
                    LirOpcode sub_op = (t.kind() == TypeKind::I32x8) ? LirOpcode::Psubd : LirOpcode::Psubq;

                    auto zero_lo = std::make_unique<LirInst>(LirOpcode::Xorps);
                    zero_lo->add_def(LirOperand::vreg(dst.lo, 16));
                    zero_lo->add_use(LirOperand::vreg(dst.lo, 16));
                    zero_lo->add_use(LirOperand::vreg(dst.lo, 16));
                    lir_bb.append_inst(std::move(zero_lo));

                    auto sub_lo = std::make_unique<LirInst>(sub_op);
                    sub_lo->add_def(LirOperand::vreg(dst.lo, 16));
                    sub_lo->add_use(LirOperand::vreg(dst.lo, 16));
                    sub_lo->add_use(LirOperand::vreg(v0.lo, 16));
                    sub_lo->mir_origin = &inst;
                    lir_bb.append_inst(std::move(sub_lo));

                    auto zero_hi = std::make_unique<LirInst>(LirOpcode::Xorps);
                    zero_hi->add_def(LirOperand::vreg(dst.hi, 16));
                    zero_hi->add_use(LirOperand::vreg(dst.hi, 16));
                    zero_hi->add_use(LirOperand::vreg(dst.hi, 16));
                    lir_bb.append_inst(std::move(zero_hi));

                    auto sub_hi = std::make_unique<LirInst>(sub_op);
                    sub_hi->add_def(LirOperand::vreg(dst.hi, 16));
                    sub_hi->add_use(LirOperand::vreg(dst.hi, 16));
                    sub_hi->add_use(LirOperand::vreg(v0.hi, 16));
                    sub_hi->mir_origin = &inst;
                    lir_bb.append_inst(std::move(sub_hi));
                }
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));

            if (t.kind() == TypeKind::F32x4) {
                auto neg = std::make_unique<LirInst>(LirOpcode::Fneg4s);
                neg->add_def(LirOperand::vreg(dst, 16));
                neg->add_use(LirOperand::vreg(v0, 16));
                neg->mir_origin = &inst;
                lir_bb.append_inst(std::move(neg));
            } else if (t.kind() == TypeKind::F64x2) {
                auto neg = std::make_unique<LirInst>(LirOpcode::Fneg2d);
                neg->add_def(LirOperand::vreg(dst, 16));
                neg->add_use(LirOperand::vreg(v0, 16));
                neg->mir_origin = &inst;
                lir_bb.append_inst(std::move(neg));
            } else {
                auto zero = std::make_unique<LirInst>(LirOpcode::Xorps);
                zero->add_def(LirOperand::vreg(dst, 16));
                zero->add_use(LirOperand::vreg(dst, 16));
                zero->add_use(LirOperand::vreg(dst, 16));
                lir_bb.append_inst(std::move(zero));

                LirOpcode sub_op = (t.kind() == TypeKind::I32x4) ? LirOpcode::Psubd : LirOpcode::Psubq;
                auto sub = std::make_unique<LirInst>(sub_op);
                sub->add_def(LirOperand::vreg(dst, 16));
                sub->add_use(LirOperand::vreg(dst, 16));
                sub->add_use(LirOperand::vreg(v0, 16));
                sub->mir_origin = &inst;
                lir_bb.append_inst(std::move(sub));
            }
            break;
        }

        case Opcode::vload: {
            Type t = inst.type();
            VReg base = get_vreg(inst.operand(0));
            int32_t offset = inst.offset();

            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());

                auto load_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                load_lo->add_def(LirOperand::vreg(dst.lo, 16));
                load_lo->add_use(LirOperand::mem(base, offset, 16));
                load_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(load_lo));

                auto load_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                load_hi->add_def(LirOperand::vreg(dst.hi, 16));
                load_hi->add_use(LirOperand::mem(base, offset + 16, 16));
                load_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(load_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
            auto load = std::make_unique<LirInst>(LirOpcode::Movups);
            load->add_def(LirOperand::vreg(dst, sz));
            load->add_use(LirOperand::mem(base, offset, sz));
            load->mir_origin = &inst;
            lir_bb.append_inst(std::move(load));
            break;
        }

        case Opcode::vstore: {
            Type t = inst.memory_type();
            VReg base = get_vreg(inst.operand(0));
            int32_t offset = inst.offset();

            if (t.is_v256()) {
                VRegPair val = get_vreg_pair(inst.operand(1));

                auto store_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                store_lo->add_def(LirOperand::mem(base, offset, 16));
                store_lo->add_use(LirOperand::vreg(val.lo, 16));
                store_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(store_lo));

                auto store_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                store_hi->add_def(LirOperand::mem(base, offset + 16, 16));
                store_hi->add_use(LirOperand::vreg(val.hi, 16));
                store_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(store_hi));
                break;
            }

            VReg val = get_vreg(inst.operand(1));
            uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
            auto store = std::make_unique<LirInst>(LirOpcode::Movups);
            store->add_def(LirOperand::mem(base, offset, sz));
            store->add_use(LirOperand::vreg(val, sz));
            store->mir_origin = &inst;
            lir_bb.append_inst(std::move(store));
            break;
        }

        case Opcode::vbroadcast: {
            Type t = inst.type();

            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VReg src = get_vreg(inst.operand(0));
                LirOpcode bop = LirOpcode::Vbroadcastss;
                uint8_t src_sz = 4;
                if (t.kind() == TypeKind::F64x4) {
                    bop = LirOpcode::Vbroadcastsd;
                    src_sz = 8;
                } else if (t.kind() == TypeKind::I32x8) {
                    bop = LirOpcode::Vpbroadcastd;
                    src_sz = 4;
                } else if (t.kind() == TypeKind::I64x4) {
                    bop = LirOpcode::Vpbroadcastq;
                    src_sz = 8;
                }
                auto bcast = std::make_unique<LirInst>(bop);
                bcast->add_def(LirOperand::vreg(dst.lo, 16));
                bcast->add_use(LirOperand::vreg(src, src_sz));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));

                emit_movaps(dst.hi, dst.lo);
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));

            if (t.kind() == TypeKind::F32x4) {
                auto bcast = std::make_unique<LirInst>(LirOpcode::Vbroadcastss);
                bcast->add_def(LirOperand::vreg(dst, 16));
                bcast->add_use(LirOperand::vreg(src, 4));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));
            } else if (t.kind() == TypeKind::F64x2) {
                auto bcast = std::make_unique<LirInst>(LirOpcode::Vbroadcastsd);
                bcast->add_def(LirOperand::vreg(dst, 16));
                bcast->add_use(LirOperand::vreg(src, 8));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));
            } else if (t.kind() == TypeKind::I32x4) {
                auto bcast = std::make_unique<LirInst>(LirOpcode::Vpbroadcastd);
                bcast->add_def(LirOperand::vreg(dst, 16));
                bcast->add_use(LirOperand::vreg(src, 4));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));
            } else if (t.kind() == TypeKind::I64x2) {
                auto bcast = std::make_unique<LirInst>(LirOpcode::Vpbroadcastq);
                bcast->add_def(LirOperand::vreg(dst, 16));
                bcast->add_use(LirOperand::vreg(src, 8));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));
            }
            break;
        }

        case Opcode::vextract_lane: {
            VReg dst = get_vreg(inst.result());
            uint32_t lane = inst.lane();
            Type src_t = inst.operand(0)->type();

            VReg src = VReg{};
            if (src_t.is_v256()) {
                VRegPair src_pair = get_vreg_pair(inst.operand(0));
                uint32_t half_lanes = (src_t.kind() == TypeKind::F32x8 || src_t.kind() == TypeKind::I32x8) ? 4 : 2;
                if (lane < half_lanes) {
                    src = src_pair.lo;
                } else {
                    src = src_pair.hi;
                    lane -= half_lanes;
                }
            } else {
                src = get_vreg(inst.operand(0));
            }

            if (src_t.kind() == TypeKind::F32x4 || src_t.kind() == TypeKind::F32x8) {
                if (lane == 0) {
                    auto movss = std::make_unique<LirInst>(LirOpcode::Movss);
                    movss->add_def(LirOperand::vreg(dst, 4));
                    movss->add_use(LirOperand::vreg(src, 4));
                    movss->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movss));
                } else {
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Pshufd);
                    shuf->add_def(LirOperand::vreg(dst, 4));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::imm(lane & 3, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else if (src_t.kind() == TypeKind::F64x2 || src_t.kind() == TypeKind::F64x4) {
                if (lane == 0) {
                    auto movsd = std::make_unique<LirInst>(LirOpcode::Movsd);
                    movsd->add_def(LirOperand::vreg(dst, 8));
                    movsd->add_use(LirOperand::vreg(src, 8));
                    movsd->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movsd));
                } else {
                    emit_movaps(dst, src);
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                    shuf->add_def(LirOperand::vreg(dst, 8));
                    shuf->add_use(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::imm(1, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else if (src_t.kind() == TypeKind::I32x4 || src_t.kind() == TypeKind::I32x8) {
                if (lane == 0) {
                    auto movd = std::make_unique<LirInst>(LirOpcode::Movd_gx);
                    movd->add_def(LirOperand::vreg(dst, 4));
                    movd->add_use(LirOperand::vreg(src, 4));
                    movd->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movd));
                } else {
                    auto pext = std::make_unique<LirInst>(LirOpcode::Pextrd);
                    pext->add_def(LirOperand::vreg(dst, 4));
                    pext->add_use(LirOperand::vreg(src, 16));
                    pext->add_use(LirOperand::imm(lane, 1));
                    pext->mir_origin = &inst;
                    lir_bb.append_inst(std::move(pext));
                }
            } else if (src_t.kind() == TypeKind::I64x2 || src_t.kind() == TypeKind::I64x4) {
                if (lane == 0) {
                    auto movq = std::make_unique<LirInst>(LirOpcode::Movd_gx);
                    movq->add_def(LirOperand::vreg(dst, 8));
                    movq->add_use(LirOperand::vreg(src, 8));
                    movq->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movq));
                } else {
                    auto pext = std::make_unique<LirInst>(LirOpcode::Pextrq);
                    pext->add_def(LirOperand::vreg(dst, 8));
                    pext->add_use(LirOperand::vreg(src, 16));
                    pext->add_use(LirOperand::imm(lane, 1));
                    pext->mir_origin = &inst;
                    lir_bb.append_inst(std::move(pext));
                }
            }
            break;
        }

        case Opcode::vinsert_lane: {
            Type t = inst.type();
            VReg val = get_vreg(inst.operand(1));
            uint32_t lane = inst.lane();

            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair vec = get_vreg_pair(inst.operand(0));
                uint32_t half_lanes = (t.kind() == TypeKind::F32x8 || t.kind() == TypeKind::I32x8) ? 4 : 2;

                if (lane < half_lanes) {
                    emit_movaps(dst.hi, vec.hi);
                    emit_movaps(dst.lo, vec.lo);
                    if (t.kind() == TypeKind::F32x8) {
                        auto ins = std::make_unique<LirInst>(LirOpcode::Insertps);
                        ins->add_def(LirOperand::vreg(dst.lo, 16));
                        ins->add_use(LirOperand::vreg(dst.lo, 16));
                        ins->add_use(LirOperand::vreg(val, 16));
                        ins->add_use(LirOperand::imm((lane & 3) << 4, 1));
                        ins->mir_origin = &inst;
                        lir_bb.append_inst(std::move(ins));
                    } else if (t.kind() == TypeKind::F64x4) {
                        if (lane == 0) {
                            // {val[0], dst.lo[1]}: a scalar fmov would clear lane 1.
                            auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                            shuf->add_def(LirOperand::vreg(dst.lo, 16));
                            shuf->add_use(LirOperand::vreg(val, 16));
                            shuf->add_use(LirOperand::vreg(dst.lo, 16));
                            shuf->add_use(LirOperand::imm(0x02, 1));
                            shuf->mir_origin = &inst;
                            lir_bb.append_inst(std::move(shuf));
                        } else {
                            auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                            shuf->add_def(LirOperand::vreg(dst.lo, 16));
                            shuf->add_use(LirOperand::vreg(dst.lo, 16));
                            shuf->add_use(LirOperand::vreg(val, 16));
                            shuf->add_use(LirOperand::imm(0x00, 1));
                            shuf->mir_origin = &inst;
                            lir_bb.append_inst(std::move(shuf));
                        }
                    } else if (t.kind() == TypeKind::I32x8) {
                        auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrd);
                        pinsr->add_def(LirOperand::vreg(dst.lo, 16));
                        pinsr->add_use(LirOperand::vreg(dst.lo, 16));
                        pinsr->add_use(LirOperand::vreg(val, 4));
                        pinsr->add_use(LirOperand::imm(lane, 1));
                        pinsr->mir_origin = &inst;
                        lir_bb.append_inst(std::move(pinsr));
                    } else if (t.kind() == TypeKind::I64x4) {
                        auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrq);
                        pinsr->add_def(LirOperand::vreg(dst.lo, 16));
                        pinsr->add_use(LirOperand::vreg(dst.lo, 16));
                        pinsr->add_use(LirOperand::vreg(val, 8));
                        pinsr->add_use(LirOperand::imm(lane, 1));
                        pinsr->mir_origin = &inst;
                        lir_bb.append_inst(std::move(pinsr));
                    }
                } else {
                    emit_movaps(dst.lo, vec.lo);
                    emit_movaps(dst.hi, vec.hi);
                    uint32_t adj_lane = lane - half_lanes;
                    if (t.kind() == TypeKind::F32x8) {
                        auto ins = std::make_unique<LirInst>(LirOpcode::Insertps);
                        ins->add_def(LirOperand::vreg(dst.hi, 16));
                        ins->add_use(LirOperand::vreg(dst.hi, 16));
                        ins->add_use(LirOperand::vreg(val, 16));
                        ins->add_use(LirOperand::imm((adj_lane & 3) << 4, 1));
                        ins->mir_origin = &inst;
                        lir_bb.append_inst(std::move(ins));
                    } else if (t.kind() == TypeKind::F64x4) {
                        if (adj_lane == 0) {
                            // {val[0], dst.hi[1]}: a scalar fmov would clear lane 1.
                            auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                            shuf->add_def(LirOperand::vreg(dst.hi, 16));
                            shuf->add_use(LirOperand::vreg(val, 16));
                            shuf->add_use(LirOperand::vreg(dst.hi, 16));
                            shuf->add_use(LirOperand::imm(0x02, 1));
                            shuf->mir_origin = &inst;
                            lir_bb.append_inst(std::move(shuf));
                        } else {
                            auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                            shuf->add_def(LirOperand::vreg(dst.hi, 16));
                            shuf->add_use(LirOperand::vreg(dst.hi, 16));
                            shuf->add_use(LirOperand::vreg(val, 16));
                            shuf->add_use(LirOperand::imm(0x00, 1));
                            shuf->mir_origin = &inst;
                            lir_bb.append_inst(std::move(shuf));
                        }
                    } else if (t.kind() == TypeKind::I32x8) {
                        auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrd);
                        pinsr->add_def(LirOperand::vreg(dst.hi, 16));
                        pinsr->add_use(LirOperand::vreg(dst.hi, 16));
                        pinsr->add_use(LirOperand::vreg(val, 4));
                        pinsr->add_use(LirOperand::imm(adj_lane, 1));
                        pinsr->mir_origin = &inst;
                        lir_bb.append_inst(std::move(pinsr));
                    } else if (t.kind() == TypeKind::I64x4) {
                        auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrq);
                        pinsr->add_def(LirOperand::vreg(dst.hi, 16));
                        pinsr->add_use(LirOperand::vreg(dst.hi, 16));
                        pinsr->add_use(LirOperand::vreg(val, 8));
                        pinsr->add_use(LirOperand::imm(adj_lane, 1));
                        pinsr->mir_origin = &inst;
                        lir_bb.append_inst(std::move(pinsr));
                    }
                }
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg vec = get_vreg(inst.operand(0));

            emit_movaps(dst, vec);

            if (t.kind() == TypeKind::F32x4) {
                auto ins = std::make_unique<LirInst>(LirOpcode::Insertps);
                ins->add_def(LirOperand::vreg(dst, 16));
                ins->add_use(LirOperand::vreg(dst, 16));
                ins->add_use(LirOperand::vreg(val, 16));
                ins->add_use(LirOperand::imm((lane & 3) << 4, 1));
                ins->mir_origin = &inst;
                lir_bb.append_inst(std::move(ins));
            } else if (t.kind() == TypeKind::F64x2) {
                if (lane == 0) {
                    // {val[0], dst[1]}: a scalar fmov would clear lane 1.
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                    shuf->add_def(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(val, 16));
                    shuf->add_use(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::imm(0x02, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                } else {
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                    shuf->add_def(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(val, 16));
                    shuf->add_use(LirOperand::imm(0x00, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else if (t.kind() == TypeKind::I32x4) {
                auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrd);
                pinsr->add_def(LirOperand::vreg(dst, 16));
                pinsr->add_use(LirOperand::vreg(dst, 16));
                pinsr->add_use(LirOperand::vreg(val, 4));
                pinsr->add_use(LirOperand::imm(lane, 1));
                pinsr->mir_origin = &inst;
                lir_bb.append_inst(std::move(pinsr));
            } else if (t.kind() == TypeKind::I64x2) {
                auto pinsr = std::make_unique<LirInst>(LirOpcode::Pinsrq);
                pinsr->add_def(LirOperand::vreg(dst, 16));
                pinsr->add_use(LirOperand::vreg(dst, 16));
                pinsr->add_use(LirOperand::vreg(val, 8));
                pinsr->add_use(LirOperand::imm(lane, 1));
                pinsr->mir_origin = &inst;
                lir_bb.append_inst(std::move(pinsr));
            }
            break;
        }

        case Opcode::vshuffle: {
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            VReg v1 = get_vreg(inst.operand(1));
            uint32_t mask = inst.shuffle_mask();
            Type t = inst.type();

            if (t.kind() == TypeKind::F32x4) {
                emit_movaps(dst, v0);
                auto shuf = std::make_unique<LirInst>(LirOpcode::Shufps);
                shuf->add_def(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(v1, 16));
                shuf->add_use(LirOperand::imm(mask, 1));
                shuf->mir_origin = &inst;
                lir_bb.append_inst(std::move(shuf));
            } else if (t.kind() == TypeKind::I32x4) {
                if (v0 == v1) {
                    auto pshuf = std::make_unique<LirInst>(LirOpcode::Pshufd);
                    pshuf->add_def(LirOperand::vreg(dst, 16));
                    pshuf->add_use(LirOperand::vreg(v0, 16));
                    pshuf->add_use(LirOperand::vreg(v0, 16));
                    pshuf->add_use(LirOperand::imm(mask, 1));
                    pshuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(pshuf));
                } else {
                    emit_movaps(dst, v0);
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Shufps);
                    shuf->add_def(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(v1, 16));
                    shuf->add_use(LirOperand::imm(mask, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else {
                emit_movaps(dst, v0);
                auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                shuf->add_def(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(v1, 16));
                shuf->add_use(LirOperand::imm(mask, 1));
                shuf->mir_origin = &inst;
                lir_bb.append_inst(std::move(shuf));
            }
            break;
        }

        case Opcode::vzero: {
            Type t = inst.type();
            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                auto z_lo = std::make_unique<LirInst>(LirOpcode::Xorps);
                z_lo->add_def(LirOperand::vreg(dst.lo, 16));
                z_lo->add_use(LirOperand::vreg(dst.lo, 16));
                z_lo->add_use(LirOperand::vreg(dst.lo, 16));
                z_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(z_lo));

                auto z_hi = std::make_unique<LirInst>(LirOpcode::Xorps);
                z_hi->add_def(LirOperand::vreg(dst.hi, 16));
                z_hi->add_use(LirOperand::vreg(dst.hi, 16));
                z_hi->add_use(LirOperand::vreg(dst.hi, 16));
                z_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(z_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            auto z = std::make_unique<LirInst>(LirOpcode::Xorps);
            z->add_def(LirOperand::vreg(dst, 16));
            z->add_use(LirOperand::vreg(dst, 16));
            z->add_use(LirOperand::vreg(dst, 16));
            z->mir_origin = &inst;
            lir_bb.append_inst(std::move(z));
            break;
        }

        case Opcode::vfma: {
            Type t = inst.type();
            if (t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair a = get_vreg_pair(inst.operand(0));
                VRegPair b = get_vreg_pair(inst.operand(1));
                VRegPair c = get_vreg_pair(inst.operand(2));

                emit_movaps(dst.lo, a.lo);
                LirOpcode fma_op = (t.kind() == TypeKind::F64x4) ? LirOpcode::Vfmadd213pd : LirOpcode::Vfmadd213ps;
                auto fma_lo = std::make_unique<LirInst>(fma_op);
                fma_lo->add_def(LirOperand::vreg(dst.lo, 16));
                fma_lo->add_use(LirOperand::vreg(dst.lo, 16));
                fma_lo->add_use(LirOperand::vreg(b.lo, 16));
                fma_lo->add_use(LirOperand::vreg(c.lo, 16));
                fma_lo->mir_origin = &inst;
                lir_bb.append_inst(std::move(fma_lo));

                emit_movaps(dst.hi, a.hi);
                auto fma_hi = std::make_unique<LirInst>(fma_op);
                fma_hi->add_def(LirOperand::vreg(dst.hi, 16));
                fma_hi->add_use(LirOperand::vreg(dst.hi, 16));
                fma_hi->add_use(LirOperand::vreg(b.hi, 16));
                fma_hi->add_use(LirOperand::vreg(c.hi, 16));
                fma_hi->mir_origin = &inst;
                lir_bb.append_inst(std::move(fma_hi));
                break;
            }

            VReg dst = get_vreg(inst.result());
            VReg a = get_vreg(inst.operand(0));
            VReg b = get_vreg(inst.operand(1));
            VReg c = get_vreg(inst.operand(2));

            emit_movaps(dst, a);

            LirOpcode fma_op = (t.kind() == TypeKind::F64x2)
                             ? LirOpcode::Vfmadd213pd : LirOpcode::Vfmadd213ps;

            auto fma = std::make_unique<LirInst>(fma_op);
            fma->add_def(LirOperand::vreg(dst, 16));
            fma->add_use(LirOperand::vreg(dst, 16));
            fma->add_use(LirOperand::vreg(b, 16));
            fma->add_use(LirOperand::vreg(c, 16));
            fma->mir_origin = &inst;
            lir_bb.append_inst(std::move(fma));
            break;
        }

        case Opcode::fma_f32:
        case Opcode::fma_f64: {
            uint8_t sz = (inst.opcode() == Opcode::fma_f32) ? 4 : 8;
            VReg dst = get_vreg(inst.result());
            VReg a = get_vreg(inst.operand(0));
            VReg b = get_vreg(inst.operand(1));
            VReg c = get_vreg(inst.operand(2));

            if (dst != a) {
                auto mov = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
                mov->add_def(LirOperand::vreg(dst, sz));
                mov->add_use(LirOperand::vreg(a, sz));
                lir_bb.append_inst(std::move(mov));
            }

            LirOpcode fma_op = (sz == 4) ? LirOpcode::Vfmadd213ss : LirOpcode::Vfmadd213sd;
            auto fma = std::make_unique<LirInst>(fma_op);
            fma->add_def(LirOperand::vreg(dst, sz));
            fma->add_use(LirOperand::vreg(dst, sz));
            fma->add_use(LirOperand::vreg(b, sz));
            fma->add_use(LirOperand::vreg(c, sz));
            fma->mir_origin = &inst;
            lir_bb.append_inst(std::move(fma));
            break;
        }

        default:
            codegen::throw_unsupported("aarch64 isel (vector)", opcode_name(inst.opcode()));
    }
}

} // namespace brass::aarch64
