#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_vector_instruction(const Instruction& inst, LirBlock& lir_bb) {
    auto emit_movaps = [&](VReg dst, VReg src) {
        if (dst != src) {
            auto mov = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov->add_def(LirOperand::vreg(dst, 16));
            mov->add_use(LirOperand::vreg(src, 16));
            lir_bb.append_inst(std::move(mov));
        }
    };
    // dst(ymm) = op(a, b), three-operand VEX form.
    auto emit_v256_binop = [&](LirOpcode op, VReg dst, VReg a, VReg b) {
        auto binop = std::make_unique<LirInst>(op);
        binop->add_def(LirOperand::vreg(dst, 32));
        binop->add_use(LirOperand::vreg(a, 32));
        binop->add_use(LirOperand::vreg(b, 32));
        binop->mir_origin = &inst;
        lir_bb.append_inst(std::move(binop));
    };
    // xmm = the high 128 bits of a ymm.
    auto extract_high_half = [&](VReg ymm) {
        VReg half = lir_fn_->allocate_vreg(RegClass::XMM, 16);
        auto ex = std::make_unique<LirInst>(LirOpcode::Vextractf128);
        ex->add_def(LirOperand::vreg(half, 16));
        ex->add_use(LirOperand::vreg(ymm, 32));
        ex->add_use(LirOperand::imm(1, 1));
        lir_bb.append_inst(std::move(ex));
        return half;
    };
    // A ymm of all-ones (shift_op == Nop) or of per-lane sign bits
    // (all-ones shifted left by `shift` in each lane): built in an xmm with
    // pcmpeqd [+ pslld/psllq], then its low lane broadcast across the ymm.
    auto splat_mask_256 = [&](LirOpcode shift_op, int64_t shift, LirOpcode bcast_op) {
        VReg m = lir_fn_->allocate_vreg(RegClass::XMM, 16);
        auto cmp = std::make_unique<LirInst>(LirOpcode::Pcmpeqd);
        cmp->add_def(LirOperand::vreg(m, 16));
        cmp->add_use(LirOperand::vreg(m, 16));
        cmp->add_use(LirOperand::vreg(m, 16));
        lir_bb.append_inst(std::move(cmp));
        if (shift_op != LirOpcode::Nop) {
            auto sh = std::make_unique<LirInst>(shift_op);
            sh->add_def(LirOperand::vreg(m, 16));
            sh->add_use(LirOperand::vreg(m, 16));
            sh->add_use(LirOperand::imm(shift, 1));
            lir_bb.append_inst(std::move(sh));
        }
        VReg wide = lir_fn_->allocate_vreg(RegClass::XMM, 32);
        auto bc = std::make_unique<LirInst>(bcast_op);
        bc->add_def(LirOperand::vreg(wide, 32));
        bc->add_use(LirOperand::vreg(m, 16));
        lir_bb.append_inst(std::move(bc));
        return wide;
    };

    switch (inst.opcode()) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vmin:
        case Opcode::vmax: {
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            VReg v1 = get_vreg(inst.operand(1));
            Type t = inst.type();

            if (inst.opcode() == Opcode::vmul && t.kind() == TypeKind::I64x2) {
                // Emulate i64x2 multiplication via 64-bit imul
                VReg r0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg r1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg s0 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg s1 = lir_fn_->allocate_vreg(RegClass::GPR, 8);

                auto pext0 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext0->add_def(LirOperand::vreg(r0, 8));
                pext0->add_use(LirOperand::vreg(v0, 16));
                pext0->add_use(LirOperand::imm(0, 1));
                lir_bb.append_inst(std::move(pext0));

                auto pext1 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext1->add_def(LirOperand::vreg(r1, 8));
                pext1->add_use(LirOperand::vreg(v1, 16));
                pext1->add_use(LirOperand::imm(0, 1));
                lir_bb.append_inst(std::move(pext1));

                auto mul0 = std::make_unique<LirInst>(LirOpcode::Imul);
                mul0->add_def(LirOperand::vreg(r0, 8));
                mul0->add_use(LirOperand::vreg(r0, 8));
                mul0->add_use(LirOperand::vreg(r1, 8));
                lir_bb.append_inst(std::move(mul0));

                auto movq0 = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                movq0->add_def(LirOperand::vreg(dst, 16));
                movq0->add_use(LirOperand::vreg(r0, 8));
                lir_bb.append_inst(std::move(movq0));

                auto pext2 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext2->add_def(LirOperand::vreg(s0, 8));
                pext2->add_use(LirOperand::vreg(v0, 16));
                pext2->add_use(LirOperand::imm(1, 1));
                lir_bb.append_inst(std::move(pext2));

                auto pext3 = std::make_unique<LirInst>(LirOpcode::Pextrq);
                pext3->add_def(LirOperand::vreg(s1, 8));
                pext3->add_use(LirOperand::vreg(v1, 16));
                pext3->add_use(LirOperand::imm(1, 1));
                lir_bb.append_inst(std::move(pext3));

                auto mul1 = std::make_unique<LirInst>(LirOpcode::Imul);
                mul1->add_def(LirOperand::vreg(s0, 8));
                mul1->add_use(LirOperand::vreg(s0, 8));
                mul1->add_use(LirOperand::vreg(s1, 8));
                lir_bb.append_inst(std::move(mul1));

                auto pinsr1 = std::make_unique<LirInst>(LirOpcode::Pinsrq);
                pinsr1->add_def(LirOperand::vreg(dst, 16));
                pinsr1->add_use(LirOperand::vreg(dst, 16));
                pinsr1->add_use(LirOperand::vreg(s0, 8));
                pinsr1->add_use(LirOperand::imm(1, 1));
                pinsr1->mir_origin = &inst;
                lir_bb.append_inst(std::move(pinsr1));
                break;
            }

            if (t.is_v256()) {
                LirOpcode op = LirOpcode::Nop;
                if (inst.opcode() == Opcode::vadd) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vaddps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vaddpd;
                    else if (t.kind() == TypeKind::I32x8) op = LirOpcode::Vpaddd;
                    else if (t.kind() == TypeKind::I64x4) op = LirOpcode::Vpaddq;
                } else if (inst.opcode() == Opcode::vsub) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vsubps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vsubpd;
                    else if (t.kind() == TypeKind::I32x8) op = LirOpcode::Vpsubd;
                    else if (t.kind() == TypeKind::I64x4) op = LirOpcode::Vpsubq;
                } else if (inst.opcode() == Opcode::vmul) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vmulps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vmulpd;
                    else if (t.kind() == TypeKind::I32x8) op = LirOpcode::Vpmulld;
                } else if (inst.opcode() == Opcode::vdiv) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vdivps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vdivpd;
                } else if (inst.opcode() == Opcode::vmin) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vminps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vminpd;
                } else if (inst.opcode() == Opcode::vmax) {
                    if (t.kind() == TypeKind::F32x8) op = LirOpcode::Vmaxps;
                    else if (t.kind() == TypeKind::F64x4) op = LirOpcode::Vmaxpd;
                }
                // No 256-bit instruction for this op/type pair (i64x4 vmul,
                // integer vdiv, ...): an error, never the 128-bit path, which
                // would compute on half the vector or not at all.
                if (op == LirOpcode::Nop) {
                    codegen::throw_unsupported("x64 isel (vector)",
                                               std::string(opcode_name(inst.opcode())) + " " + brass::to_string(t));
                }
                auto binop = std::make_unique<LirInst>(op);
                binop->add_def(LirOperand::vreg(dst, 32));
                binop->add_use(LirOperand::vreg(v0, 32));
                binop->add_use(LirOperand::vreg(v1, 32));
                binop->mir_origin = &inst;
                lir_bb.append_inst(std::move(binop));
                break;
            }

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
            if (op == LirOpcode::Nop) {
                // e.g. i64x2 vdiv / vmin: a Nop here used to leave the result
                // equal to operand 0.
                codegen::throw_unsupported("x64 isel (vector)",
                                           std::string(opcode_name(inst.opcode())) + " " + brass::to_string(t));
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
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            Type t = inst.type();
            LirOpcode op = LirOpcode::Nop;
            switch (t.kind()) {
                case TypeKind::F32x4: op = LirOpcode::Sqrtps; break;
                case TypeKind::F64x2: op = LirOpcode::Sqrtpd; break;
                case TypeKind::F32x8: op = LirOpcode::Vsqrtps; break;
                case TypeKind::F64x4: op = LirOpcode::Vsqrtpd; break;
                default:
                    codegen::throw_unsupported("x64 isel (vector)", "vsqrt " + brass::to_string(t));
            }
            const uint8_t sz = t.is_v256() ? 32 : 16;

            auto s = std::make_unique<LirInst>(op);
            s->add_def(LirOperand::vreg(dst, sz));
            s->add_use(LirOperand::vreg(v0, sz));
            s->mir_origin = &inst;
            lir_bb.append_inst(std::move(s));
            break;
        }

        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor: {
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            VReg v1 = get_vreg(inst.operand(1));

            if (inst.type().is_v256()) {
                LirOpcode op = LirOpcode::Vpand;
                if (inst.opcode() == Opcode::vor) op = LirOpcode::Vpor;
                else if (inst.opcode() == Opcode::vxor) op = LirOpcode::Vpxor;

                auto binop = std::make_unique<LirInst>(op);
                binop->add_def(LirOperand::vreg(dst, 32));
                binop->add_use(LirOperand::vreg(v0, 32));
                binop->add_use(LirOperand::vreg(v1, 32));
                binop->mir_origin = &inst;
                lir_bb.append_inst(std::move(binop));
                break;
            }

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
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            if (inst.type().is_v256()) {
                // x ^ all-ones, the ones widened from an xmm to a ymm.
                VReg ones = splat_mask_256(LirOpcode::Nop, 0, LirOpcode::Vpbroadcastq);
                emit_v256_binop(LirOpcode::Vpxor, dst, v0, ones);
                break;
            }
            VReg ones = lir_fn_->allocate_vreg(RegClass::XMM, 16);

            auto cmp = std::make_unique<LirInst>(LirOpcode::Pcmpeqd);
            cmp->add_def(LirOperand::vreg(ones, 16));
            cmp->add_use(LirOperand::vreg(ones, 16));
            cmp->add_use(LirOperand::vreg(ones, 16));
            lir_bb.append_inst(std::move(cmp));

            emit_movaps(dst, v0);

            auto xor_op = std::make_unique<LirInst>(LirOpcode::Pxor);
            xor_op->add_def(LirOperand::vreg(dst, 16));
            xor_op->add_use(LirOperand::vreg(dst, 16));
            xor_op->add_use(LirOperand::vreg(ones, 16));
            xor_op->mir_origin = &inst;
            lir_bb.append_inst(std::move(xor_op));
            break;
        }

        case Opcode::vneg: {
            VReg dst = get_vreg(inst.result());
            VReg v0 = get_vreg(inst.operand(0));
            Type t = inst.type();

            if (t.is_v256()) {
                switch (t.kind()) {
                    case TypeKind::F32x8:   // flip each sign bit
                        emit_v256_binop(LirOpcode::Vxorps, dst, v0,
                                        splat_mask_256(LirOpcode::Pslld, 31, LirOpcode::Vpbroadcastd));
                        break;
                    case TypeKind::F64x4:
                        emit_v256_binop(LirOpcode::Vxorpd, dst, v0,
                                        splat_mask_256(LirOpcode::Psllq, 63, LirOpcode::Vpbroadcastq));
                        break;
                    case TypeKind::I32x8:
                    case TypeKind::I64x4: {   // 0 - x
                        VReg zero = lir_fn_->allocate_vreg(RegClass::XMM, 32);
                        auto z = std::make_unique<LirInst>(LirOpcode::Vpxor);
                        z->add_def(LirOperand::vreg(zero, 32));
                        z->add_use(LirOperand::vreg(zero, 32));
                        z->add_use(LirOperand::vreg(zero, 32));
                        lir_bb.append_inst(std::move(z));
                        emit_v256_binop(t.kind() == TypeKind::I32x8 ? LirOpcode::Vpsubd : LirOpcode::Vpsubq,
                                        dst, zero, v0);
                        break;
                    }
                    default:
                        codegen::throw_unsupported("x64 isel (vector)", "vneg " + brass::to_string(t));
                }
                break;
            }

            if (t.kind() == TypeKind::F32x4) {
                VReg mask = lir_fn_->allocate_vreg(RegClass::XMM, 16);
                auto cmp = std::make_unique<LirInst>(LirOpcode::Pcmpeqd);
                cmp->add_def(LirOperand::vreg(mask, 16));
                cmp->add_use(LirOperand::vreg(mask, 16));
                cmp->add_use(LirOperand::vreg(mask, 16));
                lir_bb.append_inst(std::move(cmp));

                auto shift = std::make_unique<LirInst>(LirOpcode::Pslld);
                shift->add_def(LirOperand::vreg(mask, 16));
                shift->add_use(LirOperand::vreg(mask, 16));
                shift->add_use(LirOperand::imm(31, 1));
                lir_bb.append_inst(std::move(shift));

                emit_movaps(dst, v0);

                auto xor_op = std::make_unique<LirInst>(LirOpcode::Xorps);
                xor_op->add_def(LirOperand::vreg(dst, 16));
                xor_op->add_use(LirOperand::vreg(dst, 16));
                xor_op->add_use(LirOperand::vreg(mask, 16));
                xor_op->mir_origin = &inst;
                lir_bb.append_inst(std::move(xor_op));
            } else if (t.kind() == TypeKind::F64x2) {
                VReg mask = lir_fn_->allocate_vreg(RegClass::XMM, 16);
                auto cmp = std::make_unique<LirInst>(LirOpcode::Pcmpeqd);
                cmp->add_def(LirOperand::vreg(mask, 16));
                cmp->add_use(LirOperand::vreg(mask, 16));
                cmp->add_use(LirOperand::vreg(mask, 16));
                lir_bb.append_inst(std::move(cmp));

                auto shift = std::make_unique<LirInst>(LirOpcode::Psllq);
                shift->add_def(LirOperand::vreg(mask, 16));
                shift->add_use(LirOperand::vreg(mask, 16));
                shift->add_use(LirOperand::imm(63, 1));
                lir_bb.append_inst(std::move(shift));

                emit_movaps(dst, v0);

                auto xor_op = std::make_unique<LirInst>(LirOpcode::Xorpd);
                xor_op->add_def(LirOperand::vreg(dst, 16));
                xor_op->add_use(LirOperand::vreg(dst, 16));
                xor_op->add_use(LirOperand::vreg(mask, 16));
                xor_op->mir_origin = &inst;
                lir_bb.append_inst(std::move(xor_op));
            } else {
                if (t.kind() != TypeKind::I32x4 && t.kind() != TypeKind::I64x2) {
                    codegen::throw_unsupported("x64 isel (vector)", "vneg " + brass::to_string(t));
                }
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
            VReg dst = get_vreg(inst.result());
            VReg base = get_vreg(inst.operand(0));
            int32_t offset = inst.offset();
            uint8_t sz = static_cast<uint8_t>(inst.type().size_in_bytes());
            LirOpcode op = (sz == 32) ? LirOpcode::Vmovups : LirOpcode::Movups;

            auto load = std::make_unique<LirInst>(op);
            load->add_def(LirOperand::vreg(dst, sz));
            load->add_use(LirOperand::mem(base, offset, sz));
            load->mir_origin = &inst;
            lir_bb.append_inst(std::move(load));
            break;
        }

        case Opcode::vstore: {
            VReg base = get_vreg(inst.operand(0));
            VReg val = get_vreg(inst.operand(1));
            int32_t offset = inst.offset();
            uint8_t sz = static_cast<uint8_t>(inst.memory_type().size_in_bytes());
            LirOpcode op = (sz == 32) ? LirOpcode::Vmovups : LirOpcode::Movups;

            auto store = std::make_unique<LirInst>(op);
            store->add_def(LirOperand::mem(base, offset, sz));
            store->add_use(LirOperand::vreg(val, sz));
            store->mir_origin = &inst;
            lir_bb.append_inst(std::move(store));
            break;
        }

        case Opcode::vbroadcast: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            Type t = inst.type();

            if (t.is_v256()) {
                // VBROADCASTSS/SD and VPBROADCASTD/Q take an xmm (or memory)
                // source, never a GPR: an integer scalar lives in a GPR, so it
                // is moved into an xmm first.
                LirOpcode bop = LirOpcode::Nop;
                uint8_t src_sz = 0;
                bool from_gpr = false;
                switch (t.kind()) {
                    case TypeKind::F32x8: bop = LirOpcode::Vbroadcastss; src_sz = 4; break;
                    case TypeKind::F64x4: bop = LirOpcode::Vbroadcastsd; src_sz = 8; break;
                    case TypeKind::I32x8: bop = LirOpcode::Vpbroadcastd; src_sz = 4; from_gpr = true; break;
                    case TypeKind::I64x4: bop = LirOpcode::Vpbroadcastq; src_sz = 8; from_gpr = true; break;
                    default:
                        codegen::throw_unsupported("x64 isel (vector)", "vbroadcast to " + brass::to_string(t));
                }
                VReg bsrc = src;
                if (from_gpr) {
                    bsrc = lir_fn_->allocate_vreg(RegClass::XMM, 16);
                    auto movd = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                    movd->add_def(LirOperand::vreg(bsrc, 16));
                    movd->add_use(LirOperand::vreg(src, src_sz));
                    lir_bb.append_inst(std::move(movd));
                    src_sz = 16;
                }
                auto bcast = std::make_unique<LirInst>(bop);
                bcast->add_def(LirOperand::vreg(dst, 32));
                bcast->add_use(LirOperand::vreg(bsrc, src_sz));
                bcast->mir_origin = &inst;
                lir_bb.append_inst(std::move(bcast));
                break;
            }

            if (t.kind() == TypeKind::F32x4) {
                auto movss = std::make_unique<LirInst>(LirOpcode::Movss);
                movss->add_def(LirOperand::vreg(dst, 4));
                movss->add_use(LirOperand::vreg(src, 4));
                lir_bb.append_inst(std::move(movss));

                auto shuf = std::make_unique<LirInst>(LirOpcode::Shufps);
                shuf->add_def(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::imm(0x00, 1));
                shuf->mir_origin = &inst;
                lir_bb.append_inst(std::move(shuf));
            } else if (t.kind() == TypeKind::F64x2) {
                auto movddup = std::make_unique<LirInst>(LirOpcode::Movddup);
                movddup->add_def(LirOperand::vreg(dst, 16));
                movddup->add_use(LirOperand::vreg(src, 8));
                movddup->mir_origin = &inst;
                lir_bb.append_inst(std::move(movddup));
            } else if (t.kind() == TypeKind::I32x4) {
                auto movd = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                movd->add_def(LirOperand::vreg(dst, 16));
                movd->add_use(LirOperand::vreg(src, 4));
                lir_bb.append_inst(std::move(movd));

                auto shuf = std::make_unique<LirInst>(LirOpcode::Pshufd);
                shuf->add_def(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::vreg(dst, 16));
                shuf->add_use(LirOperand::imm(0x00, 1));
                shuf->mir_origin = &inst;
                lir_bb.append_inst(std::move(shuf));
            } else if (t.kind() == TypeKind::I64x2) {
                auto movq = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                movq->add_def(LirOperand::vreg(dst, 16));
                movq->add_use(LirOperand::vreg(src, 8));
                lir_bb.append_inst(std::move(movq));

                auto movddup = std::make_unique<LirInst>(LirOpcode::Movddup);
                movddup->add_def(LirOperand::vreg(dst, 16));
                movddup->add_use(LirOperand::vreg(dst, 16));
                movddup->mir_origin = &inst;
                lir_bb.append_inst(std::move(movddup));
            } else {
                codegen::throw_unsupported("x64 isel (vector)", "vbroadcast to " + brass::to_string(t));
            }
            break;
        }

        case Opcode::vextract_lane: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint32_t lane = inst.lane();
            Type src_t = inst.operand(0)->type();
            if (lane >= src_t.vector_lanes()) {
                codegen::throw_unsupported("x64 isel (vector)", "vextract_lane " + std::to_string(lane) +
                                                                    " of " + brass::to_string(src_t));
            }
            if (src_t.is_v256()) {
                // The SSE extracts see only the low 128 bits (the xmm view
                // of the ymm): a lane in the high half is taken from that
                // half after VEXTRACTF128.
                const uint32_t half_lanes = src_t.vector_lanes() / 2;
                if (lane >= half_lanes) {
                    src = extract_high_half(src);
                    lane -= half_lanes;
                }
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
            } else {
                codegen::throw_unsupported("x64 isel (vector)", "vextract_lane of " + brass::to_string(src_t));
            }
            break;
        }

        case Opcode::vinsert_lane: {
            VReg dst = get_vreg(inst.result());
            VReg vec = get_vreg(inst.operand(0));
            VReg val = get_vreg(inst.operand(1));
            uint32_t lane = inst.lane();
            Type t = inst.type();
            if (lane >= t.vector_lanes()) {
                codegen::throw_unsupported("x64 isel (vector)", "vinsert_lane " + std::to_string(lane) +
                                                                    " of " + brass::to_string(t));
            }

            // 256-bit: insert into the 128-bit half holding the lane, then
            // put that half back with VINSERTF128. `wide` is the final ymm.
            VReg wide{};
            uint32_t half = 0;
            if (t.is_v256()) {
                wide = dst;
                const uint32_t half_lanes = t.vector_lanes() / 2;
                half = lane / half_lanes;
                lane %= half_lanes;
                if (half == 1) {
                    dst = extract_high_half(vec);
                } else {
                    dst = lir_fn_->allocate_vreg(RegClass::XMM, 16);
                    auto lo = std::make_unique<LirInst>(LirOpcode::Movaps);
                    lo->add_def(LirOperand::vreg(dst, 16));
                    lo->add_use(LirOperand::vreg(vec, 16));
                    lir_bb.append_inst(std::move(lo));
                }
                t = t.element_type().kind() == TypeKind::F32 ? Type::f32x4()
                  : t.element_type().kind() == TypeKind::F64 ? Type::f64x2()
                  : t.element_type().kind() == TypeKind::I32 ? Type::i32x4()
                                                              : Type::i64x2();
            } else {
                emit_movaps(dst, vec);
            }

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
                    // movsd xmm, xmm merges: lane 1 of dst survives. Say so
                    // with a use of dst (the emitter reads uses[0]), or the
                    // copy of the vector into dst is a dead def and lane 1
                    // is whatever the register held.
                    auto movsd = std::make_unique<LirInst>(LirOpcode::Movsd);
                    movsd->add_def(LirOperand::vreg(dst, 8));
                    movsd->add_use(LirOperand::vreg(val, 8));
                    movsd->add_use(LirOperand::vreg(dst, 16));
                    movsd->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movsd));
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
            } else {
                codegen::throw_unsupported("x64 isel (vector)", "vinsert_lane of " + brass::to_string(t));
            }
            if (wide.is_valid()) {
                auto ins = std::make_unique<LirInst>(LirOpcode::Vinsertf128);
                ins->add_def(LirOperand::vreg(wide, 32));
                ins->add_use(LirOperand::vreg(vec, 32));
                ins->add_use(LirOperand::vreg(dst, 16));
                ins->add_use(LirOperand::imm(half, 1));
                ins->mir_origin = &inst;
                lir_bb.append_inst(std::move(ins));
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
                // SHUFPD's two 1-bit selectors: 2-lane types only. A 256-bit
                // shuffle has no lowering here (it used to shuffle the low
                // 128 bits and leave the rest as operand 0).
                if (t.kind() != TypeKind::F64x2 && t.kind() != TypeKind::I64x2) {
                    codegen::throw_unsupported("x64 isel (vector)", "vshuffle " + brass::to_string(t));
                }
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
            VReg dst = get_vreg(inst.result());
            Type t = inst.type();
            if (t.is_v256()) {
                auto z = std::make_unique<LirInst>(LirOpcode::Vxorps);
                z->add_def(LirOperand::vreg(dst, 32));
                z->add_use(LirOperand::vreg(dst, 32));
                z->add_use(LirOperand::vreg(dst, 32));
                z->mir_origin = &inst;
                lir_bb.append_inst(std::move(z));
            } else {
                auto z = std::make_unique<LirInst>(LirOpcode::Xorps);
                z->add_def(LirOperand::vreg(dst, 16));
                z->add_use(LirOperand::vreg(dst, 16));
                z->add_use(LirOperand::vreg(dst, 16));
                z->mir_origin = &inst;
                lir_bb.append_inst(std::move(z));
            }
            break;
        }

        case Opcode::vfma: {
            uint8_t sz = static_cast<uint8_t>(inst.type().size_in_bytes());
            VReg dst = get_vreg(inst.result());
            VReg a = get_vreg(inst.operand(0));
            VReg b = get_vreg(inst.operand(1));
            VReg c = get_vreg(inst.operand(2));
            if (!inst.type().element_type().is_float()) {
                codegen::throw_unsupported("x64 isel (vector)", "vfma " + brass::to_string(inst.type()));
            }

            if (dst != a) {
                auto mov = std::make_unique<LirInst>(sz == 32 ? LirOpcode::Vmovaps : LirOpcode::Movaps);
                mov->add_def(LirOperand::vreg(dst, sz));
                mov->add_use(LirOperand::vreg(a, sz));
                lir_bb.append_inst(std::move(mov));
            }

            LirOpcode fma_op = (inst.type().kind() == TypeKind::F64x2 || inst.type().kind() == TypeKind::F64x4)
                             ? LirOpcode::Vfmadd213pd : LirOpcode::Vfmadd213ps;

            auto fma = std::make_unique<LirInst>(fma_op);
            fma->add_def(LirOperand::vreg(dst, sz));
            fma->add_use(LirOperand::vreg(dst, sz));
            fma->add_use(LirOperand::vreg(b, sz));
            fma->add_use(LirOperand::vreg(c, sz));
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
            codegen::throw_unsupported("x64 isel (vector)", opcode_name(inst.opcode()));
    }
}

} // namespace brass::x64
