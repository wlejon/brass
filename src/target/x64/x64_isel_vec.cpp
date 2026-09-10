#include <brass/target/x64/x64_isel.hpp>
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
                if (op != LirOpcode::Nop) {
                    auto binop = std::make_unique<LirInst>(op);
                    binop->add_def(LirOperand::vreg(dst, 32));
                    binop->add_use(LirOperand::vreg(v0, 32));
                    binop->add_use(LirOperand::vreg(v1, 32));
                    binop->mir_origin = &inst;
                    lir_bb.append_inst(std::move(binop));
                    break;
                }
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
                bcast->add_def(LirOperand::vreg(dst, 32));
                bcast->add_use(LirOperand::vreg(src, src_sz));
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
            }
            break;
        }

        case Opcode::vextract_lane: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint32_t lane = inst.lane();
            Type src_t = inst.operand(0)->type();

            if (src_t.kind() == TypeKind::F32x4) {
                if (lane == 0) {
                    auto movss = std::make_unique<LirInst>(LirOpcode::Movss);
                    movss->add_def(LirOperand::vreg(dst, 4));
                    movss->add_use(LirOperand::vreg(src, 4));
                    movss->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movss));
                } else {
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Pshufd);
                    shuf->add_def(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::imm(lane & 3, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else if (src_t.kind() == TypeKind::F64x2) {
                if (lane == 0) {
                    auto movsd = std::make_unique<LirInst>(LirOpcode::Movsd);
                    movsd->add_def(LirOperand::vreg(dst, 8));
                    movsd->add_use(LirOperand::vreg(src, 8));
                    movsd->mir_origin = &inst;
                    lir_bb.append_inst(std::move(movsd));
                } else {
                    emit_movaps(dst, src);
                    auto shuf = std::make_unique<LirInst>(LirOpcode::Shufpd);
                    shuf->add_def(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(dst, 16));
                    shuf->add_use(LirOperand::vreg(src, 16));
                    shuf->add_use(LirOperand::imm(1, 1));
                    shuf->mir_origin = &inst;
                    lir_bb.append_inst(std::move(shuf));
                }
            } else if (src_t.kind() == TypeKind::I32x4) {
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
            } else if (src_t.kind() == TypeKind::I64x2) {
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
            VReg dst = get_vreg(inst.result());
            VReg vec = get_vreg(inst.operand(0));
            VReg val = get_vreg(inst.operand(1));
            uint32_t lane = inst.lane();
            Type t = inst.type();

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
                    auto movsd = std::make_unique<LirInst>(LirOpcode::Movsd);
                    movsd->add_def(LirOperand::vreg(dst, 8));
                    movsd->add_use(LirOperand::vreg(val, 8));
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
            break;
    }
}

} // namespace brass::x64
