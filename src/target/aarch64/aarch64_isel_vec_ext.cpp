#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_vector_lane_instruction(const Instruction& inst, LirBlock& lir_bb) {
    auto unsupported = [&](Type t) {
        codegen::throw_unsupported("aarch64 isel (vector)",
                                   std::string(opcode_name(inst.opcode())) + " " + brass::to_string(t));
    };
    auto vreg_xmm = [&]() { return lir_fn_->allocate_vreg(RegClass::XMM, 16); };

    auto emit_copy16 = [&](VReg dst, VReg src) {
        if (dst == src) return;
        auto mov = std::make_unique<LirInst>(LirOpcode::Movaps);
        mov->add_def(LirOperand::vreg(dst, 16));
        mov->add_use(LirOperand::vreg(src, 16));
        lir_bb.append_inst(std::move(mov));
    };
    auto emit2_16 = [&](LirOpcode op, VReg dst, VReg src, int imm = -1) {
        auto i = std::make_unique<LirInst>(op);
        i->add_def(LirOperand::vreg(dst, 16));
        i->add_use(LirOperand::vreg(src, 16));
        if (imm >= 0) i->add_use(LirOperand::imm(imm, 1));
        i->mir_origin = &inst;
        lir_bb.append_inst(std::move(i));
    };
    auto emit3_16 = [&](LirOpcode op, VReg dst, VReg a, VReg b, int imm = -1) {
        auto i = std::make_unique<LirInst>(op);
        i->add_def(LirOperand::vreg(dst, 16));
        i->add_use(LirOperand::vreg(a, 16));
        i->add_use(LirOperand::vreg(b, 16));
        if (imm >= 0) i->add_use(LirOperand::imm(imm, 1));
        i->mir_origin = &inst;
        lir_bb.append_inst(std::move(i));
    };

    const Type t0 = inst.operand(0)->type();
    const bool is_v256 = t0.is_v256();
    const bool wide_lanes = t0.element_type().size_in_bytes() == 8;

    switch (inst.opcode()) {
        case Opcode::vcmp_eq:
        case Opcode::vcmp_ne:
        case Opcode::vcmp_lt:
        case Opcode::vcmp_le:
        case Opcode::vcmp_gt:
        case Opcode::vcmp_ge: {
            auto lower_cmp_16 = [&](VReg dst, VReg a, VReg b) {
                if (t0.element_type().is_float()) {
                    int pred = 0;
                    switch (inst.opcode()) {
                        case Opcode::vcmp_eq: pred = 0x00; break;
                        case Opcode::vcmp_ne: pred = 0x04; break;
                        case Opcode::vcmp_lt: pred = 0x11; break;
                        case Opcode::vcmp_le: pred = 0x12; break;
                        case Opcode::vcmp_gt: pred = 0x1E; break;
                        default:              pred = 0x1D; break;
                    }
                    emit3_16(wide_lanes ? LirOpcode::Vcmppd : LirOpcode::Vcmpps, dst, a, b, pred);
                    return;
                }
                const LirOpcode eq = wide_lanes ? LirOpcode::Vpcmpeqq : LirOpcode::Vpcmpeqd;
                const LirOpcode gt = wide_lanes ? LirOpcode::Vpcmpgtq : LirOpcode::Vpcmpgtd;
                switch (inst.opcode()) {
                    case Opcode::vcmp_eq: emit3_16(eq, dst, a, b); break;
                    case Opcode::vcmp_gt: emit3_16(gt, dst, a, b); break;
                    case Opcode::vcmp_lt: emit3_16(gt, dst, b, a); break;
                    default: {
                        VReg r = vreg_xmm();
                        if (inst.opcode() == Opcode::vcmp_ne) emit3_16(eq, r, a, b);
                        else if (inst.opcode() == Opcode::vcmp_le) emit3_16(gt, r, a, b);
                        else emit3_16(gt, r, b, a);
                        emit2_16(LirOpcode::Pnot, dst, r);
                        break;
                    }
                }
            };
            if (is_v256) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair a = get_vreg_pair(inst.operand(0));
                VRegPair b = get_vreg_pair(inst.operand(1));
                lower_cmp_16(dst.lo, a.lo, b.lo);
                lower_cmp_16(dst.hi, a.hi, b.hi);
            } else {
                lower_cmp_16(get_vreg(inst.result()), get_vreg(inst.operand(0)), get_vreg(inst.operand(1)));
            }
            break;
        }

        case Opcode::vselect: {
            const Type out_t = inst.type();
            const LirOpcode bop = (out_t.element_type().size_in_bytes() == 8) ? LirOpcode::Vblendvpd
                                                                              : LirOpcode::Vblendvps;
            auto emit_sel_16 = [&](VReg dst, VReg mask, VReg a, VReg b) {
                auto i = std::make_unique<LirInst>(bop);
                i->add_def(LirOperand::vreg(dst, 16));
                i->add_use(LirOperand::vreg(b, 16));
                i->add_use(LirOperand::vreg(a, 16));
                i->add_use(LirOperand::vreg(mask, 16));
                i->mir_origin = &inst;
                lir_bb.append_inst(std::move(i));
            };
            if (out_t.is_v256()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair m = get_vreg_pair(inst.operand(0));
                VRegPair a = get_vreg_pair(inst.operand(1));
                VRegPair b = get_vreg_pair(inst.operand(2));
                emit_sel_16(dst.lo, m.lo, a.lo, b.lo);
                emit_sel_16(dst.hi, m.hi, a.hi, b.hi);
            } else {
                emit_sel_16(get_vreg(inst.result()), get_vreg(inst.operand(0)),
                            get_vreg(inst.operand(1)), get_vreg(inst.operand(2)));
            }
            break;
        }

        case Opcode::vshl:
        case Opcode::vlshr:
        case Opcode::vashr: {
            auto try_const_shift = [&](const Value* sv, uint64_t& out_amt) -> bool {
                if (!sv || !sv->is_instruction()) return false;
                const Instruction* def = sv->defining_instruction();
                if (def->opcode() == Opcode::vzero) {
                    out_amt = 0;
                    return true;
                }
                if (def->opcode() == Opcode::vbroadcast && def->operand_count() >= 1) {
                    const Value* sc = def->operand(0);
                    if (!sc || !sc->is_instruction()) return false;
                    const Instruction* sdef = sc->defining_instruction();
                    if (sdef->opcode() == Opcode::iconst_i32) {
                        out_amt = static_cast<uint32_t>(sdef->imm_i32());
                        return true;
                    }
                    if (sdef->opcode() == Opcode::iconst_i64) {
                        out_amt = static_cast<uint64_t>(sdef->imm_i64());
                        return true;
                    }
                }
                return false;
            };

            uint64_t imm_amt = 0;
            const uint64_t lane_bits = wide_lanes ? 64u : 32u;
            const bool use_imm = try_const_shift(inst.operand(1), imm_amt) && imm_amt < lane_bits;
            LirOpcode imm_op = LirOpcode::Nop;
            if (use_imm && imm_amt > 0) {
                if (inst.opcode() == Opcode::vshl) imm_op = wide_lanes ? LirOpcode::Psllq : LirOpcode::Pslld;
                else if (inst.opcode() == Opcode::vlshr) imm_op = wide_lanes ? LirOpcode::Psrlq : LirOpcode::Psrld;
                else imm_op = wide_lanes ? LirOpcode::Psraq : LirOpcode::Psrad;
            }
            const LirOpcode reg_op =
                (inst.opcode() == Opcode::vshl)  ? (wide_lanes ? LirOpcode::Vpsllvq : LirOpcode::Vpsllvd)
              : (inst.opcode() == Opcode::vlshr) ? (wide_lanes ? LirOpcode::Vpsrlvq : LirOpcode::Vpsrlvd)
                                                 : (wide_lanes ? LirOpcode::Vpsravq : LirOpcode::Vpsravd);

            auto lower_sh_16 = [&](VReg dst, VReg v, VReg s) {
                if (use_imm && imm_amt == 0) {
                    emit_copy16(dst, v);
                } else if (use_imm) {
                    emit2_16(imm_op, dst, v, static_cast<int>(imm_amt));
                } else {
                    emit3_16(reg_op, dst, v, s);
                }
            };
            if (is_v256) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                VRegPair s = get_vreg_pair(inst.operand(1));
                lower_sh_16(dst.lo, v.lo, s.lo);
                lower_sh_16(dst.hi, v.hi, s.hi);
            } else {
                lower_sh_16(get_vreg(inst.result()), get_vreg(inst.operand(0)), get_vreg(inst.operand(1)));
            }
            break;
        }

        case Opcode::vconvert: {
            const Type to = inst.type();
            const Type ef = t0.element_type();
            const Type et = to.element_type();
            if (ef == Type::i32() && et == Type::f32()) {
                if (is_v256) {
                    VRegPair dst = get_vreg_pair(inst.result());
                    VRegPair v = get_vreg_pair(inst.operand(0));
                    emit2_16(LirOpcode::Vcvtdq2ps, dst.lo, v.lo);
                    emit2_16(LirOpcode::Vcvtdq2ps, dst.hi, v.hi);
                } else {
                    emit2_16(LirOpcode::Vcvtdq2ps, get_vreg(inst.result()), get_vreg(inst.operand(0)));
                }
            } else if (ef == Type::f32() && et == Type::i32()) {
                if (is_v256) {
                    VRegPair dst = get_vreg_pair(inst.result());
                    VRegPair v = get_vreg_pair(inst.operand(0));
                    emit2_16(LirOpcode::Vcvttps2dq, dst.lo, v.lo);
                    emit2_16(LirOpcode::Vcvttps2dq, dst.hi, v.hi);
                } else {
                    emit2_16(LirOpcode::Vcvttps2dq, get_vreg(inst.result()), get_vreg(inst.operand(0)));
                }
            } else if (ef == Type::i32() && et == Type::f64()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VReg v = get_vreg(inst.operand(0));
                emit2_16(LirOpcode::Vcvtdq2pd, dst.lo, v, 0);
                emit2_16(LirOpcode::Vcvtdq2pd, dst.hi, v, 1);
            } else if (ef == Type::f64() && et == Type::i32()) {
                VReg dst = get_vreg(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                emit3_16(LirOpcode::Vcvttpd2dq, dst, v.lo, v.hi);
            } else if (ef == Type::f32() && et == Type::f64()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VReg v = get_vreg(inst.operand(0));
                emit2_16(LirOpcode::Vcvtps2pd, dst.lo, v, 0);
                emit2_16(LirOpcode::Vcvtps2pd, dst.hi, v, 1);
            } else if (ef == Type::f64() && et == Type::f32()) {
                VReg dst = get_vreg(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                emit3_16(LirOpcode::Vcvtpd2ps, dst, v.lo, v.hi);
            } else if (ef == Type::i32() && et == Type::i64()) {
                VRegPair dst = get_vreg_pair(inst.result());
                VReg v = get_vreg(inst.operand(0));
                emit2_16(LirOpcode::Vpmovsxdq, dst.lo, v, 0);
                emit2_16(LirOpcode::Vpmovsxdq, dst.hi, v, 1);
            } else if (ef == Type::i64() && et == Type::i32() && is_v256) {
                VReg dst = get_vreg(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                emit3_16(LirOpcode::Vshufps, dst, v.lo, v.hi, 0x88);
            } else {
                unsupported(to);
            }
            break;
        }

        case Opcode::vbitcast: {
            if (is_v256) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                emit_copy16(dst.lo, v.lo);
                emit_copy16(dst.hi, v.hi);
            } else {
                emit_copy16(get_vreg(inst.result()), get_vreg(inst.operand(0)));
            }
            break;
        }

        case Opcode::vfloor:
        case Opcode::vceil:
        case Opcode::vtrunc:
        case Opcode::vround: {
            const int mode = inst.opcode() == Opcode::vfloor ? 0x9
                           : inst.opcode() == Opcode::vceil  ? 0xA
                           : inst.opcode() == Opcode::vtrunc ? 0xB : 0x8;
            const LirOpcode rop = wide_lanes ? LirOpcode::Vroundpd : LirOpcode::Vroundps;
            if (is_v256) {
                VRegPair dst = get_vreg_pair(inst.result());
                VRegPair v = get_vreg_pair(inst.operand(0));
                emit2_16(rop, dst.lo, v.lo, mode);
                emit2_16(rop, dst.hi, v.hi, mode);
            } else {
                emit2_16(rop, get_vreg(inst.result()), get_vreg(inst.operand(0)), mode);
            }
            break;
        }

        case Opcode::vany: {
            VReg dst = get_vreg(inst.result());
            const LirOpcode msk_op = wide_lanes ? LirOpcode::Vmovmskpd : LirOpcode::Vmovmskps;
            VReg bits = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            if (is_v256) {
                VRegPair vp = get_vreg_pair(inst.operand(0));
                VReg b_lo = lir_fn_->allocate_vreg(RegClass::GPR, 4);
                VReg b_hi = lir_fn_->allocate_vreg(RegClass::GPR, 4);

                auto m0 = std::make_unique<LirInst>(msk_op);
                m0->add_def(LirOperand::vreg(b_lo, 4));
                m0->add_use(LirOperand::vreg(vp.lo, 16));
                lir_bb.append_inst(std::move(m0));

                auto m1 = std::make_unique<LirInst>(msk_op);
                m1->add_def(LirOperand::vreg(b_hi, 4));
                m1->add_use(LirOperand::vreg(vp.hi, 16));
                lir_bb.append_inst(std::move(m1));

                auto or_inst = std::make_unique<LirInst>(LirOpcode::Or32);
                or_inst->add_def(LirOperand::vreg(bits, 4));
                or_inst->add_use(LirOperand::vreg(b_lo, 4));
                or_inst->add_use(LirOperand::vreg(b_hi, 4));
                lir_bb.append_inst(std::move(or_inst));
            } else {
                auto m = std::make_unique<LirInst>(msk_op);
                m->add_def(LirOperand::vreg(bits, 4));
                m->add_use(LirOperand::vreg(get_vreg(inst.operand(0)), 16));
                lir_bb.append_inst(std::move(m));
            }

            auto test = std::make_unique<LirInst>(LirOpcode::Test32);
            test->add_use(LirOperand::vreg(bits, 4));
            test->add_use(LirOperand::vreg(bits, 4));
            lir_bb.append_inst(std::move(test));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = brass::x64::Condition::NE;
            setcc->add_def(LirOperand::vreg(dst, 4));
            setcc->mir_origin = &inst;
            lir_bb.append_inst(std::move(setcc));
            break;
        }

        default:
            unsupported(t0);
    }
}

} // namespace brass::aarch64
