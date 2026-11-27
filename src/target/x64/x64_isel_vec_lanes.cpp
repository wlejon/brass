// x64 lowering of the lane compares, selects, shifts, conversions, bitcasts,
// roundings and vany (opcodes.hpp), to the VEX forms of lir.hpp at the
// vector's own width.
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_vector_lanes(const Instruction& inst, LirBlock& lir_bb) {
    auto unsupported = [&](Type t) {
        codegen::throw_unsupported("x64 isel (vector)",
                                   std::string(opcode_name(inst.opcode())) + " " + brass::to_string(t));
    };
    auto vreg_xmm = [&](uint8_t sz) { return lir_fn_->allocate_vreg(RegClass::XMM, sz); };
    // dst = op(a, b[, imm]) at `sz` bytes.
    auto emit3 = [&](LirOpcode op, VReg dst, VReg a, VReg b, uint8_t sz, int imm = -1) {
        auto i = std::make_unique<LirInst>(op);
        i->add_def(LirOperand::vreg(dst, sz));
        i->add_use(LirOperand::vreg(a, sz));
        i->add_use(LirOperand::vreg(b, sz));
        if (imm >= 0) i->add_use(LirOperand::imm(imm, 1));
        i->mir_origin = &inst;
        lir_bb.append_inst(std::move(i));
    };
    // dst = op(src[, imm]), the two widths given apart for the conversions.
    auto emit2 = [&](LirOpcode op, VReg dst, uint8_t dsz, VReg src, uint8_t ssz, int imm = -1) {
        auto i = std::make_unique<LirInst>(op);
        i->add_def(LirOperand::vreg(dst, dsz));
        i->add_use(LirOperand::vreg(src, ssz));
        if (imm >= 0) i->add_use(LirOperand::imm(imm, 1));
        i->mir_origin = &inst;
        lir_bb.append_inst(std::move(i));
    };
    auto emit_copy = [&](VReg dst, VReg src, uint8_t sz) {
        if (dst == src) return;
        auto mov = std::make_unique<LirInst>(sz == 32 ? LirOpcode::Vmovaps : LirOpcode::Movaps);
        mov->add_def(LirOperand::vreg(dst, sz));
        mov->add_use(LirOperand::vreg(src, sz));
        lir_bb.append_inst(std::move(mov));
    };
    // dst = a ^ b: VEX.256 vpxor, or SSE pxor on a copy of a.
    auto emit_xor = [&](VReg dst, VReg a, VReg b, uint8_t sz) {
        if (sz == 32) {
            emit3(LirOpcode::Vpxor, dst, a, b, 32);
            return;
        }
        emit_copy(dst, a, 16);
        auto x = std::make_unique<LirInst>(LirOpcode::Pxor);
        x->add_def(LirOperand::vreg(dst, 16));
        x->add_use(LirOperand::vreg(dst, 16));
        x->add_use(LirOperand::vreg(b, 16));
        x->mir_origin = &inst;
        lir_bb.append_inst(std::move(x));
    };
    // All ones at `sz` bytes (vpcmpeqd x, x, x).
    auto ones = [&](uint8_t sz) {
        VReg m = vreg_xmm(sz);
        emit3(LirOpcode::Vpcmpeqd, m, m, m, sz);
        return m;
    };

    const Type t0 = inst.operand(0)->type();
    const uint8_t sz0 = static_cast<uint8_t>(t0.size_in_bytes());
    const size_t elem_sz = t0.element_type().size_in_bytes();
    const bool wide_lanes = elem_sz == 8;

    switch (inst.opcode()) {
        case Opcode::vcmp_eq: case Opcode::vcmp_ne: case Opcode::vcmp_lt:
        case Opcode::vcmp_le: case Opcode::vcmp_gt: case Opcode::vcmp_ge: {
            VReg dst = get_vreg(inst.result());
            VReg a = get_vreg(inst.operand(0));
            VReg b = get_vreg(inst.operand(1));
            if (t0.element_type().is_float()) {
                // Ordered and signalling-free predicates, except NEQ_UQ.
                int pred = 0;
                switch (inst.opcode()) {
                    case Opcode::vcmp_eq: pred = 0x00; break;   // EQ_OQ
                    case Opcode::vcmp_ne: pred = 0x04; break;   // NEQ_UQ
                    case Opcode::vcmp_lt: pred = 0x11; break;   // LT_OQ
                    case Opcode::vcmp_le: pred = 0x12; break;   // LE_OQ
                    case Opcode::vcmp_gt: pred = 0x1E; break;   // GT_OQ
                    default:              pred = 0x1D; break;   // GE_OQ
                }
                emit3(wide_lanes ? LirOpcode::Vcmppd : LirOpcode::Vcmpps, dst, a, b, sz0, pred);
                break;
            }
            const LirOpcode eq = (elem_sz == 1) ? LirOpcode::Vpcmpeqb
                               : (elem_sz == 2) ? LirOpcode::Vpcmpeqw
                               : wide_lanes ? LirOpcode::Vpcmpeqq : LirOpcode::Vpcmpeqd;
            const LirOpcode gt = (elem_sz == 1) ? LirOpcode::Vpcmpgtb
                               : (elem_sz == 2) ? LirOpcode::Vpcmpgtw
                               : wide_lanes ? LirOpcode::Vpcmpgtq : LirOpcode::Vpcmpgtd;
            switch (inst.opcode()) {
                case Opcode::vcmp_eq: emit3(eq, dst, a, b, sz0); break;
                case Opcode::vcmp_gt: emit3(gt, dst, a, b, sz0); break;
                case Opcode::vcmp_lt: emit3(gt, dst, b, a, sz0); break;
                default: {   // the complements: ne = !eq, le = !gt, ge = !lt
                    VReg r = vreg_xmm(sz0);
                    if (inst.opcode() == Opcode::vcmp_ne) emit3(eq, r, a, b, sz0);
                    else if (inst.opcode() == Opcode::vcmp_le) emit3(gt, r, a, b, sz0);
                    else emit3(gt, r, b, a, sz0);
                    emit_xor(dst, r, ones(sz0), sz0);
                    break;
                }
            }
            break;
        }

        case Opcode::vselect: {
            VReg dst = get_vreg(inst.result());
            VReg mask = get_vreg(inst.operand(0));
            VReg a = get_vreg(inst.operand(1));
            VReg b = get_vreg(inst.operand(2));
            const Type t = inst.type();
            const uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
            const size_t esz = t.element_type().size_in_bytes();
            if (esz <= 2) {
                VReg bmask = mask;
                if (esz == 2) {
                    bmask = vreg_xmm(16);
                    emit_copy(bmask, mask, 16);
                    auto s = std::make_unique<LirInst>(LirOpcode::Psraw);
                    s->add_def(LirOperand::vreg(bmask, 16));
                    s->add_use(LirOperand::vreg(bmask, 16));
                    s->add_use(LirOperand::imm(15, 1));
                    lir_bb.append_inst(std::move(s));
                }
                auto i = std::make_unique<LirInst>(LirOpcode::Vpblendvb);
                i->add_def(LirOperand::vreg(dst, 16));
                i->add_use(LirOperand::vreg(b, 16));
                i->add_use(LirOperand::vreg(a, 16));
                i->add_use(LirOperand::vreg(bmask, 16));
                i->mir_origin = &inst;
                lir_bb.append_inst(std::move(i));
                break;
            }
            auto i = std::make_unique<LirInst>(esz == 8 ? LirOpcode::Vblendvpd : LirOpcode::Vblendvps);
            i->add_def(LirOperand::vreg(dst, sz));
            i->add_use(LirOperand::vreg(b, sz));
            i->add_use(LirOperand::vreg(a, sz));
            i->add_use(LirOperand::vreg(mask, sz));
            i->mir_origin = &inst;
            lir_bb.append_inst(std::move(i));
            break;
        }

        case Opcode::vshl: case Opcode::vlshr: case Opcode::vashr: {
            VReg dst = get_vreg(inst.result());
            VReg v = get_vreg(inst.operand(0));
            VReg s = get_vreg(inst.operand(1));
            auto emit_two_imm = [&](LirOpcode sop, VReg r, int imm) {
                auto i = std::make_unique<LirInst>(sop);
                i->add_def(LirOperand::vreg(r, 16));
                i->add_use(LirOperand::vreg(r, 16));
                i->add_use(LirOperand::imm(imm, 1));
                lir_bb.append_inst(std::move(i));
            };
            auto shift_chunk4 = [&](VReg v_chunk, VReg s_chunk, bool is_byte) {
                LirOpcode vext = (inst.opcode() == Opcode::vashr)
                    ? (is_byte ? LirOpcode::Vpmovsxbd : LirOpcode::Vpmovsxwd)
                    : (is_byte ? LirOpcode::Vpmovzxbd : LirOpcode::Vpmovzxwd);
                LirOpcode sext = is_byte ? LirOpcode::Vpmovzxbd : LirOpcode::Vpmovzxwd;
                LirOpcode sh_op = inst.opcode() == Opcode::vshl ? LirOpcode::Vpsllvd
                                : inst.opcode() == Opcode::vlshr ? LirOpcode::Vpsrlvd : LirOpcode::Vpsravd;
                VReg vw = vreg_xmm(16), sw = vreg_xmm(16), sh = vreg_xmm(16);
                emit2(vext, vw, 16, v_chunk, 16);
                emit2(sext, sw, 16, s_chunk, 16);
                emit3(sh_op, sh, vw, sw, 16);
                if (inst.opcode() != Opcode::vashr) {
                    const int sbits = is_byte ? 24 : 16;
                    emit_two_imm(LirOpcode::Pslld, sh, sbits);
                    emit_two_imm(LirOpcode::Psrad, sh, sbits);
                }
                return sh;
            };
            if (elem_sz == 2) {
                VReg v_hi = vreg_xmm(16), s_hi = vreg_xmm(16);
                emit3(LirOpcode::Pshufd, v_hi, v, v, 16, 0x4E);
                emit3(LirOpcode::Pshufd, s_hi, s, s, 16, 0x4E);
                emit3(LirOpcode::Vpackssdw, dst, shift_chunk4(v, s, false), shift_chunk4(v_hi, s_hi, false), 16);
                break;
            }
            if (elem_sz == 1) {
                const int perms[4] = {0x00, 0x55, 0xAA, 0xFF};
                VReg q[4];
                for (int k = 0; k < 4; ++k) {
                    VReg vc = (k == 0) ? v : vreg_xmm(16);
                    VReg sc = (k == 0) ? s : vreg_xmm(16);
                    if (k > 0) {
                        emit3(LirOpcode::Pshufd, vc, v, v, 16, perms[k]);
                        emit3(LirOpcode::Pshufd, sc, s, s, 16, perms[k]);
                    }
                    q[k] = shift_chunk4(vc, sc, true);
                }
                VReg lo8 = vreg_xmm(16), hi8 = vreg_xmm(16);
                emit3(LirOpcode::Vpackssdw, lo8, q[0], q[1], 16);
                emit3(LirOpcode::Vpackssdw, hi8, q[2], q[3], 16);
                emit3(LirOpcode::Vpacksswb, dst, lo8, hi8, 16);
                break;
            }
            if (!wide_lanes) {
                const LirOpcode op = inst.opcode() == Opcode::vshl ? LirOpcode::Vpsllvd
                                   : inst.opcode() == Opcode::vlshr ? LirOpcode::Vpsrlvd : LirOpcode::Vpsravd;
                emit3(op, dst, v, s, sz0);
                break;
            }
            if (inst.opcode() != Opcode::vashr) {
                emit3(inst.opcode() == Opcode::vshl ? LirOpcode::Vpsllvq : LirOpcode::Vpsrlvq, dst, v, s, sz0);
                break;
            }
            // No vpsravq before AVX-512: ((v ^ sign) >>> s) ^ sign, the sign
            // lanes all ones where v < 0 (0 > v), which an amount past 63
            // leaves as the result.
            VReg zero = vreg_xmm(sz0);
            emit3(LirOpcode::Vpcmpgtq, zero, v, v, sz0);
            VReg sign = vreg_xmm(sz0);
            emit3(LirOpcode::Vpcmpgtq, sign, zero, v, sz0);
            VReg flipped = vreg_xmm(sz0);
            emit_xor(flipped, v, sign, sz0);
            VReg shifted = vreg_xmm(sz0);
            emit3(LirOpcode::Vpsrlvq, shifted, flipped, s, sz0);
            emit_xor(dst, shifted, sign, sz0);
            break;
        }

        case Opcode::vconvert: {
            VReg dst = get_vreg(inst.result());
            VReg v = get_vreg(inst.operand(0));
            const Type to = inst.type();
            const uint8_t dsz = static_cast<uint8_t>(to.size_in_bytes());
            const Type ef = t0.element_type();
            const Type et = to.element_type();
            if (ef == Type::i32() && et == Type::f32()) emit2(LirOpcode::Vcvtdq2ps, dst, dsz, v, sz0);
            else if (ef == Type::f32() && et == Type::i32()) emit2(LirOpcode::Vcvttps2dq, dst, dsz, v, sz0);
            else if (ef == Type::i32() && et == Type::f64()) emit2(LirOpcode::Vcvtdq2pd, dst, dsz, v, sz0);
            else if (ef == Type::f64() && et == Type::i32()) emit2(LirOpcode::Vcvttpd2dq, dst, dsz, v, sz0);
            else if (ef == Type::f32() && et == Type::f64()) emit2(LirOpcode::Vcvtps2pd, dst, dsz, v, sz0);
            else if (ef == Type::f64() && et == Type::f32()) emit2(LirOpcode::Vcvtpd2ps, dst, dsz, v, sz0);
            else if (ef == Type::i32() && et == Type::i64()) emit2(LirOpcode::Vpmovsxdq, dst, dsz, v, sz0);
            else if (ef == Type::i64() && et == Type::i32() && t0.is_v256()) {
                // The low half of each lane: lanes 0 and 2 of each 128-bit half.
                VReg hi = vreg_xmm(16);
                auto ex = std::make_unique<LirInst>(LirOpcode::Vextractf128);
                ex->add_def(LirOperand::vreg(hi, 16));
                ex->add_use(LirOperand::vreg(v, 32));
                ex->add_use(LirOperand::imm(1, 1));
                lir_bb.append_inst(std::move(ex));
                emit3(LirOpcode::Vshufps, dst, v, hi, 16, 0x88);
            } else {
                unsupported(to);
            }
            break;
        }

        case Opcode::vbitcast: {
            VReg dst = get_vreg(inst.result());
            VReg v = get_vreg(inst.operand(0));
            emit_copy(dst, v, sz0);
            break;
        }

        case Opcode::vfloor: case Opcode::vceil: case Opcode::vtrunc: case Opcode::vround: {
            VReg dst = get_vreg(inst.result());
            VReg v = get_vreg(inst.operand(0));
            // The rounding control, with bit 3 suppressing the precision exception.
            const int mode = inst.opcode() == Opcode::vfloor ? 0x9
                           : inst.opcode() == Opcode::vceil ? 0xA
                           : inst.opcode() == Opcode::vtrunc ? 0xB : 0x8;
            emit2(wide_lanes ? LirOpcode::Vroundpd : LirOpcode::Vroundps, dst, sz0, v, sz0, mode);
            break;
        }

        case Opcode::vany: {
            VReg dst = get_vreg(inst.result());
            VReg v = get_vreg(inst.operand(0));
            VReg bits = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            const LirOpcode mop = (elem_sz <= 2) ? LirOpcode::Vpmovmskb
                                : (wide_lanes ? LirOpcode::Vmovmskpd : LirOpcode::Vmovmskps);
            auto msk = std::make_unique<LirInst>(mop);
            msk->add_def(LirOperand::vreg(bits, 4));
            msk->add_use(LirOperand::vreg(v, sz0));
            lir_bb.append_inst(std::move(msk));

            if (elem_sz == 2) {
                auto and_msk = std::make_unique<LirInst>(LirOpcode::And32);
                and_msk->add_def(LirOperand::vreg(bits, 4));
                and_msk->add_use(LirOperand::vreg(bits, 4));
                and_msk->add_use(LirOperand::imm(0xAAAA, 4));
                lir_bb.append_inst(std::move(and_msk));
            }

            auto test = std::make_unique<LirInst>(LirOpcode::Test32);
            test->add_use(LirOperand::vreg(bits, 4));
            test->add_use(LirOperand::vreg(bits, 4));
            lir_bb.append_inst(std::move(test));

            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = Condition::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, 4));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
            break;
        }

        default:
            unsupported(t0);
    }
}

} // namespace brass::x64
