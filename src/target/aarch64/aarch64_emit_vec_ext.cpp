#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64EmitContext::emit_vec_lane_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Vpcmpeqb:
            enc_.vec_cmeq_16b(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        case LirOpcode::Vpcmpeqw:
            enc_.vec_cmeq_8h(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        case LirOpcode::Vpcmpgtb:
            enc_.vec_cmgt_16b(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        case LirOpcode::Vpcmpgtw:
            enc_.vec_cmgt_8h(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        case LirOpcode::Vpcmpeqd: {
            enc_.vec_cmeq_4s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }
        case LirOpcode::Vpcmpeqq: {
            enc_.vec_cmeq_2d(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }
        case LirOpcode::Vpcmpgtd: {
            enc_.vec_cmgt_4s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }
        case LirOpcode::Vpcmpgtq: {
            enc_.vec_cmgt_2d(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }
        case LirOpcode::Vpminsd: {
            enc_.vec_smin_4s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }
        case LirOpcode::Vpmaxsd: {
            enc_.vec_smax_4s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses[1]));
            break;
        }

        case LirOpcode::Vcmpps:
        case LirOpcode::Vcmppd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR a = to_fpr(inst.uses[0]);
            FPR b = to_fpr(inst.uses[1]);
            int pred = static_cast<int>(inst.uses[2].imm_int);
            bool d = (inst.opcode == LirOpcode::Vcmppd);
            switch (pred) {
                case 0x00: d ? enc_.vec_fcmeq_2d(dst, a, b) : enc_.vec_fcmeq_4s(dst, a, b); break;
                case 0x04:
                    d ? enc_.vec_fcmeq_2d(dst, a, b) : enc_.vec_fcmeq_4s(dst, a, b);
                    enc_.vec_not(dst, dst);
                    break;
                case 0x11: d ? enc_.vec_fcmgt_2d(dst, b, a) : enc_.vec_fcmgt_4s(dst, b, a); break;
                case 0x12: d ? enc_.vec_fcmge_2d(dst, b, a) : enc_.vec_fcmge_4s(dst, b, a); break;
                case 0x1E: d ? enc_.vec_fcmgt_2d(dst, a, b) : enc_.vec_fcmgt_4s(dst, a, b); break;
                case 0x1D: d ? enc_.vec_fcmge_2d(dst, a, b) : enc_.vec_fcmge_4s(dst, a, b); break;
                default: enc_.movi_zero(dst); break;
            }
            break;
        }

        case LirOpcode::Vpblendvb:
        case LirOpcode::Vblendvw:
        case LirOpcode::Vblendvps:
        case LirOpcode::Vblendvpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR b = to_fpr(inst.uses[0]);
            FPR a = to_fpr(inst.uses[1]);
            FPR mask = to_fpr(inst.uses[2]);
            if (inst.opcode == LirOpcode::Vpblendvb) enc_.vec_cmlt_zero_16b(FPR::V30, mask);
            else if (inst.opcode == LirOpcode::Vblendvw) enc_.vec_cmlt_zero_8h(FPR::V30, mask);
            else if (inst.opcode == LirOpcode::Vblendvpd) enc_.vec_cmlt_zero_2d(FPR::V30, mask);
            else enc_.vec_cmlt_zero_4s(FPR::V30, mask);
            enc_.vec_bsl(FPR::V30, a, b);
            enc_.vec_orr(dst, FPR::V30, FPR::V30);
            break;
        }

        case LirOpcode::Psllb:
        case LirOpcode::Psrlb:
        case LirOpcode::Psrab: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (shift == 0) {
                if (dst != src) enc_.vec_orr(dst, src, src);
            } else if (shift >= 8u) {
                if (inst.opcode == LirOpcode::Psrab) enc_.vec_sshr_imm_16b(dst, src, 8u);
                else enc_.movi_zero(dst);
            } else if (inst.opcode == LirOpcode::Psllb) {
                enc_.vec_shl_imm_16b(dst, src, static_cast<uint8_t>(shift));
            } else if (inst.opcode == LirOpcode::Psrlb) {
                enc_.vec_ushr_imm_16b(dst, src, static_cast<uint8_t>(shift));
            } else {
                enc_.vec_sshr_imm_16b(dst, src, static_cast<uint8_t>(shift));
            }
            break;
        }

        case LirOpcode::Psllw:
        case LirOpcode::Psrlw:
        case LirOpcode::Psraw: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (shift == 0) {
                if (dst != src) enc_.vec_orr(dst, src, src);
            } else if (shift >= 16u) {
                if (inst.opcode == LirOpcode::Psraw) enc_.vec_sshr_imm_8h(dst, src, 16u);
                else enc_.movi_zero(dst);
            } else if (inst.opcode == LirOpcode::Psllw) {
                enc_.vec_shl_imm_8h(dst, src, static_cast<uint8_t>(shift));
            } else if (inst.opcode == LirOpcode::Psrlw) {
                enc_.vec_ushr_imm_8h(dst, src, static_cast<uint8_t>(shift));
            } else {
                enc_.vec_sshr_imm_8h(dst, src, static_cast<uint8_t>(shift));
            }
            break;
        }

        case LirOpcode::Psrld:
        case LirOpcode::Psrad: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (shift == 0) {
                if (dst != src) enc_.vec_orr(dst, src, src);
            } else if (shift >= 32u) {
                if (inst.opcode == LirOpcode::Psrld) enc_.movi_zero(dst);
                else enc_.vec_sshr_imm_4s(dst, src, 32u);
            } else {
                if (inst.opcode == LirOpcode::Psrld) enc_.vec_ushr_imm_4s(dst, src, static_cast<uint8_t>(shift));
                else enc_.vec_sshr_imm_4s(dst, src, static_cast<uint8_t>(shift));
            }
            break;
        }

        case LirOpcode::Psrlq:
        case LirOpcode::Psraq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (shift == 0) {
                if (dst != src) enc_.vec_orr(dst, src, src);
            } else if (shift >= 64u) {
                if (inst.opcode == LirOpcode::Psrlq) enc_.movi_zero(dst);
                else enc_.vec_sshr_imm_2d(dst, src, 64u);
            } else {
                if (inst.opcode == LirOpcode::Psrlq) enc_.vec_ushr_imm_2d(dst, src, static_cast<uint8_t>(shift));
                else enc_.vec_sshr_imm_2d(dst, src, static_cast<uint8_t>(shift));
            }
            break;
        }

        case LirOpcode::Vpsllvb:
        case LirOpcode::Vpsrlvb:
        case LirOpcode::Vpsravb: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR val = to_fpr(inst.uses[0]);
            FPR amt = to_fpr(inst.uses[1]);
            bool is_arith = (inst.opcode == LirOpcode::Vpsravb);
            enc_.vec_movi_16b(FPR::V30, is_arith ? 7u : 8u);
            enc_.vec_umin_16b(FPR::V30, amt, FPR::V30);
            if (inst.opcode == LirOpcode::Vpsllvb) {
                enc_.vec_ushl_16b(dst, val, FPR::V30);
            } else {
                enc_.vec_neg_16b(FPR::V30, FPR::V30);
                if (is_arith) enc_.vec_sshl_16b(dst, val, FPR::V30);
                else enc_.vec_ushl_16b(dst, val, FPR::V30);
            }
            break;
        }

        case LirOpcode::Vpsllvw:
        case LirOpcode::Vpsrlvw:
        case LirOpcode::Vpsravw: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR val = to_fpr(inst.uses[0]);
            FPR amt = to_fpr(inst.uses[1]);
            bool is_arith = (inst.opcode == LirOpcode::Vpsravw);
            enc_.vec_movi_8h(FPR::V30, is_arith ? 15u : 16u, 0);
            enc_.vec_umin_8h(FPR::V30, amt, FPR::V30);
            if (inst.opcode == LirOpcode::Vpsllvw) {
                enc_.vec_ushl_8h(dst, val, FPR::V30);
            } else {
                enc_.vec_neg_8h(FPR::V30, FPR::V30);
                if (is_arith) enc_.vec_sshl_8h(dst, val, FPR::V30);
                else enc_.vec_ushl_8h(dst, val, FPR::V30);
            }
            break;
        }

        case LirOpcode::Vpsllvd:
        case LirOpcode::Vpsrlvd:
        case LirOpcode::Vpsravd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR val = to_fpr(inst.uses[0]);
            FPR amt = to_fpr(inst.uses[1]);
            bool is_arith = (inst.opcode == LirOpcode::Vpsravd);
            enc_.vec_movi_4s(FPR::V30, is_arith ? 31u : 32u, 0);
            enc_.vec_umin_4s(FPR::V30, amt, FPR::V30);
            if (inst.opcode == LirOpcode::Vpsllvd) {
                enc_.vec_ushl_4s(dst, val, FPR::V30);
            } else {
                enc_.vec_neg_4s(FPR::V30, FPR::V30);
                if (is_arith) enc_.vec_sshl_4s(dst, val, FPR::V30);
                else enc_.vec_ushl_4s(dst, val, FPR::V30);
            }
            break;
        }

        case LirOpcode::Vpsllvq:
        case LirOpcode::Vpsrlvq:
        case LirOpcode::Vpsravq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR val = to_fpr(inst.uses[0]);
            FPR amt = to_fpr(inst.uses[1]);
            bool is_arith = (inst.opcode == LirOpcode::Vpsravq);
            enc_.mov(GPR::X16, uint64_t{is_arith ? 63u : 64u});
            enc_.vec_dup_2d(FPR::V31, GPR::X16);
            enc_.vec_cmhs_2d(FPR::V30, amt, FPR::V31);
            enc_.vec_bsl(FPR::V30, FPR::V31, amt);
            if (inst.opcode == LirOpcode::Vpsllvq) {
                enc_.vec_ushl_2d(dst, val, FPR::V30);
            } else {
                enc_.vec_neg_2d(FPR::V30, FPR::V30);
                if (is_arith) enc_.vec_sshl_2d(dst, val, FPR::V30);
                else enc_.vec_ushl_2d(dst, val, FPR::V30);
            }
            break;
        }

        case LirOpcode::Vroundps:
        case LirOpcode::Vroundpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            int mode = static_cast<int>(inst.uses[1].imm_int) & 3;
            bool d = (inst.opcode == LirOpcode::Vroundpd);
            switch (mode) {
                case 1:  d ? enc_.vec_frintm_2d(dst, src) : enc_.vec_frintm_4s(dst, src); break;
                case 2:  d ? enc_.vec_frintp_2d(dst, src) : enc_.vec_frintp_4s(dst, src); break;
                case 3:  d ? enc_.vec_frintz_2d(dst, src) : enc_.vec_frintz_4s(dst, src); break;
                default: d ? enc_.vec_frintn_2d(dst, src) : enc_.vec_frintn_4s(dst, src); break;
            }
            break;
        }

        case LirOpcode::Vcvtdq2ps: {
            enc_.vec_scvtf_4s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]));
            break;
        }

        case LirOpcode::Vcvttps2dq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            enc_.vec_movi_4s(FPR::V30, 0x4Fu, 24);
            enc_.vec_fcmgt_4s(FPR::V30, FPR::V30, src);
            enc_.vec_fcvtzs_4s(dst, src);
            enc_.vec_movi_4s(FPR::V31, 0x80u, 24);
            enc_.vec_bif(dst, FPR::V31, FPR::V30);
            break;
        }

        case LirOpcode::Vcvtdq2pd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            bool hi_half = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1));
            if (hi_half) enc_.vec_sxtl2_2d(dst, src);
            else enc_.vec_sxtl_2d(dst, src);
            enc_.vec_scvtf_2d(dst, dst);
            break;
        }

        case LirOpcode::Vcvtps2pd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            bool hi_half = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1));
            if (hi_half) enc_.vec_fcvtl2_2d(dst, src);
            else enc_.vec_fcvtl_2d(dst, src);
            break;
        }

        case LirOpcode::Vpmovsxdq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            bool hi_half = (inst.uses.size() >= 2 && inst.uses[1].is_imm_int() && (inst.uses[1].imm_int & 1));
            if (hi_half) enc_.vec_sxtl2_2d(dst, src);
            else enc_.vec_sxtl_2d(dst, src);
            break;
        }

        case LirOpcode::Vcvttpd2dq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src0 = to_fpr(inst.uses[0]);
            FPR src1 = to_fpr(inst.uses[1]);
            // V30 = 4x32 in-range mask (true iff +2^31 > lane and not NaN).
            enc_.movz(GPR::X16, 0x41E0u, 3);
            enc_.vec_dup_2d(FPR::V31, GPR::X16);
            enc_.vec_fcmgt_2d(FPR::V30, FPR::V31, src0);
            enc_.vec_xtn_2s(FPR::V30, FPR::V30);
            enc_.vec_fcmgt_2d(FPR::V31, FPR::V31, src1);
            enc_.vec_xtn2_4s(FPR::V30, FPR::V31);

            if (src0 == src1) {
                enc_.vec_fcvtzs_2d(FPR::V31, src0);
                enc_.vec_sqxtn_2s(dst, FPR::V31);
                enc_.vec_sqxtn2_4s(dst, FPR::V31);
            } else if (dst != src1) {
                enc_.vec_fcvtzs_2d(FPR::V31, src0);
                enc_.vec_sqxtn_2s(dst, FPR::V31);
                enc_.vec_fcvtzs_2d(FPR::V31, src1);
                enc_.vec_sqxtn2_4s(dst, FPR::V31);
            } else {
                enc_.vec_fcvtzs_2d(dst, src1);
                enc_.vec_fcvtzs_2d(FPR::V31, src0);
                enc_.vec_sqxtn_2s(FPR::V31, FPR::V31);
                enc_.vec_sqxtn2_4s(FPR::V31, dst);
                enc_.vec_orr(dst, FPR::V31, FPR::V31);
            }
            enc_.vec_movi_4s(FPR::V31, 0x80u, 24);
            enc_.vec_bif(dst, FPR::V31, FPR::V30);
            break;
        }

        case LirOpcode::Vcvtpd2ps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src0 = to_fpr(inst.uses[0]);
            FPR src1 = to_fpr(inst.uses[1]);
            enc_.vec_fcvtn_2s(FPR::V31, src0);
            enc_.vec_fcvtn2_4s(FPR::V31, src1);
            enc_.vec_orr(dst, FPR::V31, FPR::V31);
            break;
        }

        case LirOpcode::Vshufps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR v0 = to_fpr(inst.uses[0]);
            FPR v1 = to_fpr(inst.uses[1]);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            if (mask == 0x88u) {
                enc_.vec_xtn_2s(FPR::V31, v0);
                enc_.vec_xtn2_4s(FPR::V31, v1);
                enc_.vec_orr(dst, FPR::V31, FPR::V31);
                break;
            }
            uint32_t l0 = mask & 3u;
            uint32_t l1 = (mask >> 2) & 3u;
            uint32_t l2 = (mask >> 4) & 3u;
            uint32_t l3 = (mask >> 6) & 3u;
            if (v0 == v1 && l0 == l1 && l1 == l2 && l2 == l3) {
                buffer_.emit_inst(0x4E040400u | (l0 << 19) | (reg_code(v0) << 5) | reg_code(dst));
            } else {
                FPR tmp = FPR::V31;
                buffer_.emit_inst(0x6E040400u | (0u << 19) | (l0 << 13) | (reg_code(v0) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (1u << 19) | (l1 << 13) | (reg_code(v0) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (2u << 19) | (l2 << 13) | (reg_code(v1) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (3u << 19) | (l3 << 13) | (reg_code(v1) << 5) | reg_code(tmp));
                enc_.vec_orr(dst, tmp, tmp);
            }
            break;
        }

        case LirOpcode::Vmovmskps: {
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            enc_.fmov_to_gpr(GPR::X16, src);
            enc_.vec_dup_d1(FPR::V31, src);
            enc_.fmov_to_gpr(GPR::X17, FPR::V31);

            enc_.lsr(GPR::X16, GPR::X16, 31);
            enc_.and32_imm(dst, GPR::X16, 1u);
            enc_.lsr(GPR::X16, GPR::X16, 31);
            enc_.and32_imm(GPR::X16, GPR::X16, 2u);
            enc_.orr32(dst, dst, GPR::X16);

            enc_.lsr(GPR::X17, GPR::X17, 29);
            enc_.and32_imm(GPR::X16, GPR::X17, 4u);
            enc_.orr32(dst, dst, GPR::X16);
            enc_.lsr(GPR::X17, GPR::X17, 31);
            enc_.and32_imm(GPR::X17, GPR::X17, 8u);
            enc_.orr32(dst, dst, GPR::X17);
            break;
        }

        case LirOpcode::Vmovmskpd: {
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            enc_.fmov_to_gpr(GPR::X16, src);
            enc_.vec_dup_d1(FPR::V31, src);
            enc_.fmov_to_gpr(GPR::X17, FPR::V31);
            enc_.lsr(GPR::X16, GPR::X16, 63);
            enc_.lsr(GPR::X17, GPR::X17, 62);
            enc_.and32_imm(GPR::X17, GPR::X17, 2u);
            enc_.orr32(dst, GPR::X16, GPR::X17);
            break;
        }

        case LirOpcode::Vany8: {
            GPR dst = to_gpr(inst.defs[0]);
            enc_.vec_cmlt_zero_16b(FPR::V30, to_fpr(inst.uses[0]));
            enc_.vec_umaxv_16b(FPR::V30, FPR::V30);
            enc_.fmov_to_gpr32(dst, FPR::V30);
            enc_.lsr32(dst, dst, 7);
            break;
        }

        case LirOpcode::Vany16: {
            GPR dst = to_gpr(inst.defs[0]);
            enc_.vec_cmlt_zero_8h(FPR::V30, to_fpr(inst.uses[0]));
            enc_.vec_umaxv_8h(FPR::V30, FPR::V30);
            enc_.fmov_to_gpr32(dst, FPR::V30);
            enc_.lsr32(dst, dst, 15);
            break;
        }

        default:
            codegen::throw_unsupported("aarch64 emit (vector lanes)", codegen::to_string(inst.opcode));
    }
}

} // namespace brass::aarch64
