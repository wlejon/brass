#include <brass/target/aarch64/aarch64_emit.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64EmitContext::emit_vec_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Movaps:
        case LirOpcode::Movups:
        case LirOpcode::Vmovaps:
        case LirOpcode::Vmovups: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;

            if (dst.is_preg()) {
                FPR dst_f = dst.preg_val.as_aarch64_fpr();
                if (src.is_preg()) enc_.vec_orr(dst_f, src.preg_val.as_aarch64_fpr(), src.preg_val.as_aarch64_fpr());
                else enc_.ldr_q(dst_f, ensure_accessible_mem(to_mem_address(src)));
            } else {
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
                if (src.is_preg()) enc_.str_q(src.preg_val.as_aarch64_fpr(), dst_mem);
                else {
                    enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                    enc_.str_q(FPR::V31, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Movd_xg: {
            FPR dst = to_fpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            if (inst.defs[0].size == 8 || inst.uses[0].size == 8) enc_.fmov_from_gpr(dst, src);
            else enc_.fmov_from_gpr32(dst, src);
            break;
        }

        case LirOpcode::Movd_gx: {
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            if (inst.defs[0].size == 8 || inst.uses[0].size == 8) enc_.fmov_to_gpr(dst, src);
            else enc_.fmov_to_gpr32(dst, src);
            break;
        }

        case LirOpcode::Addps:
        case LirOpcode::Vaddps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fadd_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fadd_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Paddd:
        case LirOpcode::Vpaddd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_add_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_add_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Subps:
        case LirOpcode::Vsubps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fsub_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fsub_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Psubd:
        case LirOpcode::Vpsubd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_sub_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_sub_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Mulps:
        case LirOpcode::Vmulps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmul_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmul_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pmulld:
        case LirOpcode::Vpmulld: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_mul_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_mul_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Divps:
        case LirOpcode::Vdivps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fdiv_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fdiv_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Minps:
        case LirOpcode::Vminps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmin_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmin_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pminsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_smin_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_smin_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Maxps:
        case LirOpcode::Vmaxps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmax_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmax_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pmaxsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_smax_4s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_smax_4s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Sqrtps: {
            FPR dst = to_fpr(inst.defs[0]);
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.vec_fsqrt_4s(dst, to_fpr(src));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                enc_.vec_fsqrt_4s(dst, FPR::V31);
            }
            break;
        }

        case LirOpcode::Fneg4s: {
            FPR dst = to_fpr(inst.defs[0]);
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.vec_fneg_4s(dst, to_fpr(src));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                enc_.vec_fneg_4s(dst, FPR::V31);
            }
            break;
        }

        case LirOpcode::Addpd:
        case LirOpcode::Vaddpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fadd_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fadd_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Paddq:
        case LirOpcode::Vpaddq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_add_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_add_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Subpd:
        case LirOpcode::Vsubpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fsub_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fsub_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Psubq:
        case LirOpcode::Vpsubq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_sub_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_sub_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Mulpd:
        case LirOpcode::Vmulpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmul_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmul_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Divpd:
        case LirOpcode::Vdivpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fdiv_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fdiv_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Minpd:
        case LirOpcode::Vminpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmin_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmin_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Maxpd:
        case LirOpcode::Vmaxpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_fmax_2d(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_fmax_2d(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Sqrtpd: {
            FPR dst = to_fpr(inst.defs[0]);
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.vec_fsqrt_2d(dst, to_fpr(src));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                enc_.vec_fsqrt_2d(dst, FPR::V31);
            }
            break;
        }

        case LirOpcode::Vfmadd213ps:
        case LirOpcode::Vfmadd231ps:
        case LirOpcode::Vfmadd213pd:
        case LirOpcode::Vfmadd231pd: {
            bool is_double = (inst.opcode == LirOpcode::Vfmadd213pd || inst.opcode == LirOpcode::Vfmadd231pd);
            bool is_213 = (inst.opcode == LirOpcode::Vfmadd213ps || inst.opcode == LirOpcode::Vfmadd213pd);
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 3) ? to_fpr(inst.uses[0]) : dst;
            size_t s2_idx = (inst.uses.size() >= 3) ? 1 : 0;
            size_t s3_idx = (inst.uses.size() >= 3) ? 2 : 1;
            FPR src2 = to_fpr(inst.uses[s2_idx]);
            const auto& src3_op = inst.uses[s3_idx];

            if (is_213) {
                // 213: dst = (src1 * src2) + src3
                // In AArch64, fmla Vd, Vn, Vm computes Vd = Vd + (Vn * Vm).
                if (src3_op.is_preg()) {
                    FPR src3 = to_fpr(src3_op);
                    enc_.vec_orr(FPR::V31, src3, src3);
                    if (is_double) enc_.vec_fmla_2d(FPR::V31, src1, src2);
                    else enc_.vec_fmla_4s(FPR::V31, src1, src2);
                    enc_.vec_orr(dst, FPR::V31, FPR::V31);
                } else {
                    enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
                    if (is_double) enc_.vec_fmla_2d(FPR::V31, src1, src2);
                    else enc_.vec_fmla_4s(FPR::V31, src1, src2);
                    enc_.vec_orr(dst, FPR::V31, FPR::V31);
                }
            } else {
                // 231: dst = (src2 * src3) + src1
                FPR acc = dst;
                bool aliases_other_src = (dst == src2 || (src3_op.is_preg() && dst == to_fpr(src3_op)));
                if (aliases_other_src) {
                    enc_.vec_orr(FPR::V30, src1, src1);
                    acc = FPR::V30;
                } else if (dst != src1) {
                    enc_.vec_orr(dst, src1, src1);
                }

                if (src3_op.is_preg()) {
                    FPR src3 = to_fpr(src3_op);
                    if (is_double) enc_.vec_fmla_2d(acc, src2, src3);
                    else enc_.vec_fmla_4s(acc, src2, src3);
                } else {
                    enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
                    if (is_double) enc_.vec_fmla_2d(acc, src2, FPR::V31);
                    else enc_.vec_fmla_4s(acc, src2, FPR::V31);
                }
                if (acc != dst) {
                    enc_.vec_orr(dst, acc, acc);
                }
            }
            break;
        }

        case LirOpcode::Fneg2d: {
            FPR dst = to_fpr(inst.defs[0]);
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.vec_fneg_2d(dst, to_fpr(src));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                enc_.vec_fneg_2d(dst, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pand:
        case LirOpcode::Vpand:
        case LirOpcode::Vandps:
        case LirOpcode::Vandpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_and(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_and(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Por:
        case LirOpcode::Vpor:
        case LirOpcode::Vorps:
        case LirOpcode::Vorpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_orr(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_orr(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pxor:
        case LirOpcode::Vpxor:
        case LirOpcode::Vxorps:
        case LirOpcode::Vxorpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.vec_eor(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.vec_eor(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Pinsrd: {
            // Lane 0..3: ins v.s[imm], w_src
            FPR dst = to_fpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[1]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 3;
            buffer_.emit_inst(0x4E041C00u | (lane << 19) | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pinsrq: {
            // Lane 0..1: ins v.d[imm], x_src
            FPR dst = to_fpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[1]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 1;
            buffer_.emit_inst(0x4E081C00u | (lane << 20) | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pextrd: {
            // umov w_dst, v.s[imm]
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 3;
            buffer_.emit_inst(0x0E043C00u | (lane << 19) | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pextrq: {
            // umov x_dst, v.d[imm]
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 1;
            buffer_.emit_inst(0x4E083C00u | (lane << 20) | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Extractps: {
            // umov w_dst, v.s[imm]
            GPR dst = to_gpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t lane = static_cast<uint32_t>(inst.uses.back().imm_int) & 3;
            buffer_.emit_inst(0x0E043C00u | (lane << 19) | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Insertps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[1]);
            uint32_t imm = static_cast<uint32_t>(inst.uses.back().imm_int);
            uint32_t src_lane = (imm >> 6) & 3;
            uint32_t dst_lane = (imm >> 4) & 3;
            uint32_t zmask = imm & 0xFu;
            if (inst.uses.size() >= 2 && inst.uses[0].is_preg()) {
                FPR src0 = to_fpr(inst.uses[0]);
                if (dst != src0) {
                    enc_.vec_orr(dst, src0, src0);
                }
            }
            // ins dst.s[dst_lane], src.s[src_lane]
            buffer_.emit_inst(0x6E040400u | (dst_lane << 19) | (src_lane << 13) | (reg_code(src) << 5) | reg_code(dst));
            if (zmask != 0) {
                for (uint32_t i = 0; i < 4; ++i) {
                    if (zmask & (1u << i)) {
                        // ins dst.s[i], wzr
                        buffer_.emit_inst(0x4E041C00u | (i << 19) | (31u << 5) | reg_code(dst));
                    }
                }
            }
            break;
        }

        case LirOpcode::Pnot: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            buffer_.emit_inst(0x6E205800u | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Vbroadcastss: {
            // dup v_dst.4s, v_src.s[0]
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            buffer_.emit_inst(0x4E040400u | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Vbroadcastsd: {
            // dup v_dst.2d, v_src.d[0]
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            buffer_.emit_inst(0x4E080400u | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Vpbroadcastd: {
            FPR dst = to_fpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            buffer_.emit_inst(0x4E040C00u | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Vpbroadcastq: {
            FPR dst = to_fpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            buffer_.emit_inst(0x4E080C00u | (static_cast<uint32_t>(reg_code(src)) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pshufd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            uint32_t l0 = mask & 3;
            uint32_t l1 = (mask >> 2) & 3;
            uint32_t l2 = (mask >> 4) & 3;
            uint32_t l3 = (mask >> 6) & 3;
            if (l0 == l1 && l1 == l2 && l2 == l3) {
                buffer_.emit_inst(0x4E040400u | (l0 << 19) | (reg_code(src) << 5) | reg_code(dst));
            } else {
                FPR tmp = (src == FPR::V31) ? FPR::V30 : FPR::V31;
                buffer_.emit_inst(0x6E040400u | (0u << 19) | (l0 << 13) | (reg_code(src) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (1u << 19) | (l1 << 13) | (reg_code(src) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (2u << 19) | (l2 << 13) | (reg_code(src) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (3u << 19) | (l3 << 13) | (reg_code(src) << 5) | reg_code(tmp));
                enc_.vec_orr(dst, tmp, tmp);
            }
            break;
        }

        case LirOpcode::Shufps: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR v0 = to_fpr(inst.uses[0]);
            FPR v1 = to_fpr(inst.uses[1]);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            uint32_t l0 = mask & 3;
            uint32_t l1 = (mask >> 2) & 3;
            uint32_t l2 = (mask >> 4) & 3;
            uint32_t l3 = (mask >> 6) & 3;
            if (v0 == v1 && l0 == l1 && l1 == l2 && l2 == l3) {
                buffer_.emit_inst(0x4E040400u | (l0 << 19) | (reg_code(v0) << 5) | reg_code(dst));
            } else {
                FPR tmp = (v0 == FPR::V31 || v1 == FPR::V31) ? FPR::V30 : FPR::V31;
                buffer_.emit_inst(0x6E040400u | (0u << 19) | (l0 << 13) | (reg_code(v0) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (1u << 19) | (l1 << 13) | (reg_code(v0) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (2u << 19) | (l2 << 13) | (reg_code(v1) << 5) | reg_code(tmp));
                buffer_.emit_inst(0x6E040400u | (3u << 19) | (l3 << 13) | (reg_code(v1) << 5) | reg_code(tmp));
                enc_.vec_orr(dst, tmp, tmp);
            }
            break;
        }

        case LirOpcode::Shufpd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR v0 = to_fpr(inst.uses[0]);
            FPR v1 = to_fpr(inst.uses[1]);
            uint32_t mask = static_cast<uint32_t>(inst.uses.back().imm_int);
            uint32_t l0 = mask & 1;
            uint32_t l1 = (mask >> 1) & 1;
            FPR tmp = (v0 == FPR::V31 || v1 == FPR::V31) ? FPR::V30 : FPR::V31;
            buffer_.emit_inst(0x6E080400u | (0u << 20) | (l0 << 14) | (reg_code(v0) << 5) | reg_code(tmp));
            buffer_.emit_inst(0x6E080400u | (1u << 20) | (l1 << 14) | (reg_code(v1) << 5) | reg_code(tmp));
            enc_.vec_orr(dst, tmp, tmp);
            break;
        }

        case LirOpcode::Movddup: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            buffer_.emit_inst(0x4E080400u | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pcmpeqd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = to_fpr(inst.uses[0]);
            FPR src2 = to_fpr(inst.uses[1]);
            buffer_.emit_inst(0x6EA08C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Pslld: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int) & 31;
            buffer_.emit_inst(0x4F205400u | (shift << 16) | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        case LirOpcode::Psllq: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src = to_fpr(inst.uses[0]);
            uint32_t shift = static_cast<uint32_t>(inst.uses.back().imm_int) & 63;
            buffer_.emit_inst(0x4F405400u | (shift << 16) | (reg_code(src) << 5) | reg_code(dst));
            break;
        }

        default:
            break;
    }
}


} // namespace brass::aarch64
