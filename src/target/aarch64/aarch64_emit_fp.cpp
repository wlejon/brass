#include <brass/target/aarch64/aarch64_emit.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64EmitContext::emit_fp_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Movsd: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;

            bool is_single = (dst.size == 4 || src.size == 4);
            if (dst.is_preg()) {
                FPR dst_f = dst.preg_val.as_aarch64_fpr();
                if (src.is_preg()) {
                    if (is_single) enc_.fmov_s(dst_f, src.preg_val.as_aarch64_fpr());
                    else enc_.fmov(dst_f, src.preg_val.as_aarch64_fpr());
                } else {
                    if (is_single) enc_.ldr_s(dst_f, ensure_accessible_mem(to_mem_address(src)));
                    else enc_.ldr(dst_f, ensure_accessible_mem(to_mem_address(src)));
                }
            } else {
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
                if (src.is_preg()) {
                    if (is_single) enc_.str_s(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_fpr(), dst_mem);
                } else {
                    if (is_single) {
                        enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                        enc_.str_s(FPR::V31, dst_mem);
                    } else {
                        enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                        enc_.str(FPR::V31, dst_mem);
                    }
                }
            }
            break;
        }

        case LirOpcode::Movss: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;

            if (dst.is_preg()) {
                FPR dst_f = dst.preg_val.as_aarch64_fpr();
                if (src.is_preg()) enc_.fmov_s(dst_f, src.preg_val.as_aarch64_fpr());
                else enc_.ldr_s(dst_f, ensure_accessible_mem(to_mem_address(src)));
            } else {
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
                if (src.is_preg()) enc_.str_s(src.preg_val.as_aarch64_fpr(), dst_mem);
                else {
                    enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src)));
                    enc_.str_s(FPR::V31, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Movq_xg: enc_.fmov_from_gpr(to_fpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Movq_gx: enc_.fmov_to_gpr(to_gpr(inst.defs[0]), to_fpr(inst.uses[0])); break;

        case LirOpcode::Addsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fadd(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fadd(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Addss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fadd_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fadd_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Subsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fsub(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fsub(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Subss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fsub_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fsub_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Mulsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmul(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmul(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Mulss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmul_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmul_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Divsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fdiv(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fdiv(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Divss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fdiv_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fdiv_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Sqrtsd: enc_.fsqrt(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Sqrtss: enc_.fsqrt_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;

        case LirOpcode::Floor32: enc_.frintm_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Floor64: enc_.frintm(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Ceil32: enc_.frintp_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Ceil64: enc_.frintp(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Round32: enc_.frinta_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Round64: enc_.frinta(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Fabs32: enc_.fabs_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;
        case LirOpcode::Fabs64: enc_.fabs(to_fpr(inst.defs[0]), to_fpr(inst.uses.back())); break;

        case LirOpcode::Minss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmin_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmin_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Minsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmin(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmin(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Maxss: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmax_s(dst, src1, to_fpr(src2));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmax_s(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Maxsd: {
            FPR dst = to_fpr(inst.defs[0]);
            FPR src1 = (inst.uses.size() >= 2) ? to_fpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) enc_.fmax(dst, src1, to_fpr(src2));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src2)));
                enc_.fmax(dst, src1, FPR::V31);
            }
            break;
        }

        case LirOpcode::Vfmadd213ss:
        case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd:
        case LirOpcode::Vfmadd231sd: {
            bool is_double = (inst.opcode == LirOpcode::Vfmadd213sd || inst.opcode == LirOpcode::Vfmadd231sd);
            bool is_213 = (inst.opcode == LirOpcode::Vfmadd213ss || inst.opcode == LirOpcode::Vfmadd213sd);
            FPR dst = to_fpr(inst.defs[0]);
            size_t s2_idx = (inst.uses.size() >= 3) ? 1 : 0;
            size_t s3_idx = (inst.uses.size() >= 3) ? 2 : 1;
            FPR src2 = to_fpr(inst.uses[s2_idx]);
            const auto& src3_op = inst.uses[s3_idx];
            FPR src3 = FPR::V31;
            if (src3_op.is_preg()) {
                src3 = to_fpr(src3_op);
            } else {
                if (is_double) enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
                else enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
            }

            FPR sn = is_213 ? dst : src2;
            FPR sm = is_213 ? src2 : src3;
            FPR sa = is_213 ? src3 : dst;

            if (is_double) enc_.fmadd_d(dst, sn, sm, sa);
            else enc_.fmadd_s(dst, sn, sm, sa);
            break;
        }

        case LirOpcode::Ucomisd: {
            FPR op0 = to_fpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.fcmp(op0, to_fpr(op1));
            else {
                enc_.ldr(FPR::V31, ensure_accessible_mem(to_mem_address(op1)));
                enc_.fcmp(op0, FPR::V31);
            }
            break;
        }

        case LirOpcode::Ucomiss: {
            FPR op0 = to_fpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.fcmp_s(op0, to_fpr(op1));
            else {
                enc_.ldr_s(FPR::V31, ensure_accessible_mem(to_mem_address(op1)));
                enc_.fcmp_s(op0, FPR::V31);
            }
            break;
        }

        case LirOpcode::Xorpd:
        case LirOpcode::Xorps:
            enc_.vec_eor(to_fpr(inst.defs[0]), to_fpr(inst.uses[0]), to_fpr(inst.uses.back()));
            break;

        case LirOpcode::Fneg:
            enc_.fneg(to_fpr(inst.defs[0]), to_fpr(inst.uses.back()));
            break;
        case LirOpcode::Fneg32:
            enc_.fneg_s(to_fpr(inst.defs[0]), to_fpr(inst.uses.back()));
            break;

        case LirOpcode::Cvtsi2sd: enc_.scvtf_d(to_fpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvtsi2sd32: enc_.scvtf_d32(to_fpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvttsd2si: enc_.fcvtzs_d(to_gpr(inst.defs[0]), to_fpr(inst.uses[0])); break;
        case LirOpcode::Cvttsd2si32: enc_.fcvtzs_d32(to_gpr(inst.defs[0]), to_fpr(inst.uses[0])); break;
        case LirOpcode::Cvtsi2ss: enc_.scvtf_s(to_fpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvtsi2ss32: enc_.scvtf_s32(to_fpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvttss2si: enc_.fcvtzs_s(to_gpr(inst.defs[0]), to_fpr(inst.uses[0])); break;
        case LirOpcode::Cvttss2si32: enc_.fcvtzs_s32(to_gpr(inst.defs[0]), to_fpr(inst.uses[0])); break;
        case LirOpcode::Cvtsd2ss: enc_.fcvt_s_d(to_fpr(inst.defs[0]), to_fpr(inst.uses[0])); break;
        case LirOpcode::Cvtss2sd: enc_.fcvt_d_s(to_fpr(inst.defs[0]), to_fpr(inst.uses[0])); break;

        default:
            break;
    }
}

} // namespace brass::aarch64
