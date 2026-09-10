#include <brass/codegen/emit_context.hpp>
#include <stdexcept>

namespace brass::codegen {

using namespace brass::x64;

void EmitContext::emit_vec_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Movaps: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return;
            }
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.movaps(dst_x, src.preg_val.as_xmm());
                else enc_.movaps(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.movaps(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.movaps(XMM::XMM5, to_mem_address(src));
                    enc_.movaps(dst_mem, XMM::XMM5);
                }
            }
            break;
        }
        case LirOpcode::Movups: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.movups(dst_x, src.preg_val.as_xmm());
                else enc_.movups(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.movups(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.movups(XMM::XMM5, to_mem_address(src));
                    enc_.movups(dst_mem, XMM::XMM5);
                }
            }
            break;
        }
        case LirOpcode::Vmovaps: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return;
            }
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.vmovaps(dst_x, src.preg_val.as_xmm());
                else enc_.vmovaps(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.vmovaps(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.vmovaps(XMM::XMM5, to_mem_address(src));
                    enc_.vmovaps(dst_mem, XMM::XMM5);
                }
            }
            break;
        }
        case LirOpcode::Vmovups: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.vmovups(dst_x, src.preg_val.as_xmm());
                else enc_.vmovups(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.vmovups(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.vmovups(XMM::XMM5, to_mem_address(src));
                    enc_.vmovups(dst_mem, XMM::XMM5);
                }
            }
            break;
        }
        case LirOpcode::Movd_xg: {
            XMM dst = to_xmm(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            if (inst.uses[0].size == 8) enc_.movq(dst, src);
            else enc_.movd(dst, src);
            break;
        }
        case LirOpcode::Movd_gx: {
            GPR dst = to_gpr(inst.defs[0]);
            XMM src = to_xmm(inst.uses[0]);
            if (inst.defs[0].size == 8) enc_.movq(dst, src);
            else enc_.movd(dst, src);
            break;
        }

        #define EMIT_VEC_BINOP(OpcodeEnum, EncMethod)                              \
        case LirOpcode::OpcodeEnum: {                                              \
            XMM dst = to_xmm(inst.defs[0]);                                        \
            const auto& src = inst.uses.back();                                    \
            if (src.is_preg()) enc_.EncMethod(dst, to_xmm(src));                   \
            else enc_.EncMethod(dst, to_mem_address(src));                         \
            break;                                                                 \
        }

        EMIT_VEC_BINOP(Addps, addps)
        EMIT_VEC_BINOP(Subps, subps)
        EMIT_VEC_BINOP(Mulps, mulps)
        EMIT_VEC_BINOP(Divps, divps)
        EMIT_VEC_BINOP(Minps, minps)
        EMIT_VEC_BINOP(Maxps, maxps)

        EMIT_VEC_BINOP(Addpd, addpd)
        EMIT_VEC_BINOP(Subpd, subpd)
        EMIT_VEC_BINOP(Mulpd, mulpd)
        EMIT_VEC_BINOP(Divpd, divpd)
        EMIT_VEC_BINOP(Minpd, minpd)
        EMIT_VEC_BINOP(Maxpd, maxpd)

        EMIT_VEC_BINOP(Paddd, paddd)
        EMIT_VEC_BINOP(Psubd, psubd)
        EMIT_VEC_BINOP(Pmulld, pmulld)
        EMIT_VEC_BINOP(Pminsd, pminsd)
        EMIT_VEC_BINOP(Pmaxsd, pmaxsd)

        EMIT_VEC_BINOP(Paddq, paddq)
        EMIT_VEC_BINOP(Psubq, psubq)

        EMIT_VEC_BINOP(Pand, pand)
        EMIT_VEC_BINOP(Por, por)
        EMIT_VEC_BINOP(Pxor, pxor)
        EMIT_VEC_BINOP(Pandn, pandn)
        EMIT_VEC_BINOP(Pcmpeqd, pcmpeqd)
        EMIT_VEC_BINOP(Xorps, xorps)

        #undef EMIT_VEC_BINOP

        #define EMIT_AVX2_BINOP(OpcodeEnum, EncMethod)                              \
        case LirOpcode::OpcodeEnum: {                                              \
            XMM dst = to_xmm(inst.defs[0]);                                        \
            XMM src1 = to_xmm(inst.uses[0]);                                       \
            const auto& src2 = (inst.uses.size() > 1) ? inst.uses[1] : inst.uses[0]; \
            if (src2.is_preg()) enc_.EncMethod(dst, src1, to_xmm(src2));           \
            else enc_.EncMethod(dst, src1, to_mem_address(src2));                  \
            break;                                                                 \
        }

        EMIT_AVX2_BINOP(Vaddps, vaddps)
        EMIT_AVX2_BINOP(Vsubps, vsubps)
        EMIT_AVX2_BINOP(Vmulps, vmulps)
        EMIT_AVX2_BINOP(Vdivps, vdivps)
        EMIT_AVX2_BINOP(Vminps, vminps)
        EMIT_AVX2_BINOP(Vmaxps, vmaxps)
        EMIT_AVX2_BINOP(Vaddpd, vaddpd)
        EMIT_AVX2_BINOP(Vsubpd, vsubpd)
        EMIT_AVX2_BINOP(Vmulpd, vmulpd)
        EMIT_AVX2_BINOP(Vdivpd, vdivpd)
        EMIT_AVX2_BINOP(Vminpd, vminpd)
        EMIT_AVX2_BINOP(Vmaxpd, vmaxpd)
        EMIT_AVX2_BINOP(Vpaddd, vpaddd)
        EMIT_AVX2_BINOP(Vpsubd, vpsubd)
        EMIT_AVX2_BINOP(Vpmulld, vpmulld)
        EMIT_AVX2_BINOP(Vpaddq, vpaddq)
        EMIT_AVX2_BINOP(Vpsubq, vpsubq)
        EMIT_AVX2_BINOP(Vandps, vandps)
        EMIT_AVX2_BINOP(Vorps, vorps)
        EMIT_AVX2_BINOP(Vxorps, vxorps)
        EMIT_AVX2_BINOP(Vandpd, vandpd)
        EMIT_AVX2_BINOP(Vorpd, vorpd)
        EMIT_AVX2_BINOP(Vxorpd, vxorpd)
        EMIT_AVX2_BINOP(Vpand, vpand)
        EMIT_AVX2_BINOP(Vpor, vpor)
        EMIT_AVX2_BINOP(Vpxor, vpxor)

        #undef EMIT_AVX2_BINOP

        #define EMIT_AVX2_BROADCAST(OpcodeEnum, EncMethod)                         \
        case LirOpcode::OpcodeEnum: {                                              \
            XMM dst = to_xmm(inst.defs[0]);                                        \
            const auto& src = inst.uses[0];                                        \
            if (src.is_preg()) enc_.EncMethod(dst, to_xmm(src));                   \
            else enc_.EncMethod(dst, to_mem_address(src));                         \
            break;                                                                 \
        }

        EMIT_AVX2_BROADCAST(Vbroadcastss, vbroadcastss)
        EMIT_AVX2_BROADCAST(Vbroadcastsd, vbroadcastsd)
        EMIT_AVX2_BROADCAST(Vpbroadcastd, vpbroadcastd)
        EMIT_AVX2_BROADCAST(Vpbroadcastq, vpbroadcastq)

        #undef EMIT_AVX2_BROADCAST

        #define EMIT_FMA_VEC(OpcodeEnum, EncMethod)                                \
        case LirOpcode::OpcodeEnum: {                                              \
            XMM dst = to_xmm(inst.defs[0]);                                        \
            size_t s2_idx = (inst.uses.size() >= 3) ? 1 : 0;                      \
            size_t s3_idx = (inst.uses.size() >= 3) ? 2 : 1;                      \
            XMM src2 = to_xmm(inst.uses[s2_idx]);                                  \
            const auto& src3 = inst.uses[s3_idx];                                  \
            bool is_256 = (inst.defs[0].size == 32);                               \
            if (src3.is_preg()) enc_.EncMethod(dst, src2, to_xmm(src3), is_256);   \
            else enc_.EncMethod(dst, src2, to_mem_address(src3), is_256);          \
            break;                                                                 \
        }

        EMIT_FMA_VEC(Vfmadd213ps, vfmadd213ps)
        EMIT_FMA_VEC(Vfmadd231ps, vfmadd231ps)
        EMIT_FMA_VEC(Vfmadd213pd, vfmadd213pd)
        EMIT_FMA_VEC(Vfmadd231pd, vfmadd231pd)

        #undef EMIT_FMA_VEC

        #define EMIT_FMA_SCALAR(OpcodeEnum, EncMethod)                             \
        case LirOpcode::OpcodeEnum: {                                              \
            XMM dst = to_xmm(inst.defs[0]);                                        \
            size_t s2_idx = (inst.uses.size() >= 3) ? 1 : 0;                      \
            size_t s3_idx = (inst.uses.size() >= 3) ? 2 : 1;                      \
            XMM src2 = to_xmm(inst.uses[s2_idx]);                                  \
            const auto& src3 = inst.uses[s3_idx];                                  \
            if (src3.is_preg()) enc_.EncMethod(dst, src2, to_xmm(src3));           \
            else enc_.EncMethod(dst, src2, to_mem_address(src3));                  \
            break;                                                                 \
        }

        EMIT_FMA_SCALAR(Vfmadd213ss, vfmadd213ss)
        EMIT_FMA_SCALAR(Vfmadd231ss, vfmadd231ss)
        EMIT_FMA_SCALAR(Vfmadd213sd, vfmadd213sd)
        EMIT_FMA_SCALAR(Vfmadd231sd, vfmadd231sd)

        #undef EMIT_FMA_SCALAR

        case LirOpcode::Sqrtps: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sqrtps(dst, to_xmm(src));
            else enc_.sqrtps(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Sqrtpd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sqrtpd(dst, to_xmm(src));
            else enc_.sqrtpd(dst, to_mem_address(src));
            break;
        }

        case LirOpcode::Pslld: {
            XMM dst = to_xmm(inst.defs[0]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pslld(dst, imm);
            break;
        }
        case LirOpcode::Psllq: {
            XMM dst = to_xmm(inst.defs[0]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.psllq(dst, imm);
            break;
        }
        case LirOpcode::Shufps: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src0 = to_xmm(inst.uses[0]);
            XMM src1 = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            if (dst != src0) {
                enc_.movaps(dst, src0);
            }
            enc_.shufps(dst, src1, imm);
            break;
        }
        case LirOpcode::Shufpd: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src0 = to_xmm(inst.uses[0]);
            XMM src1 = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            if (dst != src0) {
                enc_.movaps(dst, src0);
            }
            enc_.shufpd(dst, src1, imm);
            break;
        }
        case LirOpcode::Pshufd: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pshufd(dst, src, imm);
            break;
        }
        case LirOpcode::Movddup: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src = to_xmm(inst.uses.back());
            enc_.movddup(dst, src);
            break;
        }
        case LirOpcode::Pinsrd: {
            XMM dst = to_xmm(inst.defs[0]);
            GPR src = to_gpr(inst.uses[1]);
            uint8_t lane = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pinsrd(dst, src, lane);
            break;
        }
        case LirOpcode::Pextrd: {
            GPR dst = to_gpr(inst.defs[0]);
            XMM src = to_xmm(inst.uses[0]);
            uint8_t lane = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pextrd(dst, src, lane);
            break;
        }
        case LirOpcode::Pinsrq: {
            XMM dst = to_xmm(inst.defs[0]);
            GPR src = to_gpr(inst.uses[1]);
            uint8_t lane = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pinsrq(dst, src, lane);
            break;
        }
        case LirOpcode::Pextrq: {
            GPR dst = to_gpr(inst.defs[0]);
            XMM src = to_xmm(inst.uses[0]);
            uint8_t lane = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.pextrq(dst, src, lane);
            break;
        }
        case LirOpcode::Insertps: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.insertps(dst, src, imm);
            break;
        }
        case LirOpcode::Extractps: {
            GPR dst = to_gpr(inst.defs[0]);
            XMM src = to_xmm(inst.uses[0]);
            uint8_t lane = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.extractps(dst, src, lane);
            break;
        }
        default:
            break;
    }
}

} // namespace brass::codegen
