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
            XMM src = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.shufps(dst, src, imm);
            break;
        }
        case LirOpcode::Shufpd: {
            XMM dst = to_xmm(inst.defs[0]);
            XMM src = to_xmm(inst.uses[1]);
            uint8_t imm = static_cast<uint8_t>(inst.uses.back().imm_int);
            enc_.shufpd(dst, src, imm);
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
