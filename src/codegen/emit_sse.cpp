#include <brass/codegen/emit_context.hpp>

namespace brass::codegen {

using namespace brass::x64;

void EmitContext::emit_sse_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Movsd: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return; // Skip self-moves
            }
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.movsd(dst_x, src.preg_val.as_xmm());
                else enc_.movsd(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.movsd(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.movsd(XMM::XMM15, to_mem_address(src));
                    enc_.movsd(dst_mem, XMM::XMM15);
                }
            }
            break;
        }
        case LirOpcode::Movss: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return; // Skip self-moves
            }
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) enc_.movss(dst_x, src.preg_val.as_xmm());
                else enc_.movss(dst_x, to_mem_address(src));
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) enc_.movss(dst_mem, src.preg_val.as_xmm());
                else {
                    enc_.movss(XMM::XMM15, to_mem_address(src));
                    enc_.movss(dst_mem, XMM::XMM15);
                }
            }
            break;
        }
        case LirOpcode::Movq_xg: enc_.movq(to_xmm(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Movq_gx: enc_.movq(to_gpr(inst.defs[0]), to_xmm(inst.uses[0])); break;
        case LirOpcode::Addsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.addsd(dst, to_xmm(src));
            else enc_.addsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Addss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.addss(dst, to_xmm(src));
            else enc_.addss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Subsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.subsd(dst, to_xmm(src));
            else enc_.subsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Subss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.subss(dst, to_xmm(src));
            else enc_.subss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Mulsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.mulsd(dst, to_xmm(src));
            else enc_.mulsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Mulss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.mulss(dst, to_xmm(src));
            else enc_.mulss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Divsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.divsd(dst, to_xmm(src));
            else enc_.divsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Divss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.divss(dst, to_xmm(src));
            else enc_.divss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Sqrtsd: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sqrtsd(to_xmm(inst.defs[0]), to_xmm(src));
            else enc_.sqrtsd(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Sqrtss: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sqrtss(to_xmm(inst.defs[0]), to_xmm(src));
            else enc_.sqrtss(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Ucomisd:
            if (inst.uses[1].is_mem() || inst.uses[1].is_spill_slot()) enc_.ucomisd(to_xmm(inst.uses[0]), to_mem_address(inst.uses[1]));
            else enc_.ucomisd(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
            break;
        case LirOpcode::Ucomiss:
            if (inst.uses[1].is_mem() || inst.uses[1].is_spill_slot()) enc_.ucomiss(to_xmm(inst.uses[0]), to_mem_address(inst.uses[1]));
            else enc_.ucomiss(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
            break;
        case LirOpcode::Xorpd: enc_.xorpd(to_xmm(inst.defs[0]), to_xmm(inst.uses.back())); break;
        case LirOpcode::Cvtsi2sd: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtsi2sd(to_xmm(inst.defs[0]), to_gpr(src));
            else enc_.cvtsi2sd(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvtsi2sd32: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtsi2sd32(to_xmm(inst.defs[0]), to_gpr(src));
            else enc_.cvtsi2sd32(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvttsd2si: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvttsd2si(to_gpr(inst.defs[0]), to_xmm(src));
            else enc_.cvttsd2si(to_gpr(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvttsd2si32: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvttsd2si32(to_gpr(inst.defs[0]), to_xmm(src));
            else enc_.cvttsd2si32(to_gpr(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvtsi2ss: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtsi2ss(to_xmm(inst.defs[0]), to_gpr(src));
            else enc_.cvtsi2ss(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvtsi2ss32: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtsi2ss32(to_xmm(inst.defs[0]), to_gpr(src));
            else enc_.cvtsi2ss32(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvttss2si: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvttss2si(to_gpr(inst.defs[0]), to_xmm(src));
            else enc_.cvttss2si(to_gpr(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvttss2si32: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvttss2si32(to_gpr(inst.defs[0]), to_xmm(src));
            else enc_.cvttss2si32(to_gpr(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvtsd2ss: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtsd2ss(to_xmm(inst.defs[0]), to_xmm(src));
            else enc_.cvtsd2ss(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Cvtss2sd: {
            const auto& src = inst.uses[0];
            if (src.is_preg()) enc_.cvtss2sd(to_xmm(inst.defs[0]), to_xmm(src));
            else enc_.cvtss2sd(to_xmm(inst.defs[0]), to_mem_address(src));
            break;
        }
        case LirOpcode::Floor32: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundss(dst, to_xmm(src), 0x01);
            else enc_.roundss(dst, to_mem_address(src), 0x01);
            break;
        }
        case LirOpcode::Floor64: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundsd(dst, to_xmm(src), 0x01);
            else enc_.roundsd(dst, to_mem_address(src), 0x01);
            break;
        }
        case LirOpcode::Ceil32: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundss(dst, to_xmm(src), 0x02);
            else enc_.roundss(dst, to_mem_address(src), 0x02);
            break;
        }
        case LirOpcode::Ceil64: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundsd(dst, to_xmm(src), 0x02);
            else enc_.roundsd(dst, to_mem_address(src), 0x02);
            break;
        }
        case LirOpcode::Round32: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundss(dst, to_xmm(src), 0x00);
            else enc_.roundss(dst, to_mem_address(src), 0x00);
            break;
        }
        case LirOpcode::Round64: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.roundsd(dst, to_xmm(src), 0x00);
            else enc_.roundsd(dst, to_mem_address(src), 0x00);
            break;
        }
        case LirOpcode::Fabs32: {
            XMM dst = to_xmm(inst.defs[0]);
            enc_.mov32(GPR::R11, 0x7FFFFFFF);
            enc_.movq(XMM::XMM15, GPR::R11);
            enc_.andpd(dst, XMM::XMM15);
            break;
        }
        case LirOpcode::Fabs64: {
            XMM dst = to_xmm(inst.defs[0]);
            enc_.mov64(GPR::R11, 0x7FFFFFFFFFFFFFFFULL);
            enc_.movq(XMM::XMM15, GPR::R11);
            enc_.andpd(dst, XMM::XMM15);
            break;
        }
        case LirOpcode::Minss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.minss(dst, to_xmm(src));
            else enc_.minss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Minsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.minsd(dst, to_xmm(src));
            else enc_.minsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Maxss: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.maxss(dst, to_xmm(src));
            else enc_.maxss(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Maxsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.maxsd(dst, to_xmm(src));
            else enc_.maxsd(dst, to_mem_address(src));
            break;
        }
        default: break;
    }
}

} // namespace brass::codegen
