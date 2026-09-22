#include <brass/target/aarch64/aarch64_emit.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64EmitContext::emit_alu_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Add: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.add(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.add(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.sub(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.add(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.add(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Add32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.add32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.add32(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.sub32(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.add32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.add32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Sub: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.sub(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.sub(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.add(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.sub(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.sub(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Sub32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.sub32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.sub32(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.add32(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.sub32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.sub32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Adds: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.adds(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.adds(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.subs(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.adds(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.adds(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Adds32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.adds32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.adds32(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.subs32(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.adds32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.adds32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Subs: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.subs(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.subs(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.adds(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.subs(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.subs(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Subs32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.subs32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                if (src2.imm_int >= 0 && src2.imm_int <= 4095) {
                    enc_.subs32(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else if (src2.imm_int < 0 && -src2.imm_int <= 4095) {
                    enc_.adds32(dst, src1, static_cast<uint32_t>(-src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.subs32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.subs32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Imul: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.mul(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                enc_.mul(dst, src1, GPR::X16);
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.mul(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Imul32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.mul32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                enc_.mul32(dst, src1, GPR::X16);
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.mul32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Smulh: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.smulh(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.smulh(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Umulh: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.umulh(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.umulh(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Idiv: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.sdiv(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.sdiv(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Idiv32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.sdiv32(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.sdiv32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Div: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.udiv(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.udiv(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Div32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.udiv32(dst, src1, to_gpr(src2));
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.udiv32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Cdq:
        case LirOpcode::Cqo:
            // No-op in AArch64 (no DX:AX pair)
            break;

        case LirOpcode::And: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.and_(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint64_t>(src2.imm_int), true, n, immr, imms)) {
                    enc_.and_imm(dst, src1, static_cast<uint64_t>(src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.and_(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.and_(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::And32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.and32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint32_t>(src2.imm_int), false, n, immr, imms)) {
                    enc_.and32_imm(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.and32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.and32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Or: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.orr(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint64_t>(src2.imm_int), true, n, immr, imms)) {
                    enc_.orr_imm(dst, src1, static_cast<uint64_t>(src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.orr(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.orr(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Or32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.orr32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint32_t>(src2.imm_int), false, n, immr, imms)) {
                    enc_.orr32_imm(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.orr32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.orr32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Xor: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.eor(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint64_t>(src2.imm_int), true, n, immr, imms)) {
                    enc_.eor_imm(dst, src1, static_cast<uint64_t>(src2.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                    enc_.eor(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.eor(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Xor32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_preg()) {
                enc_.eor32(dst, src1, to_gpr(src2));
            } else if (src2.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint32_t>(src2.imm_int), false, n, immr, imms)) {
                    enc_.eor32_imm(dst, src1, static_cast<uint32_t>(src2.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                    enc_.eor32(dst, src1, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.eor32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Not: enc_.mvn(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Not32: enc_.mvn32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Neg: {
            if (to_fpr(inst.defs[0]) != FPR::None) {
                FPR src = (inst.uses.empty() || to_fpr(inst.uses[0]) == FPR::None) ? to_fpr(inst.defs[0]) : to_fpr(inst.uses[0]);
                enc_.fneg(to_fpr(inst.defs[0]), src);
            } else {
                GPR src = (inst.uses.empty() || to_gpr(inst.uses[0]) == GPR::None) ? to_gpr(inst.defs[0]) : to_gpr(inst.uses[0]);
                enc_.neg(to_gpr(inst.defs[0]), src);
            }
            break;
        }
        case LirOpcode::Neg32: {
            if (to_fpr(inst.defs[0]) != FPR::None) {
                FPR src = (inst.uses.empty() || to_fpr(inst.uses[0]) == FPR::None) ? to_fpr(inst.defs[0]) : to_fpr(inst.uses[0]);
                enc_.fneg_s(to_fpr(inst.defs[0]), src);
            } else {
                GPR src = (inst.uses.empty() || to_gpr(inst.uses[0]) == GPR::None) ? to_gpr(inst.defs[0]) : to_gpr(inst.uses[0]);
                enc_.neg32(to_gpr(inst.defs[0]), src);
            }
            break;
        }

        case LirOpcode::Shl: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.lsl(dst, src1, static_cast<uint8_t>(src2.imm_int & 63));
            else enc_.lsl(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Shl32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.lsl32(dst, src1, static_cast<uint8_t>(src2.imm_int & 31));
            else enc_.lsl32(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Shr: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.lsr(dst, src1, static_cast<uint8_t>(src2.imm_int & 63));
            else enc_.lsr(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Shr32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.lsr32(dst, src1, static_cast<uint8_t>(src2.imm_int & 31));
            else enc_.lsr32(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Sar: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.asr(dst, src1, static_cast<uint8_t>(src2.imm_int & 63));
            else enc_.asr(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Sar32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src1 = (inst.uses.size() >= 2) ? to_gpr(inst.uses[0]) : dst;
            const auto& src2 = inst.uses.back();
            if (src2.is_imm_int()) enc_.asr32(dst, src1, static_cast<uint8_t>(src2.imm_int & 31));
            else enc_.asr32(dst, src1, to_gpr(src2));
            break;
        }

        case LirOpcode::Lzcnt: enc_.clz(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Lzcnt32: enc_.clz32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;

        case LirOpcode::Tzcnt: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.rbit(dst, src);
            enc_.clz(dst, dst);
            break;
        }

        case LirOpcode::Tzcnt32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.rbit32(dst, src);
            enc_.clz32(dst, dst);
            break;
        }

        case LirOpcode::Bsr: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.clz(GPR::X16, src);
            enc_.mov(dst, 63);
            enc_.sub(dst, dst, GPR::X16);
            break;
        }

        case LirOpcode::Bsr32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.clz32(GPR::X16, src);
            enc_.mov32(dst, 31);
            enc_.sub32(dst, dst, GPR::X16);
            break;
        }

        case LirOpcode::Bsf: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.rbit(GPR::X16, src);
            enc_.clz(dst, GPR::X16);
            break;
        }

        case LirOpcode::Bsf32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.rbit32(GPR::X16, src);
            enc_.clz32(dst, GPR::X16);
            break;
        }

        case LirOpcode::Popcnt: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.fmov_from_gpr(FPR::V31, src);
            buffer_.emit_inst(0x0E205800u | (31 << 5) | 31); // cnt v31.8b, v31.8b
            buffer_.emit_inst(0x2E303800u | (31 << 5) | 31); // uaddlv h31, v31.8b
            enc_.fmov_to_gpr(dst, FPR::V31);
            break;
        }

        case LirOpcode::Popcnt32: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses[0]);
            enc_.fmov_from_gpr32(FPR::V31, src);
            buffer_.emit_inst(0x0E205800u | (31 << 5) | 31); // cnt v31.8b, v31.8b
            buffer_.emit_inst(0x2E303800u | (31 << 5) | 31); // uaddlv h31, v31.8b
            enc_.fmov_to_gpr32(dst, FPR::V31);
            break;
        }

        case LirOpcode::Cmp: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.cmp(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                if (op1.imm_int >= 0 && op1.imm_int <= 4095) {
                    enc_.cmp(op0, static_cast<uint32_t>(op1.imm_int));
                } else if (op1.imm_int < 0 && -op1.imm_int <= 4095) {
                    enc_.cmn(op0, static_cast<uint32_t>(-op1.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(op1.imm_int));
                    enc_.cmp(op0, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(op1)));
                enc_.cmp(op0, GPR::X16);
            }
            break;
        }

        case LirOpcode::Cmp32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.cmp32(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                if (op1.imm_int >= 0 && op1.imm_int <= 4095) {
                    enc_.cmp32(op0, static_cast<uint32_t>(op1.imm_int));
                } else if (op1.imm_int < 0 && -op1.imm_int <= 4095) {
                    enc_.cmn32(op0, static_cast<uint32_t>(-op1.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(op1.imm_int));
                    enc_.cmp32(op0, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(op1)));
                enc_.cmp32(op0, GPR::X16);
            }
            break;
        }

        case LirOpcode::Test: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.tst(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint64_t>(op1.imm_int), true, n, immr, imms)) {
                    enc_.tst_imm(op0, static_cast<uint64_t>(op1.imm_int));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(op1.imm_int));
                    enc_.tst(op0, GPR::X16);
                }
            } else {
                enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(op1)));
                enc_.tst(op0, GPR::X16);
            }
            break;
        }

        case LirOpcode::Test32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.tst32(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                uint32_t n, immr, imms;
                if (AArch64Encoder::encode_logical_immediate(static_cast<uint32_t>(op1.imm_int), false, n, immr, imms)) {
                    enc_.tst32_imm(op0, static_cast<uint32_t>(op1.imm_int));
                } else {
                    enc_.mov32(GPR::X16, static_cast<uint32_t>(op1.imm_int));
                    enc_.tst32(op0, GPR::X16);
                }
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(op1)));
                enc_.tst32(op0, GPR::X16);
            }
            break;
        }

        case LirOpcode::Setcc:
            enc_.cset(to_gpr(inst.defs[0]), to_aarch64_cond(inst.condition));
            break;

        case LirOpcode::Cmovcc: {
            GPR dst = to_gpr(inst.defs[0]);
            GPR src = to_gpr(inst.uses.back());
            Condition cond = to_aarch64_cond(inst.condition);
            if (inst.defs[0].size == 4) {
                enc_.csel32(dst, src, dst, cond);
            } else {
                enc_.csel(dst, src, dst, cond);
            }
            break;
        }

        default:
            break;
    }
}


} // namespace brass::aarch64
