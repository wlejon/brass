#include <brass/codegen/emit_context.hpp>
#include <stdexcept>

namespace brass::codegen {

using namespace brass::x64;

void EmitContext::emit_mov_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Mov: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return; // Skip self-moves
            }

            if (dst.is_preg()) {
                GPR dst_gpr = dst.preg_val.as_gpr();
                if (src.is_preg()) {
                    enc_.mov(dst_gpr, src.preg_val.as_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov(dst_gpr, src.imm_int);
                } else {
                    enc_.mov(dst_gpr, to_mem_address(src));
                }
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) {
                    enc_.mov(dst_mem, src.preg_val.as_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov(dst_mem, static_cast<int32_t>(src.imm_int));
                } else {
                    enc_.mov(GPR::R11, to_mem_address(src));
                    enc_.mov(dst_mem, GPR::R11);
                }
            }
            break;
        }
        case LirOpcode::Mov32: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return; // Skip self-moves
            }

            if (inst.is_patchable && dst.is_preg() && src.is_imm_int()) {
                GPR dst_gpr = dst.preg_val.as_gpr();
                size_t imm_off = (static_cast<uint8_t>(dst_gpr) >= 8) ? 2 : 1;
                size_t pad = runtime::compute_cache_line_padding(buffer_.size(), imm_off, 4);
                if (pad > 0) {
                    buffer_.emit_nops(pad);
                }
                size_t site_start = buffer_.size();
                enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Const32,
                    site_start,
                    imm_off,
                    buffer_.size() - site_start,
                    src.imm_int
                );
                break;
            }

            if (dst.is_preg()) {
                GPR dst_gpr = dst.preg_val.as_gpr();
                if (src.is_preg()) {
                    enc_.mov32(dst_gpr, src.preg_val.as_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                } else {
                    enc_.mov32(dst_gpr, to_mem_address(src));
                }
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) {
                    enc_.mov32(dst_mem, src.preg_val.as_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov32(dst_mem, static_cast<int32_t>(src.imm_int));
                } else {
                    enc_.mov32(GPR::R11, to_mem_address(src));
                    enc_.mov32(dst_mem, GPR::R11);
                }
            }
            break;
        }
        case LirOpcode::Movabs: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_symbol()) {
                enc_.movabs(dst_gpr, inst.uses[0].symbol_name);
            } else if (inst.is_patchable) {
                size_t imm_off = 2;
                size_t pad = runtime::compute_cache_line_padding(buffer_.size(), imm_off, 8);
                if (pad > 0) {
                    buffer_.emit_nops(pad);
                }
                size_t site_start = buffer_.size();
                enc_.movabs(dst_gpr, static_cast<uint64_t>(inst.uses[0].imm_int));
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Const64,
                    site_start,
                    imm_off,
                    buffer_.size() - site_start,
                    inst.uses[0].imm_int
                );
            } else {
                enc_.movabs(dst_gpr, static_cast<uint64_t>(inst.uses[0].imm_int));
            }
            break;
        }
        case LirOpcode::Movsxd: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.movsxd(dst_gpr, inst.uses[0].preg_val.as_gpr());
            } else {
                enc_.movsxd(dst_gpr, to_mem_address(inst.uses[0]));
            }
            break;
        }
        case LirOpcode::Movzx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.movzx8(dst_gpr, inst.uses[0].preg_val.as_gpr());
            } else {
                enc_.movzx8(dst_gpr, to_mem_address(inst.uses[0]));
            }
            break;
        }
        case LirOpcode::Movzx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.movzx16(dst_gpr, inst.uses[0].preg_val.as_gpr());
            } else {
                enc_.movzx16(dst_gpr, to_mem_address(inst.uses[0]));
            }
            break;
        }
        case LirOpcode::Movsx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.movsx8(dst_gpr, inst.uses[0].preg_val.as_gpr());
            } else {
                enc_.movsx8(dst_gpr, to_mem_address(inst.uses[0]));
            }
            break;
        }
        case LirOpcode::Movsx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.movsx16(dst_gpr, inst.uses[0].preg_val.as_gpr());
            } else {
                enc_.movsx16(dst_gpr, to_mem_address(inst.uses[0]));
            }
            break;
        }
        default:
            break;
    }
}

void EmitContext::emit_alu_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Add: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.add(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.add(dst, static_cast<int32_t>(src.imm_int));
            else enc_.add(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Add32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.add32(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.add32(dst, static_cast<int32_t>(src.imm_int));
            else enc_.add32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Sub: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sub(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.sub(dst, static_cast<int32_t>(src.imm_int));
            else enc_.sub(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Sub32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.sub32(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.sub32(dst, static_cast<int32_t>(src.imm_int));
            else enc_.sub32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Imul: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.imul(dst, to_gpr(src));
            else if (src.is_imm_int()) {
                if (inst.uses.size() >= 2 && inst.uses[0].is_preg() && inst.uses[0].preg_val != inst.defs[0].preg_val) {
                    enc_.imul(dst, to_gpr(inst.uses[0]), static_cast<int32_t>(src.imm_int));
                } else {
                    enc_.imul(dst, static_cast<int32_t>(src.imm_int));
                }
            } else enc_.imul(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Imul32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.imul32(dst, to_gpr(src));
            else if (src.is_imm_int()) {
                if (inst.uses.size() >= 2 && inst.uses[0].is_preg() && inst.uses[0].preg_val != inst.defs[0].preg_val) {
                    enc_.imul32(dst, to_gpr(inst.uses[0]), static_cast<int32_t>(src.imm_int));
                } else {
                    enc_.imul32(dst, static_cast<int32_t>(src.imm_int));
                }
            } else enc_.imul32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Idiv:
            if (inst.uses.back().is_preg()) enc_.idiv(to_gpr(inst.uses.back()));
            else enc_.idiv(to_mem_address(inst.uses.back()));
            break;
        case LirOpcode::Idiv32:
            if (inst.uses.back().is_preg()) enc_.idiv32(to_gpr(inst.uses.back()));
            else enc_.idiv32(to_mem_address(inst.uses.back()));
            break;
        case LirOpcode::Div:
            if (inst.uses.back().is_preg()) enc_.div(to_gpr(inst.uses.back()));
            else enc_.div(to_mem_address(inst.uses.back()));
            break;
        case LirOpcode::Div32:
            if (inst.uses.back().is_preg()) enc_.div32(to_gpr(inst.uses.back()));
            else enc_.div32(to_mem_address(inst.uses.back()));
            break;
        case LirOpcode::Cdq: enc_.cdq(); break;
        case LirOpcode::Cqo: enc_.cqo(); break;
        case LirOpcode::And: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.and_(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.and_(dst, static_cast<int32_t>(src.imm_int));
            else enc_.and_(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::And32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.and32(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.and32(dst, static_cast<int32_t>(src.imm_int));
            else enc_.and32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Or: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.or_(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.or_(dst, static_cast<int32_t>(src.imm_int));
            else enc_.or_(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Or32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.or32(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.or32(dst, static_cast<int32_t>(src.imm_int));
            else enc_.or32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Xor: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.xor_(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.xor_(dst, static_cast<int32_t>(src.imm_int));
            else enc_.xor_(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Xor32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.xor32(dst, to_gpr(src));
            else if (src.is_imm_int()) enc_.xor32(dst, static_cast<int32_t>(src.imm_int));
            else enc_.xor32(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Not: enc_.not_(to_gpr(inst.defs[0])); break;
        case LirOpcode::Not32: enc_.not32(to_gpr(inst.defs[0])); break;
        case LirOpcode::Neg: enc_.neg(to_gpr(inst.defs[0])); break;
        case LirOpcode::Neg32: enc_.neg32(to_gpr(inst.defs[0])); break;
        case LirOpcode::Shl: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.shl(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.shl(dst);
            break;
        }
        case LirOpcode::Shl32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.shl32(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.shl32(dst);
            break;
        }
        case LirOpcode::Shr: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.shr(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.shr(dst);
            break;
        }
        case LirOpcode::Shr32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.shr32(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.shr32(dst);
            break;
        }
        case LirOpcode::Sar: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.sar(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.sar(dst);
            break;
        }
        case LirOpcode::Sar32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) enc_.sar32(dst, static_cast<uint8_t>(src.imm_int));
            else enc_.sar32(dst);
            break;
        }
        case LirOpcode::Popcnt: enc_.popcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Popcnt32: enc_.popcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Lzcnt: enc_.lzcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Lzcnt32: enc_.lzcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Tzcnt: enc_.tzcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Tzcnt32: enc_.tzcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Bsr: enc_.bsr(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Bsr32: enc_.bsr32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Bsf: enc_.bsf(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Bsf32: enc_.bsf32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cmp: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.cmp(op0, to_gpr(op1));
            else if (op1.is_imm_int()) enc_.cmp(op0, static_cast<int32_t>(op1.imm_int));
            else enc_.cmp(op0, to_mem_address(op1));
            break;
        }
        case LirOpcode::Cmp32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.cmp32(op0, to_gpr(op1));
            else if (op1.is_imm_int()) enc_.cmp32(op0, static_cast<int32_t>(op1.imm_int));
            else enc_.cmp32(op0, to_mem_address(op1));
            break;
        }
        case LirOpcode::Test: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.test(op0, to_gpr(op1));
            else if (op1.is_imm_int()) enc_.test(op0, static_cast<int32_t>(op1.imm_int));
            break;
        }
        case LirOpcode::Test32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) enc_.test32(op0, to_gpr(op1));
            else if (op1.is_imm_int()) enc_.test32(op0, static_cast<int32_t>(op1.imm_int));
            break;
        }
        case LirOpcode::Setcc:
            enc_.setcc(inst.condition, to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Cmovcc:
            if (inst.defs[0].size == 4) enc_.cmovcc32(inst.condition, to_gpr(inst.defs[0]), to_gpr(inst.uses.back()));
            else enc_.cmovcc(inst.condition, to_gpr(inst.defs[0]), to_gpr(inst.uses.back()));
            break;
        default:
            break;
    }
}

} // namespace brass::codegen
