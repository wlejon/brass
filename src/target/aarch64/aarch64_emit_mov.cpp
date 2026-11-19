#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64EmitContext::emit_mov_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Mov: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return;
            }

            if (dst.is_preg()) {
                GPR dst_gpr = dst.preg_val.as_aarch64_gpr();
                if (src.is_preg()) {
                    enc_.mov(dst_gpr, src.preg_val.as_aarch64_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov(dst_gpr, static_cast<uint64_t>(src.imm_int));
                } else {
                    if (dst.size == 1 || src.size == 1) enc_.ldrb(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 1));
                    else if (dst.size == 2 || src.size == 2) enc_.ldrh(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 2));
                    else if (dst.size == 4 || src.size == 4) enc_.ldr32(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 4));
                    else if (src.is_spill_slot() && dst.size == 8 && src.size == 8) {
                        if (const auto val = constant_slot_value(src.spill_slot)) {
                            emit_load_constant_gpr(dst_gpr, *val);
                        } else {
                            enc_.ldr(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 8));
                        }
                    } else {
                        enc_.ldr(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 8));
                    }
                }
            } else {
                int store_bytes = 8;
                if (src.is_imm_int()) {
                    if (dst.size == 1) store_bytes = 1;
                    else if (dst.size == 2) store_bytes = 2;
                    else if (dst.size == 4) store_bytes = 4;
                } else {
                    if (dst.size == 1 || src.size == 1) store_bytes = 1;
                    else if (dst.size == 2 || src.size == 2) store_bytes = 2;
                    else if (dst.size == 4 || src.size == 4) store_bytes = 4;
                }
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17, store_bytes);
                GPR data_scratch = (dst_mem.base == GPR::X16 || dst_mem.index == GPR::X16) ? GPR::X15 : GPR::X16;
                if (src.is_preg()) {
                    if (dst.size == 1 || src.size == 1) enc_.strb(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 2 || src.size == 2) enc_.strh(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else if (src.is_imm_int()) {
                    if (src.imm_int == 0) {
                        if (dst.size == 1) enc_.strb(GPR::XZR, dst_mem);
                        else if (dst.size == 2) enc_.strh(GPR::XZR, dst_mem);
                        else if (dst.size == 4) enc_.str32(GPR::XZR, dst_mem);
                        else enc_.str(GPR::XZR, dst_mem);
                    } else if (dst.size == 1) {
                        enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int & 0xff));
                        enc_.strb(data_scratch, dst_mem);
                    } else if (dst.size == 2) {
                        enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int & 0xffff));
                        enc_.strh(data_scratch, dst_mem);
                    } else if (dst.size == 4) {
                        enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int));
                        enc_.str32(data_scratch, dst_mem);
                    } else {
                        enc_.mov(data_scratch, static_cast<uint64_t>(src.imm_int));
                        enc_.str(data_scratch, dst_mem);
                    }
                } else {
                    GPR src_scratch = (dst_mem.base == GPR::X17 || dst_mem.index == GPR::X17) ? GPR::X15 : GPR::X17;
                    if (src_scratch == data_scratch) src_scratch = (data_scratch == GPR::X16) ? GPR::X15 : GPR::X16;
                    if (src.is_spill_slot() && src.size == 8) {
                        if (const auto val = constant_slot_value(src.spill_slot)) {
                            emit_load_constant_gpr(data_scratch, *val, src_scratch);
                        } else {
                            enc_.ldr(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch, 8));
                        }
                    } else {
                        enc_.ldr(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch, 8));
                    }
                    if (dst.size == 1 || src.size == 1) enc_.strb(data_scratch, dst_mem);
                    else if (dst.size == 2 || src.size == 2) enc_.strh(data_scratch, dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str32(data_scratch, dst_mem);
                    else enc_.str(data_scratch, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Mov32: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

            // A self mov32 is kept: writing Wd zeroes the upper half of Xd.

            if (inst.is_patchable && dst.is_preg() && src.is_imm_int()) {
                GPR dst_gpr = dst.preg_val.as_aarch64_gpr();
                size_t site_start = buffer_.size();
                // LDR Wt, [PC, #8] -> imm19 = 2 words: 0x18000000u | (2u << 5) | reg_code(dst_gpr)
                buffer_.emit_inst(0x18000000u | (2u << 5) | reg_code(dst_gpr));
                // B #8 (skip the 4-byte literal) -> 0x14000002u
                buffer_.emit_inst(0x14000002u);
                // 4-byte literal in memory
                buffer_.emit_inst(static_cast<uint32_t>(src.imm_int));
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Const32,
                    site_start,
                    8, // imm_offset is 8 (points directly to the literal word!)
                    12,
                    src.imm_int
                );
                break;
            }

            if (dst.is_preg()) {
                GPR dst_gpr = dst.preg_val.as_aarch64_gpr();
                if (src.is_preg()) {
                    enc_.mov32(dst_gpr, src.preg_val.as_aarch64_gpr());
                } else if (src.is_imm_int()) {
                    enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                } else {
                    if (dst.size == 1 || src.size == 1) enc_.ldrb(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 1));
                    else if (dst.size == 2 || src.size == 2) enc_.ldrh(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 2));
                    else enc_.ldr32(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 4));
                }
            } else {
                int store_bytes = 4;
                if (src.is_imm_int()) {
                    if (dst.size == 1) store_bytes = 1;
                    else if (dst.size == 2) store_bytes = 2;
                } else {
                    if (dst.size == 1 || src.size == 1) store_bytes = 1;
                    else if (dst.size == 2 || src.size == 2) store_bytes = 2;
                }
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17, store_bytes);
                GPR data_scratch = (dst_mem.base == GPR::X16 || dst_mem.index == GPR::X16) ? GPR::X15 : GPR::X16;
                if (src.is_preg()) {
                    if (dst.size == 1 || src.size == 1) enc_.strb(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 2 || src.size == 2) enc_.strh(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else if (src.is_imm_int()) {
                    const uint32_t imm32 = static_cast<uint32_t>(src.imm_int);
                    if (imm32 == 0) {
                        if (dst.size == 1) enc_.strb(GPR::XZR, dst_mem);
                        else if (dst.size == 2) enc_.strh(GPR::XZR, dst_mem);
                        else enc_.str32(GPR::XZR, dst_mem);
                    } else if (dst.size == 1) {
                        enc_.mov32(data_scratch, imm32 & 0xffu);
                        enc_.strb(data_scratch, dst_mem);
                    } else if (dst.size == 2) {
                        enc_.mov32(data_scratch, imm32 & 0xffffu);
                        enc_.strh(data_scratch, dst_mem);
                    } else {
                        enc_.mov32(data_scratch, imm32);
                        enc_.str32(data_scratch, dst_mem);
                    }
                } else {
                    GPR src_scratch = (dst_mem.base == GPR::X17 || dst_mem.index == GPR::X17) ? GPR::X15 : GPR::X17;
                    if (src_scratch == data_scratch) src_scratch = (data_scratch == GPR::X16) ? GPR::X15 : GPR::X16;
                    if (src.size == 1) enc_.ldrb(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch, 1));
                    else if (src.size == 2) enc_.ldrh(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch, 2));
                    else enc_.ldr32(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch, 4));
                    if (dst.size == 1) enc_.strb(data_scratch, dst_mem);
                    else if (dst.size == 2) enc_.strh(data_scratch, dst_mem);
                    else enc_.str32(data_scratch, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Movabs: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_symbol()) {
                enc_.load_symbol_address(dst_gpr, inst.uses[0].symbol_name);
            } else if (inst.is_patchable) {
                // Ensure 8-byte natural alignment for the 64-bit literal at site_start + 8
                if (((buffer_.size() + 8) % 8) != 0) {
                    enc_.nop();
                }
                size_t site_start = buffer_.size();
                // LDR Xt, [PC, #8] -> imm19 = 2 words: 0x58000000u | (2u << 5) | reg_code(dst_gpr)
                buffer_.emit_inst(0x58000000u | (2u << 5) | reg_code(dst_gpr));
                // B #12 (skip the 8-byte literal) -> 0x14000003u
                buffer_.emit_inst(0x14000003u);
                // 8-byte literal in memory
                uint64_t v = static_cast<uint64_t>(inst.uses[0].imm_int);
                buffer_.emit_inst(static_cast<uint32_t>(v & 0xFFFFFFFFu));
                buffer_.emit_inst(static_cast<uint32_t>(v >> 32));
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Const64,
                    site_start,
                    8, // imm_offset is 8 (points directly to the literal qword!)
                    16,
                    inst.uses[0].imm_int
                );
            } else {
                const uint64_t v = static_cast<uint64_t>(inst.uses[0].imm_int);
                if (!AArch64Encoder::single_inst_mov_imm(v)) {
                    if (const auto off = pinned_constant(v)) {
                        enc_.ldr(dst_gpr, ensure_accessible_mem(ptr(kPinnedTlsGpr, *off), GPR::X16, 8));
                        break;
                    }
                }
                enc_.mov(dst_gpr, v);
            }
            break;
        }

        case LirOpcode::Movsxd: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxtw(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsw(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0]), GPR::X16, 4));
            }
            break;
        }

        case LirOpcode::Movzx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.uxtb(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrb(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0]), GPR::X16, 1));
            }
            break;
        }

        case LirOpcode::Movzx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.uxth(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrh(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0]), GPR::X16, 2));
            }
            break;
        }

        case LirOpcode::Movsx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxtb(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsb(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0]), GPR::X16, 1));
            }
            break;
        }

        case LirOpcode::Movsx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxth(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsh(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0]), GPR::X16, 2));
            }
            break;
        }

        default:
            throw_unsupported("aarch64 emit (mov)", to_string(inst.opcode));
    }
}


} // namespace brass::aarch64
