#include <brass/target/aarch64/aarch64_emit.hpp>
#include <iostream>
#include <stdexcept>
#include <cassert>

namespace brass::aarch64 {

using namespace brass::codegen;

static Condition to_aarch64_cond(x64::Condition cond) noexcept {
    switch (cond) {
        case x64::Condition::O:   return Condition::VS;
        case x64::Condition::NO:  return Condition::VC;
        case x64::Condition::B:   return Condition::CC;
        case x64::Condition::AE:  return Condition::CS;
        case x64::Condition::E:   return Condition::EQ;
        case x64::Condition::NE:  return Condition::NE;
        case x64::Condition::BE:  return Condition::LS;
        case x64::Condition::A:   return Condition::HI;
        case x64::Condition::S:   return Condition::MI;
        case x64::Condition::NS:  return Condition::PL;
        case x64::Condition::P:   return Condition::VS;
        case x64::Condition::NP:  return Condition::VC;
        case x64::Condition::L:   return Condition::LT;
        case x64::Condition::GE:  return Condition::GE;
        case x64::Condition::LE:  return Condition::LE;
        case x64::Condition::G:   return Condition::GT;
        default:                  return Condition::AL;
    }
}

AArch64EmitContext::AArch64EmitContext(const LirFunction& fn, const Target& target)
    : fn_(fn), target_(target), enc_(buffer_) {}

GPR AArch64EmitContext::to_gpr(const LirOperand& op) const {
    if (op.is_preg() && op.preg_val.is_gpr()) {
        return op.preg_val.as_aarch64_gpr();
    }
    return GPR::None;
}

FPR AArch64EmitContext::to_fpr(const LirOperand& op) const {
    if (op.is_preg() && op.preg_val.is_xmm()) {
        return op.preg_val.as_aarch64_fpr();
    }
    return FPR::None;
}

MemAddress AArch64EmitContext::to_mem_address(const LirOperand& op) {
    if (op.is_spill_slot()) {
        return AArch64FrameLayout::spill_slot_address(op.spill_slot, fn_.frame);
    }
    if (op.is_mem()) {
        const auto& m = op.mem_val;
        GPR base = m.base_preg.is_valid() ? m.base_preg.as_aarch64_gpr() : GPR::None;
        GPR index = m.index_preg.is_valid() ? m.index_preg.as_aarch64_gpr() : GPR::None;

        if (base != GPR::None && index != GPR::None) {
            uint8_t shift = 0;
            if (m.scale == x64::Scale::Eight) shift = 3;
            else if (m.scale == x64::Scale::Four) shift = 2;
            else if (m.scale == x64::Scale::Two) shift = 1;

            if (m.disp != 0) {
                if (m.disp >= 0 && m.disp <= 4095) {
                    enc_.add(GPR::X16, base, static_cast<uint32_t>(m.disp));
                } else if (m.disp < 0 && -m.disp <= 4095) {
                    enc_.sub(GPR::X16, base, static_cast<uint32_t>(-m.disp));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(m.disp));
                    enc_.add(GPR::X16, base, GPR::X16);
                }
                return MemAddress::base_index(GPR::X16, index, ExtendType::UXTX, shift);
            }
            return MemAddress::base_index(base, index, ExtendType::UXTX, shift);
        } else if (base != GPR::None) {
            return ptr(base, m.disp);
        } else if (index != GPR::None) {
            return ptr(index, m.disp);
        } else {
            return ptr(GPR::FP, m.disp);
        }
    }
    return ptr(GPR::FP, 0);
}

MemAddress AArch64EmitContext::ensure_accessible_mem(const MemAddress& mem, GPR scratch) {
    if (mem.mode != AddrMode::Offset) return mem;
    int64_t off = mem.offset;
    if (off >= -256 && off <= 16380 && (off >= 0 ? (off % 8 == 0) : true)) {
        return mem;
    }
    enc_.mov(scratch, static_cast<uint64_t>(off));
    enc_.add(scratch, mem.base, scratch);
    return ptr(scratch, 0);
}

AArch64CompilationResult AArch64EmitContext::compile() {
    AArch64CompilationResult result;
    safepoints_.clear();
    stack_map_records_.clear();
    block_labels_.clear();
    pending_exception_scopes_.clear();
    patch_sites_.clear();

    // 1. Compute frame layout
    codegen::FrameInfo mutable_frame = fn_.frame;
    AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

    // 2. Create labels for all blocks
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
    }

    // 3. Emit prologue at entry
    AArch64FrameLayout::emit_prologue(enc_, mutable_frame, fn_.calling_conv);

    // 4. Emit blocks
    for (size_t b_idx = 0; b_idx < fn_.blocks.size(); ++b_idx) {
        const auto& block = fn_.blocks[b_idx];
        if (b_idx > 0) {
            buffer_.align(16);
        }
        buffer_.bind(block_labels_[block->id]);
        result.block_offsets[block->id] = buffer_.size();

        uint32_t next_block_id = (b_idx + 1 < fn_.blocks.size()) ? fn_.blocks[b_idx + 1]->id : UINT32_MAX;
        size_t n_insts = block->instructions.size();

        bool has_jcc_jmp = false;
        bool has_trailing_jmp = false;

        if (n_insts >= 2) {
            const auto& second_last = *block->instructions[n_insts - 2];
            const auto& last = *block->instructions[n_insts - 1];
            if (second_last.opcode == LirOpcode::Jcc && last.opcode == LirOpcode::Jmp &&
                !second_last.uses.empty() && second_last.uses[0].is_label() &&
                !last.uses.empty() && last.uses[0].is_label()) {
                has_jcc_jmp = true;
            }
        }

        if (!has_jcc_jmp && n_insts >= 1) {
            const auto& last = *block->instructions[n_insts - 1];
            if (last.opcode == LirOpcode::Jmp && !last.uses.empty() && last.uses[0].is_label()) {
                has_trailing_jmp = true;
            }
        }

        size_t limit = n_insts;
        if (has_jcc_jmp) limit = n_insts - 2;
        else if (has_trailing_jmp) limit = n_insts - 1;

        for (size_t i_idx = 0; i_idx < limit; ++i_idx) {
            const auto& inst = *block->instructions[i_idx];
            size_t cur_offset = buffer_.size();
            DebugLoc loc = inst.loc;
            if (!loc.is_valid() && inst.mir_origin) {
                loc = inst.mir_origin->loc();
            }
            if (loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(cur_offset), loc);
            }
            emit_instruction(inst, b_idx == 0, i_idx == 0);
        }

        if (has_jcc_jmp) {
            const auto& jcc = *block->instructions[n_insts - 2];
            const auto& jmp = *block->instructions[n_insts - 1];
            if (jcc.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jcc.loc);
            }
            uint32_t true_target = jcc.uses[0].label_id;
            uint32_t false_target = jmp.uses[0].label_id;

            if (false_target == next_block_id) {
                enc_.b(to_aarch64_cond(jcc.condition), block_labels_[true_target]);
            } else if (true_target == next_block_id) {
                enc_.b(to_aarch64_cond(invert(jcc.condition)), block_labels_[false_target]);
            } else {
                enc_.b(to_aarch64_cond(jcc.condition), block_labels_[true_target]);
                enc_.b(block_labels_[false_target]);
            }
        } else if (has_trailing_jmp) {
            const auto& jmp = *block->instructions[n_insts - 1];
            if (jmp.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jmp.loc);
            }
            uint32_t target = jmp.uses[0].label_id;
            if (target != next_block_id) {
                enc_.b(block_labels_[target]);
            }
        }
    }

    // 4.5 Specialized OSR secondary prologue
    if (fn_.osr_entry.enabled && block_labels_.find(fn_.osr_entry.loop_header_id) != block_labels_.end()) {
        buffer_.align(16);
        result.osr_entry_offset = buffer_.size();

        AArch64FrameLayout::emit_prologue(enc_, mutable_frame, fn_.calling_conv);

        // AAPCS64 1st argument (OsrMigrationFrame*) is X0
        enc_.mov(GPR::X16, GPR::X0);

        for (size_t i = 0; i < fn_.osr_entry.live_in_vregs.size(); ++i) {
            VReg vr = fn_.osr_entry.live_in_vregs[i];
            const VRegInfo& info = fn_.get_vreg_info(vr);
            int32_t slot_offset = static_cast<int32_t>(16 + i * 16);

            if (!info.is_spilled && info.assigned_preg.is_valid()) {
                PReg preg = info.assigned_preg;
                if (preg.is_gpr()) {
                    if (vr.size == 4) {
                        enc_.ldr32(preg.as_aarch64_gpr(), ptr(GPR::X16, slot_offset));
                    } else {
                        enc_.ldr(preg.as_aarch64_gpr(), ptr(GPR::X16, slot_offset));
                    }
                } else if (preg.is_xmm()) {
                    if (vr.size == 4) {
                        enc_.ldr_s(preg.as_aarch64_fpr(), ptr(GPR::X16, slot_offset));
                    } else if (vr.size == 16) {
                        enc_.ldr_q(preg.as_aarch64_fpr(), ptr(GPR::X16, slot_offset));
                    } else {
                        enc_.ldr(preg.as_aarch64_fpr(), ptr(GPR::X16, slot_offset));
                    }
                }
            } else if (info.is_spilled && info.assigned_spill_slot >= 0) {
                MemAddress stack_addr = AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, mutable_frame);
                stack_addr = ensure_accessible_mem(stack_addr, GPR::X17);
                if (vr.is_xmm()) {
                    enc_.ldr(FPR::V31, ptr(GPR::X16, slot_offset));
                    enc_.str(FPR::V31, stack_addr);
                } else {
                    enc_.ldr(GPR::X17, ptr(GPR::X16, slot_offset));
                    enc_.str(GPR::X17, stack_addr);
                }
            }
        }

        enc_.b(block_labels_[fn_.osr_entry.loop_header_id]);
    }

    result.code_buffer = std::move(buffer_);
    result.safepoints = std::move(safepoints_);
    result.stack_map.function_name = fn_.name;
    result.stack_map.code_size = static_cast<uint32_t>(result.code_buffer.size());
    result.stack_map.records = std::move(stack_map_records_);
    result.entry_offset = 0;
    result.debug_table.set_function_name(std::string(fn_.name));
    result.debug_table.set_code_size(static_cast<uint32_t>(result.code_buffer.size()));

    // 5. Build resume table
    for (const auto& rp : fn_.resume_entries) {
        auto it = result.block_offsets.find(rp.second);
        if (it != result.block_offsets.end()) {
            std::string block_name;
            auto* blk = fn_.get_block_by_id(rp.second);
            if (blk) block_name = blk->name;
            result.resume_table.add_entry(rp.first, it->second, block_name);
        }
    }
    result.patch_sites = std::move(patch_sites_);

    // 6. Build exception table
    result.exception_table.set_function_name(std::string(fn_.name));
    result.exception_table.set_code_size(static_cast<uint32_t>(result.code_buffer.size()));
    codegen::FrameInfo fn_frame = fn_.frame;
    AArch64FrameLayout::compute_layout(fn_frame, fn_.calling_conv);
    result.exception_table.set_frame_size(static_cast<uint32_t>(fn_frame.total_frame_size));
    result.exception_table.set_saved_callee_gprs(fn_frame.saved_callee_gprs);

    for (const auto& scope : pending_exception_scopes_) {
        auto it = result.block_offsets.find(scope.unwind_block_id);
        if (it != result.block_offsets.end()) {
            result.exception_table.add_scope(
                static_cast<uint32_t>(scope.call_start),
                static_cast<uint32_t>(scope.call_end),
                static_cast<uint32_t>(it->second)
            );
        }
    }

    return result;
}

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
                    enc_.ldr(dst_gpr, ensure_accessible_mem(to_mem_address(src)));
                }
            } else {
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
                GPR data_scratch = (dst_mem.base == GPR::X16 || dst_mem.index == GPR::X16) ? GPR::X15 : GPR::X16;
                if (src.is_preg()) {
                    enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else if (src.is_imm_int()) {
                    enc_.mov(data_scratch, static_cast<uint64_t>(src.imm_int));
                    enc_.str(data_scratch, dst_mem);
                } else {
                    GPR src_scratch = (dst_mem.base == GPR::X17 || dst_mem.index == GPR::X17) ? GPR::X15 : GPR::X17;
                    if (src_scratch == data_scratch) src_scratch = (data_scratch == GPR::X16) ? GPR::X15 : GPR::X16;
                    enc_.ldr(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch));
                    enc_.str(data_scratch, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Mov32: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

            if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) {
                return;
            }

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
                    enc_.ldr32(dst_gpr, ensure_accessible_mem(to_mem_address(src)));
                }
            } else {
                MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
                GPR data_scratch = (dst_mem.base == GPR::X16 || dst_mem.index == GPR::X16) ? GPR::X15 : GPR::X16;
                if (src.is_preg()) {
                    enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else if (src.is_imm_int()) {
                    enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int));
                    enc_.str32(data_scratch, dst_mem);
                } else {
                    GPR src_scratch = (dst_mem.base == GPR::X17 || dst_mem.index == GPR::X17) ? GPR::X15 : GPR::X17;
                    if (src_scratch == data_scratch) src_scratch = (data_scratch == GPR::X16) ? GPR::X15 : GPR::X16;
                    enc_.ldr32(data_scratch, ensure_accessible_mem(to_mem_address(src), src_scratch));
                    enc_.str32(data_scratch, dst_mem);
                }
            }
            break;
        }

        case LirOpcode::Movabs: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_symbol()) {
                const std::string& sym = inst.uses[0].symbol_name;
                size_t off1 = buffer_.size();
                buffer_.emit_inst(0x90000000u | reg_code(dst_gpr));
                buffer_.add_relocation(off1, RelocationKind::Page21, sym);
                size_t off2 = buffer_.size();
                enc_.add(dst_gpr, dst_gpr, 0);
                buffer_.add_relocation(off2, RelocationKind::PageOff12, sym);
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
                enc_.mov(dst_gpr, static_cast<uint64_t>(inst.uses[0].imm_int));
            }
            break;
        }

        case LirOpcode::Movsxd: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxtw(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsw(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0])));
            }
            break;
        }

        case LirOpcode::Movzx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.uxtb(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrb(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0])));
            }
            break;
        }

        case LirOpcode::Movzx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.uxth(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrh(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0])));
            }
            break;
        }

        case LirOpcode::Movsx8: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxtb(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsb(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0])));
            }
            break;
        }

        case LirOpcode::Movsx16: {
            GPR dst_gpr = inst.defs[0].preg_val.as_aarch64_gpr();
            if (inst.uses[0].is_preg()) {
                enc_.sxth(dst_gpr, inst.uses[0].preg_val.as_aarch64_gpr());
            } else {
                enc_.ldrsh(dst_gpr, ensure_accessible_mem(to_mem_address(inst.uses[0])));
            }
            break;
        }

        default:
            break;
    }
}

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
                enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                enc_.and_(dst, src1, GPR::X16);
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
                enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                enc_.and32(dst, src1, GPR::X16);
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
                enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                enc_.orr(dst, src1, GPR::X16);
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
                enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                enc_.orr32(dst, src1, GPR::X16);
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
                enc_.mov(GPR::X16, static_cast<uint64_t>(src2.imm_int));
                enc_.eor(dst, src1, GPR::X16);
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
                enc_.mov32(GPR::X16, static_cast<uint32_t>(src2.imm_int));
                enc_.eor32(dst, src1, GPR::X16);
            } else {
                enc_.ldr32(GPR::X16, ensure_accessible_mem(to_mem_address(src2)));
                enc_.eor32(dst, src1, GPR::X16);
            }
            break;
        }

        case LirOpcode::Not: enc_.mvn(to_gpr(inst.defs[0]), to_gpr(inst.uses[0])); break;
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
                enc_.mov(GPR::X16, static_cast<uint64_t>(op1.imm_int));
                enc_.tst(op0, GPR::X16);
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
                enc_.mov32(GPR::X16, static_cast<uint32_t>(op1.imm_int));
                enc_.tst32(op0, GPR::X16);
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

        default:
            break;
    }
}

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
            size_t s2_idx = (inst.uses.size() >= 3) ? 1 : 0;
            size_t s3_idx = (inst.uses.size() >= 3) ? 2 : 1;
            FPR src2 = to_fpr(inst.uses[s2_idx]);
            const auto& src3_op = inst.uses[s3_idx];

            if (is_213) {
                // 213: dst = (dst * src2) + src3
                // In AArch64, fmla Vd, Vn, Vm computes Vd = Vd + (Vn * Vm).
                if (src3_op.is_preg()) {
                    FPR src3 = to_fpr(src3_op);
                    if (src3 == dst) {
                        if (is_double) enc_.vec_fmla_2d(dst, dst, src2);
                        else enc_.vec_fmla_4s(dst, dst, src2);
                    } else {
                        enc_.vec_orr(FPR::V31, src3, src3);
                        if (is_double) enc_.vec_fmla_2d(FPR::V31, dst, src2);
                        else enc_.vec_fmla_4s(FPR::V31, dst, src2);
                        enc_.vec_orr(dst, FPR::V31, FPR::V31);
                    }
                } else {
                    enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
                    if (is_double) enc_.vec_fmla_2d(FPR::V31, dst, src2);
                    else enc_.vec_fmla_4s(FPR::V31, dst, src2);
                    enc_.vec_orr(dst, FPR::V31, FPR::V31);
                }
            } else {
                // 231: dst = (src2 * src3) + dst
                if (src3_op.is_preg()) {
                    FPR src3 = to_fpr(src3_op);
                    if (is_double) enc_.vec_fmla_2d(dst, src2, src3);
                    else enc_.vec_fmla_4s(dst, src2, src3);
                } else {
                    enc_.ldr_q(FPR::V31, ensure_accessible_mem(to_mem_address(src3_op)));
                    if (is_double) enc_.vec_fmla_2d(dst, src2, FPR::V31);
                    else enc_.vec_fmla_4s(dst, src2, FPR::V31);
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
                FPR tmp = (src == FPR::V31) ? FPR::V29 : FPR::V31;
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
                FPR tmp = (v0 == FPR::V31 || v1 == FPR::V31) ? FPR::V29 : FPR::V31;
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
            FPR tmp = (v0 == FPR::V31 || v1 == FPR::V31) ? FPR::V29 : FPR::V31;
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

void AArch64EmitContext::emit_parallel_copy(const LirInst& inst) {
    size_t n = inst.defs.size();
    if (n == 0 || n != inst.uses.size()) return;

    struct Move {
        LirOperand dst;
        LirOperand src;
        bool done = false;
    };

    std::vector<Move> moves;
    for (size_t i = 0; i < n; ++i) {
        if (inst.defs[i].is_preg() && inst.uses[i].is_preg() &&
            inst.defs[i].preg_val == inst.uses[i].preg_val) {
            continue;
        }
        if (inst.defs[i].is_spill_slot() && inst.uses[i].is_spill_slot() &&
            inst.defs[i].spill_slot == inst.uses[i].spill_slot) {
            continue;
        }
        moves.push_back({inst.defs[i], inst.uses[i], false});
    }

    auto emit_move = [&](const LirOperand& dst, const LirOperand& src) {
        if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;
        if (dst.is_spill_slot() && src.is_spill_slot() && dst.spill_slot == src.spill_slot) return;

        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr()) {
                GPR dst_gpr = dst.preg_val.as_aarch64_gpr();
                if (src.is_preg()) {
                    GPR src_gpr = src.preg_val.as_aarch64_gpr();
                    if (dst.size == 4 && src.size == 4) enc_.mov32(dst_gpr, src_gpr);
                    else enc_.mov(dst_gpr, src_gpr);
                } else if (src.is_imm_int()) {
                    if (dst.size == 4) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                    else enc_.mov(dst_gpr, static_cast<uint64_t>(src.imm_int));
                } else {
                    if (dst.size == 4) enc_.ldr32(dst_gpr, ensure_accessible_mem(to_mem_address(src)));
                    else enc_.ldr(dst_gpr, ensure_accessible_mem(to_mem_address(src)));
                }
            } else {
                FPR dst_fpr = dst.preg_val.as_aarch64_fpr();
                if (dst.size == 16 || src.size == 16) {
                    if (src.is_preg()) enc_.vec_orr(dst_fpr, src.preg_val.as_aarch64_fpr(), src.preg_val.as_aarch64_fpr());
                    else enc_.ldr_q(dst_fpr, ensure_accessible_mem(to_mem_address(src)));
                } else if (dst.size == 4 || src.size == 4) {
                    if (src.is_preg()) enc_.fmov_s(dst_fpr, src.preg_val.as_aarch64_fpr());
                    else enc_.ldr_s(dst_fpr, ensure_accessible_mem(to_mem_address(src)));
                } else {
                    if (src.is_preg()) enc_.fmov(dst_fpr, src.preg_val.as_aarch64_fpr());
                    else enc_.ldr(dst_fpr, ensure_accessible_mem(to_mem_address(src)));
                }
            }
        } else {
            MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst), GPR::X17);
            GPR data_scratch = (dst_mem.base == GPR::X16 || dst_mem.index == GPR::X16) ? GPR::X15 : GPR::X16;
            if (src.is_preg()) {
                if (src.preg_val.is_gpr()) {
                    if (dst.size == 4 || src.size == 4) enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else {
                    if (dst.size == 16 || src.size == 16) enc_.str_q(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str_s(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_fpr(), dst_mem);
                }
            } else if (src.is_imm_int()) {
                if (dst.size == 4) {
                    enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int));
                    enc_.str32(data_scratch, dst_mem);
                } else {
                    enc_.mov(data_scratch, static_cast<uint64_t>(src.imm_int));
                    enc_.str(data_scratch, dst_mem);
                }
            } else {
                GPR src_scratch = (dst_mem.base == GPR::X17 || dst_mem.index == GPR::X17) ? GPR::X15 : GPR::X17;
                if (src_scratch == data_scratch) src_scratch = (data_scratch == GPR::X16) ? GPR::X15 : GPR::X16;
                MemAddress src_mem = ensure_accessible_mem(to_mem_address(src), src_scratch);
                if (dst.size == 16 || src.size == 16) {
                    enc_.ldr_q(FPR::V31, src_mem);
                    enc_.str_q(FPR::V31, dst_mem);
                } else if (dst.size == 4 || src.size == 4) {
                    enc_.ldr32(data_scratch, src_mem);
                    enc_.str32(data_scratch, dst_mem);
                } else {
                    enc_.ldr(data_scratch, src_mem);
                    enc_.str(data_scratch, dst_mem);
                }
            }
        }
    };

    auto peel_acyclic = [&]() -> bool {
        bool progress = false;
        for (auto& m : moves) {
            if (m.done) continue;
            bool dst_used = false;
            for (const auto& other : moves) {
                if (other.done) continue;
                if (m.dst.is_preg() && other.src.is_preg() &&
                    m.dst.preg_val == other.src.preg_val) {
                    dst_used = true;
                    break;
                }
                if (m.dst.is_spill_slot() && other.src.is_spill_slot() &&
                    m.dst.spill_slot == other.src.spill_slot) {
                    dst_used = true;
                    break;
                }
            }
            if (!dst_used) {
                emit_move(m.dst, m.src);
                m.done = true;
                progress = true;
            }
        }
        return progress;
    };

    while (true) {
        while (peel_acyclic()) {}

        size_t start_idx = SIZE_MAX;
        for (size_t i = 0; i < moves.size(); ++i) {
            if (!moves[i].done) {
                start_idx = i;
                break;
            }
        }
        if (start_idx == SIZE_MAX) break;

        // Trace permutation cycle
        std::vector<size_t> cycle;
        size_t curr = start_idx;
        while (true) {
            cycle.push_back(curr);
            size_t next_idx = SIZE_MAX;
            for (size_t i = 0; i < moves.size(); ++i) {
                if (!moves[i].done) {
                    bool match = false;
                    if (moves[i].dst.is_preg() && moves[curr].src.is_preg() &&
                        moves[i].dst.preg_val == moves[curr].src.preg_val) {
                        match = true;
                    } else if (moves[i].dst.is_spill_slot() && moves[curr].src.is_spill_slot() &&
                               moves[i].dst.spill_slot == moves[curr].src.spill_slot) {
                        match = true;
                    }
                    if (match) {
                        next_idx = i;
                        break;
                    }
                }
            }
            if (next_idx == SIZE_MAX || next_idx == start_idx) {
                break;
            }
            bool already_in_cycle = false;
            for (size_t c : cycle) {
                if (c == next_idx) {
                    already_in_cycle = true;
                    break;
                }
            }
            if (already_in_cycle) break;
            curr = next_idx;
        }

        if (!cycle.empty()) {
            size_t idx0 = cycle[0];
            auto& m0 = moves[idx0];
            bool is_fpr = (m0.dst.is_preg() && m0.dst.preg_val.is_xmm()) ||
                          (m0.src.is_preg() && m0.src.preg_val.is_xmm()) ||
                          (m0.dst.size == 16);
            PReg scratch = is_fpr ? PReg::aarch64_fpr(FPR::V31) : PReg::aarch64_gpr(GPR::X16);
            uint8_t sz = m0.dst.size;

            emit_move(LirOperand::preg(scratch, sz), m0.dst);
            size_t last_idx = cycle.back();
            moves[last_idx].src = LirOperand::preg(scratch, sz);
        } else {
            emit_move(moves[start_idx].dst, moves[start_idx].src);
            moves[start_idx].done = true;
        }
    }
}

void AArch64EmitContext::emit_control_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Jmp:
            enc_.b(block_labels_[inst.uses[0].label_id]);
            break;

        case LirOpcode::Jcc:
            enc_.b(to_aarch64_cond(inst.condition), block_labels_[inst.uses[0].label_id]);
            break;

        case LirOpcode::Call: {
            size_t call_start = buffer_.size();
            const auto& sym_op = inst.uses.back();
            std::string callee = inst.callee_symbol.empty() ? sym_op.symbol_name : inst.callee_symbol;
            if (inst.is_patchable) {
                size_t site_start = buffer_.size();
                enc_.bl(callee);
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Call,
                    site_start,
                    0,
                    4,
                    0,
                    callee
                );
            } else {
                enc_.bl(callee);
            }
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            codegen::FrameInfo mutable_frame = fn_.frame;
            AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(mutable_frame.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, mutable_frame).offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), mutable_frame).offset);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }

        case LirOpcode::CallIndirect: {
            size_t call_start = buffer_.size();
            enc_.blr(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            codegen::FrameInfo mutable_frame = fn_.frame;
            AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(mutable_frame.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, mutable_frame).offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), mutable_frame).offset);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }

        case LirOpcode::Ret: {
            codegen::FrameInfo mutable_frame = fn_.frame;
            AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
            AArch64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            break;
        }

        case LirOpcode::Push:
            enc_.str(to_gpr(inst.uses[0]), pre_idx(GPR::SP, -16));
            break;

        case LirOpcode::Pop:
            enc_.ldr(to_gpr(inst.defs[0]), post_idx(GPR::SP, 16));
            break;

        case LirOpcode::Lea: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& op = inst.uses[0];
            if (op.is_spill_slot()) {
                MemAddress addr = AArch64FrameLayout::spill_slot_address(op.spill_slot, fn_.frame);
                if (addr.offset >= 0 && addr.offset <= 4095) {
                    enc_.add(dst, GPR::FP, static_cast<uint32_t>(addr.offset));
                } else {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(addr.offset));
                    enc_.add(dst, GPR::FP, GPR::X16);
                }
            } else if (op.is_mem()) {
                const auto& m = op.mem_val;
                GPR base = m.base_preg.is_valid() ? m.base_preg.as_aarch64_gpr() : GPR::None;
                GPR index = m.index_preg.is_valid() ? m.index_preg.as_aarch64_gpr() : GPR::None;
                if (base != GPR::None && index != GPR::None) {
                    uint8_t shift = 0;
                    if (m.scale == x64::Scale::Eight) shift = 3;
                    else if (m.scale == x64::Scale::Four) shift = 2;
                    else if (m.scale == x64::Scale::Two) shift = 1;

                    if (shift > 0) {
                        enc_.lsl(GPR::X16, index, shift);
                        enc_.add(dst, base, GPR::X16);
                    } else {
                        enc_.add(dst, base, index);
                    }
                    if (m.disp != 0) {
                        if (m.disp >= 0 && m.disp <= 4095) enc_.add(dst, dst, static_cast<uint32_t>(m.disp));
                        else if (m.disp < 0 && -m.disp <= 4095) enc_.sub(dst, dst, static_cast<uint32_t>(-m.disp));
                        else {
                            enc_.mov(GPR::X16, static_cast<uint64_t>(m.disp));
                            enc_.add(dst, dst, GPR::X16);
                        }
                    }
                } else if (base != GPR::None) {
                    if (m.disp >= 0 && m.disp <= 4095) enc_.add(dst, base, static_cast<uint32_t>(m.disp));
                    else if (m.disp < 0 && -m.disp <= 4095) enc_.sub(dst, base, static_cast<uint32_t>(-m.disp));
                    else {
                        enc_.mov(GPR::X16, static_cast<uint64_t>(m.disp));
                        enc_.add(dst, base, GPR::X16);
                    }
                } else if (index != GPR::None) {
                    if (m.disp >= 0 && m.disp <= 4095) enc_.add(dst, index, static_cast<uint32_t>(m.disp));
                    else if (m.disp < 0 && -m.disp <= 4095) enc_.sub(dst, index, static_cast<uint32_t>(-m.disp));
                    else {
                        enc_.mov(GPR::X16, static_cast<uint64_t>(m.disp));
                        enc_.add(dst, index, GPR::X16);
                    }
                } else {
                    if (m.disp >= 0 && m.disp <= 4095) enc_.add(dst, GPR::FP, static_cast<uint32_t>(m.disp));
                    else if (m.disp < 0 && -m.disp <= 4095) enc_.sub(dst, GPR::FP, static_cast<uint32_t>(-m.disp));
                    else {
                        enc_.mov(GPR::X16, static_cast<uint64_t>(m.disp));
                        enc_.add(dst, GPR::FP, GPR::X16);
                    }
                }
            }
            break;
        }

        case LirOpcode::Safepoint: {
            enc_.bl("brass_gc_safepoint");
            size_t return_offset = buffer_.size();

            codegen::FrameInfo mutable_frame = fn_.frame;
            AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

            SafepointRecord rec;
            rec.code_offset = return_offset;
            rec.safepoint_id = inst.safepoint_id;

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(mutable_frame.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, mutable_frame).offset);
                    rec.live_gcref_spill_offsets.push_back(offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    rec.live_gcref_registers.push_back(static_cast<x64::GPR>(info.assigned_preg.code));
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), mutable_frame).offset);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            safepoints_.push_back(std::move(rec));
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }

        case LirOpcode::GuardExit: {
            size_t num_uses = inst.uses.size();
            size_t slots_bytes = num_uses * 8;
            size_t total_alloc = (slots_bytes + 15) & ~size_t(15);

            if (total_alloc > 0) enc_.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));

            for (size_t i = 0; i < num_uses; ++i) {
                const auto& op = inst.uses[i];
                int64_t slot_offset = static_cast<int64_t>(i * 8);

                if (op.is_preg()) {
                    if (op.preg_val.is_gpr()) {
                        GPR src_gpr = op.preg_val.as_aarch64_gpr();
                        if (op.size == 4) {
                            enc_.sxtw(GPR::X16, src_gpr);
                            enc_.str(GPR::X16, ptr(GPR::SP, slot_offset));
                        } else {
                            enc_.str(src_gpr, ptr(GPR::SP, slot_offset));
                        }
                    } else if (op.preg_val.is_xmm()) {
                        enc_.str(op.preg_val.as_aarch64_fpr(), ptr(GPR::SP, slot_offset));
                    }
                } else if (op.is_spill_slot() || op.is_mem()) {
                    MemAddress src_mem = ensure_accessible_mem(to_mem_address(op));
                    enc_.ldr(GPR::X16, src_mem);
                    enc_.str(GPR::X16, ptr(GPR::SP, slot_offset));
                } else if (op.is_imm_int()) {
                    enc_.mov(GPR::X16, static_cast<uint64_t>(op.imm_int));
                    enc_.str(GPR::X16, ptr(GPR::SP, slot_offset));
                }
            }

            uint32_t rid = inst.resume_id;
            uint32_t rsn = inst.deopt_reason == 0 ? 1 : inst.deopt_reason;
            uint32_t cnt = static_cast<uint32_t>(num_uses);

            // AAPCS64 parameter registers: X0, X1, X2, X3
            enc_.mov32(GPR::X0, rid);
            enc_.mov32(GPR::X1, rsn);
            enc_.mov32(GPR::X2, cnt);
            if (num_uses > 0) enc_.mov(GPR::X3, GPR::SP);
            else enc_.mov(GPR::X3, GPR::XZR);

            enc_.bl("brass_deopt_exit");

            if (!inst.exit_symbol.empty()) {
                if (total_alloc > 0) enc_.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));
                enc_.bl("brass_get_thread_deopt_frame");
                // X0 now holds DeoptFrame*
                enc_.add(GPR::X1, GPR::X0, static_cast<uint32_t>(offsetof(runtime::DeoptFrame, slots)));
                enc_.mov32(GPR::X0, rid);
                enc_.bl(inst.exit_symbol);

                codegen::FrameInfo mutable_frame = fn_.frame;
                AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
                AArch64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            } else {
                if (total_alloc > 0) enc_.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));
                codegen::FrameInfo mutable_frame = fn_.frame;
                AArch64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
                AArch64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            }
            break;
        }

        default:
            break;
    }
}

void AArch64EmitContext::emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst) {
    (void)is_entry_block;
    (void)is_first_inst;

    switch (inst.opcode) {
        case LirOpcode::Nop:
            enc_.nop();
            break;

        case LirOpcode::Mov:
        case LirOpcode::Mov32:
        case LirOpcode::Movabs:
        case LirOpcode::Movsxd:
        case LirOpcode::Movzx8:
        case LirOpcode::Movzx16:
        case LirOpcode::Movsx8:
        case LirOpcode::Movsx16:
            emit_mov_instruction(inst);
            break;

        case LirOpcode::Add:
        case LirOpcode::Add32:
        case LirOpcode::Adds:
        case LirOpcode::Adds32:
        case LirOpcode::Sub:
        case LirOpcode::Sub32:
        case LirOpcode::Subs:
        case LirOpcode::Subs32:
        case LirOpcode::Imul:
        case LirOpcode::Imul32:
        case LirOpcode::Smulh:
        case LirOpcode::Umulh:
        case LirOpcode::Idiv:
        case LirOpcode::Idiv32:
        case LirOpcode::Div:
        case LirOpcode::Div32:
        case LirOpcode::Cdq:
        case LirOpcode::Cqo:
        case LirOpcode::And:
        case LirOpcode::And32:
        case LirOpcode::Or:
        case LirOpcode::Or32:
        case LirOpcode::Xor:
        case LirOpcode::Xor32:
        case LirOpcode::Not:
        case LirOpcode::Not32:
        case LirOpcode::Neg:
        case LirOpcode::Neg32:
        case LirOpcode::Shl:
        case LirOpcode::Shl32:
        case LirOpcode::Shr:
        case LirOpcode::Shr32:
        case LirOpcode::Sar:
        case LirOpcode::Sar32:
        case LirOpcode::Popcnt:
        case LirOpcode::Popcnt32:
        case LirOpcode::Lzcnt:
        case LirOpcode::Lzcnt32:
        case LirOpcode::Tzcnt:
        case LirOpcode::Tzcnt32:
        case LirOpcode::Bsr:
        case LirOpcode::Bsr32:
        case LirOpcode::Bsf:
        case LirOpcode::Bsf32:
        case LirOpcode::Cmp:
        case LirOpcode::Cmp32:
        case LirOpcode::Test:
        case LirOpcode::Test32:
        case LirOpcode::Setcc:
        case LirOpcode::Cmovcc:
            emit_alu_instruction(inst);
            break;

        case LirOpcode::Movsd:
        case LirOpcode::Movss:
        case LirOpcode::Movq_gx:
        case LirOpcode::Movq_xg:
        case LirOpcode::Addsd:
        case LirOpcode::Addss:
        case LirOpcode::Subsd:
        case LirOpcode::Subss:
        case LirOpcode::Mulsd:
        case LirOpcode::Mulss:
        case LirOpcode::Divsd:
        case LirOpcode::Divss:
        case LirOpcode::Sqrtsd:
        case LirOpcode::Sqrtss:
        case LirOpcode::Vfmadd213ss:
        case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd:
        case LirOpcode::Vfmadd231sd:
        case LirOpcode::Ucomisd:
        case LirOpcode::Ucomiss:
        case LirOpcode::Xorpd:
        case LirOpcode::Fneg:
        case LirOpcode::Fneg32:
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si:
        case LirOpcode::Cvttsd2si32:
            emit_fp_instruction(inst);
            break;

        case LirOpcode::Movaps:
        case LirOpcode::Movups:
        case LirOpcode::Movd_xg:
        case LirOpcode::Movd_gx:
        case LirOpcode::Addps:
        case LirOpcode::Subps:
        case LirOpcode::Mulps:
        case LirOpcode::Divps:
        case LirOpcode::Minps:
        case LirOpcode::Maxps:
        case LirOpcode::Sqrtps:
        case LirOpcode::Vfmadd213ps:
        case LirOpcode::Vfmadd231ps:
        case LirOpcode::Fneg4s:
        case LirOpcode::Addpd:
        case LirOpcode::Subpd:
        case LirOpcode::Mulpd:
        case LirOpcode::Divpd:
        case LirOpcode::Minpd:
        case LirOpcode::Maxpd:
        case LirOpcode::Sqrtpd:
        case LirOpcode::Vfmadd213pd:
        case LirOpcode::Vfmadd231pd:
        case LirOpcode::Fneg2d:
        case LirOpcode::Paddd:
        case LirOpcode::Psubd:
        case LirOpcode::Pmulld:
        case LirOpcode::Pminsd:
        case LirOpcode::Pmaxsd:
        case LirOpcode::Paddq:
        case LirOpcode::Psubq:
        case LirOpcode::Pand:
        case LirOpcode::Por:
        case LirOpcode::Pxor:
        case LirOpcode::Pnot:
        case LirOpcode::Pinsrd:
        case LirOpcode::Pinsrq:
        case LirOpcode::Pextrd:
        case LirOpcode::Pextrq:
        case LirOpcode::Extractps:
        case LirOpcode::Insertps:
        case LirOpcode::Xorps:
        case LirOpcode::Vbroadcastss:
        case LirOpcode::Vbroadcastsd:
        case LirOpcode::Vpbroadcastd:
        case LirOpcode::Vpbroadcastq:
        case LirOpcode::Pshufd:
        case LirOpcode::Shufps:
        case LirOpcode::Shufpd:
        case LirOpcode::Movddup:
        case LirOpcode::Pcmpeqd:
        case LirOpcode::Pslld:
        case LirOpcode::Psllq:
        case LirOpcode::Vmovaps:
        case LirOpcode::Vmovups:
        case LirOpcode::Vaddps:
        case LirOpcode::Vsubps:
        case LirOpcode::Vmulps:
        case LirOpcode::Vdivps:
        case LirOpcode::Vminps:
        case LirOpcode::Vmaxps:
        case LirOpcode::Vaddpd:
        case LirOpcode::Vsubpd:
        case LirOpcode::Vmulpd:
        case LirOpcode::Vdivpd:
        case LirOpcode::Vminpd:
        case LirOpcode::Vmaxpd:
        case LirOpcode::Vpaddd:
        case LirOpcode::Vpsubd:
        case LirOpcode::Vpmulld:
        case LirOpcode::Vpaddq:
        case LirOpcode::Vpsubq:
        case LirOpcode::Vandps:
        case LirOpcode::Vorps:
        case LirOpcode::Vxorps:
        case LirOpcode::Vandpd:
        case LirOpcode::Vorpd:
        case LirOpcode::Vxorpd:
        case LirOpcode::Vpand:
        case LirOpcode::Vpor:
        case LirOpcode::Vpxor:
            emit_vec_instruction(inst);
            break;

        case LirOpcode::ParallelCopy:
            emit_parallel_copy(inst);
            break;

        default:
            emit_control_instruction(inst);
            break;
    }
}

AArch64CompilationResult compile_lir_to_aarch64(const LirFunction& fn, const Target& target) {
    AArch64EmitContext ctx(fn, target);
    return ctx.compile();
}

} // namespace brass::aarch64
