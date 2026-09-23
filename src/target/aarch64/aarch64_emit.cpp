#include <brass/target/aarch64/aarch64_emit.hpp>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;


AArch64EmitContext::AArch64EmitContext(const LirFunction& fn, const Target& target)
    : fn_(fn), target_(target), enc_(buffer_) {
    frame_ = fn_.frame;
    AArch64FrameLayout::compute_layout(frame_, fn_.calling_conv);
}

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
        if (op.spill_slot < 0) {
            int32_t caller_offset = -1 - op.spill_slot;
            return AArch64FrameLayout::incoming_arg_address(caller_offset, frame_);
        }
        return AArch64FrameLayout::spill_slot_address(op.spill_slot, frame_);
    }
    if (op.is_local_slot()) {
        return AArch64FrameLayout::local_frame_address(op.local_offset, frame_);
    }
    if (op.is_mem()) {
        const auto& m = op.mem_val;
        GPR base = m.base_preg.is_valid() ? m.base_preg.as_aarch64_gpr() : GPR::None;
        if (base == GPR::FP && m.disp < 0 && !m.index_preg.is_valid() && !m.index_vreg.is_valid()) {
            int32_t caller_offset = -1 - m.disp;
            return AArch64FrameLayout::incoming_arg_address(caller_offset, frame_);
        }
        GPR index = m.index_preg.is_valid() ? m.index_preg.as_aarch64_gpr() : GPR::None;

        if (base != GPR::None && index != GPR::None) {
            uint8_t shift = 0;
            if (m.scale == x64::Scale::Eight) shift = 3;
            else if (m.scale == x64::Scale::Four) shift = 2;
            else if (m.scale == x64::Scale::Two) shift = 1;

            if (m.disp != 0) {
                GPR disp_scratch = (base == GPR::X16 || index == GPR::X16) ? GPR::X17 : GPR::X16;
                if (m.disp >= 0 && m.disp <= 4095) {
                    enc_.add(disp_scratch, base, static_cast<uint32_t>(m.disp));
                } else if (m.disp < 0 && -m.disp <= 4095) {
                    enc_.sub(disp_scratch, base, static_cast<uint32_t>(-m.disp));
                } else {
                    enc_.mov(disp_scratch, static_cast<uint64_t>(m.disp));
                    enc_.add(disp_scratch, base, disp_scratch);
                }
                return MemAddress::base_index(disp_scratch, index, ExtendType::UXTX, shift);
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

MemAddress AArch64EmitContext::ensure_accessible_mem(const MemAddress& mem, GPR scratch, int size_bytes) {
    if (mem.mode != AddrMode::Offset) return mem;
    int64_t off = mem.offset;
    if (off >= -256 && off <= 255) {
        return mem;
    }
    if (size_bytes > 0 && off >= 0 && (off % size_bytes == 0) && (off / size_bytes <= 4095)) {
        return mem;
    }
    if (off >= 0 && off <= 4095) {
        enc_.add(scratch, mem.base, static_cast<uint32_t>(off));
    } else if (off < 0 && -off <= 4095) {
        enc_.sub(scratch, mem.base, static_cast<uint32_t>(-off));
    } else {
        enc_.mov(scratch, static_cast<uint64_t>(off));
        enc_.add(scratch, mem.base, scratch);
    }
    return ptr(scratch, 0);
}

AArch64CompilationResult AArch64EmitContext::compile() {
    // Short branches whose label ends up out of reach are re-emitted in the
    // long form on the next pass (CodeBuffer::relax_requests). Every pass
    // turns at least one more site long and a long site never asks again, so
    // this ends; in practice a function under 1 MB takes one pass.
    std::vector<uint32_t> long_sites;
    for (;;) {
        AArch64CompilationResult result = compile_pass(long_sites);
        const auto& requests = result.code_buffer.relax_requests();
        if (requests.empty()) return result;
        const size_t before = long_sites.size();
        for (uint32_t site : requests) {
            if (std::find(long_sites.begin(), long_sites.end(), site) == long_sites.end()) {
                long_sites.push_back(site);
            }
        }
        if (long_sites.size() == before) {
            throw std::runtime_error("AArch64 emit: branch relaxation made no progress in '" + fn_.name + "'");
        }
    }
}

AArch64CompilationResult AArch64EmitContext::compile_pass(const std::vector<uint32_t>& long_branch_sites) {
    AArch64CompilationResult result;
    buffer_ = CodeBuffer();
    buffer_.add_long_branch_sites(long_branch_sites);
    safepoints_.clear();
    stack_map_records_.clear();
    block_labels_.clear();
    pending_exception_scopes_.clear();
    patch_sites_.clear();

    // 1. Compute frame layout
    frame_ = fn_.frame;
    AArch64FrameLayout::compute_layout(frame_, fn_.calling_conv);

    // 2. Create labels for all blocks
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
    }

    // 3. Emit prologue at entry
    AArch64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);

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

        AArch64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);

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
                MemAddress stack_addr = AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_);
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
    result.exception_table.set_frame_size(static_cast<uint32_t>(frame_.total_frame_size));
    result.exception_table.set_saved_callee_gprs(frame_.saved_callee_gprs);

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

void AArch64EmitContext::emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst) {
    (void)is_entry_block;
    (void)is_first_inst;

    switch (inst.opcode) {
        case LirOpcode::Nop:
            enc_.nop();
            break;
        case LirOpcode::Trap:
            enc_.brk(0);
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
        case LirOpcode::Xorps:
        case LirOpcode::Fneg:
        case LirOpcode::Fneg32:
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si:
        case LirOpcode::Cvttsd2si32:
        case LirOpcode::Cvtsi2ss:
        case LirOpcode::Cvtsi2ss32:
        case LirOpcode::Cvttss2si:
        case LirOpcode::Cvttss2si32:
        case LirOpcode::Cvtsd2ss:
        case LirOpcode::Cvtss2sd:
        case LirOpcode::Floor32:
        case LirOpcode::Floor64:
        case LirOpcode::Ceil32:
        case LirOpcode::Ceil64:
        case LirOpcode::Round32:
        case LirOpcode::Round64:
        case LirOpcode::Fabs32:
        case LirOpcode::Fabs64:
        case LirOpcode::Minss:
        case LirOpcode::Minsd:
        case LirOpcode::Maxss:
        case LirOpcode::Maxsd:
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

        case LirOpcode::Jmp:
        case LirOpcode::Jcc:
        case LirOpcode::Call:
        case LirOpcode::CallIndirect:
        case LirOpcode::Ret:
        case LirOpcode::Push:
        case LirOpcode::Pop:
        case LirOpcode::Lea:
        case LirOpcode::Safepoint:
        case LirOpcode::GuardExit:
            emit_control_instruction(inst);
            break;

        default:
            throw_unsupported("aarch64 emit", to_string(inst.opcode));
    }
}

AArch64CompilationResult compile_lir_to_aarch64(const LirFunction& fn, const Target& target) {
    AArch64EmitContext ctx(fn, target);
    return ctx.compile();
}

} // namespace brass::aarch64
