#include <brass/codegen/emit_context.hpp>
#include <iostream>
#include <stdexcept>

namespace brass::codegen {

using namespace brass::x64;

EmitContext::EmitContext(const LirFunction& fn, const Target& target)
    : fn_(fn), target_(target), enc_(buffer_) {}

GPR EmitContext::to_gpr(const LirOperand& op) const {
    if (op.is_preg() && op.preg_val.is_gpr()) {
        return op.preg_val.as_gpr();
    }
    return GPR::None;
}

XMM EmitContext::to_xmm(const LirOperand& op) const {
    if (op.is_preg() && op.preg_val.is_xmm()) {
        return op.preg_val.as_xmm();
    }
    return XMM::None;
}

MemAddress EmitContext::to_mem_address(const LirOperand& op) const {
    if (op.is_spill_slot()) {
        return X64FrameLayout::spill_slot_address(op.spill_slot, fn_.frame);
    }
    if (op.is_mem()) {
        const auto& m = op.mem_val;
        GPR base = m.base_preg.is_valid() ? m.base_preg.as_gpr() : GPR::None;
        GPR index = m.index_preg.is_valid() ? m.index_preg.as_gpr() : GPR::None;

        if (base != GPR::None && index != GPR::None) {
            return ptr(base, index, m.scale, m.disp);
        } else if (base != GPR::None) {
            return ptr(base, m.disp);
        } else if (index != GPR::None) {
            return MemAddress::index_disp(index, m.scale, m.disp);
        } else {
            return ptr(GPR::RBP, m.disp);
        }
    }
    return ptr(GPR::RBP, 0);
}

CompilationResult EmitContext::compile() {
    CompilationResult result;
    safepoints_.clear();
    stack_map_records_.clear();
    block_labels_.clear();

    // 1. Compute frame layout
    codegen::FrameInfo mutable_frame = fn_.frame;
    X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

    // 2. Create labels for all blocks
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
    }

    // 3. Emit prologue at entry
    X64FrameLayout::emit_prologue(enc_, mutable_frame, fn_.calling_conv);

    // 4. Emit blocks
    for (size_t b_idx = 0; b_idx < fn_.blocks.size(); ++b_idx) {
        const auto& block = fn_.blocks[b_idx];
        if (b_idx > 0) {
            // Ensure loop headers and branch targets are 16-byte aligned.
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
                enc_.j(jcc.condition, block_labels_[true_target]);
            } else if (true_target == next_block_id) {
                enc_.j(invert(jcc.condition), block_labels_[false_target]);
            } else {
                enc_.j(jcc.condition, block_labels_[true_target]);
                enc_.jmp(block_labels_[false_target]);
            }
        } else if (has_trailing_jmp) {
            const auto& jmp = *block->instructions[n_insts - 1];
            if (jmp.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jmp.loc);
            }
            uint32_t target = jmp.uses[0].label_id;
            if (target != next_block_id) {
                enc_.jmp(block_labels_[target]);
            }
        }
    }

    // 4.5 Emit specialized OSR secondary prologue
    if (fn_.osr_entry.enabled && block_labels_.find(fn_.osr_entry.loop_header_id) != block_labels_.end()) {
        buffer_.align(16);
        result.osr_entry_offset = buffer_.size();

        // Secondary prologue: establishes native stack frame
        X64FrameLayout::emit_prologue(enc_, mutable_frame, fn_.calling_conv);

        // Standard calling convention: 1st argument (OsrMigrationFrame*)
        // On Win64: RCX; on SysV: RDI
        GPR arg_reg = (fn_.calling_conv.kind() == CallingConvKind::Win64) ? GPR::RCX : GPR::RDI;
        // Copy to R10 (scratch register never assigned by linear scan regalloc)
        enc_.mov(GPR::R10, arg_reg);

        // Unpack migration frame slots directly into physical registers and spill slots
        for (size_t i = 0; i < fn_.osr_entry.live_in_vregs.size(); ++i) {
            VReg vr = fn_.osr_entry.live_in_vregs[i];
            const VRegInfo& info = fn_.get_vreg_info(vr);
            int32_t slot_offset = static_cast<int32_t>(16 + i * 16);

            if (!info.is_spilled && info.assigned_preg.is_valid()) {
                PReg preg = info.assigned_preg;
                if (preg.is_gpr()) {
                    if (vr.size == 4) {
                        enc_.mov32(preg.as_gpr(), ptr(GPR::R10, slot_offset));
                    } else {
                        enc_.mov(preg.as_gpr(), ptr(GPR::R10, slot_offset));
                    }
                } else if (preg.is_xmm()) {
                    if (vr.size == 4) {
                        enc_.movss(preg.as_xmm(), ptr(GPR::R10, slot_offset));
                    } else if (vr.size == 16) {
                        enc_.movups(preg.as_xmm(), ptr(GPR::R10, slot_offset));
                    } else {
                        enc_.movq(preg.as_xmm(), ptr(GPR::R10, slot_offset));
                    }
                }
            } else if (info.is_spilled && info.assigned_spill_slot >= 0) {
                MemAddress stack_addr = X64FrameLayout::spill_slot_address(info.assigned_spill_slot, mutable_frame);
                if (vr.is_xmm()) {
                    enc_.movq(XMM::XMM4, ptr(GPR::R10, slot_offset));
                    enc_.movsd(stack_addr, XMM::XMM4);
                } else {
                    enc_.mov(GPR::R11, ptr(GPR::R10, slot_offset));
                    enc_.mov(stack_addr, GPR::R11);
                }
            }
        }

        // Emits jump directly to the compiled loop header block
        enc_.jmp(block_labels_[fn_.osr_entry.loop_header_id]);
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
    x64::X64FrameLayout::compute_layout(fn_frame, fn_.calling_conv);
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
                    enc_.movsd(XMM::XMM5, to_mem_address(src));
                    enc_.movsd(dst_mem, XMM::XMM5);
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
                    enc_.movss(XMM::XMM5, to_mem_address(src));
                    enc_.movss(dst_mem, XMM::XMM5);
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
        case LirOpcode::Sqrtsd: enc_.sqrtsd(to_xmm(inst.defs[0]), to_xmm(inst.uses.back())); break;
        case LirOpcode::Sqrtss: enc_.sqrtss(to_xmm(inst.defs[0]), to_xmm(inst.uses.back())); break;
        case LirOpcode::Ucomisd:
            if (inst.uses[1].is_mem() || inst.uses[1].is_spill_slot()) enc_.ucomisd(to_xmm(inst.uses[0]), to_mem_address(inst.uses[1]));
            else enc_.ucomisd(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
            break;
        case LirOpcode::Ucomiss:
            if (inst.uses[1].is_mem() || inst.uses[1].is_spill_slot()) enc_.ucomiss(to_xmm(inst.uses[0]), to_mem_address(inst.uses[1]));
            else enc_.ucomiss(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
            break;
        case LirOpcode::Xorpd: enc_.xorpd(to_xmm(inst.defs[0]), to_xmm(inst.uses.back())); break;
        case LirOpcode::Cvtsi2sd: enc_.cvtsi2sd(to_xmm(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvtsi2sd32: enc_.cvtsi2sd32(to_xmm(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Cvttsd2si: enc_.cvttsd2si(to_gpr(inst.defs[0]), to_xmm(inst.uses[0])); break;
        case LirOpcode::Cvttsd2si32: enc_.cvttsd2si32(to_gpr(inst.defs[0]), to_xmm(inst.uses[0])); break;
        default: break;
    }
}

void EmitContext::emit_parallel_copy(const LirInst& inst) {
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
            continue; // Skip self moves
        }
        if (inst.defs[i].is_spill_slot() && inst.uses[i].is_spill_slot() &&
            inst.defs[i].spill_slot == inst.uses[i].spill_slot) {
            continue; // Skip self moves
        }
        moves.push_back({inst.defs[i], inst.uses[i], false});
    }

    auto emit_move = [&](const LirOperand& dst, const LirOperand& src) {
        if (dst.is_preg() && src.is_preg() && dst.preg_val == src.preg_val) return;
        if (dst.is_spill_slot() && src.is_spill_slot() && dst.spill_slot == src.spill_slot) return;
        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr()) {
                GPR dst_gpr = dst.preg_val.as_gpr();
                if (src.is_preg()) {
                    GPR src_gpr = src.preg_val.as_gpr();
                    if (dst.size == 4 && src.size == 4) enc_.mov32(dst_gpr, src_gpr);
                    else enc_.mov(dst_gpr, src_gpr);
                } else if (src.is_imm_int()) {
                    if (dst.size == 4) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                    else enc_.mov(dst_gpr, src.imm_int);
                } else {
                    if (dst.size == 4) enc_.mov32(dst_gpr, to_mem_address(src));
                    else enc_.mov(dst_gpr, to_mem_address(src));
                }
            } else {
                XMM dst_xmm = dst.preg_val.as_xmm();
                if (dst.size == 32 || src.size == 32) {
                    if (src.is_preg()) enc_.vmovaps(dst_xmm, src.preg_val.as_xmm());
                    else enc_.vmovups(dst_xmm, to_mem_address(src));
                } else if (dst.size == 16 || src.size == 16) {
                    if (src.is_preg()) enc_.movaps(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movups(dst_xmm, to_mem_address(src));
                } else if (dst.size == 4 && src.size == 4) {
                    if (src.is_preg()) enc_.movss(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movss(dst_xmm, to_mem_address(src));
                } else {
                    if (src.is_preg()) enc_.movsd(dst_xmm, src.preg_val.as_xmm());
                    else enc_.movsd(dst_xmm, to_mem_address(src));
                }
            }
        } else {
            MemAddress dst_mem = to_mem_address(dst);
            if (src.is_preg()) {
                if (src.preg_val.is_gpr()) {
                    if (src.size == 4) enc_.mov32(dst_mem, src.preg_val.as_gpr());
                    else enc_.mov(dst_mem, src.preg_val.as_gpr());
                } else {
                    if (dst.size == 32 || src.size == 32) {
                        enc_.vmovups(dst_mem, src.preg_val.as_xmm());
                    } else if (dst.size == 16 || src.size == 16) {
                        enc_.movups(dst_mem, src.preg_val.as_xmm());
                    } else if (dst.size == 4 && src.size == 4) {
                        enc_.movss(dst_mem, src.preg_val.as_xmm());
                    } else {
                        enc_.movsd(dst_mem, src.preg_val.as_xmm());
                    }
                }
            } else if (src.is_imm_int()) {
                if (dst.size == 4) enc_.mov32(dst_mem, static_cast<int32_t>(src.imm_int));
                else enc_.mov(dst_mem, static_cast<int32_t>(src.imm_int));
            } else {
                // Memory-to-memory move using scratch register
                MemAddress src_mem = to_mem_address(src);
                if (dst.size == 32 || src.size == 32) {
                    enc_.vmovups(XMM::XMM5, src_mem);
                    enc_.vmovups(dst_mem, XMM::XMM5);
                } else if (dst.size == 16 || src.size == 16) {
                    enc_.movups(XMM::XMM5, src_mem);
                    enc_.movups(dst_mem, XMM::XMM5);
                } else if (dst.size == 4 && src.size == 4) {
                    enc_.mov32(GPR::R11, src_mem);
                    enc_.mov32(dst_mem, GPR::R11);
                } else {
                    enc_.mov(GPR::R11, src_mem);
                    enc_.mov(dst_mem, GPR::R11);
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

        if (cycle.size() == 2) {
            size_t idx0 = cycle[0];
            size_t idx1 = cycle[1];
            auto& m0 = moves[idx0];
            auto& m1 = moves[idx1];

            if (m0.dst.is_preg() && m0.dst.preg_val.is_gpr() &&
                m1.dst.is_preg() && m1.dst.preg_val.is_gpr() &&
                m0.src.is_preg() && m0.src.preg_val.is_gpr() &&
                m1.src.is_preg() && m1.src.preg_val.is_gpr()) {
                GPR r0 = m0.dst.preg_val.as_gpr();
                GPR r1 = m1.dst.preg_val.as_gpr();
                if (m0.dst.size == 4 && m1.dst.size == 4) {
                    enc_.xchg32(r0, r1);
                } else {
                    enc_.xchg(r0, r1);
                }
                m0.done = true;
                m1.done = true;
            } else {
                bool is_xmm = (m0.dst.is_preg() && m0.dst.preg_val.is_xmm()) ||
                              (m1.dst.is_preg() && m1.dst.preg_val.is_xmm()) ||
                              (m0.src.is_preg() && m0.src.preg_val.is_xmm()) ||
                              (m1.src.is_preg() && m1.src.preg_val.is_xmm()) ||
                              (m0.dst.size == 16 || m0.dst.size == 32);
                PReg scratch = is_xmm ? PReg::xmm(XMM::XMM5) : PReg::gpr(GPR::R11);
                uint8_t sz = m0.dst.size;
                emit_move(LirOperand::preg(scratch, sz), m0.src);
                emit_move(m1.dst, m0.dst);
                emit_move(m0.dst, LirOperand::preg(scratch, sz));
                m0.done = true;
                m1.done = true;
            }
        } else if (!cycle.empty()) {
            size_t idx0 = cycle[0];
            auto& m0 = moves[idx0];
            bool is_xmm = (m0.dst.is_preg() && m0.dst.preg_val.is_xmm()) ||
                          (m0.src.is_preg() && m0.src.preg_val.is_xmm()) ||
                          (m0.dst.size == 16 || m0.dst.size == 32);
            PReg scratch = is_xmm ? PReg::xmm(XMM::XMM5) : PReg::gpr(GPR::R11);
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

void EmitContext::emit_control_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Jmp:
            enc_.jmp(block_labels_[inst.uses[0].label_id]);
            break;
        case LirOpcode::Jcc:
            enc_.j(inst.condition, block_labels_[inst.uses[0].label_id]);
            break;
        case LirOpcode::Call: {
            size_t call_start = buffer_.size();
            const auto& sym_op = inst.uses.back();
            std::string callee = inst.callee_symbol.empty() ? sym_op.symbol_name : inst.callee_symbol;
            if (inst.is_patchable) {
                size_t imm_off = 1;
                size_t pad = runtime::compute_cache_line_padding(buffer_.size(), imm_off, 4);
                if (pad > 0) buffer_.emit_nops(pad);
                size_t site_start = buffer_.size();
                enc_.call(callee);
                patch_sites_.emplace_back(
                    inst.patch_symbol,
                    runtime::PatchKind::Call,
                    site_start,
                    imm_off,
                    buffer_.size() - site_start,
                    0,
                    callee
                );
            } else {
                enc_.call(callee);
            }
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            codegen::FrameInfo mutable_frame = fn_.frame;
            X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(mutable_frame.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = mutable_frame.spill_slot_offset(info.assigned_spill_slot);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    MemAddress addr = X64FrameLayout::callee_gpr_address(info.assigned_preg.as_gpr(), mutable_frame);
                    map_rec.add_root(StackMapRootLocation::callee_saved(addr.disp, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::CallIndirect: {
            size_t call_start = buffer_.size();
            enc_.call(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            codegen::FrameInfo mutable_frame = fn_.frame;
            X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(mutable_frame.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = mutable_frame.spill_slot_offset(info.assigned_spill_slot);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    MemAddress addr = X64FrameLayout::callee_gpr_address(info.assigned_preg.as_gpr(), mutable_frame);
                    map_rec.add_root(StackMapRootLocation::callee_saved(addr.disp, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::Ret: {
            codegen::FrameInfo mutable_frame = fn_.frame;
            X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
            X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            break;
        }
        case LirOpcode::Push: enc_.push(to_gpr(inst.uses[0])); break;
        case LirOpcode::Pop: enc_.pop(to_gpr(inst.defs[0])); break;
        case LirOpcode::Lea:
            if (inst.defs[0].size == 4) enc_.lea32(to_gpr(inst.defs[0]), to_mem_address(inst.uses[0]));
            else enc_.lea(to_gpr(inst.defs[0]), to_mem_address(inst.uses[0]));
            break;
        case LirOpcode::Safepoint: {
            enc_.call("brass_gc_safepoint");
            size_t return_offset = buffer_.size();

            codegen::FrameInfo mutable_frame = fn_.frame;
            X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);

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
                    int32_t offset = mutable_frame.spill_slot_offset(info.assigned_spill_slot);
                    rec.live_gcref_spill_offsets.push_back(offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    rec.live_gcref_registers.push_back(info.assigned_preg.as_gpr());
                    MemAddress addr = X64FrameLayout::callee_gpr_address(info.assigned_preg.as_gpr(), mutable_frame);
                    map_rec.add_root(StackMapRootLocation::callee_saved(addr.disp, info.assigned_preg));
                }
            }
            safepoints_.push_back(std::move(rec));
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::GuardExit: {
            size_t num_uses = inst.uses.size();
            size_t slots_bytes = num_uses * 8;
            size_t shadow_space = (fn_.calling_conv.kind() == CallingConvKind::Win64 ? 32 : 0);
            size_t total_alloc = ((slots_bytes + shadow_space + 15) & ~size_t(15));
            if (total_alloc < 32 && fn_.calling_conv.kind() == CallingConvKind::Win64) {
                total_alloc = 32;
            }

            if (total_alloc > 0) enc_.sub(GPR::RSP, static_cast<int32_t>(total_alloc));
            int32_t slots_disp = static_cast<int32_t>(shadow_space);

            for (size_t i = 0; i < num_uses; ++i) {
                const auto& op = inst.uses[i];
                int32_t slot_offset = slots_disp + static_cast<int32_t>(i * 8);

                if (op.is_preg()) {
                    if (op.preg_val.is_gpr()) {
                        GPR src_gpr = op.preg_val.as_gpr();
                        if (op.size == 4) {
                            enc_.movsxd(GPR::R11, src_gpr);
                            enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
                        } else {
                            enc_.mov(ptr(GPR::RSP, slot_offset), src_gpr);
                        }
                    } else if (op.preg_val.is_xmm()) {
                        enc_.movsd(ptr(GPR::RSP, slot_offset), op.preg_val.as_xmm());
                    }
                } else if (op.is_spill_slot() || op.is_mem()) {
                    MemAddress src_mem = to_mem_address(op);
                    enc_.mov(GPR::R11, src_mem);
                    enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
                } else if (op.is_imm_int()) {
                    enc_.mov(GPR::R11, op.imm_int);
                    enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
                }
            }

            uint32_t rid = inst.resume_id;
            uint32_t rsn = inst.deopt_reason == 0 ? 1 : inst.deopt_reason;
            uint32_t cnt = static_cast<uint32_t>(num_uses);

            if (fn_.calling_conv.kind() == CallingConvKind::Win64) {
                enc_.mov32(GPR::RCX, rid);
                enc_.mov32(GPR::RDX, rsn);
                enc_.mov32(GPR::R8, cnt);
                if (num_uses > 0) enc_.lea(GPR::R9, ptr(GPR::RSP, slots_disp));
                else enc_.xor32(GPR::R9, GPR::R9);
            } else {
                enc_.mov32(GPR::RDI, rid);
                enc_.mov32(GPR::RSI, rsn);
                enc_.mov32(GPR::RDX, cnt);
                if (num_uses > 0) enc_.lea(GPR::RCX, ptr(GPR::RSP, slots_disp));
                else enc_.xor32(GPR::RCX, GPR::RCX);
            }

            enc_.call("brass_deopt_exit");

            if (!inst.exit_symbol.empty()) {
                if (total_alloc > 0) enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
                size_t shadow2 = (fn_.calling_conv.kind() == CallingConvKind::Win64 ? 32 : 0);
                if (shadow2 > 0) enc_.sub(GPR::RSP, static_cast<int32_t>(shadow2));
                enc_.call("brass_get_thread_deopt_frame");
                if (shadow2 > 0) enc_.add(GPR::RSP, static_cast<int32_t>(shadow2));

                if (fn_.calling_conv.kind() == CallingConvKind::Win64) {
                    enc_.lea(GPR::RDX, ptr(GPR::RAX, static_cast<int32_t>(offsetof(runtime::DeoptFrame, slots))));
                    enc_.mov32(GPR::RCX, rid);
                    enc_.sub(GPR::RSP, 32);
                    enc_.call(inst.exit_symbol);
                    enc_.add(GPR::RSP, 32);
                } else {
                    enc_.lea(GPR::RSI, ptr(GPR::RAX, static_cast<int32_t>(offsetof(runtime::DeoptFrame, slots))));
                    enc_.mov32(GPR::RDI, rid);
                    enc_.call(inst.exit_symbol);
                }

                codegen::FrameInfo mutable_frame = fn_.frame;
                X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
                X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            } else {
                if (total_alloc > 0) enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
                codegen::FrameInfo mutable_frame = fn_.frame;
                X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
                X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            }
            break;
        }
        default:
            break;
    }
}

void EmitContext::emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst) {
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
        case LirOpcode::Sub:
        case LirOpcode::Sub32:
        case LirOpcode::Imul:
        case LirOpcode::Imul32:
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
        case LirOpcode::Ucomisd:
        case LirOpcode::Ucomiss:
        case LirOpcode::Xorpd:
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si:
        case LirOpcode::Cvttsd2si32:
            emit_sse_instruction(inst);
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
        case LirOpcode::Addpd:
        case LirOpcode::Subpd:
        case LirOpcode::Mulpd:
        case LirOpcode::Divpd:
        case LirOpcode::Minpd:
        case LirOpcode::Maxpd:
        case LirOpcode::Sqrtpd:
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
        case LirOpcode::Pandn:
        case LirOpcode::Pcmpeqd:
        case LirOpcode::Pslld:
        case LirOpcode::Psllq:
        case LirOpcode::Shufps:
        case LirOpcode::Shufpd:
        case LirOpcode::Pshufd:
        case LirOpcode::Movddup:
        case LirOpcode::Pinsrd:
        case LirOpcode::Pextrd:
        case LirOpcode::Pinsrq:
        case LirOpcode::Pextrq:
        case LirOpcode::Insertps:
        case LirOpcode::Extractps:
        case LirOpcode::Xorps:
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
        case LirOpcode::Vbroadcastss:
        case LirOpcode::Vbroadcastsd:
        case LirOpcode::Vpbroadcastd:
        case LirOpcode::Vpbroadcastq:
        case LirOpcode::Vfmadd213ps:
        case LirOpcode::Vfmadd231ps:
        case LirOpcode::Vfmadd213pd:
        case LirOpcode::Vfmadd231pd:
        case LirOpcode::Vfmadd213ss:
        case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd:
        case LirOpcode::Vfmadd231sd:
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

CompilationResult compile_lir_to_x64(const LirFunction& fn, const Target& target) {
    EmitContext ctx(fn, target);
    return ctx.compile();
}

} // namespace brass::codegen
