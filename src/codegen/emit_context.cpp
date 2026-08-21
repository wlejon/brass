#include <brass/codegen/emit_context.hpp>
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
        buffer_.bind(block_labels_[block->id]);
        result.block_offsets[block->id] = buffer_.size();

        for (size_t i_idx = 0; i_idx < block->instructions.size(); ++i_idx) {
            const auto& inst = *block->instructions[i_idx];
            emit_instruction(inst, b_idx == 0, i_idx == 0);
        }
    }

    result.code_buffer = std::move(buffer_);
    result.safepoints = std::move(safepoints_);
    result.stack_map.function_name = fn_.name;
    result.stack_map.code_size = static_cast<uint32_t>(result.code_buffer.size());
    result.stack_map.records = std::move(stack_map_records_);
    result.entry_offset = 0;

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

    return result;
}

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
            if (inst.is_patchable) {
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
        case LirOpcode::Movq_gx: enc_.movq(to_xmm(inst.defs[0]), to_gpr(inst.uses[0])); break;
        case LirOpcode::Movq_xg: enc_.movq(to_gpr(inst.defs[0]), to_xmm(inst.uses[0])); break;
        case LirOpcode::Addsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.addsd(dst, to_xmm(src));
            else enc_.addsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Subsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.subsd(dst, to_xmm(src));
            else enc_.subsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Mulsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.mulsd(dst, to_xmm(src));
            else enc_.mulsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Divsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) enc_.divsd(dst, to_xmm(src));
            else enc_.divsd(dst, to_mem_address(src));
            break;
        }
        case LirOpcode::Sqrtsd: enc_.sqrtsd(to_xmm(inst.defs[0]), to_xmm(inst.uses.back())); break;
        case LirOpcode::Ucomisd:
            if (inst.uses[1].is_mem() || inst.uses[1].is_spill_slot()) enc_.ucomisd(to_xmm(inst.uses[0]), to_mem_address(inst.uses[1]));
            else enc_.ucomisd(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
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
        moves.push_back({inst.defs[i], inst.uses[i], false});
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& m : moves) {
            if (m.done) continue;
            bool dst_used = false;
            for (const auto& other : moves) {
                if (!other.done && other.src.is_preg() && m.dst.is_preg() &&
                    other.src.preg_val == m.dst.preg_val) {
                    dst_used = true;
                    break;
                }
            }
            if (!dst_used) {
                if (m.dst.is_preg()) {
                    if (m.dst.preg_val.is_gpr()) {
                        if (m.src.is_preg()) enc_.mov(m.dst.preg_val.as_gpr(), m.src.preg_val.as_gpr());
                        else enc_.mov(m.dst.preg_val.as_gpr(), to_mem_address(m.src));
                    } else {
                        if (m.src.is_preg()) enc_.movsd(m.dst.preg_val.as_xmm(), m.src.preg_val.as_xmm());
                        else enc_.movsd(m.dst.preg_val.as_xmm(), to_mem_address(m.src));
                    }
                }
                m.done = true;
                progress = true;
            }
        }
    }

    // Resolve cycles with scratch register
    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i].done) continue;
        bool is_xmm = moves[i].src.is_preg() && moves[i].src.preg_val.is_xmm();
        PReg scratch = is_xmm ? PReg::xmm(XMM::XMM15) : PReg::gpr(GPR::R11);
        if (is_xmm) enc_.movsd(scratch.as_xmm(), moves[i].src.preg_val.as_xmm());
        else enc_.mov(scratch.as_gpr(), moves[i].src.preg_val.as_gpr());

        moves[i].src = LirOperand::preg(scratch, moves[i].src.size);

        progress = true;
        while (progress) {
            progress = false;
            for (auto& m : moves) {
                if (m.done) continue;
                bool dst_used = false;
                for (const auto& other : moves) {
                    if (!other.done && other.src.is_preg() && m.dst.is_preg() &&
                        other.src.preg_val == m.dst.preg_val) {
                        dst_used = true;
                        break;
                    }
                }
                if (!dst_used) {
                    if (m.dst.is_preg()) {
                        if (m.dst.preg_val.is_gpr()) {
                            if (m.src.is_preg()) enc_.mov(m.dst.preg_val.as_gpr(), m.src.preg_val.as_gpr());
                            else enc_.mov(m.dst.preg_val.as_gpr(), to_mem_address(m.src));
                        } else {
                            if (m.src.is_preg()) enc_.movsd(m.dst.preg_val.as_xmm(), m.src.preg_val.as_xmm());
                            else enc_.movsd(m.dst.preg_val.as_xmm(), to_mem_address(m.src));
                        }
                    }
                    m.done = true;
                    progress = true;
                }
            }
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
            enc_.call(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();

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
                int32_t slot_offset = static_cast<int32_t>(slots_disp + i * 8);

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
        case LirOpcode::Subsd:
        case LirOpcode::Mulsd:
        case LirOpcode::Divsd:
        case LirOpcode::Sqrtsd:
        case LirOpcode::Ucomisd:
        case LirOpcode::Xorpd:
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si:
        case LirOpcode::Cvttsd2si32:
            emit_sse_instruction(inst);
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
