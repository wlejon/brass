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
    result.entry_offset = 0;

    return result;
}

void EmitContext::emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst) {
    (void)is_entry_block;
    (void)is_first_inst;

    switch (inst.opcode) {
        case LirOpcode::Nop:
            enc_.nop();
            break;
        case LirOpcode::Mov: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];

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
            enc_.movabs(dst_gpr, static_cast<uint64_t>(inst.uses[0].imm_int));
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
        case LirOpcode::Add: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.add(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.add(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.add(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Add32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.add32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.add32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.add32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Sub: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.sub(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.sub(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.sub(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Sub32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.sub32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.sub32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.sub32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Imul: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.imul(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.imul(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.imul(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Imul32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.imul32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.imul32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.imul32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Idiv: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.idiv(to_gpr(src));
            } else {
                enc_.idiv(to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Idiv32: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.idiv32(to_gpr(src));
            } else {
                enc_.idiv32(to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Div: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.div(to_gpr(src));
            } else {
                enc_.div(to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Div32: {
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.div32(to_gpr(src));
            } else {
                enc_.div32(to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Cdq:
            enc_.cdq();
            break;
        case LirOpcode::Cqo:
            enc_.cqo();
            break;
        case LirOpcode::And: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.and_(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.and_(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.and_(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::And32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.and32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.and32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.and32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Or: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.or_(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.or_(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.or_(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Or32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.or32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.or32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.or32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Xor: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.xor_(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.xor_(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.xor_(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Xor32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.xor32(dst, to_gpr(src));
            } else if (src.is_imm_int()) {
                enc_.xor32(dst, static_cast<int32_t>(src.imm_int));
            } else {
                enc_.xor32(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Not:
            enc_.not_(to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Not32:
            enc_.not32(to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Neg:
            enc_.neg(to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Neg32:
            enc_.neg32(to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Shl: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.shl(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.shl(dst); // by CL
            }
            break;
        }
        case LirOpcode::Shl32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.shl32(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.shl32(dst);
            }
            break;
        }
        case LirOpcode::Shr: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.shr(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.shr(dst);
            }
            break;
        }
        case LirOpcode::Shr32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.shr32(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.shr32(dst);
            }
            break;
        }
        case LirOpcode::Sar: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.sar(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.sar(dst);
            }
            break;
        }
        case LirOpcode::Sar32: {
            GPR dst = to_gpr(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_imm_int()) {
                enc_.sar32(dst, static_cast<uint8_t>(src.imm_int));
            } else {
                enc_.sar32(dst);
            }
            break;
        }
        case LirOpcode::Popcnt:
            enc_.popcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Popcnt32:
            enc_.popcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Lzcnt:
            enc_.lzcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Lzcnt32:
            enc_.lzcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Tzcnt:
            enc_.tzcnt(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Tzcnt32:
            enc_.tzcnt32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Bsr:
            enc_.bsr(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Bsr32:
            enc_.bsr32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Bsf:
            enc_.bsf(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Bsf32:
            enc_.bsf32(to_gpr(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Cmp: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.cmp(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                enc_.cmp(op0, static_cast<int32_t>(op1.imm_int));
            } else {
                enc_.cmp(op0, to_mem_address(op1));
            }
            break;
        }
        case LirOpcode::Cmp32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.cmp32(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                enc_.cmp32(op0, static_cast<int32_t>(op1.imm_int));
            } else {
                enc_.cmp32(op0, to_mem_address(op1));
            }
            break;
        }
        case LirOpcode::Test: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.test(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                enc_.test(op0, static_cast<int32_t>(op1.imm_int));
            }
            break;
        }
        case LirOpcode::Test32: {
            GPR op0 = to_gpr(inst.uses[0]);
            const auto& op1 = inst.uses[1];
            if (op1.is_preg()) {
                enc_.test32(op0, to_gpr(op1));
            } else if (op1.is_imm_int()) {
                enc_.test32(op0, static_cast<int32_t>(op1.imm_int));
            }
            break;
        }
        case LirOpcode::Setcc:
            enc_.setcc(inst.condition, to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Cmovcc:
            if (inst.defs[0].size == 4) {
                enc_.cmovcc32(inst.condition, to_gpr(inst.defs[0]), to_gpr(inst.uses.back()));
            } else {
                enc_.cmovcc(inst.condition, to_gpr(inst.defs[0]), to_gpr(inst.uses.back()));
            }
            break;
        case LirOpcode::Movsd: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) {
                    enc_.movsd(dst_x, src.preg_val.as_xmm());
                } else {
                    enc_.movsd(dst_x, to_mem_address(src));
                }
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) {
                    enc_.movsd(dst_mem, src.preg_val.as_xmm());
                } else {
                    enc_.movsd(XMM::XMM15, to_mem_address(src));
                    enc_.movsd(dst_mem, XMM::XMM15);
                }
            }
            break;
        }
        case LirOpcode::Movss: {
            const auto& dst = inst.defs[0];
            const auto& src = inst.uses[0];
            if (dst.is_preg()) {
                XMM dst_x = dst.preg_val.as_xmm();
                if (src.is_preg()) {
                    enc_.movss(dst_x, src.preg_val.as_xmm());
                } else {
                    enc_.movss(dst_x, to_mem_address(src));
                }
            } else {
                MemAddress dst_mem = to_mem_address(dst);
                if (src.is_preg()) {
                    enc_.movss(dst_mem, src.preg_val.as_xmm());
                } else {
                    enc_.movss(XMM::XMM15, to_mem_address(src));
                    enc_.movss(dst_mem, XMM::XMM15);
                }
            }
            break;
        }
        case LirOpcode::Movq_gx:
            enc_.movq(to_xmm(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Movq_xg:
            enc_.movq(to_gpr(inst.defs[0]), to_xmm(inst.uses[0]));
            break;
        case LirOpcode::Addsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.addsd(dst, to_xmm(src));
            } else {
                enc_.addsd(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Subsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.subsd(dst, to_xmm(src));
            } else {
                enc_.subsd(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Mulsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.mulsd(dst, to_xmm(src));
            } else {
                enc_.mulsd(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Divsd: {
            XMM dst = to_xmm(inst.defs[0]);
            const auto& src = inst.uses.back();
            if (src.is_preg()) {
                enc_.divsd(dst, to_xmm(src));
            } else {
                enc_.divsd(dst, to_mem_address(src));
            }
            break;
        }
        case LirOpcode::Sqrtsd: {
            XMM dst = to_xmm(inst.defs[0]);
            enc_.sqrtsd(dst, to_xmm(inst.uses.back()));
            break;
        }
        case LirOpcode::Ucomisd:
            enc_.ucomisd(to_xmm(inst.uses[0]), to_xmm(inst.uses[1]));
            break;
        case LirOpcode::Xorpd:
            enc_.xorpd(to_xmm(inst.defs[0]), to_xmm(inst.uses.back()));
            break;
        case LirOpcode::Cvtsi2sd:
            enc_.cvtsi2sd(to_xmm(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Cvtsi2sd32:
            enc_.cvtsi2sd32(to_xmm(inst.defs[0]), to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Cvttsd2si:
            enc_.cvttsd2si(to_gpr(inst.defs[0]), to_xmm(inst.uses[0]));
            break;
        case LirOpcode::Cvttsd2si32:
            enc_.cvttsd2si32(to_gpr(inst.defs[0]), to_xmm(inst.uses[0]));
            break;
        case LirOpcode::Jmp: {
            uint32_t target_label = inst.uses[0].label_id;
            enc_.jmp(block_labels_[target_label]);
            break;
        }
        case LirOpcode::Jcc: {
            uint32_t target_label = inst.uses[0].label_id;
            enc_.j(inst.condition, block_labels_[target_label]);
            break;
        }
        case LirOpcode::Call: {
            const auto& sym_op = inst.uses.back();
            enc_.call(sym_op.symbol_name);
            break;
        }
        case LirOpcode::CallIndirect: {
            const auto& target_op = inst.uses.back();
            enc_.call(to_gpr(target_op));
            break;
        }
        case LirOpcode::Ret: {
            codegen::FrameInfo mutable_frame = fn_.frame;
            X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
            X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            break;
        }
        case LirOpcode::Push:
            enc_.push(to_gpr(inst.uses[0]));
            break;
        case LirOpcode::Pop:
            enc_.pop(to_gpr(inst.defs[0]));
            break;
        case LirOpcode::Lea:
            enc_.lea(to_gpr(inst.defs[0]), to_mem_address(inst.uses[0]));
            break;
        case LirOpcode::Safepoint: {
            SafepointRecord rec;
            rec.code_offset = buffer_.size();
            rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    rec.live_gcref_spill_offsets.push_back(fn_.frame.spill_slot_offset(info.assigned_spill_slot));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    rec.live_gcref_registers.push_back(info.assigned_preg.as_gpr());
                }
            }
            safepoints_.push_back(std::move(rec));
            break;
        }
        case LirOpcode::GuardExit:
            enc_.ud2();
            break;
        case LirOpcode::ParallelCopy:
            break;
    }
}

CompilationResult compile_lir_to_x64(const LirFunction& fn, const Target& target) {
    EmitContext ctx(fn, target);
    return ctx.compile();
}

} // namespace brass::codegen
