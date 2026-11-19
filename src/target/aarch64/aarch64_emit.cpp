#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <brass/codegen/lir_flags.hpp>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <bit>

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

            // A register offset scales its index by the access size or not
            // at all (the S bit: LSL #log2(size) or LSL #0), so an index
            // scaled by anything else is applied here. bronze stores a
            // 16-bit element as two bytes at index * 2 (+1): byte accesses
            // at scale 2, which encoded as LSL #0 wrote every element at
            // index * 1.
            const uint8_t size_shift = op.size == 8 ? 3 : (op.size == 4 ? 2 : (op.size == 2 ? 1 : (op.size == 16 ? 4 : 0)));
            if (shift != 0 && shift != size_shift) {
                GPR addr_scratch = (base == GPR::X16 || index == GPR::X16) ? GPR::X17 : GPR::X16;
                enc_.lsl(addr_scratch, index, shift);
                enc_.add(addr_scratch, base, addr_scratch);
                if (m.disp >= 0 && m.disp <= 4095) {
                    if (m.disp != 0) enc_.add(addr_scratch, addr_scratch, static_cast<uint32_t>(m.disp));
                } else if (m.disp < 0 && -m.disp <= 4095) {
                    enc_.sub(addr_scratch, addr_scratch, static_cast<uint32_t>(-m.disp));
                } else {
                    GPR other = addr_scratch == GPR::X16 ? GPR::X17 : GPR::X16;
                    enc_.mov(other, static_cast<uint64_t>(m.disp));
                    enc_.add(addr_scratch, addr_scratch, other);
                }
                return ptr(addr_scratch, 0);
            }

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
    // Size not given: an offset that is a multiple of 16 and under 4096 is
    // a scaled immediate for a 1-, 2-, 4-, 8- or 16-byte access alike. The
    // default was an 8-byte access's reach (to 32760), which a byte store
    // through the same address cannot encode (a slow-site counter's arm
    // byte at 4304 aborted a large harvest build).
    if (size_bytes <= 0 && off >= 0 && off % 16 == 0 && off <= 4095) {
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

namespace {

enum class FusedBranchKind : uint8_t {
    None = 0,
    Cbz,
    Cbz32,
    Cbnz,
    Cbnz32,
    Tbz,
    Tbnz
};

struct FusedBranchInfo {
    FusedBranchKind kind = FusedBranchKind::None;
    GPR reg = GPR::None;
    uint8_t bit = 0;
};

struct BranchFusionPlan {
    std::vector<uint8_t> skip_inst;
    std::vector<FusedBranchInfo> fused;
};

FusedBranchInfo invert_fused_branch(FusedBranchInfo info) noexcept {
    switch (info.kind) {
        case FusedBranchKind::Cbz:    info.kind = FusedBranchKind::Cbnz; break;
        case FusedBranchKind::Cbnz:   info.kind = FusedBranchKind::Cbz; break;
        case FusedBranchKind::Cbz32:  info.kind = FusedBranchKind::Cbnz32; break;
        case FusedBranchKind::Cbnz32: info.kind = FusedBranchKind::Cbz32; break;
        case FusedBranchKind::Tbz:    info.kind = FusedBranchKind::Tbnz; break;
        case FusedBranchKind::Tbnz:   info.kind = FusedBranchKind::Tbz; break;
        case FusedBranchKind::None:   break;
    }
    return info;
}

BranchFusionPlan plan_branch_fusions(const LirBlock& block) {
    const size_t n = block.instructions.size();
    BranchFusionPlan plan;
    plan.skip_inst.assign(n, 0);
    plan.fused.resize(n);

    for (size_t j = 1; j < n; ++j) {
        const LirInst& jcc = *block.instructions[j];
        if (jcc.opcode != LirOpcode::Jcc || jcc.uses.empty() || !jcc.uses[0].is_label()) continue;

        const brass::x64::Condition c = jcc.condition;
        const bool is_eq_ne = (c == brass::x64::Condition::E || c == brass::x64::Condition::NE);
        const bool is_sign = (c == brass::x64::Condition::L || c == brass::x64::Condition::GE ||
                              c == brass::x64::Condition::S || c == brass::x64::Condition::NS);
        if (!is_eq_ne && !is_sign) continue;

        // No subsequent instruction in this block may read flags before overwriting them.
        bool live_after = false;
        for (size_t k = j + 1; k < n; ++k) {
            const LirInst& after = *block.instructions[k];
            if (lir_reads_flags(after, Arch::aarch64)) {
                live_after = true;
                break;
            }
            if (lir_flag_effect(after, Arch::aarch64) != FlagEffect::None) break;
        }
        if (live_after) continue;

        size_t prod_idx = SIZE_MAX;
        for (size_t k = j; k-- > 0;) {
            const LirInst& cand = *block.instructions[k];
            if (lir_reads_flags(cand, Arch::aarch64)) break;
            const FlagEffect eff = lir_flag_effect(cand, Arch::aarch64);
            if (eff == FlagEffect::Defines) {
                prod_idx = k;
                break;
            }
            if (eff == FlagEffect::Clobbers) break;
        }
        if (prod_idx == SIZE_MAX || plan.skip_inst[prod_idx]) continue;

        const LirInst& prod = *block.instructions[prod_idx];
        const bool is_test = (prod.opcode == LirOpcode::Test || prod.opcode == LirOpcode::Test32);
        const bool is_cmp = (prod.opcode == LirOpcode::Cmp || prod.opcode == LirOpcode::Cmp32);
        if ((!is_test && !is_cmp) || prod.uses.size() < 2) continue;
        if (!prod.uses[0].is_preg() || !prod.uses[0].preg_val.is_gpr()) continue;

        const GPR r = prod.uses[0].preg_val.as_aarch64_gpr();
        const uint8_t rcode = static_cast<uint8_t>(reg_code(r));
        if (r == GPR::None || r == GPR::SP || r == GPR::XZR || r == GPR::X16 || r == GPR::X17 || rcode >= 31) {
            continue;
        }

        const bool is32 = (prod.opcode == LirOpcode::Test32 || prod.opcode == LirOpcode::Cmp32);
        FusedBranchInfo info{};
        info.reg = r;

        const LirOperand& u1 = prod.uses[1];
        if ((is_test && u1.is_preg() && u1.preg_val == prod.uses[0].preg_val) ||
            (is_cmp && u1.is_imm_int() && u1.imm_int == 0)) {
            if (c == brass::x64::Condition::E) {
                info.kind = is32 ? FusedBranchKind::Cbz32 : FusedBranchKind::Cbz;
            } else if (c == brass::x64::Condition::NE) {
                info.kind = is32 ? FusedBranchKind::Cbnz32 : FusedBranchKind::Cbnz;
            } else if (c == brass::x64::Condition::L || c == brass::x64::Condition::S) {
                info.kind = FusedBranchKind::Tbnz;
                info.bit = is32 ? 31 : 63;
            } else {
                info.kind = FusedBranchKind::Tbz;
                info.bit = is32 ? 31 : 63;
            }
        } else if (is_test && u1.is_imm_int() && is_eq_ne) {
            const uint64_t mask = is32 ? static_cast<uint64_t>(static_cast<uint32_t>(u1.imm_int))
                                       : static_cast<uint64_t>(u1.imm_int);
            if (mask == 0 || (mask & (mask - 1)) != 0) continue;
            info.bit = static_cast<uint8_t>(std::countr_zero(mask));
            info.kind = (c == brass::x64::Condition::E) ? FusedBranchKind::Tbz : FusedBranchKind::Tbnz;
        } else {
            continue;
        }

        // Verify `r` is neither written nor clobbered between `prod_idx` and `j`.
        bool clobbered = false;
        const uint32_t rbit = 1u << rcode;
        for (size_t k = prod_idx + 1; k < j; ++k) {
            const LirInst& m = *block.instructions[k];
            if (m.is_call() || m.opcode == LirOpcode::ParallelCopy || m.opcode == LirOpcode::GuardExit ||
                (m.clobbered_gprs & rbit)) {
                clobbered = true;
                break;
            }
            for (const auto& d : m.defs) {
                if (d.is_preg() && d.preg_val.is_gpr() && d.preg_val.as_aarch64_gpr() == r) {
                    clobbered = true;
                    break;
                }
            }
            if (clobbered) break;
        }
        if (clobbered) continue;

        plan.skip_inst[prod_idx] = 1;
        plan.fused[j] = info;
    }
    return plan;
}

} // namespace

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
    begin_constant_pool();

    // 2. Create labels for all blocks
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
    }

    // 3. Emit prologue at entry
    AArch64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);
    zero_tagged_locals();

    // 4. Emit blocks
    const BranchThreading threading(fn_);
    threading_ = &threading;
    auto emits_nothing = [&](size_t i) {
        const auto& ins = fn_.blocks[i]->instructions;
        if (threading.skipped(i) || ins.empty()) return true;
        const ClosingBranches c = closing_branches(fn_, i);
        return ins.size() == 1 && c.count == 1 && !c.conditional && i + 1 < fn_.blocks.size() &&
               threading.target(c.other) == fn_.blocks[i + 1]->id;
    };
    bool prev_falls_through = true;

    for (size_t b_idx = 0; b_idx < fn_.blocks.size(); ++b_idx) {
        const auto& block = fn_.blocks[b_idx];
        if (threading.skipped(b_idx)) {
            buffer_.bind(block_labels_[block->id]);
            continue;
        }
        if (b_idx > 0) {
            if (threading.loop_head(b_idx)) {
                buffer_.align(16);
            } else if (!prev_falls_through && !block->cold) {
                buffer_.align(16);
            }
        }
        buffer_.bind(block_labels_[block->id]);
        result.block_offsets[block->id] = buffer_.size();

        if (block->is_landing_pad) {
            size_t outgoing_bytes = (frame_.outgoing_arg_space + 15) & ~size_t(15);
            if (outgoing_bytes == 0) {
                enc_.mov(GPR::SP, GPR::FP);
            } else {
                enc_.sub(GPR::SP, GPR::FP, static_cast<uint32_t>(outgoing_bytes));
            }
        }

        auto falls_to = [&](uint32_t target) {
            for (size_t i = b_idx + 1; i < fn_.blocks.size(); ++i) {
                if (fn_.blocks[i]->id == target) return true;
                if (!emits_nothing(i)) return false;
            }
            return false;
        };

        auto emit_fused_branch = [&](const FusedBranchInfo& fb, Label target) {
            switch (fb.kind) {
                case FusedBranchKind::Cbz:    enc_.cbz(fb.reg, target); break;
                case FusedBranchKind::Cbz32:  enc_.cbz32(fb.reg, target); break;
                case FusedBranchKind::Cbnz:   enc_.cbnz(fb.reg, target); break;
                case FusedBranchKind::Cbnz32: enc_.cbnz32(fb.reg, target); break;
                case FusedBranchKind::Tbz:    enc_.tbz(fb.reg, fb.bit, target); break;
                case FusedBranchKind::Tbnz:   enc_.tbnz(fb.reg, fb.bit, target); break;
                case FusedBranchKind::None:   break;
            }
        };

        const size_t n_insts = block->instructions.size();
        const ClosingBranches closing = closing_branches(fn_, b_idx);
        const size_t limit = n_insts - closing.count;
        const BranchFusionPlan plan = plan_branch_fusions(*block);

        for (size_t i_idx = 0; i_idx < limit; ++i_idx) {
            if (plan.skip_inst[i_idx]) continue;
            const auto& inst = *block->instructions[i_idx];
            size_t cur_offset = buffer_.size();
            DebugLoc loc = inst.loc;
            if (!loc.is_valid() && inst.mir_origin) {
                loc = inst.mir_origin->loc();
            }
            if (loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(cur_offset), loc);
            }
            if (inst.opcode == LirOpcode::Jcc && plan.fused[i_idx].kind != FusedBranchKind::None) {
                const uint32_t target = threading.target(inst.uses[0].label_id);
                emit_fused_branch(plan.fused[i_idx], block_labels_[target]);
                continue;
            }
            emit_instruction(inst, b_idx == 0, i_idx == 0);
        }

        if (closing.conditional) {
            const size_t jcc_idx = n_insts - closing.count;
            const auto& jcc = *block->instructions[jcc_idx];
            if (jcc.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jcc.loc);
            }
            const uint32_t true_target = threading.target(closing.taken);
            const uint32_t false_target = threading.target(closing.other);
            const FusedBranchInfo fb = plan.fused[jcc_idx];

            if (falls_to(false_target)) {
                if (fb.kind != FusedBranchKind::None) emit_fused_branch(fb, block_labels_[true_target]);
                else enc_.b(to_aarch64_cond(jcc.condition), block_labels_[true_target]);
                prev_falls_through = true;
            } else if (falls_to(true_target)) {
                if (fb.kind != FusedBranchKind::None) emit_fused_branch(invert_fused_branch(fb), block_labels_[false_target]);
                else enc_.b(to_aarch64_cond(invert(jcc.condition)), block_labels_[false_target]);
                prev_falls_through = true;
            } else {
                if (fb.kind != FusedBranchKind::None) emit_fused_branch(fb, block_labels_[true_target]);
                else enc_.b(to_aarch64_cond(jcc.condition), block_labels_[true_target]);
                enc_.b(block_labels_[false_target]);
                prev_falls_through = false;
            }
        } else if (closing.count == 1) {
            const auto& jmp = *block->instructions[n_insts - 1];
            if (jmp.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jmp.loc);
            }
            const uint32_t target = threading.target(closing.other);
            if (!falls_to(target)) {
                enc_.b(block_labels_[target]);
                prev_falls_through = false;
            } else {
                prev_falls_through = true;
            }
        } else {
            const LirOpcode last = n_insts > 0 ? block->instructions.back()->opcode : LirOpcode::Nop;
            prev_falls_through = !(last == LirOpcode::Ret || last == LirOpcode::Trap || last == LirOpcode::GuardExit);
        }
    }

    threading_ = nullptr;
    emit_constant_pool();

    result.code_buffer = std::move(buffer_);
    result.safepoints = std::move(safepoints_);
    result.stack_map.function_name = fn_.name;
    result.stack_map.code_size = static_cast<uint32_t>(result.code_buffer.size());
    result.stack_map.records = std::move(stack_map_records_);
    // The spill slots run up from just above the save area, and the locals
    // (the allocas) lie above them (AArch64FrameLayout).
    result.stack_map.slot_area_lo = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(0, frame_).offset);
    result.stack_map.slot_area_hi = static_cast<int32_t>(
        AArch64FrameLayout::local_frame_address(static_cast<int32_t>(frame_.local_frame_bytes), frame_).offset);
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
        case LirOpcode::KeepAlive:
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
        case LirOpcode::AddShifted:
        case LirOpcode::Add32Shifted:
        case LirOpcode::Sub:
        case LirOpcode::Sub32:
        case LirOpcode::Subs:
        case LirOpcode::Subs32:
        case LirOpcode::SubShifted:
        case LirOpcode::Sub32Shifted:
        case LirOpcode::Imul:
        case LirOpcode::Imul32:
        case LirOpcode::Madd:
        case LirOpcode::Madd32:
        case LirOpcode::Msub:
        case LirOpcode::Msub32:
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
        case LirOpcode::Csinc:
        case LirOpcode::Csneg:
        case LirOpcode::Csinv:
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
