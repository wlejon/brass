#include <brass/codegen/emit_context.hpp>
#include <brass/runtime/deopt.hpp>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstring>

namespace brass::codegen {

using namespace brass::x64;

EmitContext::EmitContext(const LirFunction& fn, const Target& target)
    : fn_(fn), target_(target), frame_(fn.frame), enc_(buffer_) {}

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
    if (op.is_spill_slot()) return X64FrameLayout::spill_slot_address(op.spill_slot, frame_);
    if (op.is_local_slot()) return X64FrameLayout::local_frame_address(op.local_offset, frame_);
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
    throw std::logic_error("x64 emission: operand of kind " + std::string(to_string(op.kind)) +
                           " used as a memory address in " + fn_.name);
}

CompilationResult EmitContext::compile() {
    CompilationResult result;
    safepoints_.clear();
    stack_map_records_.clear();
    block_labels_.clear();
    callee_gpr_to_slot_.clear();
    frame_ = fn_.frame;

    // 0. Allocate designated frame spill slots for unspilled callee-saved GPRs holding live GCRefs
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint) {
                for (const auto& v : inst->live_gcrefs) {
                    const auto& info = fn_.get_vreg_info(v);
                    if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                        uint8_t code = info.assigned_preg.code;
                        if (callee_gpr_to_slot_.find(code) == callee_gpr_to_slot_.end()) {
                            int32_t slot = static_cast<int32_t>(frame_.num_spill_slots++);
                            frame_.spill_slot_is_gcref.push_back(true);
                            callee_gpr_to_slot_[code] = slot;
                        }
                    }
                }
            }
        }
    }

    // 1. Compute frame layout
    X64FrameLayout::compute_layout(frame_, fn_.calling_conv);

    // 2. Create labels for all blocks
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
    }

    // 3. Emit prologue at entry
    X64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);

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
        X64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);

        // Standard calling convention: 1st argument (OsrMigrationFrame*)
        // On Win64: RCX; on SysV: RDI
        GPR arg_reg = (fn_.calling_conv.kind() == CallingConvKind::Win64) ? GPR::RCX : GPR::RDI;
        // Copy to R10 (scratch register never assigned by linear scan regalloc)
        enc_.mov(GPR::R10, arg_reg);

        // Unpack migration frame slots directly into physical registers and spill
        // slots. Only values live at the loop header are loaded: a migrated value
        // with no live range there (a constant folded to an immediate, a value only
        // used before the loop) keeps a stale location that may alias a
        // loop-carried value, and writing it would clobber that value.
        if (fn_.osr_entry.live_at_header.size() != fn_.osr_entry.live_in_vregs.size()) {
            throw std::logic_error("OSR entry for " + std::string(fn_.name) +
                                   ": live-at-header set was not computed by register allocation");
        }
        for (size_t i = 0; i < fn_.osr_entry.live_in_vregs.size(); ++i) {
            if (!fn_.osr_entry.live_at_header[i]) continue;
            VReg vr = fn_.osr_entry.live_in_vregs[i];
            const VRegInfo& info = fn_.get_vreg_info(vr);
            int32_t slot_offset = static_cast<int32_t>(16 + i * 16);

            if (!info.is_spilled && !info.assigned_preg.is_valid()) {
                throw std::logic_error("OSR entry for " + std::string(fn_.name) + ": v" +
                                       std::to_string(vr.id) + " is live at the loop header but has no location");
            }
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
                MemAddress stack_addr = X64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_);
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
    result.stack_adjust_regions = std::move(stack_adjust_regions_);

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


namespace {

// Where each state value goes when a guard's exit stub is called as
// stub(state values...). The verifier matched the stub's parameters to the
// state values, so each value's kind picks its argument class.
struct ExitStubArgLoc {
    bool is_float = false;
    int reg = -1;    // register index, or -1 for a stack argument
    int stack = -1;  // stack argument index
};

struct ExitStubArgPlan {
    std::vector<ExitStubArgLoc> locs;
    size_t stack_args = 0;
    // Bytes at the bottom of the guard exit's allocation for the stub call:
    // shadow space, then the stack arguments; a multiple of 16.
    size_t out_area = 0;
};

ExitStubArgPlan plan_exit_stub_args(const LirInst& inst, bool win64) {
    ExitStubArgPlan plan;
    const size_t n = inst.deopt_kinds.size();
    plan.locs.resize(n);
    size_t gpr_used = 0, xmm_used = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto kind = static_cast<runtime::DeoptValueKind>(inst.deopt_kinds[i]);
        ExitStubArgLoc loc;
        loc.is_float = kind == runtime::DeoptValueKind::Float32 || kind == runtime::DeoptValueKind::Float64;
        if (win64) {
            if (i < 4) loc.reg = static_cast<int>(i);
        } else if (loc.is_float) {
            if (xmm_used < 8) loc.reg = static_cast<int>(xmm_used++);
        } else {
            if (gpr_used < 6) loc.reg = static_cast<int>(gpr_used++);
        }
        if (loc.reg < 0) loc.stack = static_cast<int>(plan.stack_args++);
        plan.locs[i] = loc;
    }
    const size_t shadow = win64 ? 32 : 0;
    plan.out_area = (shadow + plan.stack_args * 8 + 15) & ~size_t(15);
    return plan;
}

} // namespace

void EmitContext::emit_exit_stub_call(const LirInst& inst, int32_t slots_disp) {
    // Slots hold a float in its low bits and a narrow integer sign-extended.
    // The stack arguments go into the outgoing area the guard exit allocated
    // under the record together with it, so RSP does not move here and the
    // guard exit's unwind region stays exact across the call.
    const bool win64 = fn_.calling_conv.kind() == CallingConvKind::Win64;
    static constexpr GPR kWinGpr[] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
    static constexpr GPR kSysvGpr[] = {GPR::RDI, GPR::RSI, GPR::RDX, GPR::RCX, GPR::R8, GPR::R9};
    static constexpr XMM kXmm[] = {XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3,
                                   XMM::XMM4, XMM::XMM5, XMM::XMM6, XMM::XMM7};
    const ExitStubArgPlan plan = plan_exit_stub_args(inst, win64);
    const size_t shadow = win64 ? 32 : 0;
    if (static_cast<size_t>(slots_disp) < plan.out_area) {
        throw_unsupported("x64 emit (guard exit)", "exit stub arguments overlap the deopt record");
    }
    const size_t n = plan.locs.size();
    auto slot = [&](size_t i) { return ptr(GPR::RSP, slots_disp + static_cast<int32_t>(i * 8)); };
    for (size_t i = 0; i < n; ++i) {
        if (plan.locs[i].stack < 0) continue;
        enc_.mov(GPR::R11, slot(i));
        enc_.mov(ptr(GPR::RSP, static_cast<int32_t>(shadow) + plan.locs[i].stack * 8), GPR::R11);
    }
    for (size_t i = 0; i < n; ++i) {
        const auto& loc = plan.locs[i];
        if (loc.reg < 0) continue;
        if (loc.is_float) {
            enc_.movsd(kXmm[loc.reg], slot(i));
        } else {
            enc_.mov(win64 ? kWinGpr[loc.reg] : kSysvGpr[loc.reg], slot(i));
        }
    }
    enc_.call(inst.exit_symbol);
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
            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(X64FrameLayout::spill_slot_address(it->second, frame_), info.assigned_preg.as_gpr());
                    }
                }
            }

            size_t call_start = buffer_.size();
            std::string callee = inst.callee_symbol;
            if (callee.empty()) {
                for (const auto& u : inst.uses) {
                    if (u.is_symbol()) {
                        callee = u.symbol_name;
                        break;
                    }
                }
            }
            if (callee.empty() && !inst.uses.empty()) {
                callee = inst.uses.back().symbol_name;
            }
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

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(info.assigned_preg.as_gpr(), X64FrameLayout::spill_slot_address(it->second, frame_));
                    }
                }
            }

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = frame_.spill_slot_offset(info.assigned_spill_slot);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    int32_t slot = (it != callee_gpr_to_slot_.end()) ? it->second : 0;
                    int32_t offset = frame_.spill_slot_offset(slot);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::CallIndirect: {
            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(X64FrameLayout::spill_slot_address(it->second, frame_), info.assigned_preg.as_gpr());
                    }
                }
            }

            size_t call_start = buffer_.size();
            enc_.call(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(info.assigned_preg.as_gpr(), X64FrameLayout::spill_slot_address(it->second, frame_));
                    }
                }
            }

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = frame_.spill_slot_offset(info.assigned_spill_slot);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    int32_t slot = (it != callee_gpr_to_slot_.end()) ? it->second : 0;
                    int32_t offset = frame_.spill_slot_offset(slot);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::Ret: {
            X64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
            break;
        }
        case LirOpcode::Push: enc_.push(to_gpr(inst.uses[0])); break;
        case LirOpcode::Pop: enc_.pop(to_gpr(inst.defs[0])); break;
        case LirOpcode::Lea:
            if (inst.defs[0].size == 4) enc_.lea32(to_gpr(inst.defs[0]), to_mem_address(inst.uses[0]));
            else enc_.lea(to_gpr(inst.defs[0]), to_mem_address(inst.uses[0]));
            break;
        case LirOpcode::Safepoint: {
            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(X64FrameLayout::spill_slot_address(it->second, frame_), info.assigned_preg.as_gpr());
                    }
                }
            }

            enc_.call("brass_gc_safepoint");
            size_t return_offset = buffer_.size();

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    if (it != callee_gpr_to_slot_.end()) {
                        enc_.mov(info.assigned_preg.as_gpr(), X64FrameLayout::spill_slot_address(it->second, frame_));
                    }
                }
            }

            SafepointRecord rec;
            rec.code_offset = return_offset;
            rec.safepoint_id = inst.safepoint_id;

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = frame_.spill_slot_offset(info.assigned_spill_slot);
                    rec.live_gcref_spill_offsets.push_back(offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    rec.live_gcref_registers.push_back(info.assigned_preg.as_gpr());
                    auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
                    int32_t slot = (it != callee_gpr_to_slot_.end()) ? it->second : 0;
                    int32_t offset = frame_.spill_slot_offset(slot);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            safepoints_.push_back(std::move(rec));
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }
        case LirOpcode::GuardExit: {
            // Builds a runtime::DeoptExitRecord on the stack (header, 8-byte
            // slots, kind bytes; no size limit) and calls
            // brass_deopt_exit_record with its address.
            size_t num_uses = inst.uses.size();
            if (inst.deopt_kinds.size() != num_uses) {
                throw_unsupported("x64 emit (guard exit)", "state map without per-value kinds");
            }
            // Isel names an exit symbol only for a guard with an exit stub.
            const bool has_exit_symbol = !inst.exit_symbol.empty();
            const size_t header_bytes = runtime::DeoptExitRecord::kSlotsOffset;
            const size_t slots_bytes = num_uses * 8;
            const size_t kinds_bytes = (num_uses + 7) & ~size_t(7);
            const bool win64 = fn_.calling_conv.kind() == CallingConvKind::Win64;
            const size_t shadow_space = win64 ? 32 : 0;
            // Below the record: the outgoing area of the calls made here,
            // i.e. the shadow space, plus the exit stub's stack arguments.
            // One allocation holds both, so RSP is total_alloc under the
            // prologue's frame for every call this exit makes.
            const size_t out_area = has_exit_symbol ? plan_exit_stub_args(inst, win64).out_area : shadow_space;
            const size_t record_bytes = header_bytes + slots_bytes + kinds_bytes;
            if (out_area > 0x7FFF0000u || record_bytes > 0x7FFF0000u - out_area) {
                throw_unsupported("x64 emit (guard exit)", "deopt state map too large for one frame");
            }
            size_t total_alloc = ((out_area + record_bytes + 15) & ~size_t(15));
            if (total_alloc > 0x7FFF0000u) {
                throw_unsupported("x64 emit (guard exit)", "deopt state map too large for one frame");
            }

            // Touch every page of a large record in order before moving RSP
            // (Windows commits stack lazily), like __chkstk: RSP moves once,
            // so the unwind info (the primary's before the sub, this exit's
            // region after it) is exact at every instruction.
            constexpr size_t kPage = 4096;
            for (size_t off = kPage; off < total_alloc; off += kPage) {
                enc_.mov(ptr(GPR::RSP, -static_cast<int32_t>(off)), GPR::R11);
            }
            enc_.sub(GPR::RSP, static_cast<int32_t>(total_alloc));
            // From here to the `add rsp` that releases it, RSP is
            // total_alloc under the prologue's frame (the unwind info says so).
            const size_t adjust_begin = buffer_.size();
            const int32_t rec_disp = static_cast<int32_t>(out_area);
            const int32_t slots_disp = rec_disp + static_cast<int32_t>(header_bytes);
            const int32_t kinds_disp = slots_disp + static_cast<int32_t>(slots_bytes);

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
                } else if (op.is_imm_float()) {
                    uint64_t bits = 0;
                    if (op.size == 4) {
                        float f = static_cast<float>(op.imm_float);
                        uint32_t b32 = 0;
                        std::memcpy(&b32, &f, sizeof(b32));
                        bits = b32;
                    } else {
                        std::memcpy(&bits, &op.imm_float, sizeof(bits));
                    }
                    enc_.mov(GPR::R11, static_cast<int64_t>(bits));
                    enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
                } else {
                    throw_unsupported("x64 emit (guard exit)",
                                      std::string("state value operand of kind ") + std::string(to_string(op.kind)));
                }
            }
            // Kind bytes, eight per store.
            for (size_t i = 0; i < num_uses; i += 8) {
                uint64_t packed = 0;
                for (size_t j = 0; j < 8 && i + j < num_uses; ++j) {
                    packed |= static_cast<uint64_t>(inst.deopt_kinds[i + j]) << (8 * j);
                }
                enc_.mov(GPR::R11, static_cast<int64_t>(packed));
                enc_.mov(ptr(GPR::RSP, kinds_disp + static_cast<int32_t>(i)), GPR::R11);
            }

            uint32_t rid = inst.resume_id;
            uint32_t rsn = inst.deopt_reason == 0 ? 1 : inst.deopt_reason;
            uint32_t cnt = static_cast<uint32_t>(num_uses);
            uint32_t flags = has_exit_symbol ? runtime::DeoptExitRecord::kHasExitSymbol : 0u;

            // Header: code_entry (this function's own entry), resume id,
            // reason, count, flags.
            enc_.lea(GPR::R11, fn_.name);
            enc_.mov(ptr(GPR::RSP, rec_disp + 0), GPR::R11);
            enc_.mov(GPR::R11, static_cast<int64_t>((static_cast<uint64_t>(rsn) << 32) | rid));
            enc_.mov(ptr(GPR::RSP, rec_disp + 8), GPR::R11);
            enc_.mov(GPR::R11, static_cast<int64_t>((static_cast<uint64_t>(flags) << 32) | cnt));
            enc_.mov(ptr(GPR::RSP, rec_disp + 16), GPR::R11);

            const GPR arg0 = (fn_.calling_conv.kind() == CallingConvKind::Win64) ? GPR::RCX : GPR::RDI;
            enc_.lea(arg0, ptr(GPR::RSP, rec_disp));
            enc_.call("brass_deopt_exit_record");

            auto emit_epilogue = [&]() {
                codegen::FrameInfo mutable_frame = fn_.frame;
                X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
                X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
            };
            // RAX points at the lower tier's result bits for this frame.
            auto emit_return_rax_result = [&]() {
                enc_.mov(GPR::RAX, ptr(GPR::RAX, 0));
                if (fn_.return_type.kind() == TypeKind::F64) {
                    enc_.movq(XMM::XMM0, GPR::RAX);
                } else if (fn_.return_type.kind() == TypeKind::F32) {
                    enc_.movd(XMM::XMM0, GPR::RAX);
                }
                emit_epilogue();
            };

            Label unhandled = buffer_.create_label();
            enc_.test(GPR::RAX, GPR::RAX);
            enc_.j(Condition::E, unhandled);
            // The region covers each `add rsp` itself (it has not run yet at
            // its own address) and nothing after it: the code there runs at
            // the prologue's RSP, which the gap entries describe.
            auto push_region = [&](size_t begin) {
                stack_adjust_regions_.push_back({static_cast<uint32_t>(begin),
                                                 static_cast<uint32_t>(buffer_.size()),
                                                 static_cast<uint32_t>(total_alloc)});
            };
            enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
            push_region(adjust_begin);
            emit_return_rax_result();
            buffer_.bind(unhandled);
            const size_t unhandled_begin = buffer_.size();

            if (has_exit_symbol) {
                // No handler or resumer: the exit stub finishes the call,
                // called as stub(state values...) with the values read back
                // from the record, which is still on the stack. Its return
                // registers are the function's.
                emit_exit_stub_call(inst, slots_disp);
                enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
                push_region(unhandled_begin);
                emit_epilogue();
            } else {
                // brass_deopt_exit_record aborts when nothing resumes a guard
                // without an exit stub: never reached.
                enc_.ud2();
                push_region(unhandled_begin);
            }
            break;
        }
        default:
            throw_unsupported("x64 emit (control)", to_string(inst.opcode));
    }
}

void EmitContext::emit_instruction(const LirInst& inst, bool is_entry_block, bool is_first_inst) {
    (void)is_entry_block;
    (void)is_first_inst;

    switch (inst.opcode) {
        case LirOpcode::Nop:
            enc_.nop();
            break;
        case LirOpcode::Trap:
            enc_.ud2();
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
        case LirOpcode::Vsqrtps:
        case LirOpcode::Vsqrtpd:
        case LirOpcode::Vextractf128:
        case LirOpcode::Vinsertf128:
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
            // No x64 encoding for this LIR opcode (e.g. the AArch64-only
            // Adds/Subs): a compile error, not a ud2 in the output.
            throw_unsupported("x64 emit", to_string(inst.opcode));
    }
}

CompilationResult compile_lir_to_x64(const LirFunction& fn, const Target& target) {
    EmitContext ctx(fn, target);
    return ctx.compile();
}

} // namespace brass::codegen
