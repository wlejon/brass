#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <brass/runtime/deopt.hpp>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

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

namespace {

// Opcodes whose encoding is always VEX.256 (the FMA vector forms are 256-bit
// only on 32-byte operands, which the operand scan sees).
bool is_vex256_opcode(LirOpcode op) {
    switch (op) {
        case LirOpcode::Vmovaps: case LirOpcode::Vmovups:
        case LirOpcode::Vaddps: case LirOpcode::Vsubps: case LirOpcode::Vmulps:
        case LirOpcode::Vdivps: case LirOpcode::Vminps: case LirOpcode::Vmaxps:
        case LirOpcode::Vaddpd: case LirOpcode::Vsubpd: case LirOpcode::Vmulpd:
        case LirOpcode::Vdivpd: case LirOpcode::Vminpd: case LirOpcode::Vmaxpd:
        case LirOpcode::Vsqrtps: case LirOpcode::Vsqrtpd:
        case LirOpcode::Vextractf128: case LirOpcode::Vinsertf128:
        case LirOpcode::Vpaddd: case LirOpcode::Vpsubd: case LirOpcode::Vpmulld:
        case LirOpcode::Vpaddq: case LirOpcode::Vpsubq:
        case LirOpcode::Vandps: case LirOpcode::Vorps: case LirOpcode::Vxorps:
        case LirOpcode::Vandpd: case LirOpcode::Vorpd: case LirOpcode::Vxorpd:
        case LirOpcode::Vpand: case LirOpcode::Vpor: case LirOpcode::Vpxor:
        case LirOpcode::Vbroadcastss: case LirOpcode::Vbroadcastsd:
        case LirOpcode::Vpbroadcastd: case LirOpcode::Vpbroadcastq:
            return true;
        default:
            return false;
    }
}

// A 32-byte operand is a YMM register or a 256-bit memory access; there is
// no 32-byte GPR value.
bool has_256_operand(const ScratchVector<LirOperand>& ops) {
    for (const auto& op : ops) {
        if (op.size == 32) return true;
    }
    return false;
}

bool function_dirties_upper(const LirFunction& fn) {
    for (const auto& block : fn.blocks) {
        for (const auto& inst : block->instructions) {
            if (is_vex256_opcode(inst->opcode) || has_256_operand(inst->defs) ||
                has_256_operand(inst->uses)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

void EmitContext::clear_upper_state() {
    if (dirties_upper_) enc_.vzeroupper();
}

CompilationResult EmitContext::compile() {
    // The scan is the normal path; the encoder's own record of a VEX.256
    // encoding backs it up (an emitter that uses YMM behind a narrower LIR
    // operand), re-emitting the function once with the clears in place.
    if (!dirties_upper_) dirties_upper_ = function_dirties_upper(fn_);
    CompilationResult result = compile_once();
    if (!dirties_upper_ && enc_.wrote_256()) {
        EmitContext again(fn_, target_);
        again.dirties_upper_ = true;
        return again.compile();
    }

    return relax_branches(std::move(result));
}

CompilationResult EmitContext::compile_once() {
    CompilationResult result;
    safepoints_.clear();
    stack_map_records_.clear();
    block_labels_.clear();
    callee_gpr_to_slot_.clear();
    frame_ = fn_.frame;
    begin_constant_pool();

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
    block_cold_.clear();
    for (const auto& block : fn_.blocks) {
        block_labels_[block->id] = buffer_.create_label();
        if (block->id >= block_cold_.size()) block_cold_.resize(size_t{block->id} + 1, 0);
        block_cold_[block->id] = block->cold ? 1 : 0;
    }

    // 3. Emit prologue at entry
    X64FrameLayout::emit_prologue(enc_, frame_, fn_.calling_conv);
    zero_tagged_locals();

    // 4. Emit blocks
    const BranchThreading threading(fn_);
    threading_ = &threading;
    // A target is the fall-through when only blocks that emit nothing lie
    // between: skipped ones, empty ones, and a jump to the block after.
    auto emits_nothing = [&](size_t i) {
        const auto& ins = fn_.blocks[i]->instructions;
        if (threading.skipped(i) || ins.empty()) return true;
        const ClosingBranches c = closing_branches(fn_, i);
        return ins.size() == 1 && c.count == 1 && !c.conditional && i + 1 < fn_.blocks.size() &&
               threading.target(c.other) == fn_.blocks[i + 1]->id;
    };
    // Whether the last block emitted runs into the next one's first byte.
    bool prev_falls_through = true;
    for (size_t b_idx = 0; b_idx < fn_.blocks.size(); ++b_idx) {
        const auto& block = fn_.blocks[b_idx];
        if (threading.skipped(b_idx)) {
            buffer_.bind(block_labels_[block->id]);
            continue;
        }
        if (b_idx > 0) {
            // A loop's head starts 32-byte aligned: a small loop then sits in
            // one 32-byte fetch window (on Zen one straddling two ran the
            // sieve's inner loop 1.3x slower). Any other block 16, but only
            // where nothing falls into it: padding a block that the one
            // before runs straight into executes the padding on that path,
            // which on a chain of small blocks (an and-chain's tests, an
            // inline cache's guards) was a nop or two per branch. A cold
            // block (a miss, a slow call) is not aligned at all: its padding
            // only spreads the hot code over more lines.
            if (threading.loop_head(b_idx)) {
                buffer_.align(32);
            } else if (!prev_falls_through && !block->cold) {
                buffer_.align(16);
            }
        }
        buffer_.bind(block_labels_[block->id]);
        result.block_offsets[block->id] = buffer_.size();
        emitting_cold_ = block->cold;

        if (block->is_landing_pad) {
            if (frame_.total_frame_size > 0) {
                enc_.lea(GPR::RSP, ptr(GPR::RBP, -static_cast<int32_t>(frame_.total_frame_size)));
            } else {
                enc_.mov(GPR::RSP, GPR::RBP);
            }
        }

        auto falls_to = [&](uint32_t target) {
            for (size_t i = b_idx + 1; i < fn_.blocks.size(); ++i) {
                if (fn_.blocks[i]->id == target) return true;
                if (!emits_nothing(i)) return false;
            }
            return false;
        };
        size_t n_insts = block->instructions.size();
        const ClosingBranches closing = closing_branches(fn_, b_idx);
        const size_t limit = n_insts - closing.count;

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

        if (closing.conditional) {
            const auto& jcc = *block->instructions[n_insts - closing.count];
            if (jcc.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jcc.loc);
            }
            uint32_t true_target = threading.target(closing.taken);
            uint32_t false_target = threading.target(closing.other);

            if (falls_to(false_target)) {
                emit_jump(jcc.condition, true_target);
                prev_falls_through = true;
            } else if (falls_to(true_target)) {
                emit_jump(invert(jcc.condition), false_target);
                prev_falls_through = true;
            } else {
                emit_jump(jcc.condition, true_target);
                emit_jump(Condition::None, false_target);
                prev_falls_through = false;
            }
        } else if (closing.count == 1) {
            const auto& jmp = *block->instructions[n_insts - 1];
            if (jmp.loc.is_valid()) {
                result.debug_table.add_line_entry(static_cast<uint32_t>(buffer_.size()), jmp.loc);
            }
            uint32_t target = threading.target(closing.other);
            if (!falls_to(target)) {
                emit_jump(Condition::None, target);
                prev_falls_through = false;
            } else {
                prev_falls_through = true;
            }
        } else {
            // Anything but a return or a trap may run on; assuming it does
            // only skips padding.
            const LirOpcode last = n_insts > 0 ? block->instructions[n_insts - 1]->opcode : LirOpcode::Nop;
            prev_falls_through = !(last == LirOpcode::Ret || last == LirOpcode::Trap);
        }
    }

    threading_ = nullptr;
    emit_constant_pool();
    for (size_t off : buffer_.rel8_overflows()) {
        auto it = short_patch_ord_.find(off);
        if (it != short_patch_ord_.end()) overflow_ords_.push_back(it->second);
    }
    // BRASS_BLOCK_MAP=<file>: one line per emitted block, `function <tab>
    // offset <tab> c|h <tab> block`, appended. With a profiler's samples by
    // offset in the function's symbol, where the time of a function is by
    // the names its front end gave its blocks.
    if (static const char* const map_path = std::getenv("BRASS_BLOCK_MAP"); map_path && map_path[0] != '\0') {
        static std::mutex map_mutex;
        std::string lines;
        for (const auto& block : fn_.blocks) {
            const auto it = result.block_offsets.find(block->id);
            if (it == result.block_offsets.end()) continue;
            lines += std::string(fn_.name) + '\t' + std::to_string(it->second) + '\t' + (block->cold ? 'c' : 'h') +
                     '\t' + block->name + '\n';
        }
        const std::lock_guard<std::mutex> lock(map_mutex);
        if (std::FILE* f = std::fopen(map_path, "a")) {
            std::fwrite(lines.data(), 1, lines.size(), f);
            std::fclose(f);
        }
    }
    result.code_buffer = std::move(buffer_);
    result.safepoints = std::move(safepoints_);
    result.stack_map.function_name = fn_.name;
    result.stack_map.code_size = static_cast<uint32_t>(result.code_buffer.size());
    result.stack_map.records = std::move(stack_map_records_);
    // The spill slots run down from just under the save area, and the
    // locals (the allocas) lie under them (X64FrameLayout).
    result.stack_map.slot_area_lo = X64FrameLayout::local_frame_address(0, frame_).disp;
    result.stack_map.slot_area_hi = X64FrameLayout::spill_slot_address(0, frame_).disp + 8;
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


// On Win64 the OS unwinder reads the code at a frame's return address, and
// when it is an epilogue (add rsp / lea rsp, pops, ret) it takes the frame to
// be in its epilogue: it unwinds by running that code and calls no language
// handler, so an invoke whose normal edge goes straight to a return would
// never reach its pad. A nop after the call keeps the return address off the
// epilogue, as MSVC does after calls.
void EmitContext::pad_invoke_return() {
    if (fn_.calling_conv.kind() == CallingConvKind::Win64) enc_.nop();
}

void EmitContext::spill_callee_saved_roots(const LirInst& inst) {
    for (const auto& v : inst.live_gcrefs) {
        const auto& info = fn_.get_vreg_info(v);
        if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
            auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
            if (it != callee_gpr_to_slot_.end()) {
                enc_.mov(X64FrameLayout::spill_slot_address(it->second, frame_), info.assigned_preg.as_gpr());
            }
        }
    }
}

void EmitContext::reload_callee_saved_roots(const LirInst& inst) {
    for (const auto& v : inst.live_gcrefs) {
        const auto& info = fn_.get_vreg_info(v);
        if (!info.is_spilled && info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
            auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
            if (it != callee_gpr_to_slot_.end()) {
                enc_.mov(info.assigned_preg.as_gpr(), X64FrameLayout::spill_slot_address(it->second, frame_));
            }
        }
    }
}

void EmitContext::record_gc_point(const LirInst& inst, size_t return_offset, SafepointRecord* sp) {
    StackMapRecord map_rec;
    map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
    map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
    map_rec.safepoint_id = inst.safepoint_id;

    for (const auto& v : inst.live_gcrefs) {
        const auto& info = fn_.get_vreg_info(v);
        const StackMapValueKind value = StackMapRootLocation::value_kind_of(info.vreg);
        if (info.is_spilled) {
            int32_t offset = frame_.spill_slot_offset(info.assigned_spill_slot);
            if (sp) sp->live_gcref_spill_offsets.push_back(offset);
            map_rec.add_root(StackMapRootLocation::frame_slot(offset, value));
        } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
            if (sp) sp->live_gcref_registers.push_back(info.assigned_preg.as_gpr());
            auto it = callee_gpr_to_slot_.find(info.assigned_preg.code);
            int32_t slot = (it != callee_gpr_to_slot_.end()) ? it->second : 0;
            int32_t offset = frame_.spill_slot_offset(slot);
            map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg, value));
        }
    }
    for (const auto& [start, bytes] : frame_.tagged_locals) {
        for (uint32_t off = 0; off + 8 <= bytes; off += 8) {
            const int32_t disp = X64FrameLayout::local_frame_address(static_cast<int32_t>(start + off), frame_).disp;
            map_rec.add_root(StackMapRootLocation::frame_slot(disp, StackMapValueKind::Tagged));
        }
    }
    stack_map_records_.push_back(std::move(map_rec));
}

void EmitContext::zero_tagged_locals() {
    for (const auto& [start, bytes] : frame_.tagged_locals) {
        for (uint32_t off = 0; off + 8 <= bytes; off += 8) {
            enc_.mov(X64FrameLayout::local_frame_address(static_cast<int32_t>(start + off), frame_), int32_t{0});
        }
    }
}

void EmitContext::emit_control_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Jmp:
            emit_jump(Condition::None, threading_ ? threading_->target(inst.uses[0].label_id) : inst.uses[0].label_id);
            break;
        case LirOpcode::Jcc:
            emit_jump(inst.condition, threading_ ? threading_->target(inst.uses[0].label_id) : inst.uses[0].label_id);
            break;
        case LirOpcode::Call: {
            // A 256-bit argument travels in a YMM register; no other value
            // lives in one across a call (256-bit values spanning a call
            // are spilled), so the clear loses nothing.
            if (!has_256_operand(inst.uses)) clear_upper_state();
            spill_callee_saved_roots(inst);

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
            } else if (!inst.entry_cell.empty()) {
                enc_.call_through(inst.entry_cell);
            } else {
                enc_.call(callee);
            }
            size_t return_offset = buffer_.size();
            reload_callee_saved_roots(inst);

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
                pad_invoke_return();
            }
            record_gc_point(inst, return_offset, nullptr);
            break;
        }
        case LirOpcode::CallIndirect: {
            if (!has_256_operand(inst.uses)) clear_upper_state();
            spill_callee_saved_roots(inst);
            size_t call_start = buffer_.size();
            enc_.call(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();
            reload_callee_saved_roots(inst);

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
                pad_invoke_return();
            }
            record_gc_point(inst, return_offset, nullptr);
            break;
        }
        case LirOpcode::Ret: {
            // A 256-bit result is returned in YMM0; its caller clears.
            if (!fn_.return_type.is_v256()) clear_upper_state();
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
            clear_upper_state();
            spill_callee_saved_roots(inst);
            enc_.call("brass_gc_safepoint");
            size_t return_offset = buffer_.size();
            reload_callee_saved_roots(inst);

            SafepointRecord rec;
            rec.code_offset = return_offset;
            rec.safepoint_id = inst.safepoint_id;
            record_gc_point(inst, return_offset, &rec);
            safepoints_.push_back(std::move(rec));
            break;
        }
        case LirOpcode::GuardExit:
            emit_guard_exit(inst);
            break;
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
        case LirOpcode::KeepAlive:
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
        case LirOpcode::Smulh:
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
        case LirOpcode::Vcmpps: case LirOpcode::Vcmppd:
        case LirOpcode::Vpcmpeqd: case LirOpcode::Vpcmpgtd:
        case LirOpcode::Vpcmpeqq: case LirOpcode::Vpcmpgtq:
        case LirOpcode::Vpminsd: case LirOpcode::Vpmaxsd:
        case LirOpcode::Vpsllvd: case LirOpcode::Vpsllvq:
        case LirOpcode::Vpsrlvd: case LirOpcode::Vpsrlvq: case LirOpcode::Vpsravd:
        case LirOpcode::Vshufps: case LirOpcode::Vblendvps: case LirOpcode::Vblendvpd:
        case LirOpcode::Vcvtdq2ps: case LirOpcode::Vcvttps2dq:
        case LirOpcode::Vcvtdq2pd: case LirOpcode::Vcvttpd2dq:
        case LirOpcode::Vcvtps2pd: case LirOpcode::Vcvtpd2ps: case LirOpcode::Vpmovsxdq:
        case LirOpcode::Vroundps: case LirOpcode::Vroundpd:
        case LirOpcode::Vmovmskps: case LirOpcode::Vmovmskpd:
            emit_vex_lanes_instruction(inst);
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
