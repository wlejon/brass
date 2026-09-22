#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/runtime/deopt.hpp>
#include <cassert>
#include <stdexcept>
#include <string>

namespace brass::aarch64 {

using namespace brass::codegen;

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
                    if (dst.size == 1) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int & 0xff));
                    else if (dst.size == 2) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int & 0xffff));
                    else if (dst.size == 4) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                    else enc_.mov(dst_gpr, static_cast<uint64_t>(src.imm_int));
                } else {
                    if (dst.size == 1 || src.size == 1) enc_.ldrb(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 1));
                    else if (dst.size == 2 || src.size == 2) enc_.ldrh(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 2));
                    else if (dst.size == 4 || src.size == 4) enc_.ldr32(dst_gpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 4));
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
                    if (dst.size == 1 || src.size == 1) enc_.strb(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 2 || src.size == 2) enc_.strh(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else {
                    if (dst.size == 16 || src.size == 16) enc_.str_q(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str_s(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_fpr(), dst_mem);
                }
            } else if (src.is_imm_int()) {
                if (dst.size == 1) {
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
                MemAddress src_mem = ensure_accessible_mem(to_mem_address(src), src_scratch);
                if (dst.size == 16 || src.size == 16) {
                    enc_.ldr_q(FPR::V31, src_mem);
                    enc_.str_q(FPR::V31, dst_mem);
                } else if (dst.size == 1 || src.size == 1) {
                    enc_.ldrb(data_scratch, src_mem);
                    enc_.strb(data_scratch, dst_mem);
                } else if (dst.size == 2 || src.size == 2) {
                    enc_.ldrh(data_scratch, src_mem);
                    enc_.strh(data_scratch, dst_mem);
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
            PReg scratch = is_fpr ? PReg::aarch64_fpr(FPR::V31) : PReg::aarch64_gpr(GPR::X15);
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

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_).offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), frame_).offset);
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

            StackMapRecord map_rec;
            map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
            map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
            map_rec.safepoint_id = inst.safepoint_id;

            for (const auto& v : inst.live_gcrefs) {
                const auto& info = fn_.get_vreg_info(v);
                if (info.is_spilled) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_).offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), frame_).offset);
                    map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg));
                }
            }
            stack_map_records_.push_back(std::move(map_rec));
            break;
        }

        case LirOpcode::Ret: {
            AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
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
            } else if (op.is_local_slot()) {
                MemAddress addr = AArch64FrameLayout::local_frame_address(op.local_offset, fn_.frame);
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
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_).offset);
                    rec.live_gcref_spill_offsets.push_back(offset);
                    map_rec.add_root(StackMapRootLocation::frame_slot(offset));
                } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
                    rec.live_gcref_registers.push_back(static_cast<x64::GPR>(info.assigned_preg.code));
                    int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), frame_).offset);
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
                } else if (op.is_spill_slot() || op.is_mem() || op.is_local_slot()) {
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

            if (!inst.exit_symbol.empty() && inst.exit_symbol != "@exit_stub" && inst.exit_symbol != "exit_stub") {
                if (total_alloc > 0) enc_.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));
                enc_.bl("brass_get_thread_deopt_frame");
                // X0 now holds DeoptFrame*
                enc_.add(GPR::X1, GPR::X0, static_cast<uint32_t>(offsetof(runtime::DeoptFrame, slots)));
                enc_.mov32(GPR::X0, rid);
                enc_.bl(inst.exit_symbol);

                if (fn_.return_type.kind() == TypeKind::F64) {
                    enc_.fmov_from_gpr(FPR::V0, GPR::X0);
                } else if (fn_.return_type.kind() == TypeKind::F32) {
                    enc_.fmov_from_gpr32(FPR::V0, GPR::X0);
                } else if (fn_.return_type.is_v128()) {
                    enc_.fmov_from_gpr(FPR::V0, GPR::X0);
                }

                AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
            } else {
                if (total_alloc > 0) enc_.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));
                if (inst.deopt_reason == static_cast<uint32_t>(runtime::DeoptReason::BoundsCheckFailed)) {
                    // Out-of-bounds guards must not silently exit returning uninitialized garbage.
                    // If brass_deopt_exit returned null (unhandled deopt), trap with brk(0).
                    Label handle_ok = buffer_.create_label();
                    enc_.cbnz(GPR::X0, handle_ok);
                    enc_.brk(0);
                    buffer_.bind(handle_ok);
                }
                if (fn_.return_type.kind() == TypeKind::F64) {
                    enc_.fmov_from_gpr(FPR::V0, GPR::X0);
                } else if (fn_.return_type.kind() == TypeKind::F32) {
                    enc_.fmov_from_gpr32(FPR::V0, GPR::X0);
                } else if (fn_.return_type.is_v128()) {
                    enc_.fmov_from_gpr(FPR::V0, GPR::X0);
                }

                AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
            }
            break;
        }

        default:
            assert(false && "Unhandled LIR opcode in AArch64 control emit");
            throw std::runtime_error("Unhandled LIR opcode in AArch64 control emit: " + std::to_string(static_cast<int>(inst.opcode)));
    }
}


} // namespace brass::aarch64
