#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <brass/runtime/deopt.hpp>
#include <algorithm>
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

    auto same_loc = [](const LirOperand& a, const LirOperand& b) -> bool {
        if (a.kind != b.kind) return false;
        switch (a.kind) {
            case LirOperandKind::PReg: return a.preg_val == b.preg_val;
            case LirOperandKind::SpillSlot: return a.spill_slot == b.spill_slot;
            case LirOperandKind::LocalSlot: return a.local_offset == b.local_offset;
            case LirOperandKind::Mem: return a.mem_val == b.mem_val && a.size == b.size;
            default: return false;
        }
    };

    std::vector<Move> moves;
    for (size_t i = 0; i < n; ++i) {
        bool shadowed = false;
        for (size_t j = i + 1; j < n; ++j) {
            if (same_loc(inst.defs[i], inst.defs[j])) {
                shadowed = true;
                break;
            }
        }
        if (shadowed || same_loc(inst.defs[i], inst.uses[i])) continue;
        moves.push_back({inst.defs[i], inst.uses[i], false});
    }

    auto emit_move = [&](const LirOperand& dst, const LirOperand& src) {
        if (same_loc(dst, src)) return;

        if (dst.is_preg()) {
            if (dst.preg_val.is_gpr()) {
                GPR dst_gpr = dst.preg_val.as_aarch64_gpr();
                if (src.is_preg()) {
                    if (src.preg_val.is_gpr()) {
                        GPR src_gpr = src.preg_val.as_aarch64_gpr();
                        if (dst.size == 4 && src.size == 4) enc_.mov32(dst_gpr, src_gpr);
                        else enc_.mov(dst_gpr, src_gpr);
                    } else if (dst.size == 4 && src.size == 4) {
                        enc_.fmov_to_gpr32(dst_gpr, src.preg_val.as_aarch64_fpr());
                    } else {
                        enc_.fmov_to_gpr(dst_gpr, src.preg_val.as_aarch64_fpr());
                    }
                } else if (src.is_imm_int()) {
                    if (dst.size == 1) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int & 0xff));
                    else if (dst.size == 2) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int & 0xffff));
                    else if (dst.size == 4) enc_.mov32(dst_gpr, static_cast<uint32_t>(src.imm_int));
                    else {
                        const uint64_t v = static_cast<uint64_t>(src.imm_int);
                        if (!AArch64Encoder::single_inst_mov_imm(v)) {
                            if (const auto off = pinned_constant(v)) {
                                enc_.ldr(dst_gpr, ensure_accessible_mem(ptr(kPinnedTlsGpr, *off), GPR::X16, 8));
                                return;
                            }
                        }
                        enc_.mov(dst_gpr, v);
                    }
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
                FPR dst_fpr = dst.preg_val.as_aarch64_fpr();
                if (src.is_preg() && src.preg_val.is_gpr()) {
                    if (dst.size == 4 && src.size == 4) enc_.fmov_from_gpr32(dst_fpr, src.preg_val.as_aarch64_gpr());
                    else enc_.fmov_from_gpr(dst_fpr, src.preg_val.as_aarch64_gpr());
                } else if (dst.size == 16 || src.size == 16) {
                    if (src.is_preg()) enc_.vec_orr(dst_fpr, src.preg_val.as_aarch64_fpr(), src.preg_val.as_aarch64_fpr());
                    else enc_.ldr_q(dst_fpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 16));
                } else if (dst.size == 4 || src.size == 4) {
                    if (src.is_preg()) enc_.fmov_s(dst_fpr, src.preg_val.as_aarch64_fpr());
                    else enc_.ldr_s(dst_fpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 4));
                } else {
                    if (src.is_preg()) enc_.fmov(dst_fpr, src.preg_val.as_aarch64_fpr());
                    else if (src.is_spill_slot() && dst.size == 8 && src.size == 8) {
                        if (const auto val = constant_slot_value(src.spill_slot)) {
                            emit_load_constant_fpr(dst_fpr, *val);
                        } else {
                            enc_.ldr(dst_fpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 8));
                        }
                    } else {
                        enc_.ldr(dst_fpr, ensure_accessible_mem(to_mem_address(src), GPR::X16, 8));
                    }
                }
            }
        } else {
            // A spill slot holds a narrow GPR value zero-extended to all 8
            // bytes (linear_scan_rewrite's write-back stores the whole
            // register): its reloads may be 64-bit, e.g. as an address
            // index, so a 1/2/4-byte store would leave a stale upper half.
            const bool whole_slot = dst.is_spill_slot();
            int dst_bytes = 8;
            if (src.is_preg()) {
                if (src.preg_val.is_gpr()) {
                    if (whole_slot) dst_bytes = 8;
                    else if (dst.size == 1 || src.size == 1) dst_bytes = 1;
                    else if (dst.size == 2 || src.size == 2) dst_bytes = 2;
                    else if (dst.size == 4 || src.size == 4) dst_bytes = 4;
                } else {
                    if (dst.size == 16 || src.size == 16) dst_bytes = 16;
                    else if (dst.size == 4 || src.size == 4) dst_bytes = 4;
                }
            } else if (src.is_imm_int()) {
                if (!whole_slot && (dst.size == 1 || dst.size == 2 || dst.size == 4)) {
                    dst_bytes = dst.size;
                }
            } else {
                if (dst.size == 16 || src.size == 16) dst_bytes = 16;
                else if (!whole_slot && (dst.size == 1 || src.size == 1)) dst_bytes = 1;
                else if (!whole_slot && (dst.size == 2 || src.size == 2)) dst_bytes = 2;
                else if (!whole_slot && (dst.size == 4 || src.size == 4)) dst_bytes = 4;
            }
            auto pick_pc_scratch = [](const MemAddress& m, GPR avoid = GPR::None) noexcept {
                for (GPR cand : {GPR::X16, GPR::X17, GPR::X15}) {
                    if (cand != avoid && cand != m.base && cand != m.index) return cand;
                }
                return GPR::X16;
            };
            MemAddress dst_mem = ensure_accessible_mem(to_mem_address(dst, GPR::X17), GPR::X17, dst_bytes);
            GPR data_scratch = pick_pc_scratch(dst_mem);
            if (src.is_preg()) {
                if (src.preg_val.is_gpr()) {
                    if (whole_slot) enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 1 || src.size == 1) enc_.strb(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 2 || src.size == 2) enc_.strh(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str32(src.preg_val.as_aarch64_gpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_gpr(), dst_mem);
                } else {
                    if (dst.size == 16 || src.size == 16) enc_.str_q(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else if (dst.size == 4 || src.size == 4) enc_.str_s(src.preg_val.as_aarch64_fpr(), dst_mem);
                    else enc_.str(src.preg_val.as_aarch64_fpr(), dst_mem);
                }
            } else if (src.is_imm_int()) {
                // mov wN zero-extends into xN, so a whole-slot store of the
                // X register writes the value with its upper half zeroed.
                if (dst.size == 1) {
                    enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int & 0xff));
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.strb(data_scratch, dst_mem);
                } else if (dst.size == 2) {
                    enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int & 0xffff));
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.strh(data_scratch, dst_mem);
                } else if (dst.size == 4) {
                    enc_.mov32(data_scratch, static_cast<uint32_t>(src.imm_int));
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.str32(data_scratch, dst_mem);
                } else {
                    enc_.mov(data_scratch, static_cast<uint64_t>(src.imm_int));
                    enc_.str(data_scratch, dst_mem);
                }
            } else {
                GPR src_scratch = pick_pc_scratch(dst_mem, data_scratch);
                int src_bytes = 8;
                if (dst.size == 16 || src.size == 16) src_bytes = 16;
                else if (dst.size == 1 || src.size == 1) src_bytes = 1;
                else if (dst.size == 2 || src.size == 2) src_bytes = 2;
                else if (dst.size == 4 || src.size == 4) src_bytes = 4;
                if (src.is_spill_slot() && dst.size == 8 && src.size == 8) {
                    if (const auto val = constant_slot_value(src.spill_slot)) {
                        emit_load_constant_gpr(data_scratch, *val, src_scratch);
                        enc_.str(data_scratch, dst_mem);
                        return;
                    }
                }
                MemAddress src_mem = ensure_accessible_mem(to_mem_address(src, src_scratch), src_scratch, src_bytes);
                if (dst.size == 16 || src.size == 16) {
                    enc_.ldr_q(FPR::V31, src_mem);
                    enc_.str_q(FPR::V31, dst_mem);
                } else if (dst.size == 1 || src.size == 1) {
                    enc_.ldrb(data_scratch, src_mem);   // zero-extends
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.strb(data_scratch, dst_mem);
                } else if (dst.size == 2 || src.size == 2) {
                    enc_.ldrh(data_scratch, src_mem);   // zero-extends
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.strh(data_scratch, dst_mem);
                } else if (dst.size == 4 || src.size == 4) {
                    enc_.ldr32(data_scratch, src_mem);  // zero-extends
                    if (whole_slot) enc_.str(data_scratch, dst_mem);
                    else enc_.str32(data_scratch, dst_mem);
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
                if (same_loc(m.dst, other.src)) {
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
                if (!moves[i].done && same_loc(moves[i].dst, moves[curr].src)) {
                    next_idx = i;
                    break;
                }
            }
            if (next_idx == SIZE_MAX || next_idx == start_idx) {
                break;
            }
            auto it = std::find(cycle.begin(), cycle.end(), next_idx);
            if (it != cycle.end()) {
                cycle.erase(cycle.begin(), it);
                break;
            }
            curr = next_idx;
        }

        if (!cycle.empty()) {
            size_t idx0 = cycle[0];
            auto& m0 = moves[idx0];
            bool is_fpr = false;
            for (size_t idx : cycle) {
                const auto& cm = moves[idx];
                if ((cm.dst.is_preg() && cm.dst.preg_val.is_xmm()) ||
                    (cm.src.is_preg() && cm.src.preg_val.is_xmm()) ||
                    cm.dst.size >= 16 || cm.src.size >= 16) {
                    is_fpr = true;
                    break;
                }
            }
            // The value parked here has to survive every move still to be
            // emitted, and emit_move itself uses X15/X16/X17 and V31 for a
            // memory-to-memory move (a 16-byte slot copy goes through V31).
            // X14 (the def scratch) and V30 are free at a parallel copy.
            PReg scratch = is_fpr ? PReg::aarch64_fpr(FPR::V30) : PReg::aarch64_gpr(GPR::X14);
            // The scratch parks m0.dst's current value for the move that
            // reads it (the last in the cycle), whose width can differ from
            // m0's: an i32 edge argument and an i64 one swapping registers.
            // A 32-bit park (`mov w14, w1`) drops the i64 value's upper half.
            size_t last_idx = cycle.back();
            const uint8_t sz = std::max({m0.dst.size, moves[last_idx].src.size, moves[last_idx].dst.size});
            LirOperand parked = m0.dst;
            parked.size = sz;

            emit_move(LirOperand::preg(scratch, sz), parked);
            moves[last_idx].src = LirOperand::preg(scratch, sz);
        } else {
            emit_move(moves[start_idx].dst, moves[start_idx].src);
            moves[start_idx].done = true;
        }
    }
}

void AArch64EmitContext::record_gc_point(const LirInst& inst, size_t return_offset, SafepointRecord* sp) {
    StackMapRecord map_rec;
    map_rec.instruction_offset = static_cast<uint32_t>(return_offset);
    map_rec.frame_size = static_cast<uint32_t>(frame_.total_frame_size);
    map_rec.safepoint_id = inst.safepoint_id;

    for (const auto& v : inst.live_gcrefs) {
        const auto& info = fn_.get_vreg_info(v);
        const StackMapValueKind value = StackMapRootLocation::value_kind_of(info.vreg);
        if (info.is_spilled) {
            int32_t offset = static_cast<int32_t>(AArch64FrameLayout::spill_slot_address(info.assigned_spill_slot, frame_).offset);
            if (sp) sp->live_gcref_spill_offsets.push_back(offset);
            map_rec.add_root(StackMapRootLocation::frame_slot(offset, value));
        } else if (info.assigned_preg.is_valid() && info.assigned_preg.is_gpr()) {
            if (sp) sp->live_gcref_registers.push_back(static_cast<x64::GPR>(info.assigned_preg.code));
            int32_t offset = static_cast<int32_t>(AArch64FrameLayout::callee_gpr_address(info.assigned_preg.as_aarch64_gpr(), frame_).offset);
            map_rec.add_root(StackMapRootLocation::callee_saved(offset, info.assigned_preg, value));
        }
    }
    for (const auto& [start, bytes] : frame_.tagged_locals) {
        for (uint32_t off = 0; off + 8 <= bytes; off += 8) {
            const auto disp = AArch64FrameLayout::local_frame_address(static_cast<int32_t>(start + off), frame_).offset;
            map_rec.add_root(StackMapRootLocation::frame_slot(static_cast<int32_t>(disp), StackMapValueKind::Tagged));
        }
    }
    stack_map_records_.push_back(std::move(map_rec));
}

void AArch64EmitContext::zero_tagged_locals() {
    if (frame_.tagged_locals.empty()) return;
    struct Span {
        int64_t lo;
        int64_t hi;
    };
    std::vector<Span> spans;
    spans.reserve(frame_.tagged_locals.size());
    for (const auto& [start, bytes] : frame_.tagged_locals) {
        const uint32_t whole = bytes & ~uint32_t(7);
        if (whole == 0) continue;
        const int64_t lo = AArch64FrameLayout::local_frame_address(static_cast<int32_t>(start), frame_).offset;
        spans.push_back({lo, lo + static_cast<int64_t>(whole)});
    }
    if (spans.empty()) return;
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    });
    std::vector<Span> merged;
    for (const Span& s : spans) {
        if (!merged.empty() && s.lo <= merged.back().hi) {
            merged.back().hi = std::max(merged.back().hi, s.hi);
        } else {
            merged.push_back(s);
        }
    }

    for (const Span& s : merged) {
        int64_t cur = s.lo;
        GPR chunk_base = GPR::None;
        int64_t chunk_off = 0;
        while (cur + 16 <= s.hi) {
            if (cur >= -512 && cur <= 504 && (cur % 8 == 0)) {
                enc_.stp(GPR::XZR, GPR::XZR, ptr(GPR::FP, cur));
            } else {
                if (chunk_base == GPR::None || (cur - chunk_off) < -512 || (cur - chunk_off) > 504 ||
                    ((cur - chunk_off) % 8 != 0)) {
                    chunk_off = cur;
                    if (chunk_off >= 0 && chunk_off <= 4095) {
                        enc_.add(GPR::X16, GPR::FP, static_cast<uint32_t>(chunk_off));
                    } else if (chunk_off < 0 && -chunk_off <= 4095) {
                        enc_.sub(GPR::X16, GPR::FP, static_cast<uint32_t>(-chunk_off));
                    } else {
                        enc_.mov(GPR::X16, static_cast<uint64_t>(chunk_off));
                        enc_.add(GPR::X16, GPR::FP, GPR::X16);
                    }
                    chunk_base = GPR::X16;
                }
                enc_.stp(GPR::XZR, GPR::XZR, ptr(chunk_base, cur - chunk_off));
            }
            cur += 16;
        }
        if (cur + 8 <= s.hi) {
            enc_.str(GPR::XZR, ensure_accessible_mem(ptr(GPR::FP, cur), GPR::X16, 8));
        }
    }
}

void AArch64EmitContext::emit_control_instruction(const LirInst& inst) {
    switch (inst.opcode) {
        case LirOpcode::Jmp: {
            const uint32_t id = threading_ ? threading_->target(inst.uses[0].label_id) : inst.uses[0].label_id;
            enc_.b(block_labels_[id]);
            break;
        }

        case LirOpcode::Jcc: {
            const uint32_t id = threading_ ? threading_->target(inst.uses[0].label_id) : inst.uses[0].label_id;
            enc_.b(to_aarch64_cond(inst.condition), block_labels_[id]);
            break;
        }

        case LirOpcode::Call: {
            size_t call_start = buffer_.size();
            std::string callee = !inst.callee_symbol.empty()
                                     ? inst.callee_symbol
                                     : (!inst.uses.empty() ? inst.uses.back().symbol_name : std::string{});
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
            } else if (!inst.entry_cell.empty()) {
                enc_.load_symbol_u64(GPR::X16, inst.entry_cell);
                call_start = buffer_.size();
                enc_.blr(GPR::X16);
            } else {
                enc_.bl(callee);
            }
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }
            record_gc_point(inst, return_offset, nullptr);
            break;
        }

        case LirOpcode::CallIndirect: {
            size_t call_start = buffer_.size();
            enc_.blr(to_gpr(inst.uses.back()));
            size_t return_offset = buffer_.size();

            if (inst.is_invoke && inst.unwind_block_id != UINT32_MAX) {
                pending_exception_scopes_.push_back({call_start, return_offset, inst.unwind_block_id});
            }
            record_gc_point(inst, return_offset, nullptr);
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
                        enc_.add_lsl(dst, base, index, shift);
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
            record_gc_point(inst, return_offset, &rec);
            safepoints_.push_back(std::move(rec));
            break;
        }

        case LirOpcode::GuardExit:
            emit_guard_exit(inst);
            break;

        default:
            throw_unsupported("aarch64 emit (control)", to_string(inst.opcode));
    }
}


} // namespace brass::aarch64
