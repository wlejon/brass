#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

struct PairCandidate {
    bool is_load = false;
    bool is_fpr = false;
    uint8_t size = 0; // 4, 8, or 16
    GPR gpr = GPR::None;
    FPR fpr = FPR::None;
    GPR base = GPR::None;
    int64_t offset = 0;
};

bool resolve_simple_mem(const FrameInfo& frame,
                        const std::unordered_map<int32_t, uint64_t>& constant_slots,
                        const LirOperand& op,
                        GPR& base,
                        int64_t& offset) {
    if (op.is_spill_slot()) {
        if (constant_slots.count(op.spill_slot) != 0) return false;
        MemAddress addr = AArch64FrameLayout::spill_slot_address(op.spill_slot, frame);
        if (addr.mode != AddrMode::Offset) return false;
        base = addr.base;
        offset = addr.offset;
    } else if (op.is_local_slot()) {
        MemAddress addr = AArch64FrameLayout::local_frame_address(op.local_offset, frame);
        if (addr.mode != AddrMode::Offset) return false;
        base = addr.base;
        offset = addr.offset;
    } else if (op.is_mem()) {
        const auto& m = op.mem_val;
        if (m.index_preg.is_valid() || !m.base_preg.is_valid() || !m.base_preg.is_gpr()) return false;
        base = m.base_preg.as_aarch64_gpr();
        offset = m.disp;
    } else {
        return false;
    }
    return base != GPR::None && base != GPR::XZR;
}

bool same_reg(const PairCandidate& a, const PairCandidate& b) {
    if (a.is_fpr != b.is_fpr) return false;
    return a.is_fpr ? (a.fpr == b.fpr) : (a.gpr == b.gpr);
}

std::optional<PairCandidate> extract_candidate(const FrameInfo& frame,
                                               const std::unordered_map<int32_t, uint64_t>& constant_slots,
                                               const LirInst& inst) {
    if (inst.defs.size() != 1 || inst.uses.size() != 1) return std::nullopt;
    if (inst.is_patchable || inst.safepoint_id != 0 || inst.resume_id != 0) return std::nullopt;
    if (inst.clobbered_gprs != 0 || inst.clobbered_xmms != 0) return std::nullopt;

    const LirOperand& dst = inst.defs[0];
    const LirOperand& src = inst.uses[0];
    PairCandidate c;

    // Register load from simple base+disp memory
    if (dst.is_preg()) {
        GPR base = GPR::None;
        int64_t offset = 0;
        if (!resolve_simple_mem(frame, constant_slots, src, base, offset)) return std::nullopt;

        c.is_load = true;
        c.base = base;
        c.offset = offset;
        switch (inst.opcode) {
            case LirOpcode::Mov:
                if (!dst.preg_val.is_gpr() || dst.size != 8 || src.size != 8) return std::nullopt;
                c.is_fpr = false;
                c.size = 8;
                c.gpr = dst.preg_val.as_aarch64_gpr();
                break;
            case LirOpcode::Mov32:
                if (!dst.preg_val.is_gpr() || dst.size != 4 || src.size != 4) return std::nullopt;
                c.is_fpr = false;
                c.size = 4;
                c.gpr = dst.preg_val.as_aarch64_gpr();
                break;
            case LirOpcode::Movsd:
                if (!dst.preg_val.is_xmm() || dst.size != 8 || src.size != 8) return std::nullopt;
                c.is_fpr = true;
                c.size = 8;
                c.fpr = dst.preg_val.as_aarch64_fpr();
                break;
            case LirOpcode::Movss:
                if (!dst.preg_val.is_xmm() || dst.size != 4 || src.size != 4) return std::nullopt;
                c.is_fpr = true;
                c.size = 4;
                c.fpr = dst.preg_val.as_aarch64_fpr();
                break;
            case LirOpcode::Movaps:
            case LirOpcode::Movups:
                if (!dst.preg_val.is_xmm() || dst.size != 16 || src.size != 16) return std::nullopt;
                c.is_fpr = true;
                c.size = 16;
                c.fpr = dst.preg_val.as_aarch64_fpr();
                break;
            default:
                return std::nullopt;
        }
        if (!c.is_fpr && (c.gpr == GPR::None || c.gpr == GPR::XZR || c.gpr == GPR::SP)) {
            return std::nullopt;
        }
        if (c.is_fpr && c.fpr == FPR::None) return std::nullopt;
        return c;
    }

    // Store from register or immediate 0 to simple base+disp memory
    GPR base = GPR::None;
    int64_t offset = 0;
    if (!resolve_simple_mem(frame, constant_slots, dst, base, offset)) return std::nullopt;

    c.is_load = false;
    c.base = base;
    c.offset = offset;
    switch (inst.opcode) {
        case LirOpcode::Mov:
            if (dst.size != 8) return std::nullopt;
            if (src.is_preg() && src.preg_val.is_gpr() && src.size == 8) {
                c.is_fpr = false;
                c.size = 8;
                c.gpr = src.preg_val.as_aarch64_gpr();
            } else if (src.is_imm_int() && src.imm_int == 0) {
                c.is_fpr = false;
                c.size = 8;
                c.gpr = GPR::XZR;
            } else {
                return std::nullopt;
            }
            break;
        case LirOpcode::Mov32:
            if (dst.size != 4 || dst.is_spill_slot()) return std::nullopt;
            if (src.is_preg() && src.preg_val.is_gpr() && src.size == 4) {
                c.is_fpr = false;
                c.size = 4;
                c.gpr = src.preg_val.as_aarch64_gpr();
            } else if (src.is_imm_int() && static_cast<uint32_t>(src.imm_int) == 0) {
                c.is_fpr = false;
                c.size = 4;
                c.gpr = GPR::XZR;
            } else {
                return std::nullopt;
            }
            break;
        case LirOpcode::Movsd:
            if (dst.size != 8 || !src.is_preg() || !src.preg_val.is_xmm() || src.size != 8) return std::nullopt;
            c.is_fpr = true;
            c.size = 8;
            c.fpr = src.preg_val.as_aarch64_fpr();
            break;
        case LirOpcode::Movss:
            if (dst.size != 4 || dst.is_spill_slot() || !src.is_preg() || !src.preg_val.is_xmm() || src.size != 4) {
                return std::nullopt;
            }
            c.is_fpr = true;
            c.size = 4;
            c.fpr = src.preg_val.as_aarch64_fpr();
            break;
        case LirOpcode::Movaps:
        case LirOpcode::Movups:
            if (dst.size != 16 || !src.is_preg() || !src.preg_val.is_xmm() || src.size != 16) return std::nullopt;
            c.is_fpr = true;
            c.size = 16;
            c.fpr = src.preg_val.as_aarch64_fpr();
            break;
        default:
            return std::nullopt;
    }
    if (!c.is_fpr && (c.gpr == GPR::None || c.gpr == GPR::SP)) return std::nullopt;
    if (c.is_fpr && c.fpr == FPR::None) return std::nullopt;
    return c;
}

bool ranges_disjoint(int64_t off1, uint8_t sz1, int64_t off2, uint8_t sz2) {
    const int64_t end1 = off1 + static_cast<int64_t>(sz1);
    const int64_t end2 = off2 + static_cast<int64_t>(sz2);
    return end1 <= off2 || end2 <= off1;
}

bool can_reorder_load_before(const PairCandidate& b, const std::vector<PairCandidate>& intervening) {
    for (const PairCandidate& k : intervening) {
        if (k.base != GPR::FP && k.base != b.base) return false;
        if (!b.is_fpr && b.gpr == k.base) return false;
        if (!k.is_fpr && k.gpr == b.base) return false;
        if (same_reg(b, k)) return false;
    }
    return true;
}

bool can_reorder_store_before(const PairCandidate& b, const std::vector<PairCandidate>& intervening) {
    for (const PairCandidate& k : intervening) {
        if (k.base != b.base) return false;
        if (!ranges_disjoint(k.offset, k.size, b.offset, b.size)) return false;
    }
    return true;
}

} // namespace

bool AArch64EmitContext::try_emit_load_store_pair(const LirBlock& block,
                                                  size_t start_idx,
                                                  size_t limit,
                                                  const std::vector<uint8_t>& skip_inst,
                                                  std::vector<uint8_t>& consumed) {
    const auto a = extract_candidate(frame_, constant_slot_values_, *block.instructions[start_idx]);
    if (!a) return false;

    // If the first load overwrites the base register, pairing changes the base read by later ops.
    if (a->is_load && !a->is_fpr && a->gpr == a->base) return false;

    std::vector<PairCandidate> intervening;
    const size_t scan_end = std::min(limit, start_idx + 5);
    for (size_t j = start_idx + 1; j < scan_end; ++j) {
        if (skip_inst[j] || consumed[j]) continue;
        const auto b = extract_candidate(frame_, constant_slot_values_, *block.instructions[j]);
        if (!b || b->is_load != a->is_load) break;

        bool match = (a->is_fpr == b->is_fpr && a->size == b->size && a->base == b->base);
        if (match) {
            const int64_t stride = static_cast<int64_t>(a->size);
            match = (b->offset == a->offset + stride || a->offset == b->offset + stride);
        }

        int64_t lo_off = 0;
        if (match) {
            lo_off = std::min(a->offset, b->offset);
            const int64_t scale = static_cast<int64_t>(a->size);
            if ((lo_off % scale) != 0) match = false;
            else {
                const int64_t simm7 = lo_off / scale;
                if (simm7 < -64 || simm7 > 63) match = false;
            }
        }

        if (match && a->is_load) {
            if (same_reg(*a, *b)) match = false;
            else if (!can_reorder_load_before(*b, intervening)) match = false;
        } else if (match && !a->is_load) {
            if (!can_reorder_store_before(*b, intervening)) match = false;
        }

        if (match) {
            const PairCandidate& first = (a->offset < b->offset) ? *a : *b;
            const PairCandidate& second = (a->offset < b->offset) ? *b : *a;
            const MemAddress mem = ptr(a->base, lo_off);
            if (a->is_load) {
                if (!a->is_fpr) {
                    if (a->size == 8) enc_.ldp(first.gpr, second.gpr, mem);
                    else enc_.ldp32(first.gpr, second.gpr, mem);
                } else {
                    if (a->size == 8) enc_.ldp(first.fpr, second.fpr, mem);
                    else if (a->size == 4) enc_.ldp_s(first.fpr, second.fpr, mem);
                    else enc_.ldp_q(first.fpr, second.fpr, mem);
                }
            } else {
                if (!a->is_fpr) {
                    if (a->size == 8) enc_.stp(first.gpr, second.gpr, mem);
                    else enc_.stp32(first.gpr, second.gpr, mem);
                } else {
                    if (a->size == 8) enc_.stp(first.fpr, second.fpr, mem);
                    else if (a->size == 4) enc_.stp_s(first.fpr, second.fpr, mem);
                    else enc_.stp_q(first.fpr, second.fpr, mem);
                }
            }
            consumed[j] = 1;
            return true;
        }

        intervening.push_back(*b);
    }

    return false;
}

} // namespace brass::aarch64
