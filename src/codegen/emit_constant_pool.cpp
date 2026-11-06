#include <brass/codegen/emit_context.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>

// x64 memory operands, and the function's constant pool: the slots
// FrameInfo::constant_slots names (linear_scan_constants.cpp) are never
// stored, and every read of one is a rip-relative read of its constant,
// placed 8-aligned after the function's last block. A constant the host
// keeps at the pinned TLS register (Module::pinned_constants) is read there
// instead: `[r13 + d8]` is three bytes shorter than `[rip + d32]`, and a
// load from it seven shorter than a movabs.

namespace brass::codegen {

using namespace brass::x64;

MemAddress EmitContext::to_mem_address(const LirOperand& op) const {
    if (op.is_spill_slot()) {
        if (!pool_labels_.empty()) {
            auto it = pool_labels_.find(op.spill_slot);
            if (it != pool_labels_.end()) return MemAddress::rip_label_ref(it->second.id);
        }
        if (!pinned_slots_.empty()) {
            auto it = pinned_slots_.find(op.spill_slot);
            if (it != pinned_slots_.end()) return ptr(kPinnedTlsGpr, it->second);
        }
        return X64FrameLayout::spill_slot_address(op.spill_slot, frame_);
    }
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

std::optional<int32_t> EmitContext::pinned_constant(uint64_t value) const {
    const uint32_t pinned = reg_mask(kPinnedTlsGpr);
    if (fn_.pinned_constants.empty() || !(fn_.reserved_gprs & pinned) || (fn_.forced_saved_gprs & pinned)) {
        return std::nullopt;
    }
    for (const auto& [v, off] : fn_.pinned_constants) {
        if (v == value) return off;
    }
    return std::nullopt;
}

bool EmitContext::mov_pinned_constant(GPR dst, int64_t imm) {
    if (imm >= INT32_MIN && imm <= static_cast<int64_t>(UINT32_MAX)) return false;
    const std::optional<int32_t> off = pinned_constant(static_cast<uint64_t>(imm));
    if (!off) return false;
    enc_.mov(dst, ptr(kPinnedTlsGpr, *off));
    return true;
}

void EmitContext::begin_constant_pool() {
    pool_labels_.clear();
    pinned_slots_.clear();
    for (const auto& [slot, value] : frame_.constant_slots) {
        if (const std::optional<int32_t> off = pinned_constant(static_cast<uint64_t>(value))) {
            pinned_slots_[slot] = *off;
        } else {
            pool_labels_[slot] = buffer_.create_label();
        }
    }
}

void EmitContext::emit_constant_pool() {
    if (pool_labels_.empty()) return;
    // Never executed: the last block ends in a return, a jump or a trap.
    while (buffer_.size() % 8 != 0) buffer_.emit8(0xCC);
    for (const auto& [slot, value] : frame_.constant_slots) {
        auto it = pool_labels_.find(slot);
        if (it == pool_labels_.end()) continue;
        buffer_.bind(it->second);
        buffer_.emit64(static_cast<uint64_t>(value));
    }
}

} // namespace brass::codegen
