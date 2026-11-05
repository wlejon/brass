#include <brass/codegen/emit_context.hpp>
#include <stdexcept>
#include <string>

// x64 memory operands, and the function's constant pool: the slots
// FrameInfo::constant_slots names (linear_scan_constants.cpp) are never
// stored, and every read of one is a rip-relative read of its constant,
// placed 8-aligned after the function's last block.

namespace brass::codegen {

using namespace brass::x64;

MemAddress EmitContext::to_mem_address(const LirOperand& op) const {
    if (op.is_spill_slot()) {
        if (!pool_labels_.empty()) {
            auto it = pool_labels_.find(op.spill_slot);
            if (it != pool_labels_.end()) return MemAddress::rip_label_ref(it->second.id);
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

void EmitContext::begin_constant_pool() {
    pool_labels_.clear();
    for (const auto& [slot, value] : frame_.constant_slots) pool_labels_[slot] = buffer_.create_label();
}

void EmitContext::emit_constant_pool() {
    if (pool_labels_.empty()) return;
    // Never executed: the last block ends in a return, a jump or a trap.
    while (buffer_.size() % 8 != 0) buffer_.emit8(0xCC);
    for (const auto& [slot, value] : frame_.constant_slots) {
        buffer_.bind(pool_labels_[slot]);
        buffer_.emit64(static_cast<uint64_t>(value));
    }
}

} // namespace brass::codegen
