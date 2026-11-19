#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

std::optional<int32_t> AArch64EmitContext::pinned_constant(uint64_t value) const {
    const uint32_t pinned = reg_mask(kPinnedTlsGpr);
    if (fn_.pinned_constants.empty() || !(fn_.reserved_gprs & pinned) || (fn_.forced_saved_gprs & pinned)) {
        return std::nullopt;
    }
    for (const auto& [v, off] : fn_.pinned_constants) {
        if (v == value) return off;
    }
    return std::nullopt;
}

std::optional<uint64_t> AArch64EmitContext::constant_slot_value(int32_t slot) const {
    const auto it = constant_slot_values_.find(slot);
    if (it == constant_slot_values_.end()) return std::nullopt;
    return it->second;
}

Label AArch64EmitContext::get_or_create_pool_label(uint64_t value) {
    auto [it, inserted] = pool_labels_.try_emplace(value, Label{});
    if (inserted) {
        it->second = buffer_.create_label();
        pool_order_.push_back(value);
    }
    return it->second;
}

void AArch64EmitContext::emit_load_constant_gpr(GPR dst, uint64_t val, GPR addr_scratch) {
    if (AArch64Encoder::single_inst_mov_imm(val)) {
        enc_.mov(dst, val);
        return;
    }
    if (const auto off = pinned_constant(val)) {
        enc_.ldr(dst, ensure_accessible_mem(ptr(kPinnedTlsGpr, *off), addr_scratch, 8));
        return;
    }
    enc_.ldr_literal(dst, get_or_create_pool_label(val), val);
}

void AArch64EmitContext::emit_load_constant_fpr(FPR dst, uint64_t val, GPR addr_scratch) {
    if (val == 0) {
        enc_.movi_zero(dst);
        return;
    }
    uint8_t imm8 = 0;
    if (AArch64Encoder::encode_fp64_imm8(val, imm8)) {
        enc_.fmov_imm(dst, imm8);
        return;
    }
    if (const auto off = pinned_constant(val)) {
        enc_.ldr(dst, ensure_accessible_mem(ptr(kPinnedTlsGpr, *off), addr_scratch, 8));
        return;
    }
    enc_.ldr_literal(dst, get_or_create_pool_label(val), val);
}

void AArch64EmitContext::emit_load_constant_fpr32(FPR dst, uint32_t val) {
    if (val == 0) {
        enc_.movi_zero(dst);
        return;
    }
    uint8_t imm8 = 0;
    if (AArch64Encoder::encode_fp32_imm8(val, imm8)) {
        enc_.fmov_imm_s(dst, imm8);
        return;
    }
    enc_.ldr_literal_s(dst, get_or_create_pool_label(static_cast<uint64_t>(val)), val);
}

void AArch64EmitContext::begin_constant_pool() {
    constant_slot_values_.clear();
    pool_labels_.clear();
    pool_order_.clear();
    for (const auto& [slot, value] : frame_.constant_slots) {
        constant_slot_values_[slot] = static_cast<uint64_t>(value);
    }
}

void AArch64EmitContext::emit_constant_pool() {
    if (pool_order_.empty()) return;
    // Never executed: placed past the end of the last emitted block.
    buffer_.align(8);
    for (uint64_t val : pool_order_) {
        buffer_.bind(pool_labels_[val]);
        buffer_.emit64(val);
    }
}

} // namespace brass::aarch64
