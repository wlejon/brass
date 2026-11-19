#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <stdexcept>

namespace brass::aarch64 {

void AArch64Encoder::ldr_literal(GPR dst, Label target, uint64_t fallback_imm) {
    const uint32_t site = buffer_.next_short_branch_site();
    if (buffer_.is_long_branch_site(site)) {
        mov(dst, fallback_imm);
        return;
    }
    const size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        const int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        const int64_t word_disp = disp >> 2;
        if (word_disp < -(1 << 18) || word_disp > ((1 << 18) - 1)) {
            mov(dst, fallback_imm);
            return;
        }
        const uint32_t imm19 = static_cast<uint32_t>(word_disp) & 0x0007FFFFu;
        buffer_.emit_inst(0x58000000u | (imm19 << 5) | reg_code(dst));
    } else {
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19, site);
        buffer_.emit_inst(0x58000000u | reg_code(dst));
    }
}

void AArch64Encoder::ldr_literal(FPR dst, Label target, uint64_t fallback_imm) {
    const uint32_t site = buffer_.next_short_branch_site();
    if (buffer_.is_long_branch_site(site)) {
        mov(GPR::X16, fallback_imm);
        fmov_from_gpr(dst, GPR::X16);
        return;
    }
    const size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        const int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        const int64_t word_disp = disp >> 2;
        if (word_disp < -(1 << 18) || word_disp > ((1 << 18) - 1)) {
            mov(GPR::X16, fallback_imm);
            fmov_from_gpr(dst, GPR::X16);
            return;
        }
        const uint32_t imm19 = static_cast<uint32_t>(word_disp) & 0x0007FFFFu;
        buffer_.emit_inst(0x5C000000u | (imm19 << 5) | reg_code(dst));
    } else {
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19, site);
        buffer_.emit_inst(0x5C000000u | reg_code(dst));
    }
}

void AArch64Encoder::load_symbol_u64(GPR dst, const std::string& symbol) {
    const uint32_t rd = reg_code(dst);
    buffer_.add_relocation(buffer_.size(), RelocationKind::Page21, symbol);
    buffer_.emit_inst(0x90000000u | rd);              // adrp dst, symbol@PAGE
    buffer_.add_relocation(buffer_.size(), RelocationKind::LdSt64Lo12, symbol);
    buffer_.emit_inst(0xF9400000u | (rd << 5) | rd);  // ldr dst, [dst, symbol@PAGEOFF]
}

namespace {

void emit_add_sub_shifted_impl(AArch64Encoder& enc, CodeBuffer& buf, bool is_64, bool is_sub,
                               GPR dst, GPR src1, GPR src2, ShiftType shift, uint8_t amount) {
    if (src2 == GPR::SP) {
        throw std::invalid_argument("AArch64Encoder: SP cannot be the shifted operand");
    }
    const uint8_t max_amt = is_64 ? 63 : 31;
    const uint8_t amt = static_cast<uint8_t>(amount & max_amt);
    if (amt == 0) {
        if (is_64) {
            if (is_sub) enc.sub(dst, src1, src2);
            else enc.add(dst, src1, src2);
        } else {
            if (is_sub) enc.sub32(dst, src1, src2);
            else enc.add32(dst, src1, src2);
        }
        return;
    }
    if (shift == ShiftType::ROR) {
        throw std::invalid_argument("AArch64Encoder: ADD/SUB shifted-register does not support ROR");
    }
    if (dst == GPR::SP || src1 == GPR::SP) {
        GPR scratch = (src1 == GPR::X16 || src2 == GPR::X16) ? GPR::X17 : GPR::X16;
        if (is_64) {
            if (shift == ShiftType::LSL) enc.lsl(scratch, src2, amt);
            else if (shift == ShiftType::LSR) enc.lsr(scratch, src2, amt);
            else enc.asr(scratch, src2, amt);
            if (is_sub) enc.sub(dst, src1, scratch);
            else enc.add(dst, src1, scratch);
        } else {
            if (shift == ShiftType::LSL) enc.lsl32(scratch, src2, amt);
            else if (shift == ShiftType::LSR) enc.lsr32(scratch, src2, amt);
            else enc.asr32(scratch, src2, amt);
            if (is_sub) enc.sub32(dst, src1, scratch);
            else enc.add32(dst, src1, scratch);
        }
        return;
    }
    const uint32_t sf = is_64 ? 1u : 0u;
    const uint32_t op = is_sub ? 1u : 0u;
    const uint32_t base = (sf << 31) | (op << 30) | (0x0Bu << 24) |
                          (static_cast<uint32_t>(shift) << 22);
    buf.emit_inst(base | (reg_code(src2) << 16) | (static_cast<uint32_t>(amt) << 10) |
                  (reg_code(src1) << 5) | reg_code(dst));
}

} // namespace

void AArch64Encoder::add_shifted(GPR dst, GPR src1, GPR src2, ShiftType shift, uint8_t amount) {
    emit_add_sub_shifted_impl(*this, buffer_, true, false, dst, src1, src2, shift, amount);
}

void AArch64Encoder::add32_shifted(GPR dst, GPR src1, GPR src2, ShiftType shift, uint8_t amount) {
    emit_add_sub_shifted_impl(*this, buffer_, false, false, dst, src1, src2, shift, amount);
}

void AArch64Encoder::sub_shifted(GPR dst, GPR src1, GPR src2, ShiftType shift, uint8_t amount) {
    emit_add_sub_shifted_impl(*this, buffer_, true, true, dst, src1, src2, shift, amount);
}

void AArch64Encoder::sub32_shifted(GPR dst, GPR src1, GPR src2, ShiftType shift, uint8_t amount) {
    emit_add_sub_shifted_impl(*this, buffer_, false, true, dst, src1, src2, shift, amount);
}

void AArch64Encoder::add_lsl(GPR dst, GPR src1, GPR src2, uint8_t amount) {
    add_shifted(dst, src1, src2, ShiftType::LSL, amount);
}

} // namespace brass::aarch64
