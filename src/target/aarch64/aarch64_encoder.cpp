#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <stdexcept>

namespace brass::aarch64 {

// =============================================================================
// Helper: Add / Sub Immediate
// =============================================================================

void AArch64Encoder::emit_add_sub_imm(bool is_64, bool is_sub, bool set_flags, GPR dst, GPR src, uint32_t imm) {
    uint32_t sf = is_64 ? 1u : 0u;
    uint32_t op = is_sub ? 1u : 0u;
    uint32_t s  = set_flags ? 1u : 0u;

    uint32_t base = (sf << 31) | (op << 30) | (s << 29) | (0x11u << 24);

    if (imm <= 4095) {
        uint32_t inst = base | (imm << 10) | (reg_code(src) << 5) | reg_code(dst);
        buffer_.emit_inst(inst);
    } else if ((imm & 0xFFFu) == 0 && (imm >> 12) <= 4095) {
        uint32_t inst = base | (1u << 22) | ((imm >> 12) << 10) | (reg_code(src) << 5) | reg_code(dst);
        buffer_.emit_inst(inst);
    } else if (set_flags) {
        if (dst == GPR::XZR || reg_code(dst) == 31) {
            throw std::invalid_argument("AArch64Encoder: flag-setting add/sub immediate cannot be split with XZR destination");
        }
        // Materialize full constant into scratch register X16 so flags are set soundly in a single instruction
        GPR scratch = (src == GPR::X16) ? GPR::X17 : GPR::X16;
        if (is_64) {
            mov(scratch, static_cast<uint64_t>(imm));
            if (is_sub) {
                subs(dst, src, scratch);
            } else {
                adds(dst, src, scratch);
            }
        } else {
            mov32(scratch, imm);
            if (is_sub) {
                subs32(dst, src, scratch);
            } else {
                adds32(dst, src, scratch);
            }
        }
    } else if ((imm & 0xFF000000u) != 0) {
        // Bits 24..31 cannot fit in a 2-instruction 12-bit high/low split.
        // Materialize full constant into scratch register X16 rather than silently truncating.
        GPR scratch = (src == GPR::X16) ? GPR::X17 : GPR::X16;
        if (is_64) {
            mov(scratch, static_cast<uint64_t>(imm));
            if (is_sub) {
                sub(dst, src, scratch);
            } else {
                add(dst, src, scratch);
            }
        } else {
            mov32(scratch, imm);
            if (is_sub) {
                sub32(dst, src, scratch);
            } else {
                add32(dst, src, scratch);
            }
        }
    } else {
        // Non-flag-setting immediate with bits 24..31 == 0: split into high (shifted) and low
        uint32_t low = imm & 0xFFFu;
        uint32_t high = imm & 0x00FFF000u;
        if (high != 0) {
            uint32_t inst1 = base | (1u << 22) | ((high >> 12) << 10) | (reg_code(src) << 5) | reg_code(dst);
            buffer_.emit_inst(inst1);
            if (low != 0) {
                uint32_t inst2 = base | (low << 10) | (reg_code(dst) << 5) | reg_code(dst);
                buffer_.emit_inst(inst2);
            }
        } else {
            uint32_t inst = base | (low << 10) | (reg_code(src) << 5) | reg_code(dst);
            buffer_.emit_inst(inst);
        }
    }
}

// =============================================================================
// ALU: ADD, ADDS, SUB, SUBS, CMP, CMN, NEG, NEGS
// =============================================================================

void AArch64Encoder::add(GPR dst, GPR src1, GPR src2) {
    if (dst == GPR::SP || src1 == GPR::SP) {
        constexpr uint32_t option_uxtx = 0b011u;
        buffer_.emit_inst(0x8B200000u | (reg_code(src2) << 16) | (option_uxtx << 13) |
                          (reg_code(src1) << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x8B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
    }
}

void AArch64Encoder::add32(GPR dst, GPR src1, GPR src2) {
    if (dst == GPR::SP || src1 == GPR::SP) {
        constexpr uint32_t option_uxtw = 0b010u;
        buffer_.emit_inst(0x0B200000u | (reg_code(src2) << 16) | (option_uxtw << 13) |
                          (reg_code(src1) << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x0B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
    }
}

void AArch64Encoder::add(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(true, false, false, dst, src, imm);
}

void AArch64Encoder::add32(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(false, false, false, dst, src, imm);
}

void AArch64Encoder::adds(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xAB000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::adds32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x2B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::adds(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(true, false, true, dst, src, imm);
}

void AArch64Encoder::adds32(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(false, false, true, dst, src, imm);
}

void AArch64Encoder::sub(GPR dst, GPR src1, GPR src2) {
    if (dst == GPR::SP || src1 == GPR::SP) {
        constexpr uint32_t option_uxtx = 0b011u;
        buffer_.emit_inst(0xCB200000u | (reg_code(src2) << 16) | (option_uxtx << 13) |
                          (reg_code(src1) << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0xCB000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
    }
}

void AArch64Encoder::sub32(GPR dst, GPR src1, GPR src2) {
    if (dst == GPR::SP || src1 == GPR::SP) {
        constexpr uint32_t option_uxtw = 0b010u;
        buffer_.emit_inst(0x4B200000u | (reg_code(src2) << 16) | (option_uxtw << 13) |
                          (reg_code(src1) << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x4B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
    }
}

void AArch64Encoder::sub(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(true, true, false, dst, src, imm);
}

void AArch64Encoder::sub32(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(false, true, false, dst, src, imm);
}

void AArch64Encoder::subs(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xEB000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::subs32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x6B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::subs(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(true, true, true, dst, src, imm);
}

void AArch64Encoder::subs32(GPR dst, GPR src, uint32_t imm) {
    emit_add_sub_imm(false, true, true, dst, src, imm);
}

void AArch64Encoder::cmp(GPR src1, GPR src2) {
    subs(GPR::XZR, src1, src2);
}

void AArch64Encoder::cmp32(GPR src1, GPR src2) {
    subs32(GPR::XZR, src1, src2);
}

void AArch64Encoder::cmp(GPR src, uint32_t imm) {
    subs(GPR::XZR, src, imm);
}

void AArch64Encoder::cmp32(GPR src, uint32_t imm) {
    subs32(GPR::XZR, src, imm);
}

void AArch64Encoder::cmn(GPR src1, GPR src2) {
    adds(GPR::XZR, src1, src2);
}

void AArch64Encoder::cmn32(GPR src1, GPR src2) {
    adds32(GPR::XZR, src1, src2);
}

void AArch64Encoder::cmn(GPR src, uint32_t imm) {
    adds(GPR::XZR, src, imm);
}

void AArch64Encoder::cmn32(GPR src, uint32_t imm) {
    adds32(GPR::XZR, src, imm);
}

void AArch64Encoder::neg(GPR dst, GPR src) {
    sub(dst, GPR::XZR, src);
}

void AArch64Encoder::neg32(GPR dst, GPR src) {
    sub32(dst, GPR::XZR, src);
}

void AArch64Encoder::negs(GPR dst, GPR src) {
    subs(dst, GPR::XZR, src);
}

void AArch64Encoder::negs32(GPR dst, GPR src) {
    subs32(dst, GPR::XZR, src);
}

// Logical operations (register & immediate) are implemented in aarch64_encoder_logical.cpp

// =============================================================================
// Multiply & Divide
// =============================================================================

void AArch64Encoder::mul(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9B007C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::mul32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1B007C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::madd(GPR dst, GPR src1, GPR src2, GPR acc) {
    buffer_.emit_inst(0x9B000000u | (reg_code(src2) << 16) | (reg_code(acc) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::madd32(GPR dst, GPR src1, GPR src2, GPR acc) {
    buffer_.emit_inst(0x1B000000u | (reg_code(src2) << 16) | (reg_code(acc) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::msub(GPR dst, GPR src1, GPR src2, GPR acc) {
    buffer_.emit_inst(0x9B008000u | (reg_code(src2) << 16) | (reg_code(acc) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::msub32(GPR dst, GPR src1, GPR src2, GPR acc) {
    buffer_.emit_inst(0x1B008000u | (reg_code(src2) << 16) | (reg_code(acc) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::smulh(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9B407C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::umulh(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9BC07C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::sdiv(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC00C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::sdiv32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC00C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::udiv(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC00800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::udiv32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC00800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

// =============================================================================
// Shifts and Bit Manipulation
// =============================================================================

void AArch64Encoder::lsl(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC02000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::lsl32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC02000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::lsl(GPR dst, GPR src, uint8_t shift) {
    shift &= 63u;
    uint32_t immr = (-static_cast<int>(shift)) & 63u;
    uint32_t imms = 63u - shift;
    buffer_.emit_inst(0xD3400000u | (immr << 16) | (imms << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::lsl32(GPR dst, GPR src, uint8_t shift) {
    shift &= 31u;
    uint32_t immr = (-static_cast<int>(shift)) & 31u;
    uint32_t imms = 31u - shift;
    buffer_.emit_inst(0x53000000u | (immr << 16) | (imms << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::lsr(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC02400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::lsr32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC02400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::lsr(GPR dst, GPR src, uint8_t shift) {
    shift &= 63u;
    buffer_.emit_inst(0xD3400000u | (shift << 16) | (63u << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::lsr32(GPR dst, GPR src, uint8_t shift) {
    shift &= 31u;
    buffer_.emit_inst(0x53000000u | (shift << 16) | (31u << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::asr(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC02800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::asr32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC02800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::asr(GPR dst, GPR src, uint8_t shift) {
    shift &= 63u;
    buffer_.emit_inst(0x93400000u | (shift << 16) | (63u << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::asr32(GPR dst, GPR src, uint8_t shift) {
    shift &= 31u;
    buffer_.emit_inst(0x13000000u | (shift << 16) | (31u << 10) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::ror(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x9AC02C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::ror32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x1AC02C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::clz(GPR dst, GPR src) {
    buffer_.emit_inst(0xDAC01000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::clz32(GPR dst, GPR src) {
    buffer_.emit_inst(0x5AC01000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::rbit(GPR dst, GPR src) {
    buffer_.emit_inst(0xDAC00000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::rbit32(GPR dst, GPR src) {
    buffer_.emit_inst(0x5AC00000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::rev(GPR dst, GPR src) {
    buffer_.emit_inst(0xDAC00C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::rev32(GPR dst, GPR src) {
    buffer_.emit_inst(0x5AC00800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::rev16(GPR dst, GPR src) {
    buffer_.emit_inst(0x5AC00400u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::sxtb(GPR dst, GPR src) {
    buffer_.emit_inst(0x13001C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::sxth(GPR dst, GPR src) {
    buffer_.emit_inst(0x13003C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::sxtw(GPR dst, GPR src) {
    buffer_.emit_inst(0x93407C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::uxtb(GPR dst, GPR src) {
    buffer_.emit_inst(0x53001C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::uxth(GPR dst, GPR src) {
    buffer_.emit_inst(0x53003C00u | (reg_code(src) << 5) | reg_code(dst));
}

// =============================================================================
// Move & Constant Loading
// =============================================================================

void AArch64Encoder::mov(GPR dst, GPR src) {
    if (dst == GPR::SP || src == GPR::SP) {
        add(dst, src, 0);
    } else {
        orr(dst, GPR::XZR, src);
    }
}

void AArch64Encoder::mov32(GPR dst, GPR src) {
    if (dst == GPR::SP || src == GPR::SP) {
        add32(dst, src, 0);
    } else {
        orr32(dst, GPR::XZR, src);
    }
}

void AArch64Encoder::movz(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0xD2800000u | (static_cast<uint32_t>(hw_shift & 3) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::movz32(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0x52800000u | (static_cast<uint32_t>(hw_shift & 1) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::movk(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0xF2800000u | (static_cast<uint32_t>(hw_shift & 3) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::movk32(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0x72800000u | (static_cast<uint32_t>(hw_shift & 1) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::movn(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0x92800000u | (static_cast<uint32_t>(hw_shift & 3) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::movn32(GPR dst, uint16_t imm, uint8_t hw_shift) {
    buffer_.emit_inst(0x12800000u | (static_cast<uint32_t>(hw_shift & 1) << 21) |
                      (static_cast<uint32_t>(imm) << 5) | reg_code(dst));
}

void AArch64Encoder::mov(GPR dst, uint64_t imm) {
    if (imm == 0) {
        mov(dst, GPR::XZR);
        return;
    }

    uint16_t chunks[4] = {
        static_cast<uint16_t>(imm & 0xFFFFu),
        static_cast<uint16_t>((imm >> 16) & 0xFFFFu),
        static_cast<uint16_t>((imm >> 32) & 0xFFFFu),
        static_cast<uint16_t>((imm >> 48) & 0xFFFFu),
    };

    bool first = true;
    for (uint8_t i = 0; i < 4; ++i) {
        if (first && chunks[i] != 0) {
            movz(dst, chunks[i], i);
            first = false;
        } else if (!first && chunks[i] != 0) {
            movk(dst, chunks[i], i);
        }
    }
    if (first) {
        movz(dst, 0, 0);
    }
}

void AArch64Encoder::mov32(GPR dst, uint32_t imm) {
    uint16_t low = static_cast<uint16_t>(imm & 0xFFFFu);
    uint16_t high = static_cast<uint16_t>((imm >> 16) & 0xFFFFu);

    if (high == 0) {
        movz32(dst, low, 0);
    } else {
        movz32(dst, low, 0);
        movk32(dst, high, 1);
    }
}

// =============================================================================
// Conditional Selection & Set
// =============================================================================

void AArch64Encoder::csel(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x9A800000u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csel32(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x1A800000u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csinc(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x9A800400u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csinc32(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x1A800400u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csinv(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x9A800800u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csinv32(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x1A800800u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csneg(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x9A800C00u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::csneg32(GPR dst, GPR src1, GPR src2, Condition cond) {
    buffer_.emit_inst(0x1A800C00u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::cset(GPR dst, Condition cond) {
    csinc(dst, GPR::XZR, GPR::XZR, invert(cond));
}

void AArch64Encoder::cset32(GPR dst, Condition cond) {
    csinc32(dst, GPR::XZR, GPR::XZR, invert(cond));
}

void AArch64Encoder::csetm(GPR dst, Condition cond) {
    csinv(dst, GPR::XZR, GPR::XZR, invert(cond));
}

void AArch64Encoder::csetm32(GPR dst, Condition cond) {
    csinv32(dst, GPR::XZR, GPR::XZR, invert(cond));
}

// =============================================================================
// Control Flow: Branches, Calls, Returns
// =============================================================================

void AArch64Encoder::b(Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        int64_t wdisp = disp >> 2;
        if (wdisp < -(1 << 25) || wdisp > ((1 << 25) - 1)) {
            throw std::runtime_error("AArch64Encoder: Branch target out of 26-bit range");
        }
        uint32_t imm26 = static_cast<uint32_t>(wdisp) & 0x03FFFFFFu;
        buffer_.emit_inst(0x14000000u | imm26);
    } else {
        buffer_.emit_inst(0x14000000u);
        buffer_.record_label_fixup(target, patch_off, FixupKind::Branch26);
    }
}

void AArch64Encoder::b(Condition cond, Label target) {
    if (cond == Condition::AL) {
        b(target);
        return;
    }
    emit_short_branch(0x54000000u | static_cast<uint32_t>(cond), FixupKind::CondBranch19, target);
}

void AArch64Encoder::emit_short_branch(uint32_t inst, FixupKind kind, Label target) {
    const int bits = (kind == FixupKind::TestBranch14) ? 14 : 19;
    const int64_t lo = -(int64_t(1) << (bits - 1));
    const int64_t hi = (int64_t(1) << (bits - 1)) - 1;
    const uint32_t field = (1u << bits) - 1u;
    const uint32_t site = buffer_.next_short_branch_site();

    // The long form: the inverted condition skips the unconditional branch
    // (+8 bytes, a 2-word displacement). B.cond inverts in bit 0 of the
    // condition; CBZ/CBNZ and TBZ/TBNZ in bit 24.
    auto emit_long = [&]() {
        const uint32_t inverted = (kind == FixupKind::CondBranch19 && (inst & 0xFF000000u) == 0x54000000u)
                                      ? (inst ^ 1u)
                                      : (inst ^ (1u << 24));
        buffer_.emit_inst(inverted | (2u << 5));
        b(target);
    };

    const size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        const int64_t wdisp =
            (static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off)) >> 2;
        if (wdisp < lo || wdisp > hi) {
            emit_long();
            return;
        }
        buffer_.emit_inst(inst | ((static_cast<uint32_t>(wdisp) & field) << 5));
        return;
    }
    if (buffer_.is_long_branch_site(site)) {
        emit_long();
        return;
    }
    buffer_.emit_inst(inst);
    buffer_.record_label_fixup(target, patch_off, kind, site);
}

void AArch64Encoder::bl(Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        int64_t wdisp = disp >> 2;
        if (wdisp < -(1 << 25) || wdisp > ((1 << 25) - 1)) {
            throw std::runtime_error("AArch64Encoder: Branch target out of 26-bit range");
        }
        uint32_t imm26 = static_cast<uint32_t>(wdisp) & 0x03FFFFFFu;
        buffer_.emit_inst(0x94000000u | imm26);
    } else {
        buffer_.emit_inst(0x94000000u);
        buffer_.record_label_fixup(target, patch_off, FixupKind::Branch26);
    }
}

void AArch64Encoder::b(const std::string& symbol) {
    size_t off = buffer_.size();
    buffer_.emit_inst(0x14000000u);
    buffer_.add_relocation(off, RelocationKind::Jump26, symbol);
}

void AArch64Encoder::bl(const std::string& symbol) {
    size_t off = buffer_.size();
    buffer_.emit_inst(0x94000000u);
    buffer_.add_relocation(off, RelocationKind::Call26, symbol);
}

void AArch64Encoder::blr(GPR target) {
    buffer_.emit_inst(0xD63F0000u | (reg_code(target) << 5));
}

void AArch64Encoder::br(GPR target) {
    buffer_.emit_inst(0xD61F0000u | (reg_code(target) << 5));
}

void AArch64Encoder::ret(GPR target) {
    buffer_.emit_inst(0xD65F0000u | (reg_code(target) << 5));
}

void AArch64Encoder::cbz(GPR reg, Label target) {
    emit_short_branch(0xB4000000u | reg_code(reg), FixupKind::CondBranch19, target);
}

void AArch64Encoder::cbz32(GPR reg, Label target) {
    emit_short_branch(0x34000000u | reg_code(reg), FixupKind::CondBranch19, target);
}

void AArch64Encoder::cbnz(GPR reg, Label target) {
    emit_short_branch(0xB5000000u | reg_code(reg), FixupKind::CondBranch19, target);
}

void AArch64Encoder::cbnz32(GPR reg, Label target) {
    emit_short_branch(0x35000000u | reg_code(reg), FixupKind::CondBranch19, target);
}

namespace {
uint32_t test_branch_base(uint32_t op, GPR reg, unsigned bit) {
    if (bit > 63) throw std::invalid_argument("AArch64Encoder: TBZ/TBNZ bit index out of range");
    const uint32_t b5 = (bit >> 5) & 1u;
    const uint32_t b40 = bit & 0x1Fu;
    return op | (b5 << 31) | (b40 << 19) | reg_code(reg);
}
} // namespace

void AArch64Encoder::tbz(GPR reg, unsigned bit, Label target) {
    emit_short_branch(test_branch_base(0x36000000u, reg, bit), FixupKind::TestBranch14, target);
}

void AArch64Encoder::tbnz(GPR reg, unsigned bit, Label target) {
    emit_short_branch(test_branch_base(0x37000000u, reg, bit), FixupKind::TestBranch14, target);
}

void AArch64Encoder::load_symbol_address(GPR dst, const std::string& symbol) {
    const uint32_t rd = reg_code(dst);
    buffer_.add_relocation(buffer_.size(), RelocationKind::GotPage21, symbol);
    buffer_.emit_inst(0x90000000u | rd);                       // adrp dst, :got:symbol
    buffer_.add_relocation(buffer_.size(), RelocationKind::GotLo12, symbol);
    buffer_.emit_inst(0xF9400000u | (rd << 5) | rd);           // ldr dst, [dst, :got_lo12:symbol]
}

// =============================================================================
// Helper: Load/Store Immediate
// =============================================================================

void AArch64Encoder::emit_load_store_imm(bool is_load, uint8_t size_bytes, bool is_signed, GPR reg, const MemAddress& mem) {
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(reg);
    int64_t off = mem.offset;

    // Handle register offset
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t size_code = 0;
        uint32_t opc = is_load ? 1u : 0u;

        switch (size_bytes) {
        case 8: size_code = 3; break;
        case 4: size_code = 2; break;
        case 2: size_code = 1; break;
        case 1: size_code = 0; break;
        default: size_code = 3; break;
        }

        uint32_t inst = (size_code << 30) | (7u << 27) | (opc << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }

    // Pre-index & Post-index
    if (mem.is_pre_indexed() || mem.is_post_indexed()) {
        if (off < -256 || off > 255) {
            throw std::runtime_error("AArch64Encoder: Pre/Post-index offset out of range [-256, 255]");
        }
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        uint32_t idx_mode = mem.is_pre_indexed() ? 3u : 1u; // 3 = pre, 1 = post
        uint32_t size_code = (size_bytes == 8) ? 3u : ((size_bytes == 4) ? 2u : ((size_bytes == 2) ? 1u : 0u));
        uint32_t opc = is_load ? 1u : 0u;
        uint32_t inst = (size_code << 30) | (7u << 27) | (opc << 22) |
                        (simm9 << 12) | (idx_mode << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }

    // Standard Offset (Scaled unsigned or unscaled signed)
    if (off >= 0 && (off % size_bytes == 0) && (off / size_bytes <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / size_bytes) & 0xFFFu;
        uint32_t size_code = (size_bytes == 8) ? 3u : ((size_bytes == 4) ? 2u : ((size_bytes == 2) ? 1u : 0u));
        uint32_t opc = 0;
        if (is_load) {
            if (is_signed && size_bytes < 8) {
                opc = 2u; // signed load
            } else {
                opc = 1u;
            }
        }
        uint32_t inst = (size_code << 30) | (7u << 27) | (1u << 24) | (opc << 22) |
                        (imm12 << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
    } else if (off >= -256 && off <= 255) {
        // Unscaled LDUR/STUR
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        uint32_t size_code = (size_bytes == 8) ? 3u : ((size_bytes == 4) ? 2u : ((size_bytes == 2) ? 1u : 0u));
        uint32_t opc = is_load ? 1u : 0u;
        uint32_t inst = (size_code << 30) | (7u << 27) | (opc << 22) |
                        (simm9 << 12) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
    } else {
        throw std::runtime_error("AArch64Encoder: Load/store offset out of range");
    }
}

void AArch64Encoder::ldr(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 8, false, dst, mem);
}

void AArch64Encoder::ldr32(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 4, false, dst, mem);
}

void AArch64Encoder::ldrb(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 1, false, dst, mem);
}

void AArch64Encoder::ldrh(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 2, false, dst, mem);
}

void AArch64Encoder::ldrsb(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 1, true, dst, mem);
}

void AArch64Encoder::ldrsh(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 2, true, dst, mem);
}

void AArch64Encoder::ldrsw(GPR dst, const MemAddress& mem) {
    emit_load_store_imm(true, 4, true, dst, mem);
}

void AArch64Encoder::str(GPR src, const MemAddress& mem) {
    emit_load_store_imm(false, 8, false, src, mem);
}

void AArch64Encoder::str32(GPR src, const MemAddress& mem) {
    emit_load_store_imm(false, 4, false, src, mem);
}

void AArch64Encoder::strb(GPR src, const MemAddress& mem) {
    emit_load_store_imm(false, 1, false, src, mem);
}

void AArch64Encoder::strh(GPR src, const MemAddress& mem) {
    emit_load_store_imm(false, 2, false, src, mem);
}

// =============================================================================
// Load / Store Pair (LDP / STP)
// =============================================================================

void AArch64Encoder::emit_load_store_pair(bool is_load, bool is_64, GPR reg1, GPR reg2, const MemAddress& mem) {
    int64_t scale = is_64 ? 8 : 4;
    int64_t off = mem.offset;
    if (off % scale != 0) {
        throw std::runtime_error("AArch64Encoder: Pair offset not aligned to scale");
    }
    int64_t simm7 = off / scale;
    if (simm7 < -64 || simm7 > 63) {
        throw std::runtime_error("AArch64Encoder: Pair offset out of range [-64, 63]");
    }

    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t opc = is_64 ? 2u : 0u;
    uint32_t l = is_load ? 1u : 0u;
    uint32_t mode_bits = 2u; // signed offset
    if (mem.is_post_indexed()) mode_bits = 1u;
    else if (mem.is_pre_indexed()) mode_bits = 3u;

    uint32_t inst = (opc << 30) | (5u << 27) | (mode_bits << 23) | (l << 22) |
                    (imm7 << 15) | (reg_code(reg2) << 10) | (reg_code(mem.base) << 5) | reg_code(reg1);
    buffer_.emit_inst(inst);
}

void AArch64Encoder::ldp(GPR dst1, GPR dst2, const MemAddress& mem) {
    emit_load_store_pair(true, true, dst1, dst2, mem);
}

void AArch64Encoder::ldp32(GPR dst1, GPR dst2, const MemAddress& mem) {
    emit_load_store_pair(true, false, dst1, dst2, mem);
}

void AArch64Encoder::stp(GPR src1, GPR src2, const MemAddress& mem) {
    emit_load_store_pair(false, true, src1, src2, mem);
}

void AArch64Encoder::stp32(GPR src1, GPR src2, const MemAddress& mem) {
    emit_load_store_pair(false, false, src1, src2, mem);
}



// =============================================================================
// System & Miscellaneous
// =============================================================================

void AArch64Encoder::nop() {
    buffer_.emit_inst(0xD503201Fu);
}

void AArch64Encoder::brk(uint16_t imm) {
    buffer_.emit_inst(0xD4200000u | (static_cast<uint32_t>(imm) << 5));
}

void AArch64Encoder::brk_if_zero(GPR reg, bool is_64bit, uint16_t imm) {
    Label ok = buffer_.create_label();
    if (is_64bit) {
        cbnz(reg, ok);
    } else {
        cbnz32(reg, ok);
    }
    brk(imm);
    buffer_.bind(ok);
}

void AArch64Encoder::adr(GPR dst, Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        if (disp < -(1 << 20) || disp > ((1 << 20) - 1)) {
            throw std::runtime_error("AArch64Encoder: ADR target out of 21-bit range");
        }
        uint32_t imm21 = static_cast<uint32_t>(disp) & 0x001FFFFFu;
        uint32_t immlo = imm21 & 0x3u;
        uint32_t immhi = (imm21 >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(0x10000000u | (immlo << 29) | (immhi << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x10000000u | reg_code(dst));
        buffer_.record_label_fixup(target, patch_off, FixupKind::Adr21);
    }
}

} // namespace brass::aarch64
