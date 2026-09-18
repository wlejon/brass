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
    } else {
        // Materialize in scratch register (XZR is not usable, but if immediate is large we can split)
        // Split into high and low:
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
    buffer_.emit_inst(0x8B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::add32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x0B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
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
    buffer_.emit_inst(0xCB000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::sub32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x4B000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
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

// =============================================================================
// Logical: AND, BIC, ORR, ORN, EOR, EON, TST, MVN
// =============================================================================

void AArch64Encoder::and_(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x8A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::and32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x0A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::bic(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x8A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::bic32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x0A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orr(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xAA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orr32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x2A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orn(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xAA200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orn32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x2A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eor(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xCA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eor32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x4A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eon(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xCA200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eon32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x4A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::tst(GPR src1, GPR src2) {
    buffer_.emit_inst(0xEA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | 31u);
}

void AArch64Encoder::tst32(GPR src1, GPR src2) {
    buffer_.emit_inst(0x6A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | 31u);
}

void AArch64Encoder::mvn(GPR dst, GPR src) {
    orn(dst, GPR::XZR, src);
}

void AArch64Encoder::mvn32(GPR dst, GPR src) {
    orn32(dst, GPR::XZR, src);
}

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
        uint32_t imm26 = static_cast<uint32_t>(disp >> 2) & 0x03FFFFFFu;
        buffer_.emit_inst(0x14000000u | imm26);
    } else {
        buffer_.emit_inst(0x14000000u);
        buffer_.record_label_fixup(target, patch_off, FixupKind::Branch26);
    }
}

void AArch64Encoder::b(Condition cond, Label target) {
    size_t patch_off = buffer_.size();
    uint32_t base = 0x54000000u | static_cast<uint32_t>(cond);
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm19 = static_cast<uint32_t>(disp >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(base | (imm19 << 5));
    } else {
        buffer_.emit_inst(base);
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19);
    }
}

void AArch64Encoder::bl(Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm26 = static_cast<uint32_t>(disp >> 2) & 0x03FFFFFFu;
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
    size_t patch_off = buffer_.size();
    uint32_t base = 0xB4000000u | reg_code(reg);
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm19 = static_cast<uint32_t>(disp >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(base | (imm19 << 5));
    } else {
        buffer_.emit_inst(base);
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19);
    }
}

void AArch64Encoder::cbz32(GPR reg, Label target) {
    size_t patch_off = buffer_.size();
    uint32_t base = 0x34000000u | reg_code(reg);
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm19 = static_cast<uint32_t>(disp >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(base | (imm19 << 5));
    } else {
        buffer_.emit_inst(base);
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19);
    }
}

void AArch64Encoder::cbnz(GPR reg, Label target) {
    size_t patch_off = buffer_.size();
    uint32_t base = 0xB5000000u | reg_code(reg);
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm19 = static_cast<uint32_t>(disp >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(base | (imm19 << 5));
    } else {
        buffer_.emit_inst(base);
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19);
    }
}

void AArch64Encoder::cbnz32(GPR reg, Label target) {
    size_t patch_off = buffer_.size();
    uint32_t base = 0x35000000u | reg_code(reg);
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm19 = static_cast<uint32_t>(disp >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(base | (imm19 << 5));
    } else {
        buffer_.emit_inst(base);
        buffer_.record_label_fixup(target, patch_off, FixupKind::CondBranch19);
    }
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
// Floating-Point & SIMD
// =============================================================================

// Double precision
void AArch64Encoder::fadd(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E602800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsub(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E603800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmul(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E600800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fdiv(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E601800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmadd_d(FPR dst, FPR src1, FPR src2, FPR addend) {
    buffer_.emit_inst(0x1F400000u | (reg_code(src2) << 16) | (reg_code(addend) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsqrt(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E61C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fabs(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E60C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fneg(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E614000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcmp(FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E602000u | (reg_code(src2) << 16) | (reg_code(src1) << 5));
}

void AArch64Encoder::fcmp_zero(FPR src) {
    buffer_.emit_inst(0x1E602008u | (reg_code(src) << 5));
}

// Single precision
void AArch64Encoder::fadd_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E202800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsub_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E203800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmul_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E200800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fdiv_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E201800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmadd_s(FPR dst, FPR src1, FPR src2, FPR addend) {
    buffer_.emit_inst(0x1F000000u | (reg_code(src2) << 16) | (reg_code(addend) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsqrt_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E21C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fabs_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E20C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fneg_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E214000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcmp_s(FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E202000u | (reg_code(src2) << 16) | (reg_code(src1) << 5));
}

void AArch64Encoder::fcmp_zero_s(FPR src) {
    buffer_.emit_inst(0x1E202008u | (reg_code(src) << 5));
}

// Moves & Conversions
void AArch64Encoder::fmov(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E604000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E204000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_to_gpr(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E660000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_from_gpr(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E670000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_to_gpr32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E260000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_from_gpr32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E270000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_d_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E22C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_s_d(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E624000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_d(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E620000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_s(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E220000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_d32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E620000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_s32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E220000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_d(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E780000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_d32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E780000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_s(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E380000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_s32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E380000u | (reg_code(src) << 5) | reg_code(dst));
}

// FP Loads & Stores
void AArch64Encoder::ldr(FPR dst, const MemAddress& mem) {
    // 64-bit Dd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (3u << 30) | (7u << 27) | (1u << 26) | (1u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 8 == 0) && (off / 8 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 8) & 0xFFFu;
        buffer_.emit_inst(0xFD400000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xFC400000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP load offset out of range");
    }
}

void AArch64Encoder::ldr_s(FPR dst, const MemAddress& mem) {
    // 32-bit Sd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (2u << 30) | (7u << 27) | (1u << 26) | (1u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 4 == 0) && (off / 4 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 4) & 0xFFFu;
        buffer_.emit_inst(0xBD400000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xBC400000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP load offset out of range");
    }
}

void AArch64Encoder::ldr_q(FPR dst, const MemAddress& mem) {
    // 128-bit Qd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (0u << 30) | (7u << 27) | (1u << 26) | (3u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 16 == 0) && (off / 16 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 16) & 0xFFFu;
        buffer_.emit_inst(0x3DC00000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0x3CC00000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: 128-bit FP load offset out of range");
    }
}

void AArch64Encoder::str(FPR src, const MemAddress& mem) {
    // 64-bit Dd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (3u << 30) | (7u << 27) | (1u << 26) | (0u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 8 == 0) && (off / 8 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 8) & 0xFFFu;
        buffer_.emit_inst(0xFD000000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xFC000000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP store offset out of range");
    }
}

void AArch64Encoder::str_s(FPR src, const MemAddress& mem) {
    // 32-bit Sd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (2u << 30) | (7u << 27) | (1u << 26) | (0u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 4 == 0) && (off / 4 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 4) & 0xFFFu;
        buffer_.emit_inst(0xBD000000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xBC000000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP store offset out of range");
    }
}

void AArch64Encoder::str_q(FPR src, const MemAddress& mem) {
    // 128-bit Qd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (0u << 30) | (7u << 27) | (1u << 26) | (2u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 16 == 0) && (off / 16 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 16) & 0xFFFu;
        buffer_.emit_inst(0x3D800000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0x3C800000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: 128-bit FP store offset out of range");
    }
}

void AArch64Encoder::ldp(FPR dst1, FPR dst2, const MemAddress& mem) {
    int64_t off = mem.offset;
    if (off % 8 != 0) throw std::runtime_error("AArch64Encoder: FP LDP offset must be 8-byte aligned");
    int64_t simm7 = off / 8;
    if (simm7 < -64 || simm7 > 63) throw std::runtime_error("AArch64Encoder: FP LDP offset out of range");
    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t mode = mem.is_post_indexed() ? 1u : (mem.is_pre_indexed() ? 3u : 2u);
    buffer_.emit_inst((1u << 30) | (5u << 27) | (1u << 26) | (mode << 23) | (1u << 22) |
                      (imm7 << 15) | (reg_code(dst2) << 10) | (reg_code(mem.base) << 5) | reg_code(dst1));
}

void AArch64Encoder::stp(FPR src1, FPR src2, const MemAddress& mem) {
    int64_t off = mem.offset;
    if (off % 8 != 0) throw std::runtime_error("AArch64Encoder: FP STP offset must be 8-byte aligned");
    int64_t simm7 = off / 8;
    if (simm7 < -64 || simm7 > 63) throw std::runtime_error("AArch64Encoder: FP STP offset out of range");
    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t mode = mem.is_post_indexed() ? 1u : (mem.is_pre_indexed() ? 3u : 2u);
    buffer_.emit_inst((1u << 30) | (5u << 27) | (1u << 26) | (mode << 23) | (0u << 22) |
                      (imm7 << 15) | (reg_code(src2) << 10) | (reg_code(mem.base) << 5) | reg_code(src1));
}

// 128-bit SIMD Vector
// 128-bit SIMD Vector (Float)
void AArch64Encoder::vec_fadd_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsub_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA0D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmul_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E20DC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fdiv_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E20FC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmla_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20CC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmin_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA0F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmax_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fneg_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EA0F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsqrt_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EA1F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fadd_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsub_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE0D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmul_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E60DC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fdiv_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E60FC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmla_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60CC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmin_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE0F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmax_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fneg_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EE0F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsqrt_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EE1F800u | (reg_code(src) << 5) | reg_code(dst));
}

// 128-bit SIMD Vector (Integer & Bitwise)
void AArch64Encoder::vec_add_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sub_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_mul_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA09C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_smin_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA06C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_smax_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA06400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_add_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sub_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_and(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E201C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_orr(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA01C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_eor(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E201C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::cnt_8b(FPR dst, FPR src) {
    buffer_.emit_inst(0x0E205800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::uaddlv_h(FPR dst, FPR src) {
    buffer_.emit_inst(0x2E303800u | (reg_code(src) << 5) | reg_code(dst));
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

void AArch64Encoder::adr(GPR dst, Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t disp = static_cast<int64_t>(buffer_.label_offset(target)) - static_cast<int64_t>(patch_off);
        uint32_t imm21 = static_cast<uint32_t>(disp) & 0x001FFFFFu;
        uint32_t immlo = imm21 & 0x3u;
        uint32_t immhi = (imm21 >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(0x10000000u | (immlo << 29) | (immhi << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x10000000u | reg_code(dst));
        buffer_.record_label_fixup(target, patch_off, FixupKind::Adr21);
    }
}

void AArch64Encoder::adrp(GPR dst, Label target) {
    size_t patch_off = buffer_.size();
    if (buffer_.is_bound(target)) {
        int64_t target_page = static_cast<int64_t>(buffer_.label_offset(target)) >> 12;
        int64_t curr_page = static_cast<int64_t>(patch_off) >> 12;
        int64_t disp = target_page - curr_page;
        uint32_t imm21 = static_cast<uint32_t>(disp) & 0x001FFFFFu;
        uint32_t immlo = imm21 & 0x3u;
        uint32_t immhi = (imm21 >> 2) & 0x0007FFFFu;
        buffer_.emit_inst(0x90000000u | (immlo << 29) | (immhi << 5) | reg_code(dst));
    } else {
        buffer_.emit_inst(0x90000000u | reg_code(dst));
        buffer_.record_label_fixup(target, patch_off, FixupKind::Adrp21);
    }
}

} // namespace brass::aarch64
