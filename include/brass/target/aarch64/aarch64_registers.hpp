#pragma once

#include <cstdint>
#include <string_view>
#include <iosfwd>

namespace brass::aarch64 {

enum class GPR : uint8_t {
    X0  = 0,
    X1  = 1,
    X2  = 2,
    X3  = 3,
    X4  = 4,
    X5  = 5,
    X6  = 6,
    X7  = 7,
    X8  = 8,
    X9  = 9,
    X10 = 10,
    X11 = 11,
    X12 = 12,
    X13 = 13,
    X14 = 14,
    X15 = 15,
    X16 = 16,
    X17 = 17,
    X18 = 18,
    X19 = 19,
    X20 = 20,
    X21 = 21,
    X22 = 22,
    X23 = 23,
    X24 = 24,
    X25 = 25,
    X26 = 26,
    X27 = 27,
    X28 = 28,
    X29 = 29, // Frame pointer (FP)
    FP  = 29, // Alias for X29
    X30 = 30, // Link register (LR)
    LR  = 30, // Alias for X30
    SP  = 31, // Stack pointer
    XZR = 32, // Zero register (in register context encodes as 31)
    None = 0xFF
};

// Common aliases
inline constexpr GPR FP = GPR::X29;
inline constexpr GPR LR = GPR::X30;

enum class FPR : uint8_t {
    V0  = 0,
    V1  = 1,
    V2  = 2,
    V3  = 3,
    V4  = 4,
    V5  = 5,
    V6  = 6,
    V7  = 7,
    V8  = 8,
    V9  = 9,
    V10 = 10,
    V11 = 11,
    V12 = 12,
    V13 = 13,
    V14 = 14,
    V15 = 15,
    V16 = 16,
    V17 = 17,
    V18 = 18,
    V19 = 19,
    V20 = 20,
    V21 = 21,
    V22 = 22,
    V23 = 23,
    V24 = 24,
    V25 = 25,
    V26 = 26,
    V27 = 27,
    V28 = 28,
    V29 = 29,
    V30 = 30,
    V31 = 31,
    None = 0xFF
};

enum class OperandSize : uint8_t {
    Byte  = 1,
    Half  = 2,
    Word  = 4, // 32-bit (Wd / Sd)
    Xword = 8, // 64-bit (Xd / Dd)
    Quad  = 16 // 128-bit (Qd / Vd)
};

enum class Condition : uint8_t {
    EQ = 0,   // Equal (Z==1)
    NE = 1,   // Not equal (Z==0)
    CS = 2,   // Carry set / unsigned higher or same (C==1)
    HS = 2,   // Alias for CS
    CC = 3,   // Carry clear / unsigned lower (C==0)
    LO = 3,   // Alias for CC
    MI = 4,   // Minus / negative (N==1)
    PL = 5,   // Plus / positive or zero (N==0)
    VS = 6,   // Overflow set (V==1)
    VC = 7,   // Overflow clear (V==0)
    HI = 8,   // Unsigned higher (C==1 && Z==0)
    LS = 9,   // Unsigned lower or same (C==0 || Z==1)
    GE = 10,  // Signed greater than or equal (N==V)
    LT = 11,  // Signed less than (N!=V)
    GT = 12,  // Signed greater than (Z==0 && N==V)
    LE = 13,  // Signed less than or equal (Z==1 || N!=V)
    AL = 14,  // Always
    NV = 15,  // Always (never in ARMv7, alias in ARMv8)
    None = 0xFF
};

constexpr Condition invert(Condition cond) noexcept {
    if (cond == Condition::None || cond == Condition::AL || cond == Condition::NV) {
        return cond;
    }
    return static_cast<Condition>(static_cast<uint8_t>(cond) ^ 1);
}

constexpr uint8_t reg_code(GPR reg) noexcept {
    if (reg == GPR::SP || reg == GPR::XZR) return 31;
    return static_cast<uint8_t>(reg) & 0x1F;
}

constexpr uint8_t reg_code(FPR reg) noexcept {
    return static_cast<uint8_t>(reg) & 0x1F;
}

constexpr uint8_t reg_id(GPR reg) noexcept {
    return static_cast<uint8_t>(reg);
}

constexpr uint8_t reg_id(FPR reg) noexcept {
    return static_cast<uint8_t>(reg);
}

using RegMask = uint32_t;

constexpr RegMask reg_mask(GPR reg) noexcept {
    if (reg == GPR::None || reg == GPR::XZR) return 0;
    return static_cast<RegMask>(1u << reg_code(reg));
}

constexpr RegMask reg_mask(FPR reg) noexcept {
    if (reg == FPR::None) return 0;
    return static_cast<RegMask>(1u << reg_code(reg));
}

constexpr bool mask_has(RegMask mask, GPR reg) noexcept {
    if (reg == GPR::None || reg == GPR::XZR) return false;
    return (mask & reg_mask(reg)) != 0;
}

constexpr bool mask_has(RegMask mask, FPR reg) noexcept {
    if (reg == FPR::None) return false;
    return (mask & reg_mask(reg)) != 0;
}

constexpr RegMask all_gprs_mask() noexcept {
    return 0xFFFFFFFFu;
}

constexpr RegMask all_fprs_mask() noexcept {
    return 0xFFFFFFFFu;
}

std::string_view to_string(GPR reg, OperandSize size = OperandSize::Xword) noexcept;
std::string_view to_string(FPR reg, OperandSize size = OperandSize::Xword) noexcept;
std::string_view to_string(Condition cond) noexcept;
std::string_view to_string(OperandSize size) noexcept;

std::ostream& operator<<(std::ostream& os, GPR reg);
std::ostream& operator<<(std::ostream& os, FPR reg);
std::ostream& operator<<(std::ostream& os, Condition cond);
std::ostream& operator<<(std::ostream& os, OperandSize size);

} // namespace brass::aarch64
