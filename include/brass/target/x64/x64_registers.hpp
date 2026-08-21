#pragma once

#include <cstdint>
#include <string_view>
#include <iosfwd>

namespace brass::x64 {

enum class GPR : uint8_t {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8  = 8,
    R9  = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
    None = 0xFF
};

enum class XMM : uint8_t {
    XMM0  = 0,
    XMM1  = 1,
    XMM2  = 2,
    XMM3  = 3,
    XMM4  = 4,
    XMM5  = 5,
    XMM6  = 6,
    XMM7  = 7,
    XMM8  = 8,
    XMM9  = 9,
    XMM10 = 10,
    XMM11 = 11,
    XMM12 = 12,
    XMM13 = 13,
    XMM14 = 14,
    XMM15 = 15,
    None  = 0xFF
};

enum class OperandSize : uint8_t {
    Byte  = 1,
    Word  = 2,
    Dword = 4,
    Qword = 8
};

enum class Condition : uint8_t {
    O   = 0,  // Overflow (OF=1)
    NO  = 1,  // Not Overflow (OF=0)
    B   = 2,  // Below / Carry (CF=1)
    C   = 2,  // Carry alias
    NAE = 2,  // Not Above or Equal alias
    AE  = 3,  // Above or Equal / Not Carry (CF=0)
    NB  = 3,  // Not Below alias
    NC  = 3,  // Not Carry alias
    E   = 4,  // Equal / Zero (ZF=1)
    Z   = 4,  // Zero alias
    NE  = 5,  // Not Equal / Not Zero (ZF=0)
    NZ  = 5,  // Not Zero alias
    BE  = 6,  // Below or Equal (CF=1 or ZF=1)
    NA  = 6,  // Not Above alias
    A   = 7,  // Above (CF=0 and ZF=0)
    NBE = 7,  // Not Below or Equal alias
    S   = 8,  // Sign / Negative (SF=1)
    NS  = 9,  // Not Sign / Non-negative (SF=0)
    P   = 10, // Parity / Parity Even (PF=1)
    PE  = 10, // Parity Even alias
    NP  = 11, // Parity Odd / Not Parity (PF=0)
    PO  = 11, // Parity Odd alias
    L   = 12, // Less (SF != OF)
    NGE = 12, // Not Greater or Equal alias
    GE  = 13, // Greater or Equal (SF == OF)
    NL  = 13, // Not Less alias
    LE  = 14, // Less or Equal (ZF=1 or SF != OF)
    NG  = 14, // Not Greater alias
    G   = 15, // Greater (ZF=0 and SF == OF)
    NLE = 15, // Not Less or Equal alias
    None = 0xFF
};

constexpr Condition invert(Condition cond) noexcept {
    if (cond == Condition::None) return Condition::None;
    return static_cast<Condition>(static_cast<uint8_t>(cond) ^ 1);
}

constexpr uint8_t reg_code(GPR reg) noexcept {
    return static_cast<uint8_t>(reg) & 0x07;
}

constexpr uint8_t reg_code(XMM reg) noexcept {
    return static_cast<uint8_t>(reg) & 0x07;
}

constexpr uint8_t reg_id(GPR reg) noexcept {
    return static_cast<uint8_t>(reg);
}

constexpr uint8_t reg_id(XMM reg) noexcept {
    return static_cast<uint8_t>(reg);
}

constexpr bool is_extended(GPR reg) noexcept {
    return static_cast<uint8_t>(reg) >= 8 && static_cast<uint8_t>(reg) <= 15;
}

constexpr bool is_extended(XMM reg) noexcept {
    return static_cast<uint8_t>(reg) >= 8 && static_cast<uint8_t>(reg) <= 15;
}

using RegMask = uint16_t;

constexpr RegMask reg_mask(GPR reg) noexcept {
    if (reg == GPR::None) return 0;
    return static_cast<RegMask>(1u << static_cast<uint8_t>(reg));
}

constexpr RegMask reg_mask(XMM reg) noexcept {
    if (reg == XMM::None) return 0;
    return static_cast<RegMask>(1u << static_cast<uint8_t>(reg));
}

constexpr bool mask_has(RegMask mask, GPR reg) noexcept {
    if (reg == GPR::None) return false;
    return (mask & reg_mask(reg)) != 0;
}

constexpr bool mask_has(RegMask mask, XMM reg) noexcept {
    if (reg == XMM::None) return false;
    return (mask & reg_mask(reg)) != 0;
}

constexpr RegMask all_gprs_mask() noexcept {
    return 0xFFFF;
}

constexpr RegMask all_xmms_mask() noexcept {
    return 0xFFFF;
}

std::string_view to_string(GPR reg, OperandSize size = OperandSize::Qword) noexcept;
std::string_view to_string(XMM reg) noexcept;
std::string_view to_string(Condition cond) noexcept;
std::string_view to_string(OperandSize size) noexcept;

std::ostream& operator<<(std::ostream& os, GPR reg);
std::ostream& operator<<(std::ostream& os, XMM reg);
std::ostream& operator<<(std::ostream& os, Condition cond);
std::ostream& operator<<(std::ostream& os, OperandSize size);

} // namespace brass::x64
