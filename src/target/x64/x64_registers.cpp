#include <brass/target/x64/x64_registers.hpp>

namespace brass::x64 {

std::string_view to_string(GPR reg, OperandSize size) noexcept {
    static constexpr std::string_view names_64[] = {
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"
    };
    static constexpr std::string_view names_32[] = {
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
        "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"
    };
    static constexpr std::string_view names_16[] = {
        "ax",  "cx",  "dx",  "bx",  "sp",  "bp",  "si",  "di",
        "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"
    };
    static constexpr std::string_view names_8[] = {
        "al",  "cl",  "dl",  "bl",  "spl", "bpl", "sil", "dil",
        "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"
    };

    uint8_t id = reg_id(reg);
    if (id >= 16) return "none";

    switch (size) {
    case OperandSize::Byte:  return names_8[id];
    case OperandSize::Word:  return names_16[id];
    case OperandSize::Dword: return names_32[id];
    case OperandSize::Qword: return names_64[id];
    default: return "none";
    }
}

std::string_view to_string(XMM reg) noexcept {
    static constexpr std::string_view names_xmm[] = {
        "xmm0", "xmm1", "xmm2",  "xmm3",  "xmm4",  "xmm5",  "xmm6",  "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
    };
    uint8_t id = reg_id(reg);
    if (id >= 16) return "none";
    return names_xmm[id];
}

std::string_view to_string(Condition cond) noexcept {
    switch (cond) {
    case Condition::O:   return "o";
    case Condition::NO:  return "no";
    case Condition::B:   return "b";
    case Condition::AE:  return "ae";
    case Condition::E:   return "e";
    case Condition::NE:  return "ne";
    case Condition::BE:  return "be";
    case Condition::A:   return "a";
    case Condition::S:   return "s";
    case Condition::NS:  return "ns";
    case Condition::P:   return "p";
    case Condition::NP:  return "np";
    case Condition::L:   return "l";
    case Condition::GE:  return "ge";
    case Condition::LE:  return "le";
    case Condition::G:   return "g";
    default: return "none";
    }
}

std::string_view to_string(OperandSize size) noexcept {
    switch (size) {
    case OperandSize::Byte:  return "byte";
    case OperandSize::Word:  return "word";
    case OperandSize::Dword: return "dword";
    case OperandSize::Qword: return "qword";
    default: return "unknown";
    }
}

} // namespace brass::x64
