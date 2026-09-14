#include <brass/target/aarch64/aarch64_registers.hpp>
#include <ostream>

namespace brass::aarch64 {

std::string_view to_string(GPR reg, OperandSize size) noexcept {
    static constexpr std::string_view names_64[] = {
        "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
        "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
        "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
        "x24", "x25", "x26", "x27", "x28", "x29", "x30", "sp", "xzr"
    };

    static constexpr std::string_view names_32[] = {
        "w0",  "w1",  "w2",  "w3",  "w4",  "w5",  "w6",  "w7",
        "w8",  "w9",  "w10", "w11", "w12", "w13", "w14", "w15",
        "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23",
        "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wsp", "wzr"
    };

    uint8_t id = reg_id(reg);
    if (id >= 33) return "none";

    switch (size) {
    case OperandSize::Byte:
    case OperandSize::Half:
    case OperandSize::Word:
        return names_32[id];
    case OperandSize::Xword:
    case OperandSize::Quad:
        return names_64[id];
    default:
        return "none";
    }
}

std::string_view to_string(FPR reg, OperandSize size) noexcept {
    static constexpr std::string_view names_v[] = {
        "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
        "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    };
    static constexpr std::string_view names_d[] = {
        "d0",  "d1",  "d2",  "d3",  "d4",  "d5",  "d6",  "d7",
        "d8",  "d9",  "d10", "d11", "d12", "d13", "d14", "d15",
        "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
        "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"
    };
    static constexpr std::string_view names_s[] = {
        "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
        "s8",  "s9",  "s10", "s11", "s12", "s13", "s14", "s15",
        "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
        "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"
    };
    static constexpr std::string_view names_q[] = {
        "q0",  "q1",  "q2",  "q3",  "q4",  "q5",  "q6",  "q7",
        "q8",  "q9",  "q10", "q11", "q12", "q13", "q14", "q15",
        "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23",
        "q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"
    };

    uint8_t id = reg_id(reg);
    if (id >= 32) return "none";

    switch (size) {
    case OperandSize::Word:  return names_s[id];
    case OperandSize::Xword: return names_d[id];
    case OperandSize::Quad:  return names_q[id];
    default: return names_v[id];
    }
}

std::string_view to_string(Condition cond) noexcept {
    switch (cond) {
    case Condition::EQ: return "eq";
    case Condition::NE: return "ne";
    case Condition::CS: return "cs";
    case Condition::CC: return "cc";
    case Condition::MI: return "mi";
    case Condition::PL: return "pl";
    case Condition::VS: return "vs";
    case Condition::VC: return "vc";
    case Condition::HI: return "hi";
    case Condition::LS: return "ls";
    case Condition::GE: return "ge";
    case Condition::LT: return "lt";
    case Condition::GT: return "gt";
    case Condition::LE: return "le";
    case Condition::AL: return "al";
    case Condition::NV: return "nv";
    default: return "none";
    }
}

std::string_view to_string(OperandSize size) noexcept {
    switch (size) {
    case OperandSize::Byte:  return "byte";
    case OperandSize::Half:  return "half";
    case OperandSize::Word:  return "word";
    case OperandSize::Xword: return "xword";
    case OperandSize::Quad:  return "quad";
    default: return "unknown";
    }
}

std::ostream& operator<<(std::ostream& os, GPR reg) {
    return os << to_string(reg);
}

std::ostream& operator<<(std::ostream& os, FPR reg) {
    return os << to_string(reg);
}

std::ostream& operator<<(std::ostream& os, Condition cond) {
    return os << to_string(cond);
}

std::ostream& operator<<(std::ostream& os, OperandSize size) {
    return os << to_string(size);
}

} // namespace brass::aarch64
