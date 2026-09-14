#include <brass/target/aarch64/aarch64_operands.hpp>
#include <ostream>
#include <sstream>

namespace brass::aarch64 {

std::string_view to_string(ShiftType st) noexcept {
    switch (st) {
    case ShiftType::LSL: return "lsl";
    case ShiftType::LSR: return "lsr";
    case ShiftType::ASR: return "asr";
    case ShiftType::ROR: return "ror";
    default: return "unknown";
    }
}

std::string_view to_string(ExtendType ext) noexcept {
    switch (ext) {
    case ExtendType::UXTB: return "uxtb";
    case ExtendType::UXTH: return "uxth";
    case ExtendType::UXTW: return "uxtw";
    case ExtendType::UXTX: return "uxtx";
    case ExtendType::SXTB: return "sxtb";
    case ExtendType::SXTH: return "sxth";
    case ExtendType::SXTW: return "sxtw";
    case ExtendType::SXTX: return "sxtx";
    default: return "unknown";
    }
}

std::string_view to_string(AddrMode mode) noexcept {
    switch (mode) {
    case AddrMode::Offset:    return "offset";
    case AddrMode::PreIndex:  return "pre_index";
    case AddrMode::PostIndex: return "post_index";
    case AddrMode::RegOffset: return "reg_offset";
    case AddrMode::Literal:   return "literal";
    default: return "unknown";
    }
}

std::string to_string(const MemAddress& mem) {
    std::ostringstream ss;
    if (mem.mode == AddrMode::Literal) {
        ss << "[pc, #" << mem.offset << "]";
        return ss.str();
    }

    ss << "[" << to_string(mem.base);
    if (mem.mode == AddrMode::PostIndex) {
        ss << "], #" << mem.offset;
        return ss.str();
    }

    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        ss << ", " << to_string(mem.index);
        if (mem.extend != ExtendType::UXTX || mem.shift > 0) {
            ss << ", " << to_string(mem.extend);
            if (mem.shift > 0) {
                ss << " #" << static_cast<int>(mem.shift);
            }
        }
    } else if (mem.offset != 0) {
        ss << ", #" << mem.offset;
    }
    ss << "]";
    if (mem.mode == AddrMode::PreIndex) {
        ss << "!";
    }
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, ShiftType st) {
    return os << to_string(st);
}

std::ostream& operator<<(std::ostream& os, ExtendType ext) {
    return os << to_string(ext);
}

std::ostream& operator<<(std::ostream& os, AddrMode mode) {
    return os << to_string(mode);
}

std::ostream& operator<<(std::ostream& os, const MemAddress& mem) {
    return os << to_string(mem);
}

} // namespace brass::aarch64
