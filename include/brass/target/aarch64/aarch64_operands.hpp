#pragma once

#include <brass/target/aarch64/aarch64_registers.hpp>
#include <cstdint>
#include <string>
#include <iosfwd>

namespace brass::aarch64 {

enum class AddrMode : uint8_t {
    Offset,     // [Xn, #offset]
    PreIndex,   // [Xn, #offset]!
    PostIndex,  // [Xn], #offset
    RegOffset,  // [Xn, Xm{, extend {#amount}}]
    Literal     // [PC, #offset]
};

enum class ShiftType : uint8_t {
    LSL = 0,
    LSR = 1,
    ASR = 2,
    ROR = 3
};

enum class ExtendType : uint8_t {
    UXTB = 0,
    UXTH = 1,
    UXTW = 2,
    UXTX = 3, // LSL
    SXTB = 4,
    SXTH = 5,
    SXTW = 6,
    SXTX = 7
};

struct MemAddress {
    GPR base = GPR::None;
    GPR index = GPR::None;
    int64_t offset = 0;
    AddrMode mode = AddrMode::Offset;
    ExtendType extend = ExtendType::UXTX;
    uint8_t shift = 0; // 0 or scale factor

    constexpr MemAddress() noexcept = default;

    static constexpr MemAddress base_only(GPR b) noexcept {
        MemAddress m;
        m.base = b;
        m.mode = AddrMode::Offset;
        return m;
    }

    static constexpr MemAddress base_disp(GPR b, int64_t d) noexcept {
        MemAddress m;
        m.base = b;
        m.offset = d;
        m.mode = AddrMode::Offset;
        return m;
    }

    static constexpr MemAddress pre_indexed(GPR b, int64_t d) noexcept {
        MemAddress m;
        m.base = b;
        m.offset = d;
        m.mode = AddrMode::PreIndex;
        return m;
    }

    static constexpr MemAddress post_indexed(GPR b, int64_t d) noexcept {
        MemAddress m;
        m.base = b;
        m.offset = d;
        m.mode = AddrMode::PostIndex;
        return m;
    }

    static constexpr MemAddress base_index(GPR b, GPR idx, ExtendType ext = ExtendType::UXTX, uint8_t shift_amt = 0) noexcept {
        MemAddress m;
        m.base = b;
        m.index = idx;
        m.mode = AddrMode::RegOffset;
        m.extend = ext;
        m.shift = shift_amt;
        return m;
    }

    static constexpr MemAddress literal(int64_t d) noexcept {
        MemAddress m;
        m.offset = d;
        m.mode = AddrMode::Literal;
        return m;
    }

    constexpr bool has_base() const noexcept { return base != GPR::None; }
    constexpr bool has_index() const noexcept { return index != GPR::None; }
    constexpr bool is_pre_indexed() const noexcept { return mode == AddrMode::PreIndex; }
    constexpr bool is_post_indexed() const noexcept { return mode == AddrMode::PostIndex; }
    constexpr bool is_literal() const noexcept { return mode == AddrMode::Literal; }

    constexpr bool operator==(const MemAddress& other) const noexcept = default;
    constexpr bool operator!=(const MemAddress& other) const noexcept = default;
};

// Shorthand helper functions
inline constexpr MemAddress ptr(GPR base) noexcept {
    return MemAddress::base_only(base);
}

inline constexpr MemAddress ptr(GPR base, int64_t disp) noexcept {
    return MemAddress::base_disp(base, disp);
}

inline constexpr MemAddress pre_idx(GPR base, int64_t disp) noexcept {
    return MemAddress::pre_indexed(base, disp);
}

inline constexpr MemAddress post_idx(GPR base, int64_t disp) noexcept {
    return MemAddress::post_indexed(base, disp);
}

std::string_view to_string(ShiftType st) noexcept;
std::string_view to_string(ExtendType ext) noexcept;
std::string_view to_string(AddrMode mode) noexcept;
std::string to_string(const MemAddress& mem);

std::ostream& operator<<(std::ostream& os, ShiftType st);
std::ostream& operator<<(std::ostream& os, ExtendType ext);
std::ostream& operator<<(std::ostream& os, AddrMode mode);
std::ostream& operator<<(std::ostream& os, const MemAddress& mem);

} // namespace brass::aarch64
