#pragma once

#include <brass/target/x64/x64_registers.hpp>
#include <cstdint>
#include <string>

namespace brass::x64 {

enum class Scale : uint8_t {
    One   = 1,
    Two   = 2,
    Four  = 4,
    Eight = 8,
};

constexpr uint8_t scale_value(Scale s) noexcept {
    return static_cast<uint8_t>(s);
}

constexpr uint8_t scale_bits(Scale s) noexcept {
    switch (s) {
    case Scale::One:   return 0;
    case Scale::Two:   return 1;
    case Scale::Four:  return 2;
    case Scale::Eight: return 3;
    default: return 0;
    }
}

constexpr Scale scale_from_int(int val) noexcept {
    switch (val) {
    case 2:  return Scale::Two;
    case 4:  return Scale::Four;
    case 8:  return Scale::Eight;
    default: return Scale::One;
    }
}

struct MemAddress {
    GPR base = GPR::None;
    GPR index = GPR::None;
    Scale scale = Scale::One;
    int32_t disp = 0;
    bool is_rip_rel = false;

    constexpr MemAddress() noexcept = default;

    static constexpr MemAddress base_only(GPR b) noexcept {
        MemAddress m;
        m.base = b;
        return m;
    }

    static constexpr MemAddress base_disp(GPR b, int32_t d) noexcept {
        MemAddress m;
        m.base = b;
        m.disp = d;
        return m;
    }

    static constexpr MemAddress base_index(GPR b, GPR idx, Scale sc = Scale::One, int32_t d = 0) noexcept {
        MemAddress m;
        m.base = b;
        m.index = idx;
        m.scale = sc;
        m.disp = d;
        return m;
    }

    static constexpr MemAddress index_disp(GPR idx, Scale sc, int32_t d = 0) noexcept {
        MemAddress m;
        m.index = idx;
        m.scale = sc;
        m.disp = d;
        return m;
    }

    static constexpr MemAddress rip_relative(int32_t d) noexcept {
        MemAddress m;
        m.is_rip_rel = true;
        m.disp = d;
        return m;
    }

    constexpr bool has_base() const noexcept { return base != GPR::None; }
    constexpr bool has_index() const noexcept { return index != GPR::None; }
    constexpr bool is_rip_relative() const noexcept { return is_rip_rel; }

    constexpr bool requires_sib() const noexcept {
        if (is_rip_rel) return false;
        if (has_base() && reg_code(base) == 4) return true; // RSP or R12
        if (has_index()) return true;
        if (!has_base()) return true; // disp only
        return false;
    }

    constexpr bool operator==(const MemAddress& other) const noexcept = default;
    constexpr bool operator!=(const MemAddress& other) const noexcept = default;
};

inline constexpr MemAddress ptr(GPR base) noexcept {
    return MemAddress::base_only(base);
}

inline constexpr MemAddress ptr(GPR base, int32_t disp) noexcept {
    return MemAddress::base_disp(base, disp);
}

inline constexpr MemAddress ptr(GPR base, GPR index, Scale scale = Scale::One, int32_t disp = 0) noexcept {
    return MemAddress::base_index(base, index, scale, disp);
}

inline constexpr MemAddress ptr(GPR base, GPR index, int scale_val, int32_t disp = 0) noexcept {
    return MemAddress::base_index(base, index, scale_from_int(scale_val), disp);
}

inline constexpr MemAddress rip_rel(int32_t disp = 0) noexcept {
    return MemAddress::rip_relative(disp);
}

std::string to_string(const MemAddress& mem);

} // namespace brass::x64
