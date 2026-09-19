#pragma once

// Byte-level helpers shared by the three image writers (PE, ELF, Mach-O).
// Internal to src/target: each writer used to carry its own copy of these,
// and the import-table work made a fourth copy one too many.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace brass::target::image {

inline void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

inline void write_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

inline void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

inline void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

inline void write_i64(std::vector<uint8_t>& buf, int64_t v) {
    write_u64(buf, static_cast<uint64_t>(v));
}

inline void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t count) {
    const auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

inline void write_cstring(std::vector<uint8_t>& buf, std::string_view s) {
    buf.insert(buf.end(), s.begin(), s.end());
    buf.push_back(0);
}

// A fixed-width name field (Mach-O segment and section names): the text,
// then zeros to `fixed_len`; a longer text is cut.
inline void write_fixed_string(std::vector<uint8_t>& buf, std::string_view str, size_t fixed_len) {
    size_t copy_len = str.size() < fixed_len ? str.size() : fixed_len;
    buf.insert(buf.end(), str.begin(), str.begin() + static_cast<std::ptrdiff_t>(copy_len));
    if (copy_len < fixed_len) buf.resize(buf.size() + (fixed_len - copy_len), 0);
}

inline uint64_t align_up(uint64_t val, uint64_t align) {
    if (align <= 1) return val;
    uint64_t rem = val % align;
    return rem == 0 ? val : val + (align - rem);
}

inline void pad_to(std::vector<uint8_t>& buf, size_t align) {
    buf.resize(static_cast<size_t>(align_up(buf.size(), align)), 0);
}

inline void patch_u32(std::vector<uint8_t>& buf, size_t off, uint32_t v) {
    std::memcpy(buf.data() + off, &v, 4);
}

inline void patch_u64(std::vector<uint8_t>& buf, size_t off, uint64_t v) {
    std::memcpy(buf.data() + off, &v, 8);
}

inline uint32_t read_u32(const std::vector<uint8_t>& buf, size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, buf.data() + off, 4);
    return v;
}

inline void encode_uleb128(std::vector<uint8_t>& buf, uint64_t val) {
    do {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        if (val != 0) byte |= 0x80;
        buf.push_back(byte);
    } while (val != 0);
}

inline size_t uleb128_len(uint64_t val) {
    size_t len = 0;
    do {
        val >>= 7;
        ++len;
    } while (val != 0);
    return len;
}

inline void encode_sleb128(std::vector<uint8_t>& buf, int64_t val) {
    bool more = true;
    while (more) {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        const bool sign = (byte & 0x40) != 0;
        if ((val == 0 && !sign) || (val == -1 && sign)) {
            more = false;
        } else {
            byte |= 0x80;
        }
        buf.push_back(byte);
    }
}

// The AArch64 `adrp`/`ldr (unsigned offset, 64-bit)` pair every stub in every
// format is built from: load a pointer slot by page-relative address.
inline uint32_t aarch64_adrp(uint32_t reg, uint64_t pc, uint64_t target) {
    const int64_t page_diff = (static_cast<int64_t>(target) >> 12) - (static_cast<int64_t>(pc) >> 12);
    const uint32_t imm21 = static_cast<uint32_t>(page_diff) & 0x1FFFFFu;
    const uint32_t immlo = (imm21 & 3u) << 29;
    const uint32_t immhi = ((imm21 >> 2) & 0x7FFFFu) << 5;
    return 0x90000000u | immlo | immhi | (reg & 0x1Fu);
}

inline uint32_t aarch64_ldr_x_uoff(uint32_t dst, uint32_t base, uint64_t target) {
    const uint32_t imm12 = static_cast<uint32_t>((target & 0xFFFu) >> 3) & 0xFFFu;
    return 0xF9400000u | (imm12 << 10) | ((base & 0x1Fu) << 5) | (dst & 0x1Fu);
}

inline uint32_t aarch64_br(uint32_t reg) {
    return 0xD61F0000u | ((reg & 0x1Fu) << 5);
}

} // namespace brass::target::image
