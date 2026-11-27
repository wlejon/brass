#pragma once

#include <cstdint>
#include <cstring>

namespace brass {

// IEEE-754 binary16 and bfloat16 conversion helpers shared by MIR passes,
// interpreter, fast VM, baseline JIT, and CPU/GPU lowering.

inline float bf16_bits_to_f32(uint16_t bits) noexcept {
    const uint32_t w = static_cast<uint32_t>(bits) << 16;
    float out = 0.0f;
    std::memcpy(&out, &w, sizeof(out));
    return out;
}

inline uint16_t f32_to_bf16_bits(float val) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &val, sizeof(u));
    // Propagate NaN payload as quiet NaN (matches PTX cvt.rn.bf16.f32 / SPIR-V / MSL).
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    return static_cast<uint16_t>((u + 0x7FFFu + lsb) >> 16);
}

inline float f16_bits_to_f32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out_bits = 0;

    if (exp == 0) {
        if (frac == 0) {
            out_bits = sign;
        } else {
            int shift = 0;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                ++shift;
            }
            frac &= 0x03FFu;
            const uint32_t new_exp = static_cast<uint32_t>(127 - 15 + 1 - shift);
            out_bits = sign | (new_exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        // Inf or NaN (promote signaling NaN to quiet NaN to match x86/ARM hardware fcvt).
        const uint32_t mant = (frac == 0) ? 0u : ((frac << 13) | 0x00400000u);
        out_bits = sign | 0x7F800000u | mant;
    } else {
        out_bits = sign | ((exp + (127u - 15u)) << 23) | (frac << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

inline uint16_t f32_to_f16_bits(float val) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &val, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t abs_u = u & 0x7FFFFFFFu;

    if (abs_u > 0x7F800000u) {
        // Quiet NaN.
        return static_cast<uint16_t>(sign | 0x7E00u | ((u >> 13) & 0x01FFu));
    }
    if (abs_u == 0x7F800000u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (abs_u >= 0x47800000u) {
        // Overflow to Inf (or clamp threshold >= 65536).
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (abs_u < 0x38800000u) {
        // Subnormal or underflow to signed zero.
        if (abs_u < 0x33000000u) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t exp = (abs_u >> 23);
        const uint32_t mant = (abs_u & 0x007FFFFFu) | 0x00800000u;
        const uint32_t shift = 113u - exp + 13u; // shift right to 10-bit subnormal mantissa
        // Shift <= 24 when abs_u >= 0x33000000 (exp >= 102 -> shift <= 24).
        const uint32_t half_bit = 1u << (shift - 1u);
        const uint32_t lsb_bit = (mant >> shift) & 1u;
        const uint32_t rounded = (mant + (half_bit - 1u) + lsb_bit) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }

    // Normal binary16 range.
    const uint32_t norm = abs_u - 0x38000000u;
    const uint32_t lsb = (norm >> 13) & 1u;
    const uint32_t rounded = norm + 0x0FFFu + lsb;
    // Check if RNE rounded up into infinity (exp == 31).
    if ((rounded >> 13) >= 0x7C00u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (rounded >> 13));
}

extern "C" {
float brass_f16_to_f32(uint32_t bits) noexcept;
uint32_t brass_f32_to_f16(float val) noexcept;
float brass_bf16_to_f32(uint32_t bits) noexcept;
uint32_t brass_f32_to_bf16(float val) noexcept;
}

} // namespace brass
