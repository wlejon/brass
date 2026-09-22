#pragma once

#include <cstdint>

namespace brass::aarch64 {

/// Encodes a 32-bit or 64-bit value into ARMv8 bitmask immediate fields:
/// N (1-bit), immr (6-bit), and imms (6-bit).
/// Returns true if val is a valid ARMv8 logical immediate bitmask, false otherwise.
bool encode_logical_immediate(uint64_t val, bool is_64bit, uint32_t& n, uint32_t& immr, uint32_t& imms) noexcept;

} // namespace brass::aarch64
