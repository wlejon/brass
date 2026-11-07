#pragma once

// Bit scans over 64-bit words, for the heap's bitmaps (heap_internal.hpp).

#include <bit>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace brass::gc::detail {

// Index of the lowest set bit of a nonzero word.
inline unsigned ctz64(uint64_t word) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward64(&index, word);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctzll(word));
#endif
}

inline unsigned popcount64(uint64_t word) noexcept { return static_cast<unsigned>(std::popcount(word)); }

// Index of the highest set bit of a nonzero word.
inline unsigned clz_index64(uint64_t word) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanReverse64(&index, word);
    return static_cast<unsigned>(index);
#else
    return 63u - static_cast<unsigned>(__builtin_clzll(word));
#endif
}

} // namespace brass::gc::detail
