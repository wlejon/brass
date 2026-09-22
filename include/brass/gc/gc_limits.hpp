#pragma once

#include <cstddef>

namespace brass {

// Every GenerationalGC nursery is at least this large; the constructor rounds
// smaller requests up. The compiler relies on the floor: an object whose
// header plus payload fits in half of the smallest nursery is always
// allocated young, whatever nursery size the embedder picks at run time.
inline constexpr size_t kMinNurseryBytes = 32 * 1024;

// Header bytes plus alignment slack budgeted per allocation when reasoning
// about the nursery threshold at compile time.
inline constexpr size_t kGcAllocationOverheadBytes = 64;

// Largest `brass_gc_alloc` payload that is guaranteed to land in the nursery.
// Larger requests may go straight to tenured space, where stores into the
// fresh object need a card mark like any other old object.
inline constexpr size_t kMaxAlwaysYoungPayloadBytes =
    kMinNurseryBytes / 2 - kGcAllocationOverheadBytes;

} // namespace brass
