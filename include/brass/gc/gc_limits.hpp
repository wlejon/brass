#pragma once

#include <cstddef>

namespace brass {

// Every gc::Heap's eden is at least this large, and every heap allocates a
// payload of up to kMinLargeObjectBytes young (never straight into the
// large-object space), whatever HeapConfig says. The compiler relies on the
// floor: an object whose header plus payload fits under it is always
// allocated young by brass_gc_alloc, whatever heap the embedder configures.
inline constexpr size_t kMinEdenBytes = 64 * 1024;
inline constexpr size_t kMinLargeObjectBytes = 16 * 1024;

// Header bytes plus alignment slack budgeted per allocation when reasoning
// about the young-allocation threshold at compile time.
inline constexpr size_t kGcAllocationOverheadBytes = 64;

// Largest `brass_gc_alloc` payload that is guaranteed to be allocated young.
// Larger requests may go straight to the old generation, where stores into
// the fresh object need a card mark like any other old object.
inline constexpr size_t kMaxAlwaysYoungPayloadBytes = kMinLargeObjectBytes - kGcAllocationOverheadBytes;

} // namespace brass
