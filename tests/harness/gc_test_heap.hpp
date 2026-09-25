#pragma once

// A gc::Heap for tests: small young generation (the eden floor), so code
// that allocates a few thousand objects collects many times, bound as the
// thread's current heap for the object's lifetime so generated code's
// brass_gc_alloc / brass_gc_safepoint act on it.

#include <brass/gc/gc_limits.hpp>
#include <brass/gc/heap.hpp>

namespace brass::test {

inline gc::HeapConfig small_heap_config(uint8_t tenure_age = 2) {
    gc::HeapConfig config;
    config.eden_bytes = kMinEdenBytes;
    config.survivor_bytes = 32 * 1024;
    config.tenure_age = tenure_age;
    return config;
}

struct BoundHeap {
    gc::Heap heap;
    gc::HeapScope scope;

    explicit BoundHeap(const gc::HeapConfig& config = small_heap_config()) : heap(config), scope(heap) {}
    BoundHeap(const BoundHeap&) = delete;
    BoundHeap& operator=(const BoundHeap&) = delete;

    gc::Heap* operator->() noexcept { return &heap; }
    uint64_t minor_collections() const noexcept { return heap.stats().minor_collections; }
    uint64_t full_collections() const noexcept { return heap.stats().full_collections; }
};

} // namespace brass::test
