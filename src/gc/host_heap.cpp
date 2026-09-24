#include <brass/gc/host_heap.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace brass {

namespace {
std::atomic<HostHeap*> g_host_heap{nullptr};
} // namespace

void set_host_heap(HostHeap* heap) noexcept {
    g_host_heap.store(heap, std::memory_order_release);
}

HostHeap* host_heap() noexcept {
    return g_host_heap.load(std::memory_order_acquire);
}

uintptr_t host_heap_allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    HostHeap* heap = host_heap();
    const uintptr_t payload = heap ? heap->allocate(size, pointer_mask, type_tag) : 0;
    if (payload == 0) {
        std::fprintf(stderr,
                     "brass: fatal: the host heap could not allocate %zu bytes (type tag %u)\n",
                     size, static_cast<unsigned>(type_tag));
        std::fflush(stderr);
        std::abort();
    }
    return payload;
}

} // namespace brass
