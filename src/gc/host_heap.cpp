#include <brass/gc/host_heap.hpp>
#include <brass/runtime/coroutine.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace brass {

namespace {
std::atomic<HostHeap*> g_host_heap{nullptr};
thread_local int t_host_collection_depth = 0;
} // namespace

HostHeapCollectionScope::HostHeapCollectionScope() {
    runtime::lock_host_heap_coro_roots();
    ++t_host_collection_depth;
}

HostHeapCollectionScope::~HostHeapCollectionScope() {
    --t_host_collection_depth;
    runtime::unlock_host_heap_coro_roots();
}

bool in_host_heap_collection() noexcept {
    return t_host_collection_depth > 0;
}

void set_host_heap(HostHeap* heap) noexcept {
    g_host_heap.store(heap, std::memory_order_release);
}

HostHeap* host_heap() noexcept {
    return g_host_heap.load(std::memory_order_acquire);
}

uintptr_t host_heap_allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag,
                             uintptr_t caller_fp, uintptr_t caller_ip) {
    HostHeap* heap = host_heap();
    const uintptr_t payload = heap ? heap->allocate_at(size, pointer_mask, type_tag, caller_fp, caller_ip) : 0;
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
