#pragma once

#include <cstdint>
#include <cstddef>

namespace brass {

class HostGC;

struct ThreadLocalAllocBuffer {
    uintptr_t top = 0;               // Bump pointer (current allocation offset)
    uintptr_t end = 0;               // Limit pointer (end of current buffer chunk)
    HostGC* owner_gc = nullptr;      // Associated moving collector
    size_t total_allocated = 0;      // Cumulative bytes allocated through this TLAB
    size_t default_size = 64 * 1024; // Default chunk refill size (64 KB)

    void init(HostGC* gc, size_t default_sz = 64 * 1024);
    void refill(size_t min_bytes);
    void reset();
    uintptr_t allocate_fast(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);

    [[nodiscard]] bool can_allocate_fast(size_t size) const noexcept {
        size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
        return (top + 24 /* sizeof(HostGcHeader) */ + aligned_size) <= end;
    }

    [[nodiscard]] size_t remaining_bytes() const noexcept {
        return (end > top) ? (end - top) : 0;
    }
};

// Thread-local / global active TLAB accessors
ThreadLocalAllocBuffer* get_active_tlab() noexcept;
void set_active_tlab(ThreadLocalAllocBuffer* tlab) noexcept;

} // namespace brass

// Exported C runtime bridge symbols callable from JIT code
extern "C" {

uintptr_t brass_tlab_refill(size_t min_bytes);
uintptr_t* brass_tlab_top_ptr() noexcept;
uintptr_t* brass_tlab_end_ptr() noexcept;

}
