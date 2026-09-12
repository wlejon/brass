#include <brass/gc/tlab.hpp>
#include <brass/embedding/host_gc.hpp>
#include <cstring>
#include <algorithm>

namespace brass {

namespace {

thread_local ThreadLocalAllocBuffer g_default_tlab;
thread_local ThreadLocalAllocBuffer* g_active_tlab = nullptr;
static uintptr_t s_fallback_zero = 0;

} // namespace

void ThreadLocalAllocBuffer::init(HostGC* gc, size_t default_sz) {
    owner_gc = gc;
    default_size = default_sz;
    top = 0;
    end = 0;
    total_allocated = 0;
    if (owner_gc) {
        owner_gc->register_tlab(this);
    }
}

void ThreadLocalAllocBuffer::refill(size_t min_bytes) {
    if (!owner_gc || owner_gc->stress_mode()) {
        reset();
        return;
    }
    if (top != 0 && end != 0) {
        owner_gc->retire_tlab(top, end);
    }
    top = 0;
    end = 0;

    uintptr_t new_top = 0;
    uintptr_t new_end = 0;
    if (owner_gc->allocate_tlab(min_bytes, default_size, new_top, new_end)) {
        top = new_top;
        end = new_end;
    }
}

void ThreadLocalAllocBuffer::reset() {
    if (owner_gc && top != 0 && end != 0) {
        owner_gc->retire_tlab(top, end);
    }
    top = 0;
    end = 0;
}

uintptr_t ThreadLocalAllocBuffer::allocate_fast(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (!owner_gc || owner_gc->stress_mode()) {
        return 0;
    }
    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(HostGcHeader) + aligned_size;

    if (top + total_size > end) {
        refill(total_size);
        if (top + total_size > end) {
            return 0;
        }
    }

    uintptr_t hdr_addr = top;
    uintptr_t payload_addr = hdr_addr + sizeof(HostGcHeader);
    auto* hdr = reinterpret_cast<HostGcHeader*>(hdr_addr);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;

    std::memset(reinterpret_cast<void*>(payload_addr), 0, aligned_size);

    top += total_size;
    total_allocated += aligned_size;
    return payload_addr;
}

ThreadLocalAllocBuffer* get_active_tlab() noexcept {
    HostGC* current_gc = get_active_host_gc();
    if (!g_active_tlab) {
        g_active_tlab = &g_default_tlab;
    }
    if (g_active_tlab->owner_gc != current_gc) {
        if (g_active_tlab->owner_gc) {
            g_active_tlab->reset();
        }
        g_active_tlab->owner_gc = current_gc;
        if (current_gc) {
            g_active_tlab->init(current_gc);
        }
    }
    return g_active_tlab;
}

void set_active_tlab(ThreadLocalAllocBuffer* tlab) noexcept {
    if (g_active_tlab && g_active_tlab != tlab && g_active_tlab->owner_gc) {
        g_active_tlab->reset();
    }
    g_active_tlab = tlab;
    if (g_active_tlab && g_active_tlab->owner_gc) {
        g_active_tlab->owner_gc->register_tlab(g_active_tlab);
    }
}

} // namespace brass

extern "C" {

uintptr_t brass_tlab_refill(size_t min_bytes) {
    auto* tlab = brass::get_active_tlab();
    if (!tlab) return 0;
    tlab->refill(min_bytes);
    return tlab->top;
}

uintptr_t* brass_tlab_top_ptr() noexcept {
    auto* tlab = brass::get_active_tlab();
    return tlab ? &tlab->top : &brass::s_fallback_zero;
}

uintptr_t* brass_tlab_end_ptr() noexcept {
    auto* tlab = brass::get_active_tlab();
    return tlab ? &tlab->end : &brass::s_fallback_zero;
}

}
