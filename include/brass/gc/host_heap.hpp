#pragma once

#include <cstddef>
#include <cstdint>

namespace brass {

// The embedder's heap, as brass's own runtime sees it.
//
// brass owns GC *mechanism* — stack maps, the code registry, the stack
// walker, root enumeration of JIT, baseline and interpreter frames, the
// card-table and write-barrier contract. It does not own the heap an
// embedder's objects live in. When a host installs a HostHeap, every
// allocation brass itself makes on behalf of running code goes to it:
// `brass_gc_alloc` from generated code (every tier), the interpreters'
// allocations, and coroutine frames. Safepoints in generated code — and
// `brass_gc_collect` from generated code, which on every target has always
// been a safepoint — reach safepoint(); an interpreter's explicit
// collection reaches collect(). brass's own heaps (MiniCheneyGC, GenerationalGC,
// HostGC) are then never consulted; they remain the default for an
// embedder that brings no heap.
//
// The write barrier is not part of this interface: compiled code calls the
// symbol "brass_gc_write_barrier" through the engines' symbol tables, and a
// host registers its own barrier there (HostSymbolProvider). brass's
// default, brass_default_gc_write_barrier, is registered under that name
// only until a host overrides it, so a process holds one definition.
class HostHeap {
public:
    virtual ~HostHeap() = default;

    // `size` payload bytes. `pointer_mask` and `type_tag` describe the
    // payload's 8-byte fields exactly as for brass's heaps (bit i set: field
    // i holds a gcref; bit 63 set: every field from 63 on does). The host
    // must trace those fields and keep the object alive while any root or
    // traced field names it. Returning 0 means the host is out of memory;
    // brass then stops the process, since the callers are JIT frames that
    // cannot unwind a C++ exception.
    virtual uintptr_t allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) = 0;

    // A collection an interpreter asked for explicitly (its brass_gc_collect).
    virtual void collect() {}

    // A safepoint: an opportunity to collect, never a demand.
    virtual void safepoint() {}
};

// Process-wide, not owned; the heap must outlive all code running while it
// is installed, and installing nullptr removes it. Like HostSymbolProvider,
// one per process: a host whose heap is per thread dispatches on the
// calling thread inside allocate().
void set_host_heap(HostHeap* heap) noexcept;
HostHeap* host_heap() noexcept;

// host_heap()->allocate(), aborting with a message on a 0 answer. Only
// meaningful with a heap installed.
uintptr_t host_heap_allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag);

} // namespace brass
