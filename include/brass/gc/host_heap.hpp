#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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
// been a safepoint — reach safepoint_at() (by default safepoint()); an
// interpreter's explicit collection reaches collect(). A collecting host
// finds brass's roots on the thread with brass_enumerate_thread_roots. brass's own heaps (MiniCheneyGC, GenerationalGC,
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

    // A safepoint (or brass_gc_collect) reached from generated code, with
    // the generated frame that reached it: its frame pointer and the return
    // address into it (either 0 if unknown). brass calls this, never
    // safepoint() directly; the default forwards to safepoint(). A host that
    // collects here passes both to brass_enumerate_thread_roots to find the
    // gcrefs of the generated frames on the stack.
    virtual void safepoint_at(uintptr_t caller_fp, uintptr_t caller_ip) {
        (void)caller_fp;
        (void)caller_ip;
        safepoint();
    }
};

// Every gcref slot brass knows on the calling thread, appended to `roots`,
// for a host heap's collection (from allocate(), collect() or
// safepoint_at()). A slot may be reported more than once; each holds a
// gcref (or a tagged value whose low 48 bits are one) the host must keep
// alive and, if it moves the object, update in place. Reports:
//   - the generated (JIT, baseline) frames from (caller_fp, caller_ip)
//     upward, through the code stack maps: pass what safepoint_at received.
//     With either 0 (a collection from allocate() or collect()) none are
//     walked from here;
//   - every run of native frames below re-entered Tier-0 code
//     (NativeFramesScope), and every ThreadRootsScope, which includes the
//     frames of the fresh Tier-0 interpreters a deoptimization or a
//     native-to-Tier-0 call starts;
//   - suspended coroutine frames;
//   - the frames of the innermost running Interpreter and FastInterpreter
//     on the thread. An outer interpreter hidden under an inner one of the
//     same kind is not reached: a host that runs nested interpreters also
//     reports their collect_all_roots() itself.
// Roots the host registered on its own heap are its own business.
void brass_enumerate_thread_roots(uintptr_t caller_fp, uintptr_t caller_ip, std::vector<uintptr_t*>& roots);

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
