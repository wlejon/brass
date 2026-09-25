#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/code_stack_maps.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/gc/stack_walker.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace brass {

namespace {

// The stack maps of the code running on THIS thread: what they describe is
// the frames of the thread that allocates. Parallel loop bodies, the one
// place brass runs generated code on a thread that did not install these,
// neither allocate nor call (loop_parallel_analysis.cpp).
thread_local const ModuleStackMap* g_active_stack_maps = nullptr;

// Native allocation entry points are called from generated frames, where a
// C++ exception cannot be relied on to unwind; a missing heap or missing
// stack maps is an embedding bug, so it stops the process with a message.
[[noreturn]] void gc_fatal_no_heap() {
    std::fprintf(stderr, "brass: fatal: brass_gc_alloc called on a thread with no current gc::Heap; "
                         "bind one (gc::HeapScope) before running code that allocates\n");
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] void gc_fatal_no_maps(const char* op) {
    std::fprintf(stderr, "brass: fatal: %s may collect but no stack maps are active and the calling "
                         "code has none registered, so live gcrefs in generated frames cannot be "
                         "found; call brass_set_active_stack_maps with the running code's maps\n", op);
    std::fflush(stderr);
    std::abort();
}

// The maps a collection requested from caller_ip walks this thread's
// generated frames with (brass_stack_maps_for_caller).
const ModuleStackMap* walk_maps(uintptr_t caller_ip) noexcept {
    if (g_active_stack_maps) return g_active_stack_maps;
    static const ModuleStackMap* const registry_only = [] {
        auto* m = new ModuleStackMap();
        m->set_indexed_by_code_registry(true);
        return m;
    }();
    if (caller_ip != 0 && code_stack_maps_cover(caller_ip)) return registry_only;
    return nullptr;
}

void push_slot(void** slot, void* user_data) {
    auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
    if (slot != nullptr && *slot != nullptr) vec->push_back(reinterpret_cast<uintptr_t*>(slot));
}

} // namespace

void brass_set_active_stack_maps(const ModuleStackMap* maps) noexcept {
    g_active_stack_maps = maps;
}

const ModuleStackMap* brass_get_active_stack_maps() noexcept {
    return g_active_stack_maps;
}

const ModuleStackMap* brass_stack_maps_for_caller(uintptr_t caller_ip) noexcept {
    return walk_maps(caller_ip);
}

void brass_append_generated_frame_roots(uintptr_t caller_fp, uintptr_t caller_ip,
                                        std::vector<uintptr_t*>& roots) {
    if (caller_fp == 0 || caller_ip == 0) return;
    const auto* maps = walk_maps(caller_ip);
    if (!maps) return;
    brass_stack_walk(caller_fp, caller_ip, *maps, &push_slot, &roots);
}

void brass_enumerate_thread_roots(uintptr_t caller_fp, uintptr_t caller_ip, std::vector<uintptr_t*>& roots) {
    brass_append_generated_frame_roots(caller_fp, caller_ip, roots);
    brass_append_native_frame_roots(roots);
    if (Interpreter* interp = Interpreter::active_on_thread()) interp->collect_all_roots(roots);
    if (FastInterpreter* fast = FastInterpreter::current()) fast->collect_all_roots(roots);
}

} // namespace brass

namespace {

using brass::gc::CollectionKind;
using brass::gc::Heap;

void runtime_safepoint(uintptr_t caller_fp, uintptr_t caller_ip) {
    Heap* heap = Heap::current();
    if (!heap || heap->in_collection()) return;
    // A safepoint is an opportunity, not a demand: without stack maps the
    // generated frames' roots are unknown, so collecting here would be unsafe.
    if (!brass::walk_maps(caller_ip)) return;
    heap->safepoint_at(caller_fp, caller_ip);
}

void runtime_collect(uintptr_t caller_fp, uintptr_t caller_ip) {
    Heap* heap = Heap::current();
    if (!heap || heap->in_collection()) return;
    if (!brass::walk_maps(caller_ip)) return;
    heap->collect_at(CollectionKind::Full, caller_fp, caller_ip);
}

uintptr_t runtime_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag,
                        uintptr_t caller_fp, uintptr_t caller_ip) {
    Heap* heap = Heap::current();
    if (!heap) brass::gc_fatal_no_heap();
    const brass::gc::LayoutId layout = brass::gc::mask_layout(pointer_mask, type_tag);
    // The inline fast path never collects; the slow path may, and then the
    // caller's frames must be describable.
    Heap::AllocationBuffer& buffer = *heap->allocation_buffer();
    const size_t total = brass::gc::payload_bytes_for(size) + brass::gc::kHeaderBytes;
    if (total > static_cast<size_t>(buffer.end - buffer.top) && caller_ip != 0 &&
        !brass::walk_maps(caller_ip)) {
        brass::gc_fatal_no_maps("brass_gc_alloc");
    }
    return heap->allocate_at(size, layout, 0, 0, caller_fp, caller_ip);
}

} // namespace

extern "C" {

#if defined(_MSC_VER)

// The assembly stubs (gc_msvc_*.asm) pass the generated caller's frame.
void brass_runtime_gc_safepoint_bridge(uintptr_t caller_fp, uintptr_t caller_ip) {
    runtime_safepoint(caller_fp, caller_ip);
}

void brass_runtime_gc_collect_bridge(uintptr_t caller_fp, uintptr_t caller_ip) {
    runtime_collect(caller_fp, caller_ip);
}

uintptr_t brass_runtime_gc_alloc_bridge(size_t size, uint64_t pointer_mask, uint32_t type_tag,
                                        uintptr_t caller_fp, uintptr_t caller_ip) {
    return runtime_alloc(size, pointer_mask, type_tag, caller_fp, caller_ip);
}

#else

#define BRASS_CALLER_FRAME(fp, ip)                                                        \
    void* frame_ = __builtin_frame_address(0);                                            \
    const uintptr_t fp = frame_ ? *reinterpret_cast<uintptr_t*>(frame_) : 0;              \
    const uintptr_t ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0))

__attribute__((noinline)) void brass_gc_safepoint() {
    BRASS_CALLER_FRAME(fp, ip);
    runtime_safepoint(fp, ip);
}

__attribute__((noinline)) uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    BRASS_CALLER_FRAME(fp, ip);
    return runtime_alloc(size, pointer_mask, type_tag, fp, ip);
}

__attribute__((noinline)) void brass_gc_collect() {
    BRASS_CALLER_FRAME(fp, ip);
    runtime_collect(fp, ip);
}

#undef BRASS_CALLER_FRAME

#endif

void brass_default_gc_write_barrier(uintptr_t obj, uintptr_t val) {
    // Compiled code may pass a derived pointer as `obj`.
    if (Heap* heap = Heap::current()) heap->write_barrier_interior(obj, val);
}

uint8_t* brass_gc_card_table_base() {
    Heap* heap = Heap::current();
    return heap ? heap->card_table_base() : nullptr;
}

uintptr_t brass_gc_heap_base() {
    Heap* heap = Heap::current();
    return heap ? heap->old_base() : 0;
}

}
