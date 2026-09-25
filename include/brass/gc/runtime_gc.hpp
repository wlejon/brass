#pragma once

// The runtime entry points generated code calls into the collector with, and
// the root enumeration they share. Every entry point acts on the calling
// thread's current heap (gc::Heap::current()); see docs/gc_contract.md.

#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace brass {

// The stack maps of the code running on this thread. They add to the code
// registry (code_stack_maps.hpp), through which a stack walk finds the maps
// of all code brass loaded.
void brass_set_active_stack_maps(const ModuleStackMap* maps) noexcept;
const ModuleStackMap* brass_get_active_stack_maps() noexcept;

// The maps a collection requested by the code returning to `caller_ip`
// walks this thread's generated frames with: the thread's active maps, else
// the code registry when it covers `caller_ip`. Null when nothing describes
// the caller's frame (it is not generated code brass knows).
const ModuleStackMap* brass_stack_maps_for_caller(uintptr_t caller_ip) noexcept;

// The gcref slots of the generated frames from (caller_fp, caller_ip)
// upward, through the code stack maps; nothing when either is 0 or no maps
// describe the caller.
void brass_append_generated_frame_roots(uintptr_t caller_fp, uintptr_t caller_ip,
                                        std::vector<uintptr_t*>& roots);

// Every root slot brass holds on this thread for a collector that is not
// gc::Heap (a host runtime with its own heap): the generated frames from
// (caller_fp, caller_ip) upward, the native frames registered under re-entered
// interpreter code (native_frames.hpp), and the frames of the interpreters
// running on this thread. gc::Heap gathers the same set itself.
void brass_enumerate_thread_roots(uintptr_t caller_fp, uintptr_t caller_ip,
                                  std::vector<uintptr_t*>& roots);

} // namespace brass

// The C symbols generated code calls. brass_gc_alloc, brass_gc_safepoint and
// brass_gc_collect capture their caller's frame (an assembly stub on MSVC) so
// a collection walks the generated frames.
extern "C" {

// A collection opportunity: collects when one was requested or in stress mode.
void brass_gc_safepoint();
// A zeroed object with `pointer_mask`'s words traced as references (bit 63:
// every word from 63 on) and `type_tag`, from the thread's heap. Fatal when
// the thread has no heap.
uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag);
// A full collection of the thread's heap; nothing when it has none.
void brass_gc_collect();
// gc::Heap's write barrier on the thread's heap. JIT symbol tables register
// it under the name compiled code calls, "brass_gc_write_barrier", as a
// default a host runtime overrides with its own barrier. It is deliberately
// not defined under that name: a host runtime may define
// `brass_gc_write_barrier` itself, and two strong definitions collide when
// both are statically linked.
void brass_default_gc_write_barrier(uintptr_t obj, uintptr_t val);
// The thread heap's card table and old-generation base (card of old address
// a: base[(a - heap_base) >> 9]); null / 0 without a heap.
uint8_t* brass_gc_card_table_base();
uintptr_t brass_gc_heap_base();

}
