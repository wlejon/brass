#pragma once

#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <cstdint>
#include <cstddef>

namespace brass {

void brass_set_active_gc(MiniCheneyGC* gc) noexcept;
MiniCheneyGC* brass_get_active_gc() noexcept;

void brass_set_active_generational_gc(GenerationalGC* gc) noexcept;
GenerationalGC* brass_get_active_generational_gc() noexcept;

void brass_set_active_stack_maps(const ModuleStackMap* maps) noexcept;
const ModuleStackMap* brass_get_active_stack_maps() noexcept;

// The maps a collection requested by the code returning to `caller_ip`
// walks this thread's generated frames with: the thread's active maps, else
// the code registry when it covers `caller_ip`. Null when nothing describes
// the caller's frame (it is not generated code brass knows).
const ModuleStackMap* brass_stack_maps_for_caller(uintptr_t caller_ip) noexcept;

// Explicit safepoint trigger with passed RBP / return IP (or automatic detection if 0)
void brass_runtime_gc_safepoint(
    MiniCheneyGC* gc,
    const ModuleStackMap& stack_maps,
    uintptr_t rbp = 0,
    uintptr_t return_ip = 0
);

void brass_runtime_gc_safepoint(
    GenerationalGC* gc,
    const ModuleStackMap& stack_maps,
    uintptr_t rbp = 0,
    uintptr_t return_ip = 0
);

// Explicit allocation with moving Cheney collection and stack walking if needed
uintptr_t brass_runtime_gc_alloc(
    MiniCheneyGC* gc,
    const ModuleStackMap& stack_maps,
    size_t size,
    uint64_t pointer_mask = 0,
    uint32_t type_tag = 0,
    uintptr_t rbp = 0,
    uintptr_t return_ip = 0
);

uintptr_t brass_runtime_gc_alloc(
    GenerationalGC* gc,
    const ModuleStackMap& stack_maps,
    size_t size,
    uint64_t pointer_mask = 0,
    uint32_t type_tag = 0,
    uintptr_t rbp = 0,
    uintptr_t return_ip = 0
);

} // namespace brass

// Extern C runtime bridge symbols callable from JIT code
extern "C" {

void brass_gc_safepoint();
uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag);
void brass_gc_collect();
// brass's own generational-GC barrier. JIT symbol tables register it under
// the name compiled code calls, "brass_gc_write_barrier", as a default a host
// runtime overrides with its own barrier. It is deliberately NOT defined under
// that name: a host runtime may define `brass_gc_write_barrier` itself,
// and two strong definitions collide when both are statically linked.
void brass_default_gc_write_barrier(uintptr_t obj, uintptr_t val);
uint8_t* brass_gc_card_table_base();
uintptr_t brass_gc_heap_base();

}
