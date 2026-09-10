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

// Explicit safepoint trigger with passed RBP / return IP (or automatic detection if 0)
void brass_runtime_gc_safepoint(
    MiniCheneyGC* gc,
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

} // namespace brass

// Extern C runtime bridge symbols callable from JIT code
extern "C" {

void brass_gc_safepoint();
uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag);
void brass_gc_collect();
void brass_gc_write_barrier(uintptr_t obj, uintptr_t val);
uint8_t* brass_gc_card_table_base();
uintptr_t brass_gc_heap_base();

}
