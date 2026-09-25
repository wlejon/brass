#pragma once

// The interpreters' checked memory access: loads and stores of MIR types
// through a base address and an offset, bounds-checked against the object
// when the base lies in the heap (a derived base is resolved to its object),
// with the heap's write barrier on every word-sized store.

#include <brass/gc/heap.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/types.hpp>

#include <cstdint>

namespace brass {

// The value of type `t` at base + offset. Throws std::runtime_error for a
// null base or an access outside the heap object `base` points into.
RuntimeValue read_memory(const gc::Heap& heap, uintptr_t base, int64_t offset, Type t);
// Stores `val` as type `t` at base + offset, with the same checks.
void write_memory(gc::Heap& heap, uintptr_t base, int64_t offset, Type t, RuntimeValue val);

} // namespace brass
