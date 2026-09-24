#pragma once

#include <brass/gc/stack_map.hpp>
#include <cstdint>
#include <memory>

namespace brass {

// Process-wide index from a code address to the stack map of the function
// whose code is there. Every piece of code brass loads (a JitExecutionEngine
// module, a baseline function) registers its maps for as long as the code
// exists, so a stack walk finds the maps of each frame actually on the stack,
// whichever program, tier or thread produced it, and whatever maps the thread
// has active. Readers take an immutable snapshot and never lock.
//
// Registering keeps a copy of the maps. The returned token unregisters them
// when the last reference to it goes; its owner must drop it before the code
// is freed. (The copy itself is freed later, when the index is compacted.) Functions without an address or without records are not indexed.
// Registering code that overlaps code already registered is a fatal error:
// one of the two registrations describes code that no longer exists.
[[nodiscard]] std::shared_ptr<const void> register_code_stack_maps(const ModuleStackMap& maps);
[[nodiscard]] std::shared_ptr<const void> register_code_stack_map(const FunctionStackMap& map);

class CodeStackMapSnapshot;

// The current snapshot; lookups through it see the registrations published
// before it was taken. Hold it for the duration of one stack walk.
std::shared_ptr<const CodeStackMapSnapshot> code_stack_map_snapshot() noexcept;

// The registered function map whose code contains ip, or null.
const FunctionStackMap* find_code_stack_map(const CodeStackMapSnapshot* snapshot, uintptr_t ip) noexcept;

// Whether ip lies in registered code.
bool code_stack_maps_cover(uintptr_t ip) noexcept;

// The number of sorted segments the current snapshot searches (a lookup is a
// binary search in each). Registering and unregistering are amortized
// O(log n); this stays O(log n) in the number of registered functions.
size_t code_stack_map_segment_count() noexcept;

} // namespace brass
