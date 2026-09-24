#pragma once

#include <brass/gc/stack_map.hpp>
#include <cstdint>
#include <cstddef>
#include <functional>

namespace brass {

typedef void (*brass_root_visitor_fn)(void** root_slot, void* user_data);

struct StackFrameInfo {
    uintptr_t frame_rbp = 0;
    uintptr_t return_ip = 0;
    const FunctionStackMap* function_map = nullptr;
    const StackMapRecord* record = nullptr;
};

// Walks native stack frames starting from top_rbp and top_return_ip.
// For each frame recognized in the ModuleStackMap table, visits each mutable root pointer.
// Returns the number of frames visited.
size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
);

// As above, but stops before any frame at or above the stack address
// stop_at: a walk whose frames from there on another walk of the same
// collection covers (native_frames.cpp bounds each recorded run by the next
// outer one this way).
size_t brass_stack_walk_bounded(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at
);

// The number of compiled (C++) frames the walks on this thread have unwound
// through so far, a diagnostic for tests: a collection's count must not grow
// with the depth of the host's stack under a GeneratedCodeEntryScope.
size_t brass_stack_walk_unwind_steps() noexcept;

// C++ std::function overload for convenience
size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    const std::function<void(void**)>& visitor
);

} // namespace brass
