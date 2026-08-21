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

// C++ std::function overload for convenience
size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    const std::function<void(void**)>& visitor
);

} // namespace brass
