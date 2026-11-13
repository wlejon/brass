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

// Walks this thread's stack from the caller of this function: the compiled
// (C++) frames up to the first generated frame are passed over by their
// unwind information (Win64 unwind data, or DWARF CFI: native_unwind.hpp;
// on Windows ARM64 by their frame pointers, which that code must then keep),
// and every generated frame from there on is visited as brass_stack_walk
// visits it, up to `stop_at` as brass_stack_walk_bounded stops.
size_t brass_stack_walk_from_here(
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at = UINTPTR_MAX
);

// The number of compiled (C++) frames the walks on this thread have unwound
// through so far, a diagnostic for tests: a collection's count must not grow
// with the depth of the host's stack under a GeneratedCodeEntryScope.
size_t brass_stack_walk_unwind_steps() noexcept;

class GeneratedCodeEntryScope;

// A stack the thread is not running on: one a fiber switched away from
// (fiber.hpp), described as of the switch. It cannot change until it is
// switched back to, so a collection walks it from there like the running one.
struct SwitchedStack {
    // Where the stack's code switched away. On Windows x64 the register
    // context captured there (a CONTEXT, which the unwind data needs);
    // elsewhere the frame: the pc, the stack pointer at it and the frame
    // pointer register's value.
    const void* context = nullptr;
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    uintptr_t fp = 0;
    // The stack's extent, where it is known (Windows), and its own chain of
    // entry scopes, which the thread's does not hold while it is switched out.
    uintptr_t low = 0;
    uintptr_t high = UINTPTR_MAX;
    GeneratedCodeEntryScope* entries = nullptr;
};

// Visits the roots of every generated frame on `stack` from its switch point
// to its base, as brass_stack_walk_from_here does for the running stack.
size_t brass_stack_walk_switched(
    const SwitchedStack& stack,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
);

// brass_stack_walk_bounded for a run of frames recorded on `stack`.
size_t brass_stack_walk_bounded_on(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at,
    const SwitchedStack& stack
);

// While one lives, every walk on this thread also reports each generated
// frame whose roots it visits: the frame pointer, the frame's map, and the
// record at its return address (its current safepoint). Called before the
// frame's roots are visited. The collector's stack scan learns this way
// which frames the root walks of a collection describe exactly (heap_pin.cpp).
// Scopes nest; the innermost observes.
using brass_frame_observer_fn = void (*)(uintptr_t fp, const FunctionStackMap& map,
                                         const StackMapRecord& record, void* ctx);
class StackFrameObserverScope {
public:
    StackFrameObserverScope(brass_frame_observer_fn fn, void* ctx) noexcept;
    ~StackFrameObserverScope();
    StackFrameObserverScope(const StackFrameObserverScope&) = delete;
    StackFrameObserverScope& operator=(const StackFrameObserverScope&) = delete;

private:
    brass_frame_observer_fn fn_;
    void* ctx_;
    StackFrameObserverScope* prev_;
    friend struct StackFrameObserverAccess;
};

// C++ std::function overload for convenience
size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    const std::function<void(void**)>& visitor
);

} // namespace brass
