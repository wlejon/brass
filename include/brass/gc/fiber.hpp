#pragma once
// Stackful coroutines: machine stacks a thread switches between.
//
// brass's own coroutines (runtime/coroutine.hpp) are stackless: a lowered
// body keeps its state in a heap frame and suspends only in that body. A
// front end whose coroutines suspend from any depth of calls (Lua's
// coroutine.yield, called from a function called from the coroutine's body,
// through the runtime's C++ in between) runs each one on a stack of its own
// and switches stacks. A Fiber is that stack, and the switch is brass's
// because brass owns what a stack holds for it:
//
//  - the gcrefs of the generated frames on it. A collection visits the
//    running stack from where it is, and every other stack of the thread
//    (each fiber that is suspended, and each stack waiting for the fiber it
//    resumed to come back) from the point it switched away, to its base;
//  - the chains of NativeFramesScope / ThreadRootsScope /
//    GeneratedCodeEntryScope its frames made (native_frames.hpp), which are
//    LIFO on one stack and would not be if a thread kept one chain across
//    stacks. Each stack keeps its own, and a switch exchanges them;
//  - exceptions. A brass or C++ exception unwinds within the fiber it was
//    thrown on; the entry function is the fiber's outermost frame, and an
//    exception that would leave it ends the process, so a host catches what
//    its fibers throw in the entry.
//
// A fiber belongs to the thread that first resumes it and is resumed and
// suspended only there. Switching costs a register save and restore, no
// walk: the position a collection needs is what the switch leaves behind.
// Windows uses the system's fibers (the thread is converted the first time
// it resumes one); x86-64 and AArch64 elsewhere switch with a few
// instructions of their own.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace brass::gc {

class Fiber {
public:
    using Entry = void (*)(void* arg);

    // A fiber that runs entry(arg) the first time it is resumed, on a stack
    // of `stack_bytes` (rounded up to whole pages) with a guard page below
    // it. Null if the stack cannot be made.
    static std::unique_ptr<Fiber> create(Entry entry, void* arg, size_t stack_bytes = 256 * 1024);

    // Frees the stack. A fiber is destroyed when it is not running and not
    // waiting on a fiber it resumed; a suspended fiber's frames are dropped
    // without being unwound (no destructor on them runs), so a host that
    // needs them unwound resumes the fiber with a request to unwind and lets
    // it finish first.
    ~Fiber();
    Fiber(const Fiber&) = delete;
    Fiber& operator=(const Fiber&) = delete;

    // Runs this fiber on the calling thread until it suspends or its entry
    // returns. It must be neither finished nor already running (nor waiting
    // on a fiber it resumed).
    void resume();

    // Called on a fiber: switches back to the stack that resumed it, and
    // returns when the fiber is resumed again.
    static void suspend();

    // The fiber the calling thread is running on, or null on the thread's
    // own stack.
    static Fiber* current() noexcept;

    bool started() const noexcept;
    bool finished() const noexcept;

    // The part of one of the thread's stacks that holds frames, for a host
    // that also scans its stacks conservatively (its C++ frames have no stack
    // maps). [low, high) is the stack's live words; `high` is 0 for the
    // thread's own stack, whose base brass does not know (the host scans it
    // up to its own anchor). [regs, regs_end), when not empty, holds register
    // values the switch kept off the stack (Windows' fibers save them in the
    // fiber, not on its stack), to be scanned the same way.
    struct StackSpan {
        uintptr_t low = 0;
        uintptr_t high = 0;
        const void* regs = nullptr;
        const void* regs_end = nullptr;
    };
    // Calls fn for the running stack, from `sp` (the caller's stack pointer),
    // and for every stack of the thread that is switched out, from where it
    // switched away. When no fiber has run on the thread that is one span:
    // the thread's own stack from `sp`.
    static void for_each_stack_span(uintptr_t sp, void (*fn)(void* ctx, const StackSpan& span), void* ctx);

    struct Impl;

private:
    explicit Fiber(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace brass::gc

namespace brass {

// The roots of the generated frames and recorded runs on every stack of this
// thread other than the running one (gc::Fiber). A collection's root walk
// (heap_collect.cpp) adds them to the running stack's.
void brass_append_switched_stack_roots(std::vector<uintptr_t*>& roots);

} // namespace brass
