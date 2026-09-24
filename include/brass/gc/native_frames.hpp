#pragma once

// Native (JIT) frames that sit below re-entered Tier-0 code.
//
// A collection finds the gcrefs of native frames by walking the stack from
// the point where native code called the collector (a safepoint or an
// allocation stub). When native code instead calls back into an interpreter
// (a native-to-Tier-0 bridge, a deoptimization that resumes in Tier 0, or a
// host function that runs interpreted code), a collection that code triggers
// starts in the interpreter, whose root provider sees only interpreter
// frames: the native frames between it and the outer interpreter would keep
// from-space addresses. Every such entry records the native frame that made
// the call with a NativeFramesScope; every collector on the thread
// (MiniCheneyGC, GenerationalGC) then walks each recorded run of native
// frames with the code stack maps and reports their gcref slots as roots.
//
// A walk starts at the recorded frame and goes upward as a safepoint walk
// does; a slot reported twice (a run also reached by a safepoint's own walk)
// is harmless, since a copying collector leaves a root that already points
// at to-space unchanged.
//
// Generated code that calls a C++ function which calls generated code back
// (brass_coro_resume, or a host function registered as an external symbol)
// needs no scope on Windows x64: the walk unwinds through compiled frames
// with their unwind data. Elsewhere compiled code may omit frame pointers
// and the walk cannot cross it, so such a C++ function must record its
// generated caller with brass_capture_caller_frame and a NativeFramesScope
// around the call back; brass does so for every such call it makes.

#include <cstdint>
#include <vector>

namespace brass {

// The frame of the caller of the function that calls this: its frame
// pointer and the return address into it. False if it cannot be found.
bool brass_capture_caller_frame(uintptr_t& rbp, uintptr_t& ip) noexcept;

// Records, for its lifetime, the native frames from (rbp, ip) upward as a
// run of frames whose gcrefs every collection on this thread must update.
// Scopes nest (a bridge under a bridge); they must be destroyed in reverse
// order, which RAII on one thread guarantees.
class NativeFramesScope {
public:
    NativeFramesScope(uintptr_t rbp, uintptr_t ip) noexcept;
    ~NativeFramesScope();
    NativeFramesScope(const NativeFramesScope&) = delete;
    NativeFramesScope& operator=(const NativeFramesScope&) = delete;

private:
    uintptr_t rbp_;
    uintptr_t ip_;
    NativeFramesScope* prev_;
    friend void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots);
};

// Records, for its lifetime, roots that every collection on this thread
// must update although the collector's own root provider does not know
// them: the frames of an interpreter that allocates from a heap it does not
// own (a fresh Tier-0 interpreter finishing a deoptimization or a
// native-to-Tier-0 call in the thread's active GC). Nests as
// NativeFramesScope does.
class ThreadRootsScope {
public:
    using Provider = void (*)(void* ctx, std::vector<uintptr_t*>& roots);
    ThreadRootsScope(Provider provider, void* ctx) noexcept;
    ~ThreadRootsScope();
    ThreadRootsScope(const ThreadRootsScope&) = delete;
    ThreadRootsScope& operator=(const ThreadRootsScope&) = delete;

private:
    Provider provider_;
    void* ctx_;
    ThreadRootsScope* prev_;
    friend void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots);
};

// Marks, for its lifetime, a place where compiled (C++) code calls generated
// code: brass's invoke paths hold one around the call, and a host that calls
// a generated function pointer should too. On Windows x64, a walk that
// leaves the outermost generated frame for compiled code unwinds through it
// looking for generated frames further down, which on a plain host stack
// means every frame to the thread's base. The frames beneath a live scope
// cannot change while it lives, so the first walk that unwinds past it
// records what it found beneath (the next generated frame, or none), and
// every later walk stops there with that answer: a collection's cost stops
// depending on the host's stack depth. The record is always the result of a
// real unwind, so a scope anywhere is safe; without one a walk still unwinds
// to the base. Scopes nest; destroy them in reverse order (RAII).
class GeneratedCodeEntryScope {
public:
    GeneratedCodeEntryScope() noexcept;
    ~GeneratedCodeEntryScope();
    GeneratedCodeEntryScope(const GeneratedCodeEntryScope&) = delete;
    GeneratedCodeEntryScope& operator=(const GeneratedCodeEntryScope&) = delete;

    // Walk-internal (stack_walker.cpp).
    enum class Beneath : uint8_t { Unknown, Nothing, Generated };
    struct Memo {
        Beneath beneath = Beneath::Unknown;
        const void* maps = nullptr;  // the stack maps the answer was found with
        uintptr_t rsp = 0, rbp = 0, ip = 0;  // the generated frame found
    };
    Memo memo;
    GeneratedCodeEntryScope* outer() const noexcept { return prev_; }
    uintptr_t address() const noexcept { return reinterpret_cast<uintptr_t>(this); }

private:
    GeneratedCodeEntryScope* prev_;
};

// The innermost live GeneratedCodeEntryScope on this thread, or nullptr.
GeneratedCodeEntryScope* brass_innermost_entry_scope() noexcept;

// The gcref slots of every recorded run of native frames on this thread,
// and the roots of every ThreadRootsScope.
void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots);

} // namespace brass
