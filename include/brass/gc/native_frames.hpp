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
// A walk starts at the recorded frame and follows the frame-pointer chain
// upward as a safepoint walk does; a slot reported twice (a run also
// reached by a safepoint's own walk) is harmless, since a copying collector
// leaves a root that already points at to-space unchanged.

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

// The gcref slots of every recorded run of native frames on this thread.
void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots);

} // namespace brass
