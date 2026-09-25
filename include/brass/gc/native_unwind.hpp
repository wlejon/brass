#pragma once

// Stepping native frames by their unwind information off Windows (x86-64 and
// AArch64, ELF and Mach-O), from any frame on the thread's stack.
//
// The stack walker (stack_walker.cpp) steps through the compiled (C++) frames
// between two generated frames with it, as it does with the Win64 unwind data
// on Windows x64, so compiled code need not keep a frame-pointer chain. A
// frame is stepped by the DWARF CFI that describes its code: the process's
// registered frames (generated code's .eh_frame, which brass registers at
// load) and every loaded image's .eh_frame, found with the unwinder's
// _Unwind_Find_FDE and interpreted here for the three registers a walk needs;
// on Apple platforms, where most code carries compact unwind entries instead,
// libunwind steps a cursor set to the frame. Code no unwind information
// describes (a hand-written stub that keeps a frame record) is stepped by its
// frame record.

#include <cstdint>

namespace brass {

// One native frame: the pc it is executing (a return address for every
// frame a step produced), its stack pointer at that pc, and the value its
// frame-pointer register (rbp / x29) holds there.
struct NativeUnwindFrame {
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    uintptr_t fp = 0;
};

#if !defined(_WIN32) && (defined(__x86_64__) || defined(__aarch64__))
#define BRASS_NATIVE_UNWIND 1

// Steps `f` to the frame of its caller. `f.ip` is a return address (the
// lookup is made at the call before it) unless `ip_is_return_address` is
// false. False, leaving `f` unchanged, at the base of the stack or when the
// frame cannot be stepped.
bool brass_unwind_step(NativeUnwindFrame& f, bool ip_is_return_address = true) noexcept;

// The frame `skip` frames above the caller of this function (0: that
// caller's caller), as the unwinder recovers it. False if the stack is not
// that deep or cannot be unwound.
bool brass_capture_frame(NativeUnwindFrame& f, unsigned skip) noexcept;

// Whether ip lies in a loaded image (compiled code: the host's, brass's own
// runtime, the C++ runtime, the system) rather than in code generated at
// run time.
bool brass_ip_in_image(uintptr_t ip) noexcept;

#endif

} // namespace brass
