#pragma once

// Windows x64 frame unwinding shared by the stack walker and the native
// frame capture (internal header).

#if defined(_WIN32) && defined(_M_X64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace brass::detail {

// One frame up from `ctx`, by the function's unwind data (a leaf function
// has none: its return address is at the stack pointer). False once the
// return address is 0 (the base of the thread's stack).
inline bool win64_unwind_one(CONTEXT& ctx) noexcept {
    DWORD64 image_base = 0;
    PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
    if (!fe) {
        ctx.Rip = *reinterpret_cast<const DWORD64*>(ctx.Rsp);
        ctx.Rsp += 8;
    } else {
        void* handler_data = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fe, &ctx, &handler_data, &establisher, nullptr);
    }
    return ctx.Rip != 0;
}

// Whether ip lies in a loaded executable image (compiled code: the host's,
// brass's own runtime, the CRT, the system), as opposed to code generated
// at run time. Image code always carries unwind data for its non-leaf
// functions and need not keep a frame-pointer chain.
inline bool win64_ip_in_image(uintptr_t ip) noexcept {
    void* base = nullptr;
    return RtlPcToFileHeader(reinterpret_cast<void*>(ip), &base) != nullptr;
}

} // namespace brass::detail

#endif
