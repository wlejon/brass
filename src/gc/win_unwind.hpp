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
// return address is 0 (the base of the thread's stack) or if the frame's
// stack/frame pointer lies outside `[stack_low, stack_high]`.
inline bool win64_unwind_one(CONTEXT& ctx, uintptr_t stack_low = 0,
                             uintptr_t stack_high = UINTPTR_MAX) noexcept {
    if ((ctx.Rsp & 7u) != 0 || ctx.Rsp < stack_low || ctx.Rsp > stack_high ||
        stack_high - ctx.Rsp < 8) {
        return false;
    }
    DWORD64 image_base = 0;
    PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
    if (!fe) {
        ctx.Rip = *reinterpret_cast<const DWORD64*>(ctx.Rsp);
        ctx.Rsp += 8;
    } else {
        const RUNTIME_FUNCTION* cur_fe = fe;
        for (int chain = 0; chain < 8 && cur_fe != nullptr; ++chain) {
            const auto* info = reinterpret_cast<const uint8_t*>(image_base + cur_fe->UnwindData);
            const uint8_t flags = info[0] >> 3;
            const uint8_t prolog_size = info[1];
            const uint8_t cnt_codes = info[2];
            const uint8_t frame_reg = info[3] & 0x0Fu;
            const uint8_t frame_off = info[3] >> 4;
            if (frame_reg != 0 && (ctx.Rip - (image_base + cur_fe->BeginAddress)) >= prolog_size) {
                const DWORD64 reg_val = (&ctx.Rax)[frame_reg];
                const DWORD64 scaled = static_cast<DWORD64>(frame_off) * 16u;
                if (reg_val < scaled) return false;
                const uintptr_t est = static_cast<uintptr_t>(reg_val - scaled);
                if ((est & 7u) != 0 || est < stack_low || est > stack_high) return false;
            }
            if ((flags & UNW_FLAG_CHAININFO) == 0) break;
            const size_t codes_bytes = static_cast<size_t>((cnt_codes + 1u) & ~1u) * 2u;
            cur_fe = reinterpret_cast<const RUNTIME_FUNCTION*>(info + 4 + codes_bytes);
        }
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
