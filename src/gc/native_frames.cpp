#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#define BRASS_NATIVE_FRAMES_NOINLINE __declspec(noinline)
#else
#define BRASS_NATIVE_FRAMES_NOINLINE __attribute__((noinline))
#endif

namespace brass {

namespace {

thread_local NativeFramesScope* t_native_frames = nullptr;

#if defined(_WIN32) && defined(_M_X64)
// One frame up from `ctx`, by the function's unwind data (a leaf function
// has none: its return address is at the stack pointer).
bool unwind_one(CONTEXT& ctx) noexcept {
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
#endif

} // namespace

BRASS_NATIVE_FRAMES_NOINLINE bool brass_capture_caller_frame(uintptr_t& rbp, uintptr_t& ip) noexcept {
    rbp = 0;
    ip = 0;
#if defined(_WIN32) && defined(_M_X64)
    // This function's frame, then its caller's (the function that asked),
    // then that one's caller: the frame asked for. Unwinding restores RBP
    // as each function saved it, whatever MSVC code used it for since.
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    if (!unwind_one(ctx) || !unwind_one(ctx)) return false;
    rbp = static_cast<uintptr_t>(ctx.Rbp);
    ip = static_cast<uintptr_t>(ctx.Rip);
#elif defined(__GNUC__) || defined(__clang__)
    // Frame-pointer chain: the asking function's frame record holds its
    // caller's frame pointer and the return address into it.
    auto* asking = static_cast<uintptr_t*>(__builtin_frame_address(1));
    if (!asking) return false;
    rbp = asking[0];
    ip = asking[1];
#endif
    return rbp != 0 && ip != 0;
}

NativeFramesScope::NativeFramesScope(uintptr_t rbp, uintptr_t ip) noexcept
    : rbp_(rbp), ip_(ip), prev_(t_native_frames) {
    t_native_frames = this;
}

NativeFramesScope::~NativeFramesScope() {
    if (t_native_frames != this) {
        std::fprintf(stderr, "brass: fatal: native frame scopes destroyed out of order\n");
        std::fflush(stderr);
        std::abort();
    }
    t_native_frames = prev_;
}

void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots) {
    if (!t_native_frames) return;
    // The code registry describes every frame of code brass loaded; the
    // thread's installed maps add code registered elsewhere.
    static const ModuleStackMap* const registry_only = [] {
        auto* m = new ModuleStackMap();
        m->set_indexed_by_code_registry(true);
        return m;
    }();
    const ModuleStackMap* installed = brass_get_active_stack_maps();
    const ModuleStackMap& maps = installed ? *installed : *registry_only;
    for (const NativeFramesScope* s = t_native_frames; s != nullptr; s = s->prev_) {
        if (s->rbp_ == 0 || s->ip_ == 0) continue;
        brass_stack_walk(s->rbp_, s->ip_, maps, [](void** slot, void* user_data) {
            auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
            if (slot != nullptr && *slot != nullptr) vec->push_back(reinterpret_cast<uintptr_t*>(slot));
        }, &roots);
    }
}

} // namespace brass
