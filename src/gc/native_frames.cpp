#include <brass/gc/native_frames.hpp>
#include <brass/gc/native_unwind.hpp>
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
#include "win_unwind.hpp"

#if defined(_MSC_VER)
#define BRASS_NATIVE_FRAMES_NOINLINE __declspec(noinline)
#else
#define BRASS_NATIVE_FRAMES_NOINLINE __attribute__((noinline))
#endif

namespace brass {

namespace {

thread_local NativeFramesScope* t_native_frames = nullptr;
thread_local ThreadRootsScope* t_thread_roots = nullptr;
thread_local GeneratedCodeEntryScope* t_entry_scopes = nullptr;

#if defined(_WIN32) && defined(_M_X64)
using detail::win64_unwind_one;
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
    if (!win64_unwind_one(ctx) || !win64_unwind_one(ctx)) return false;
    rbp = static_cast<uintptr_t>(ctx.Rbp);
    ip = static_cast<uintptr_t>(ctx.Rip);
#elif defined(BRASS_NATIVE_UNWIND)
    // The same two steps by the frames' CFI: the asking function, then the
    // frame it was called from.
    NativeUnwindFrame f;
    if (!brass_capture_frame(f, 1)) return false;
    rbp = f.fp;
    ip = f.ip;
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

ThreadRootsScope::ThreadRootsScope(Provider provider, void* ctx) noexcept
    : provider_(provider), ctx_(ctx), prev_(t_thread_roots) {
    t_thread_roots = this;
}

ThreadRootsScope::~ThreadRootsScope() {
    if (t_thread_roots != this) {
        std::fprintf(stderr, "brass: fatal: thread root scopes destroyed out of order\n");
        std::fflush(stderr);
        std::abort();
    }
    t_thread_roots = prev_;
}

GeneratedCodeEntryScope::GeneratedCodeEntryScope() noexcept : prev_(t_entry_scopes) {
    t_entry_scopes = this;
}

GeneratedCodeEntryScope::~GeneratedCodeEntryScope() {
    if (t_entry_scopes != this) {
        std::fprintf(stderr, "brass: fatal: generated-code entry scopes destroyed out of order\n");
        std::fflush(stderr);
        std::abort();
    }
    t_entry_scopes = prev_;
}

GeneratedCodeEntryScope* brass_innermost_entry_scope() noexcept { return t_entry_scopes; }

void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots) {
    for (const ThreadRootsScope* s = t_thread_roots; s != nullptr; s = s->prev_) {
        if (s->provider_) s->provider_(s->ctx_, roots);
    }
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
    // Each run's walk stops where the next outer run whose walk covers the
    // frames from there on begins: one that starts at a generated frame (a
    // return address outside every image). Unbounded, N nested runs would
    // each walk everything below them: O(N^2) per collection. Where the walk
    // cannot unwind compiled frames (Windows ARM64) a run cannot be told
    // apart, and walks unbounded.
    std::vector<const NativeFramesScope*> runs;
    for (const NativeFramesScope* s = t_native_frames; s != nullptr; s = s->prev_) {
        if (s->rbp_ != 0 && s->ip_ != 0) runs.push_back(s);
    }
    uintptr_t cover = UINTPTR_MAX;  // start of the innermost covering run outside the current one
    for (size_t i = runs.size(); i-- > 0;) {
        const NativeFramesScope* s = runs[i];
        const uintptr_t stop_at = cover > s->rbp_ ? cover : UINTPTR_MAX;
        brass_stack_walk_bounded(s->rbp_, s->ip_, maps, [](void** slot, void* user_data) {
            auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
            if (slot != nullptr && *slot != nullptr) vec->push_back(reinterpret_cast<uintptr_t*>(slot));
        }, &roots, stop_at);
#if defined(_WIN32) && defined(_M_X64)
        if (!detail::win64_ip_in_image(s->ip_)) cover = s->rbp_;
#elif defined(BRASS_NATIVE_UNWIND)
        if (!brass_ip_in_image(s->ip_)) cover = s->rbp_;
#endif
    }
}

} // namespace brass
