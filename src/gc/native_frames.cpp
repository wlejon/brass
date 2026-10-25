#include <brass/gc/native_frames.hpp>
#include <brass/gc/code_stack_maps.hpp>
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

namespace {

// The maps the runs are walked with: the code registry describes every
// frame of code brass loaded; the thread's installed maps add code
// registered elsewhere.
const ModuleStackMap& run_maps() {
    static const ModuleStackMap* const registry_only = [] {
        auto* m = new ModuleStackMap();
        m->set_indexed_by_code_registry(true);
        return m;
    }();
    const ModuleStackMap* installed = brass_get_active_stack_maps();
    return installed ? *installed : *registry_only;
}

// Where a run starts, as its walk tells (stack_walker.cpp):
//   Generated  a return address with a stack map (JIT code, or AOT code in a
//              loaded image) or outside every image (a stub): the walk
//              visits every frame from there to the base.
//   Compiled   one in a loaded image with no map (a C++ frame): the walk
//              has no stack position to unwind from and visits nothing.
//   Unknown    a host where the walk cannot unwind compiled frames (Windows
//              ARM64) and follows frame pointers from anywhere.
enum class RunStart : uint8_t { Unknown, Generated, Compiled };

RunStart classify_run_start([[maybe_unused]] const ModuleStackMap& maps,
                            [[maybe_unused]] const CodeStackMapSnapshot* registered, [[maybe_unused]] uintptr_t ip) {
#if (defined(_WIN32) && defined(_M_X64)) || defined(BRASS_NATIVE_UNWIND)
    if (find_code_stack_map(registered, ip) != nullptr) return RunStart::Generated;
    if (!maps.indexed_by_code_registry() && maps.find_function_by_ip(ip) != nullptr) return RunStart::Generated;
#if defined(_WIN32) && defined(_M_X64)
    return detail::win64_ip_in_image(ip) ? RunStart::Compiled : RunStart::Generated;
#else
    return brass_ip_in_image(ip) ? RunStart::Compiled : RunStart::Generated;
#endif
#else
    return RunStart::Unknown;
#endif
}

// classify_run_start answered once per return address for one collection:
// the runs of a deep recursion are a few call sites many times over.
// Direct-mapped; a collision only asks again.
struct StartKinds {
    struct Slot {
        uintptr_t ip = 0;
        RunStart kind = RunStart::Unknown;
    };
    Slot slots[32];
    RunStart of(const ModuleStackMap& maps, const CodeStackMapSnapshot* registered, uintptr_t ip) {
        Slot& s = slots[((ip >> 2) ^ (ip >> 9)) & 31];
        if (s.ip != ip) {
            s.ip = ip;
            s.kind = classify_run_start(maps, registered, ip);
        }
        return s.kind;
    }
};

} // namespace

uintptr_t brass_native_frames_cover_start() noexcept {
    if (!t_native_frames) return UINTPTR_MAX;
    const ModuleStackMap& maps = run_maps();
    const auto registered = code_stack_map_snapshot();
    StartKinds kinds;
    for (const NativeFramesScope* s = t_native_frames; s != nullptr; s = s->prev_) {
        if (s->rbp_ == 0 || s->ip_ == 0) continue;
        if (kinds.of(maps, registered.get(), s->ip_) == RunStart::Generated) return s->rbp_;
    }
    return UINTPTR_MAX;
}

struct NativeFramesAccess {
    // The roots of one stack's chains; `stack` is null for the running one.
    static void append(const StackChains& chains, const SwitchedStack* stack, std::vector<uintptr_t*>& roots);
};

void brass_append_native_frame_roots(std::vector<uintptr_t*>& roots) {
    NativeFramesAccess::append(StackChains{t_native_frames, t_thread_roots, t_entry_scopes}, nullptr, roots);
}

void brass_append_switched_native_frame_roots(const StackChains& chains, const SwitchedStack& stack,
                                              std::vector<uintptr_t*>& roots) {
    NativeFramesAccess::append(chains, &stack, roots);
}

StackChains brass_exchange_stack_chains(const StackChains& incoming) noexcept {
    const StackChains outgoing{t_native_frames, t_thread_roots, t_entry_scopes};
    t_native_frames = incoming.native_frames;
    t_thread_roots = incoming.thread_roots;
    t_entry_scopes = incoming.entry_scopes;
    return outgoing;
}

void NativeFramesAccess::append(const StackChains& chains, const SwitchedStack* stack,
                                std::vector<uintptr_t*>& roots) {
    for (const ThreadRootsScope* s = chains.thread_roots; s != nullptr; s = s->prev_) {
        if (s->provider_) s->provider_(s->ctx_, roots);
    }
    if (!chains.native_frames) return;
    const ModuleStackMap& maps = run_maps();
    const auto registered = code_stack_map_snapshot();
    // Each run's walk stops where the next outer run whose walk covers the
    // frames from there on begins (run_covers). Unbounded, N nested runs
    // would each walk everything below them: O(N^2) per collection.
    std::vector<const NativeFramesScope*> runs;
    for (const NativeFramesScope* s = chains.native_frames; s != nullptr; s = s->prev_) {
        if (s->rbp_ != 0 && s->ip_ != 0) runs.push_back(s);
    }
    auto push = [](void** slot, void* user_data) {
        auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
        if (slot != nullptr && *slot != nullptr) vec->push_back(reinterpret_cast<uintptr_t*>(slot));
    };
    uintptr_t cover = UINTPTR_MAX;  // start of the innermost covering run outside the current one
    StartKinds kinds;
    for (size_t i = runs.size(); i-- > 0;) {
        const NativeFramesScope* s = runs[i];
        const RunStart start = kinds.of(maps, registered.get(), s->ip_);
        // A run recorded at a C++ caller (a coroutine resumed from the
        // runtime) starts in compiled code, where its walk stops at once:
        // nothing to visit. A deep async recursion records thousands.
        if (start == RunStart::Compiled) continue;
        const uintptr_t stop_at = cover > s->rbp_ ? cover : UINTPTR_MAX;
        if (stack != nullptr) {
            brass_stack_walk_bounded_on(s->rbp_, s->ip_, maps, push, &roots, stop_at, *stack);
        } else {
            brass_stack_walk_bounded(s->rbp_, s->ip_, maps, push, &roots, stop_at);
        }
        if (start == RunStart::Generated) cover = s->rbp_;
    }
}

} // namespace brass
