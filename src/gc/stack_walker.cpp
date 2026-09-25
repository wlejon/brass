#include <brass/gc/stack_walker.hpp>
#include <brass/gc/code_stack_maps.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/gc/native_unwind.hpp>
#include <iostream>

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

#if (defined(_WIN32) && defined(_M_X64)) || defined(BRASS_NATIVE_UNWIND)
#define BRASS_WALK_UNWINDS 1
#else
#define BRASS_WALK_UNWINDS 0
#endif

namespace brass {

// The walk visits (rbp, ip) pairs: ip is the return address into a frame
// and rbp is the frame pointer that frame held when it made the call.
// Generated code always keeps a frame-pointer chain, so for a generated
// frame the next pair is ([rbp], [rbp + 8]).
//
// Compiled (C++) code need not keep one: it may use rbp as an ordinary
// register, or as a frame pointer at an arbitrary offset. When a generated
// frame's return address leads into compiled code (a generated frame called
// a C++ function that called generated code back: brass_coro_resume, a host
// function), [rbp] and [rbp + 8] say nothing about the frames above. On
// Windows x64 the walk then unwinds through the compiled frames with their
// unwind data, and on the other x86-64 and AArch64 hosts with their DWARF CFI
// (native_unwind.hpp), until it returns to generated code, and resumes the
// chain there; a compiled frame's slots are never reported. Code outside
// every loaded image (generated code without a stack map: a stub) is left to
// the chain as well. Beneath the outermost generated frame lies the host's
// stack, which only a GeneratedCodeEntryScope keeps that unwind from
// crossing to the thread's base on every walk (native_frames.hpp).
//
// Where neither is available (Windows ARM64), the walk follows [rbp]
// regardless, which is right only for code built with frame pointers. Every
// transition brass makes from generated code through C++ back into generated
// or interpreted code records the generated caller with a NativeFramesScope,
// which starts a walk of its own at that frame. There a GeneratedCodeEntryScope
// bounds the chain as it bounds the unwind: the first walk that follows the
// chain past a scope (the first pair whose frame pointer lies beneath it)
// records the next generated pair it reaches beneath, or that the chain
// ended, and later walks take that answer. An unwind step here is one
// compiled frame followed.
namespace {
thread_local size_t t_unwind_steps = 0;

#if defined(_WIN32) && defined(_M_X64)
// A frame being unwound: its context, of which the walk reads three
// registers.
struct Cursor {
    CONTEXT ctx{};
    uintptr_t ip() const noexcept { return static_cast<uintptr_t>(ctx.Rip); }
    uintptr_t sp() const noexcept { return static_cast<uintptr_t>(ctx.Rsp); }
    uintptr_t fp() const noexcept { return static_cast<uintptr_t>(ctx.Rbp); }
    void set(uintptr_t ip, uintptr_t sp, uintptr_t fp) noexcept {
        ctx.Rip = ip;
        ctx.Rsp = sp;
        ctx.Rbp = fp;
    }
    // One frame up; false at the base of the stack.
    bool step() noexcept { return detail::win64_unwind_one(ctx); }
};
bool ip_in_image(uintptr_t ip) noexcept { return detail::win64_ip_in_image(ip); }
#elif defined(BRASS_NATIVE_UNWIND)
struct Cursor {
    NativeUnwindFrame f;
    uintptr_t ip() const noexcept { return f.ip; }
    uintptr_t sp() const noexcept { return f.sp; }
    uintptr_t fp() const noexcept { return f.fp; }
    void set(uintptr_t ip, uintptr_t sp, uintptr_t fp) noexcept { f = {ip, sp, fp}; }
    // One frame up; false at the base of the stack or where no step is
    // possible, which the walk cannot see past either way.
    bool step() noexcept { return brass_unwind_step(f); }
};
bool ip_in_image(uintptr_t ip) noexcept { return brass_ip_in_image(ip); }
#endif

} // namespace

size_t brass_stack_walk_unwind_steps() noexcept { return t_unwind_steps; }

size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
) {
    return brass_stack_walk_bounded(top_rbp, top_return_ip, stack_maps, visitor, user_data, UINTPTR_MAX);
}

namespace {

#if BRASS_WALK_UNWINDS
using StartFrame = Cursor;
#else
struct StartFrame {};
#endif

// The walk. `start`, when given, is a compiled frame on this thread's stack:
// the walk unwinds from it to the first generated frame and starts there
// instead of at (top_rbp, top_return_ip). Every step moves strictly toward
// the stack's base, so the walk ends without a frame limit.
size_t walk_stack(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at,
    [[maybe_unused]] const StartFrame* start
) {
    size_t frame_count = 0;
    uintptr_t cur_rbp = top_rbp;
    uintptr_t cur_return_ip = top_return_ip;
    // The frame pointer of the generated frame whose [rbp + 8] held
    // cur_return_ip; 0 for the top pair, whose stack position is unknown.
    [[maybe_unused]] uintptr_t callee_rbp = 0;

    const auto registered = code_stack_map_snapshot();

    [[maybe_unused]] uintptr_t stack_low = 0;
    [[maybe_unused]] uintptr_t stack_high = UINTPTR_MAX;
#if defined(_WIN32)
    PNT_TIB tib = reinterpret_cast<PNT_TIB>(NtCurrentTeb());
    if (tib) {
        stack_low = reinterpret_cast<uintptr_t>(tib->StackLimit);
        stack_high = reinterpret_cast<uintptr_t>(tib->StackBase);
    }
#endif

    // The registry holds the maps of all code brass loaded; the given maps
    // add code registered elsewhere (an embedder's own images).
    auto find_map = [&](uintptr_t ip) -> const FunctionStackMap* {
        const FunctionStackMap* m = find_code_stack_map(registered.get(), ip);
        if (m == nullptr && !stack_maps.indexed_by_code_registry()) {
            m = stack_maps.find_function_by_ip(ip);
        }
        return m;
    };

#if !BRASS_WALK_UNWINDS
    // Entry scopes the walk passes (above): the innermost one not yet
    // passed, and the passed ones waiting for the answer beneath them. A
    // scope above the first pair lies in a frame the walk does not visit.
    using Beneath = GeneratedCodeEntryScope::Beneath;
    GeneratedCodeEntryScope* entry = brass_innermost_entry_scope();
    while (entry && entry->address() < top_rbp) entry = entry->outer();
    constexpr size_t MAX_PENDING = 16;
    GeneratedCodeEntryScope* pending[MAX_PENDING];
    size_t npending = 0;
    auto settle = [&](Beneath beneath, uintptr_t rbp, uintptr_t ip) {
        for (size_t i = 0; i < npending; ++i) {
            GeneratedCodeEntryScope::Memo& m = pending[i]->memo;
            m.beneath = beneath;
            m.maps = &stack_maps;
            m.rsp = 0;
            m.rbp = rbp;
            m.ip = ip;
        }
        npending = 0;
    };
    bool chain_ended = false;
#else
    // Unwinds cur, a compiled frame, up to the next generated frame. Entry
    // scopes (native_frames.hpp) the unwind passes answer for the frames
    // beneath them, or learn the answer this unwind finds.
    enum class UnwindResult { Generated, Nothing, GaveUp };
    GeneratedCodeEntryScope* entry = brass_innermost_entry_scope();
    auto unwind_to_generated = [&](Cursor& cur) -> UnwindResult {
        constexpr size_t MAX_PENDING = 16;
        GeneratedCodeEntryScope* pending[MAX_PENDING];
        size_t npending = 0;
        auto on_this_stack = [&](const GeneratedCodeEntryScope* e) {
            return e->address() >= stack_low && e->address() < stack_high;
        };
        while (entry && (!on_this_stack(entry) || entry->address() < cur.sp())) entry = entry->outer();
        UnwindResult result = UnwindResult::GaveUp;
        for (;;) {
            const uintptr_t prev_sp = cur.sp();
            ++t_unwind_steps;
            if (!cur.step()) {  // base of the stack
                result = UnwindResult::Nothing;
                break;
            }
            if (cur.sp() <= prev_sp || cur.sp() < stack_low || cur.sp() > stack_high) break;
            if (cur.sp() >= stop_at) break;
            bool answered = false;
            while (entry && cur.sp() > entry->address()) {
                GeneratedCodeEntryScope::Memo& m = entry->memo;
                if (on_this_stack(entry) && m.beneath != GeneratedCodeEntryScope::Beneath::Unknown &&
                    m.maps == &stack_maps) {
                    if (m.beneath == GeneratedCodeEntryScope::Beneath::Generated) {
                        cur.set(m.ip, m.rsp, m.rbp);
                        result = UnwindResult::Generated;
                    } else {
                        result = UnwindResult::Nothing;
                    }
                    answered = true;
                    break;
                }
                if (on_this_stack(entry) && npending < MAX_PENDING) pending[npending++] = entry;
                entry = entry->outer();
            }
            if (answered) break;
            if (find_map(cur.ip()) != nullptr || !ip_in_image(cur.ip())) {
                result = UnwindResult::Generated;
                break;
            }
        }
        if (result != UnwindResult::GaveUp) {
            for (size_t i = 0; i < npending; ++i) {
                GeneratedCodeEntryScope::Memo& m = pending[i]->memo;
                m.beneath = result == UnwindResult::Generated ? GeneratedCodeEntryScope::Beneath::Generated
                                                              : GeneratedCodeEntryScope::Beneath::Nothing;
                m.maps = &stack_maps;
                m.rsp = cur.sp();
                m.rbp = cur.fp();
                m.ip = cur.ip();
            }
        }
        return result;
    };

    if (start != nullptr) {
        Cursor cur = *start;
        if (find_map(cur.ip()) == nullptr) {
            if (unwind_to_generated(cur) != UnwindResult::Generated) return 0;
        }
        if (cur.sp() >= stop_at || cur.fp() < cur.sp()) return 0;
        cur_rbp = cur.fp();
        cur_return_ip = cur.ip();
    }
#endif

    while (cur_return_ip != 0) {
        const FunctionStackMap* fn_map = find_map(cur_return_ip);

#if BRASS_WALK_UNWINDS
        if (fn_map == nullptr && ip_in_image(cur_return_ip)) {
            // A compiled frame: cur_rbp is only the value its rbp register
            // held. Its stack pointer at the call is just above the return
            // address, which sits at callee_rbp + 8.
            if (callee_rbp == 0) {
                break;  // the walk started in compiled code: no stack position
            }
            Cursor cur;
            cur.set(cur_return_ip, callee_rbp + 16, cur_rbp);
            if (cur.sp() >= stop_at) break;
            const UnwindResult res = unwind_to_generated(cur);
            if (res != UnwindResult::Generated) break;
            if (cur.sp() >= stop_at) break;
            // The generated frame's rbp lies at or above its stack pointer.
            if (cur.fp() < cur.sp()) break;
            cur_rbp = cur.fp();
            cur_return_ip = cur.ip();
            callee_rbp = 0;
            continue;
        }
#endif

        if (cur_rbp == 0 || (cur_rbp % 8) != 0) {
#if !BRASS_WALK_UNWINDS
            chain_ended = true;
#endif
            break;
        }
        if (cur_rbp >= stop_at) {
            break;
        }
        if (cur_rbp < stack_low || cur_rbp + 16 > stack_high) {
            break;
        }

#if !BRASS_WALK_UNWINDS
        if (fn_map != nullptr) {
            // The answer for the scopes passed since the last generated
            // pair. A scope this pair passes lies in a frame without a
            // frame record: no answer for it.
            settle(Beneath::Generated, cur_rbp, cur_return_ip);
            while (entry && cur_rbp > entry->address()) entry = entry->outer();
        } else {
            ++t_unwind_steps;
            const GeneratedCodeEntryScope::Memo* answer = nullptr;
            while (entry && cur_rbp > entry->address()) {
                GeneratedCodeEntryScope* passed = entry;
                entry = entry->outer();
                if (passed->memo.beneath != Beneath::Unknown && passed->memo.maps == &stack_maps) {
                    answer = &passed->memo;
                    break;
                }
                if (npending < MAX_PENDING) pending[npending++] = passed;
            }
            if (answer != nullptr) {
                if (answer->beneath == Beneath::Nothing) {
                    chain_ended = true;
                    break;
                }
                // The scopes between here and the answer lie above it.
                cur_rbp = answer->rbp;
                cur_return_ip = answer->ip;
                callee_rbp = 0;
                settle(Beneath::Generated, cur_rbp, cur_return_ip);
                while (entry && entry->address() < cur_rbp) entry = entry->outer();
                continue;
            }
        }
#endif

        if (fn_map != nullptr) {
            const StackMapRecord* rec = fn_map->find_record_by_ip(cur_return_ip);
            if (rec != nullptr) {
                for (const auto& root_loc : rec->roots) {
                    intptr_t slot_addr_int = static_cast<intptr_t>(cur_rbp) + root_loc.offset_from_rbp;
                    void** root_slot = reinterpret_cast<void**>(slot_addr_int);
                    if (visitor && root_slot) {
                        visitor(root_slot, user_data);
                    }
                }
                frame_count++;
            }
        }

        uintptr_t next_rbp = *reinterpret_cast<const uintptr_t*>(cur_rbp);
        uintptr_t next_return_ip = *reinterpret_cast<const uintptr_t*>(cur_rbp + 8);

#if BRASS_WALK_UNWINDS
        // A compiled caller's rbp is any value; the unwind above checks it.
        const bool next_checked_by_unwind = find_map(next_return_ip) == nullptr && ip_in_image(next_return_ip);
#else
        const bool next_checked_by_unwind = false;
#endif
        if (!next_checked_by_unwind) {
            if (next_rbp <= cur_rbp || (next_rbp % 8) != 0) {
#if !BRASS_WALK_UNWINDS
                chain_ended = true;
#endif
                break;
            }
            if (next_rbp < stack_low || next_rbp + 16 > stack_high) {
                break;
            }
        }

        callee_rbp = cur_rbp;
        cur_rbp = next_rbp;
        cur_return_ip = next_return_ip;
    }

#if !BRASS_WALK_UNWINDS
    // Not when stop_at ended the walk: the chain may go on.
    if (chain_ended || cur_return_ip == 0) settle(Beneath::Nothing, 0, 0);
#endif
    return frame_count;
}

} // namespace

size_t brass_stack_walk_bounded(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at
) {
    return walk_stack(top_rbp, top_return_ip, stack_maps, visitor, user_data, stop_at, nullptr);
}

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
size_t brass_stack_walk_from_here(
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
) {
#if defined(_WIN32) && defined(_M_X64)
    Cursor cur;
    RtlCaptureContext(&cur.ctx);
    return walk_stack(0, 0, stack_maps, visitor, user_data, UINTPTR_MAX, &cur);
#elif defined(BRASS_NATIVE_UNWIND)
    // This function's caller, as the unwinder recovers it; the compiled
    // frames from there to the first generated one are stepped by their CFI.
    Cursor cur;
    if (!brass_capture_frame(cur.f, 0)) return 0;
    return walk_stack(0, 0, stack_maps, visitor, user_data, UINTPTR_MAX, &cur);
#else
    // This frame's record: the caller's frame pointer, then the return
    // address into it. The compiled frames up to the first generated one
    // are followed by their frame pointers.
    const auto* fp = static_cast<const uintptr_t*>(__builtin_frame_address(0));
    if (fp == nullptr) return 0;
    return walk_stack(fp[0], fp[1], stack_maps, visitor, user_data, UINTPTR_MAX, nullptr);
#endif
}

size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    const std::function<void(void**)>& visitor
) {
    auto adapter = [](void** slot, void* udata) {
        auto* fn = static_cast<const std::function<void(void**)>*>(udata);
        if (fn && *fn) {
            (*fn)(slot);
        }
    };
    return brass_stack_walk(top_rbp, top_return_ip, stack_maps, adapter, const_cast<void*>(static_cast<const void*>(&visitor)));
}

} // namespace brass
