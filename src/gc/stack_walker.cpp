#include <brass/gc/stack_walker.hpp>
#include <brass/gc/code_stack_maps.hpp>
#include <brass/gc/native_frames.hpp>
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

namespace brass {

// The walk visits (rbp, ip) pairs: ip is the return address into a frame
// and rbp is the frame pointer that frame held when it made the call.
// Generated code always keeps a frame-pointer chain, so for a generated
// frame the next pair is ([rbp], [rbp + 8]).
//
// Compiled (C++) code need not keep one: MSVC uses rbp as an ordinary
// register or as a frame pointer at an arbitrary offset. When a generated
// frame's return address leads into compiled code (a generated frame called
// a C++ function that called generated code back: brass_coro_resume, a host
// function), [rbp] and [rbp + 8] say nothing about the frames above. On
// Windows x64 the walk then unwinds through the compiled frames with their
// unwind data until it returns to generated code, and resumes the chain
// there; a compiled frame's slots are never reported. Beneath the outermost
// generated frame lies the host's stack, which only a
// GeneratedCodeEntryScope keeps that unwind from crossing to the thread's
// base on every walk (native_frames.hpp).
//
// Elsewhere, compiled frames may omit the frame pointer, and nothing
// describes them to the walk: it follows [rbp] regardless, which is right
// only for code built with frame pointers. Every transition brass makes from
// generated code through C++ back into generated or interpreted code records
// the generated caller with a NativeFramesScope, which starts a walk of its
// own at that frame; a host function that calls generated or interpreted
// code from generated code must do the same (native_frames.hpp).
namespace {
thread_local size_t t_unwind_steps = 0;
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

size_t brass_stack_walk_bounded(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data,
    uintptr_t stop_at
) {
    size_t frame_count = 0;
    uintptr_t cur_rbp = top_rbp;
    uintptr_t cur_return_ip = top_return_ip;
    // The frame pointer of the generated frame whose [rbp + 8] held
    // cur_return_ip; 0 for the top pair, whose stack position is unknown.
    uintptr_t callee_rbp = 0;

    constexpr size_t MAX_FRAMES = 1024;
    constexpr size_t MAX_STEPS = 64 * 1024;

    const auto registered = code_stack_map_snapshot();

#if defined(_WIN32)
    uintptr_t stack_low = 0;
    uintptr_t stack_high = UINTPTR_MAX;
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

    size_t steps = 0;

#if defined(_WIN32) && defined(_M_X64)
    // Unwinds ctx, a compiled frame, up to the next generated frame. Entry
    // scopes (native_frames.hpp) the unwind passes answer for the frames
    // beneath them, or learn the answer this unwind finds.
    enum class UnwindResult { Generated, Nothing, GaveUp };
    GeneratedCodeEntryScope* entry = brass_innermost_entry_scope();
    auto unwind_to_generated = [&](CONTEXT& ctx) -> UnwindResult {
        constexpr size_t MAX_PENDING = 16;
        GeneratedCodeEntryScope* pending[MAX_PENDING];
        size_t npending = 0;
        auto on_this_stack = [&](const GeneratedCodeEntryScope* e) {
            return e->address() >= stack_low && e->address() < stack_high;
        };
        while (entry && (!on_this_stack(entry) || entry->address() < ctx.Rsp)) entry = entry->outer();
        UnwindResult result = UnwindResult::GaveUp;
        while (steps++ < MAX_STEPS) {
            const DWORD64 prev_rsp = ctx.Rsp;
            ++t_unwind_steps;
            if (!detail::win64_unwind_one(ctx)) {  // base of the stack
                result = UnwindResult::Nothing;
                break;
            }
            if (ctx.Rsp <= prev_rsp || ctx.Rsp < stack_low || ctx.Rsp > stack_high) break;
            if (ctx.Rsp >= stop_at) break;
            bool answered = false;
            while (entry && ctx.Rsp > entry->address()) {
                GeneratedCodeEntryScope::Memo& m = entry->memo;
                if (on_this_stack(entry) && m.beneath != GeneratedCodeEntryScope::Beneath::Unknown &&
                    m.maps == &stack_maps) {
                    if (m.beneath == GeneratedCodeEntryScope::Beneath::Generated) {
                        ctx.Rsp = m.rsp;
                        ctx.Rbp = m.rbp;
                        ctx.Rip = m.ip;
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
            if (find_map(ctx.Rip) != nullptr || !detail::win64_ip_in_image(ctx.Rip)) {
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
                m.rsp = static_cast<uintptr_t>(ctx.Rsp);
                m.rbp = static_cast<uintptr_t>(ctx.Rbp);
                m.ip = static_cast<uintptr_t>(ctx.Rip);
            }
        }
        return result;
    };
#endif

    while (cur_return_ip != 0 && frame_count < MAX_FRAMES && steps++ < MAX_STEPS) {
        const FunctionStackMap* fn_map = find_map(cur_return_ip);

#if defined(_WIN32) && defined(_M_X64)
        if (fn_map == nullptr && detail::win64_ip_in_image(cur_return_ip)) {
            // A compiled frame: cur_rbp is only the value its rbp register
            // held. Its stack pointer at the call is just above the return
            // address, which sits at callee_rbp + 8.
            if (callee_rbp == 0) {
                break;  // the walk started in compiled code: no stack position
            }
            CONTEXT ctx{};
            ctx.Rip = cur_return_ip;
            ctx.Rsp = callee_rbp + 16;
            ctx.Rbp = cur_rbp;
            if (ctx.Rsp >= stop_at) break;
            const UnwindResult res = unwind_to_generated(ctx);
            if (res != UnwindResult::Generated) break;
            if (ctx.Rsp >= stop_at) break;
            // The generated frame's rbp lies at or above its stack pointer.
            if (ctx.Rbp < ctx.Rsp) break;
            cur_rbp = static_cast<uintptr_t>(ctx.Rbp);
            cur_return_ip = static_cast<uintptr_t>(ctx.Rip);
            callee_rbp = 0;
            continue;
        }
#endif

        if (cur_rbp == 0 || (cur_rbp % 8) != 0 || cur_rbp >= stop_at) {
            break;
        }
#if defined(_WIN32)
        if (cur_rbp < stack_low || cur_rbp + 16 > stack_high) {
            break;
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

#if defined(_WIN32) && defined(_M_X64)
        // A compiled caller's rbp is any value; the unwind above checks it.
        const bool next_checked_by_unwind =
            find_map(next_return_ip) == nullptr && detail::win64_ip_in_image(next_return_ip);
#else
        const bool next_checked_by_unwind = false;
#endif
        if (!next_checked_by_unwind) {
            if (next_rbp <= cur_rbp || (next_rbp % 8) != 0) {
                break;
            }
#if defined(_WIN32)
            if (next_rbp < stack_low || next_rbp + 16 > stack_high) {
                break;
            }
#endif
        }

        callee_rbp = cur_rbp;
        cur_rbp = next_rbp;
        cur_return_ip = next_return_ip;
    }

    return frame_count;
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
