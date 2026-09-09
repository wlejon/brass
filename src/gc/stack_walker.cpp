#include <brass/gc/stack_walker.hpp>
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

namespace brass {

size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
) {
    size_t frame_count = 0;
    uintptr_t cur_rbp = top_rbp;
    uintptr_t cur_return_ip = top_return_ip;

    constexpr size_t MAX_FRAMES = 1024;

#if defined(_WIN32)
    uintptr_t stack_low = 0;
    uintptr_t stack_high = UINTPTR_MAX;
    PNT_TIB tib = reinterpret_cast<PNT_TIB>(NtCurrentTeb());
    if (tib) {
        stack_low = reinterpret_cast<uintptr_t>(tib->StackLimit);
        stack_high = reinterpret_cast<uintptr_t>(tib->StackBase);
    }
#endif

    while (cur_rbp != 0 && cur_return_ip != 0 && frame_count < MAX_FRAMES) {
#if defined(_WIN32)
        if (cur_rbp < stack_low || cur_rbp + 16 > stack_high) {
            break;
        }
#endif
        if ((cur_rbp % 8) != 0) {
            break;
        }

        const FunctionStackMap* fn_map = stack_maps.find_function_by_ip(cur_return_ip);
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

#if defined(_WIN32)
        if (cur_rbp + sizeof(uintptr_t) * 2 > stack_high) {
            break;
        }
#endif

        uintptr_t next_rbp = *reinterpret_cast<const uintptr_t*>(cur_rbp);
        uintptr_t next_return_ip = *reinterpret_cast<const uintptr_t*>(cur_rbp + 8);

        if (next_rbp <= cur_rbp || (next_rbp % 8) != 0) {
            break;
        }
#if defined(_WIN32)
        if (next_rbp < stack_low || next_rbp + 16 > stack_high) {
            break;
        }
#endif

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
