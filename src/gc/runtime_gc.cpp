#include <brass/gc/runtime_gc.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
extern "C" uintptr_t brass_get_rbp();
#endif

namespace brass {

namespace {

static MiniCheneyGC* g_active_gc = nullptr;
static const ModuleStackMap* g_active_stack_maps = nullptr;

inline void get_caller_frame(uintptr_t& caller_rbp, uintptr_t& caller_ip) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    void** ret_addr_slot = reinterpret_cast<void**>(_AddressOfReturnAddress());
    caller_ip = reinterpret_cast<uintptr_t>(*ret_addr_slot);
    caller_rbp = brass_get_rbp();
#elif defined(__GNUC__) || defined(__clang__)
    void* cur_frame = __builtin_frame_address(0);
    if (cur_frame) {
        caller_rbp = *reinterpret_cast<uintptr_t*>(cur_frame);
        caller_ip = *reinterpret_cast<uintptr_t*>(reinterpret_cast<uintptr_t>(cur_frame) + 8);
    }
#else
    caller_rbp = 0;
    caller_ip = 0;
#endif
}

} // namespace

void brass_set_active_gc(MiniCheneyGC* gc) noexcept {
    g_active_gc = gc;
}

MiniCheneyGC* brass_get_active_gc() noexcept {
    return g_active_gc;
}

void brass_set_active_stack_maps(const ModuleStackMap* maps) noexcept {
    g_active_stack_maps = maps;
}

const ModuleStackMap* brass_get_active_stack_maps() noexcept {
    return g_active_stack_maps;
}

void brass_runtime_gc_safepoint(
    MiniCheneyGC* gc,
    const ModuleStackMap& stack_maps,
    uintptr_t rbp,
    uintptr_t return_ip
) {
    if (!gc) return;

    uintptr_t cur_rbp = rbp;
    uintptr_t cur_ip = return_ip;

    if (cur_rbp == 0 || cur_ip == 0) {
        get_caller_frame(cur_rbp, cur_ip);
    }

    std::vector<uintptr_t*> roots;
    if (cur_rbp != 0 && cur_ip != 0) {
        brass_stack_walk(cur_rbp, cur_ip, stack_maps, [](void** slot, void* user_data) {
            auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
            if (slot != nullptr && *slot != nullptr) {
                vec->push_back(reinterpret_cast<uintptr_t*>(slot));
            }
        }, &roots);
    }

    gc->collect(roots);
}

uintptr_t brass_runtime_gc_alloc(
    MiniCheneyGC* gc,
    const ModuleStackMap& stack_maps,
    size_t size,
    uint64_t pointer_mask,
    uint32_t type_tag,
    uintptr_t rbp,
    uintptr_t return_ip
) {
    if (!gc) {
        throw std::runtime_error("MiniCheneyGC pointer is null in brass_runtime_gc_alloc");
    }

    if (gc->can_allocate_fast(size)) {
        return gc->allocate(size, pointer_mask, type_tag);
    }

    uintptr_t cur_rbp = rbp;
    uintptr_t cur_ip = return_ip;

    if (cur_rbp == 0 || cur_ip == 0) {
        get_caller_frame(cur_rbp, cur_ip);
    }

    std::vector<uintptr_t*> roots;
    if (cur_rbp != 0 && cur_ip != 0) {
        brass_stack_walk(cur_rbp, cur_ip, stack_maps, [](void** slot, void* user_data) {
            auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
            if (slot != nullptr && *slot != nullptr) {
                vec->push_back(reinterpret_cast<uintptr_t*>(slot));
            }
        }, &roots);
    }

    return gc->allocate(size, pointer_mask, type_tag, roots);
}

} // namespace brass

#if defined(_MSC_VER)

extern "C" {

void brass_runtime_gc_safepoint_bridge(uintptr_t caller_rbp, uintptr_t caller_ip) {
    auto* gc = brass::brass_get_active_gc();
    const auto* maps = brass::brass_get_active_stack_maps();
    if (!gc || !maps) return;
    brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
}

uintptr_t brass_runtime_gc_alloc_bridge(size_t size, uint64_t pointer_mask, uint32_t type_tag, uintptr_t caller_rbp, uintptr_t caller_ip) {
    auto* gc = brass::brass_get_active_gc();
    if (!gc) {
        return 0;
    }
    if (gc->can_allocate_fast(size)) {
        return gc->allocate(size, pointer_mask, type_tag);
    }
    const auto* maps = brass::brass_get_active_stack_maps();
    if (!maps) {
        return gc->allocate(size, pointer_mask, type_tag);
    }
    return brass::brass_runtime_gc_alloc(gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
}

}

#else

extern "C" {

void brass_gc_safepoint() {
    auto* gc = brass::brass_get_active_gc();
    const auto* maps = brass::brass_get_active_stack_maps();
    if (!gc || !maps) return;

    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));

    brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
}

uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    auto* gc = brass::brass_get_active_gc();
    if (!gc) {
        return 0;
    }
    if (gc->can_allocate_fast(size)) {
        return gc->allocate(size, pointer_mask, type_tag);
    }
    const auto* maps = brass::brass_get_active_stack_maps();
    if (!maps) {
        return gc->allocate(size, pointer_mask, type_tag);
    }

    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));

    return brass::brass_runtime_gc_alloc(gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
}

void brass_gc_collect() {
    auto* gc = brass::brass_get_active_gc();
    const auto* maps = brass::brass_get_active_stack_maps();
    if (!gc) return;
    if (maps) {
        void* frame = __builtin_frame_address(0);
        uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
        uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
        brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
    } else {
        gc->collect();
    }
}

}

#endif
