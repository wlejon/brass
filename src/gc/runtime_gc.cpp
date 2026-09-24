#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/code_stack_maps.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
extern "C" uintptr_t brass_get_rbp();
#endif

namespace brass {

namespace {

// The collector and stack maps of the code running on THIS thread. Per thread
// because what they describe is: a collector's roots are the frames of the
// thread that allocates, and the maps are the maps of the code on that
// thread's stack. Process-wide, a second thread installing its program's maps
// (or an OSR bridge swapping in an interpreter's heap) retargeted every other
// thread's safepoints at frames they do not have. Parallel loop bodies, the
// one place brass runs generated code on a thread that did not install
// these, neither allocate nor call (loop_parallel_analysis.cpp). The maps
// installed here only add to the code registry (code_stack_maps.hpp), through
// which a stack walk finds the maps of all code brass loaded, so frames of a
// tier or program other than the installed one still have their roots.
thread_local MiniCheneyGC* g_active_gc = nullptr;
thread_local GenerationalGC* g_active_gen_gc = nullptr;
thread_local const ModuleStackMap* g_active_stack_maps = nullptr;

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

// Native allocation entry points are called from JIT frames, where a C++
// exception cannot be relied on to unwind; a missing collector or missing
// stack maps is an embedding bug, so it stops the process with a message.
[[noreturn]] void gc_fatal_no_gc() {
    std::cerr << "brass: fatal: brass_gc_alloc called with no active GC; the embedder must "
              << "install one (brass_set_active_gc or brass_set_active_generational_gc) "
              << "before running code that allocates\n";
    std::cerr.flush();
    std::abort();
}

[[noreturn]] void gc_fatal_no_maps(const char* op) {
    std::cerr << "brass: fatal: " << op << " needs a collection but no stack maps are active "
              << "and the calling code has none registered, so live gcrefs in native frames "
              << "cannot be found; call brass_set_active_stack_maps with the running code's maps\n";
    std::cerr.flush();
    std::abort();
}

// The maps a collection requested from caller_ip walks this thread's native
// frames with. The stack walk resolves every frame of code brass loaded
// through the code registry, whatever the thread has installed; the maps the
// thread installed add code registered elsewhere. A thread that installed
// none walks with the registry alone, provided the code asking is registered
// code: then its frames, and the brass frames under it, have maps. Null when
// nothing describes the caller's frame.
const ModuleStackMap* walk_maps(uintptr_t caller_ip) noexcept {
    if (g_active_stack_maps) return g_active_stack_maps;
    static const ModuleStackMap* const registry_only = [] {
        auto* m = new ModuleStackMap();
        m->set_indexed_by_code_registry(true);
        return m;
    }();
    if (caller_ip != 0 && code_stack_maps_cover(caller_ip)) return registry_only;
    return nullptr;
}

} // namespace

void brass_set_active_gc(MiniCheneyGC* gc) noexcept {
    g_active_gc = gc;
}

MiniCheneyGC* brass_get_active_gc() noexcept {
    return g_active_gc;
}

void brass_set_active_generational_gc(GenerationalGC* gc) noexcept {
    g_active_gen_gc = gc;
}

GenerationalGC* brass_get_active_generational_gc() noexcept {
    return g_active_gen_gc;
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

    // The native frames' slots are not the only roots: an interpreter that
    // migrated into this code (OSR) holds gcrefs in its frames, reported by
    // the collector's root provider. Allocation already includes them.
    gc->collect_with_extra_roots(roots);
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

void brass_runtime_gc_safepoint(
    GenerationalGC* gc,
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
    GenerationalGC* gc,
    const ModuleStackMap& stack_maps,
    size_t size,
    uint64_t pointer_mask,
    uint32_t type_tag,
    uintptr_t rbp,
    uintptr_t return_ip
) {
    if (!gc) {
        throw std::runtime_error("GenerationalGC pointer is null in brass_runtime_gc_alloc");
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
    if (auto* gen_gc = brass::brass_get_active_generational_gc()) {
        // A safepoint is an opportunity, not a demand: without stack maps the
        // native frames' roots are unknown, so collecting here would be unsafe.
        if (const auto* maps = brass::walk_maps(caller_ip)) {
            brass::brass_runtime_gc_safepoint(gen_gc, *maps, caller_rbp, caller_ip);
        }
        return;
    }
    auto* gc = brass::brass_get_active_gc();
    if (!gc) return;
    const auto* maps = brass::walk_maps(caller_ip);
    if (!maps) return;
    brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
}

uintptr_t brass_runtime_gc_alloc_bridge(size_t size, uint64_t pointer_mask, uint32_t type_tag, uintptr_t caller_rbp, uintptr_t caller_ip) {
    if (auto* gen_gc = brass::brass_get_active_generational_gc()) {
        if (gen_gc->can_allocate_fast(size)) {
            return gen_gc->allocate(size, pointer_mask, type_tag);
        }
        const auto* maps = brass::walk_maps(caller_ip);
        if (!maps) brass::gc_fatal_no_maps("brass_gc_alloc");
        return brass::brass_runtime_gc_alloc(gen_gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
    }
    auto* gc = brass::brass_get_active_gc();
    if (!gc) brass::gc_fatal_no_gc();
    if (gc->can_allocate_fast(size)) {
        return gc->allocate(size, pointer_mask, type_tag);
    }
    const auto* maps = brass::walk_maps(caller_ip);
    if (!maps) brass::gc_fatal_no_maps("brass_gc_alloc");
    return brass::brass_runtime_gc_alloc(gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
}

void brass_default_gc_write_barrier(uintptr_t obj, uintptr_t val) {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    if (!gen_gc) return;
    if (!gen_gc->is_old(obj)) return;
    uintptr_t ptr_val = val & 0x0000FFFFFFFFFFFFULL;
    if (!gen_gc->is_young(val) && !gen_gc->is_young(ptr_val)) return;
    gen_gc->card_table().mark_card(obj);
}

uint8_t* brass_gc_card_table_base() {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    return gen_gc ? gen_gc->card_table().byte_map_base() : nullptr;
}

uintptr_t brass_gc_heap_base() {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    return gen_gc ? gen_gc->card_table().heap_base() : 0;
}
}

#else

extern "C" {

void brass_gc_safepoint() {
    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));

    // Without stack maps the native frames' roots are unknown: skip.
    const auto* maps = brass::walk_maps(caller_ip);
    if (!maps) return;
    if (auto* gen_gc = brass::brass_get_active_generational_gc()) {
        brass::brass_runtime_gc_safepoint(gen_gc, *maps, caller_rbp, caller_ip);
        return;
    }
    if (auto* gc = brass::brass_get_active_gc()) {
        brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
    }
}

uintptr_t brass_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (auto* gen_gc = brass::brass_get_active_generational_gc()) {
        if (gen_gc->can_allocate_fast(size)) {
            return gen_gc->allocate(size, pointer_mask, type_tag);
        }
        void* frame = __builtin_frame_address(0);
        uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
        uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
        const auto* maps = brass::walk_maps(caller_ip);
        if (!maps) brass::gc_fatal_no_maps("brass_gc_alloc");
        return brass::brass_runtime_gc_alloc(gen_gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
    }
    auto* gc = brass::brass_get_active_gc();
    if (!gc) brass::gc_fatal_no_gc();
    if (gc->can_allocate_fast(size)) {
        return gc->allocate(size, pointer_mask, type_tag);
    }
    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    const auto* maps = brass::walk_maps(caller_ip);
    if (!maps) brass::gc_fatal_no_maps("brass_gc_alloc");

    return brass::brass_runtime_gc_alloc(gc, *maps, size, pointer_mask, type_tag, caller_rbp, caller_ip);
}

void brass_gc_collect() {
    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));

    // Same as a safepoint (the MSVC stub routes both through one bridge):
    // without stack maps the native frames' roots are unknown, so skip.
    const auto* maps = brass::walk_maps(caller_ip);
    if (!maps) return;
    if (auto* gen_gc = brass::brass_get_active_generational_gc()) {
        brass::brass_runtime_gc_safepoint(gen_gc, *maps, caller_rbp, caller_ip);
    } else if (auto* gc = brass::brass_get_active_gc()) {
        brass::brass_runtime_gc_safepoint(gc, *maps, caller_rbp, caller_ip);
    }
}

void brass_default_gc_write_barrier(uintptr_t obj, uintptr_t val) {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    if (!gen_gc) return;
    if (!gen_gc->is_old(obj)) return;
    uintptr_t ptr_val = val & 0x0000FFFFFFFFFFFFULL;
    if (!gen_gc->is_young(val) && !gen_gc->is_young(ptr_val)) return;
    gen_gc->card_table().mark_card(obj);
}

uint8_t* brass_gc_card_table_base() {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    return gen_gc ? gen_gc->card_table().byte_map_base() : nullptr;
}

uintptr_t brass_gc_heap_base() {
    auto* gen_gc = brass::brass_get_active_generational_gc();
    return gen_gc ? gen_gc->card_table().heap_base() : 0;
}

}

#endif
