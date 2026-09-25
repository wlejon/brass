#include <brass/embedding/brass_c_api.h>
#include <brass/embedding/embedding.hpp>
#include <brass/gc/heap.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/mir/module.hpp>

struct brass_engine_t {
    brass::HostEngine engine;
};

struct brass_compiled_module_t {
    std::unique_ptr<brass::CompiledModule> module;
};

struct brass_module_t {
    std::unique_ptr<brass::Module> module;
};

struct brass_heap_t {
    brass::gc::Heap heap;
    explicit brass_heap_t(const brass::gc::HeapConfig& config) : heap(config) {}
};

namespace {
// The thread's bound C handle, beside gc::Heap::current() (which it binds).
thread_local brass_heap_t* t_bound_heap = nullptr;
} // namespace

extern "C" {

brass_engine_t* brass_engine_create(void) {
    try {
        return new brass_engine_t();
    } catch (...) {
        return nullptr;
    }
}

void brass_engine_destroy(brass_engine_t* engine) {
    try {
        delete engine;
    } catch (...) {
    }
}

void brass_engine_register_symbol(brass_engine_t* engine, const char* name, void* address) {
    if (!engine || !name) return;
    try {
        engine->engine.register_external_symbol(name, address);
    } catch (...) {
    }
}

brass_module_t* brass_embed_module_create(const char* name) {
    try {
        auto* mod = new brass_module_t();
        mod->module = std::make_unique<brass::Module>(name ? name : "anonymous");
        return mod;
    } catch (...) {
        return nullptr;
    }
}

void brass_embed_module_destroy(brass_module_t* module) {
    try {
        delete module;
    } catch (...) {
    }
}

void brass_embed_module_add_external_symbol(brass_module_t* module, const char* name) {
    if (!module || !module->module || !name) return;
    try {
        module->module->add_external_symbol(name);
    } catch (...) {
    }
}

brass_compiled_module_t* brass_engine_compile_module(brass_engine_t* engine, const brass_module_t* module) {
    if (!engine || !module || !module->module) return nullptr;
    try {
        auto compiled = engine->engine.compile(*module->module);
        if (!compiled) return nullptr;
        auto* res = new brass_compiled_module_t();
        res->module = std::move(compiled);
        return res;
    } catch (...) {
        return nullptr;
    }
}

void brass_embed_compiled_module_destroy(brass_compiled_module_t* module) {
    try {
        delete module;
    } catch (...) {
    }
}

void* brass_embed_compiled_module_get_symbol(const brass_compiled_module_t* module, const char* name) {
    if (!module || !module->module || !name) return nullptr;
    try {
        return module->module->get_symbol_address(name);
    } catch (...) {
        return nullptr;
    }
}

int brass_compiled_module_patch_const32(brass_compiled_module_t* module, const char* site_name, int32_t new_val) {
    if (!module || !module->module || !site_name) return 0;
    try {
        return module->module->patch_constant(site_name, new_val) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int brass_compiled_module_patch_const64(brass_compiled_module_t* module, const char* site_name, int64_t new_val) {
    if (!module || !module->module || !site_name) return 0;
    try {
        return module->module->patch_constant(site_name, new_val) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int brass_compiled_module_patch_call(brass_compiled_module_t* module, const char* site_name, const void* new_target) {
    if (!module || !module->module || !site_name) return 0;
    try {
        return module->module->patch_call(site_name, new_target) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int brass_compiled_module_patch_call_target(brass_compiled_module_t* module, const char* site_name, const char* new_target_fn) {
    if (!module || !module->module || !site_name || !new_target_fn) return 0;
    try {
        return module->module->patch_call(site_name, std::string_view(new_target_fn)) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

size_t brass_compiled_module_walk_stack(
    const brass_compiled_module_t* module,
    uintptr_t rbp,
    uintptr_t return_ip,
    brass_c_root_visitor_fn visitor,
    void* user_data
) {
    if (!module || !module->module || !visitor) return 0;
    try {
        return module->module->walk_stack(rbp, return_ip, visitor, user_data);
    } catch (...) {
        return 0;
    }
}

brass_heap_t* brass_heap_create(size_t young_bytes) {
    try {
        brass::gc::HeapConfig config;
        if (young_bytes > 0) config.eden_bytes = young_bytes;
        // Raw addresses (tag 0) and NaN-boxed gcref values are references.
        config.reference_tags = {0, static_cast<uint16_t>(brass::HostValue::TAG_GCREF >> 48)};
        return new brass_heap_t(config);
    } catch (...) {
        return nullptr;
    }
}

void brass_heap_destroy(brass_heap_t* heap) {
    if (!heap) return;
    if (t_bound_heap == heap) {
        t_bound_heap = nullptr;
        brass::gc::Heap::set_current(nullptr);
    }
    delete heap;
}

brass_heap_t* brass_heap_bind(brass_heap_t* heap) {
    brass_heap_t* previous = t_bound_heap;
    t_bound_heap = heap;
    brass::gc::Heap::set_current(heap ? &heap->heap : nullptr);
    return previous;
}

uintptr_t brass_heap_allocate(brass_heap_t* heap, size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (!heap) return 0;
    try {
        return heap->heap.allocate_masked(size, pointer_mask, type_tag);
    } catch (...) {
        return 0;
    }
}

brass_value_t brass_heap_allocate_value(brass_heap_t* heap, size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    const uintptr_t object = brass_heap_allocate(heap, size, pointer_mask, type_tag);
    return object ? brass::HostValue::from_gcref(object).raw() : brass::HostValue::null_val().raw();
}

void brass_heap_collect(brass_heap_t* heap, int full) {
    if (!heap) return;
    try {
        heap->heap.collect(full ? brass::gc::CollectionKind::Full : brass::gc::CollectionKind::Minor);
    } catch (...) {
    }
}

void brass_heap_add_root(brass_heap_t* heap, uint64_t* slot) {
    if (!heap || !slot) return;
    try {
        heap->heap.add_root(slot);
    } catch (...) {
    }
}

void brass_heap_remove_root(brass_heap_t* heap, uint64_t* slot) {
    if (!heap || !slot) return;
    heap->heap.remove_root(slot);
}

void brass_heap_write_barrier(brass_heap_t* heap, uintptr_t object, uint64_t value) {
    if (heap) heap->heap.write_barrier_interior(object, value);
}

void brass_heap_set_stress(brass_heap_t* heap, int mode) {
    if (!heap) return;
    using brass::gc::StressMode;
    const StressMode modes[] = {StressMode::None, StressMode::Minor, StressMode::Full, StressMode::Alternate};
    heap->heap.set_stress(mode >= 0 && mode <= 3 ? modes[mode] : StressMode::None);
}

int brass_heap_get_stress(const brass_heap_t* heap) {
    if (!heap) return 0;
    switch (heap->heap.stress()) {
        case brass::gc::StressMode::Minor: return 1;
        case brass::gc::StressMode::Full: return 2;
        case brass::gc::StressMode::Alternate: return 3;
        default: return 0;
    }
}

size_t brass_heap_collection_count(const brass_heap_t* heap) {
    return heap ? static_cast<size_t>(heap->heap.collection_count()) : 0;
}

brass_value_t brass_value_from_f64(double d) {
    return brass::HostValue::from_f64(d).raw();
}

brass_value_t brass_value_from_i32(int32_t i) {
    return brass::HostValue::from_i32(i).raw();
}

brass_value_t brass_value_from_bool(int b) {
    return brass::HostValue::from_bool(b != 0).raw();
}

brass_value_t brass_value_null(void) {
    return brass::HostValue::null_val().raw();
}

brass_value_t brass_value_undefined(void) {
    return brass::HostValue::undefined_val().raw();
}

brass_value_t brass_value_from_gcref(uintptr_t ptr) {
    return brass::HostValue::from_gcref(ptr).raw();
}

brass_value_t brass_value_from_pointer(const void* ptr) {
    return brass::HostValue::from_pointer(ptr).raw();
}

brass_value_t brass_value_from_raw(uint64_t raw) {
    return brass::HostValue::from_raw(raw).raw();
}

int brass_value_is_f64(brass_value_t v) {
    return brass::HostValue(v).is_f64() ? 1 : 0;
}

int brass_value_is_i32(brass_value_t v) {
    return brass::HostValue(v).is_i32() ? 1 : 0;
}

int brass_value_is_bool(brass_value_t v) {
    return brass::HostValue(v).is_bool() ? 1 : 0;
}

int brass_value_is_null(brass_value_t v) {
    return brass::HostValue(v).is_null() ? 1 : 0;
}

int brass_value_is_undefined(brass_value_t v) {
    return brass::HostValue(v).is_undefined() ? 1 : 0;
}

int brass_value_is_gcref(brass_value_t v) {
    return brass::HostValue(v).is_gcref() ? 1 : 0;
}

int brass_value_is_pointer(brass_value_t v) {
    return brass::HostValue(v).is_pointer() ? 1 : 0;
}

double brass_value_as_f64(brass_value_t v) {
    return brass::HostValue(v).as_f64();
}

int32_t brass_value_as_i32(brass_value_t v) {
    return brass::HostValue(v).as_i32();
}

int brass_value_as_bool(brass_value_t v) {
    return brass::HostValue(v).as_bool() ? 1 : 0;
}

uintptr_t brass_value_as_gcref(brass_value_t v) {
    return brass::HostValue(v).as_gcref();
}

void* brass_value_as_pointer(brass_value_t v) {
    return brass::HostValue(v).as_pointer<void>();
}

uint64_t brass_value_raw(brass_value_t v) {
    return brass::HostValue(v).raw();
}

brass_value_t brass_value_update_gcref(brass_value_t v, uintptr_t new_ptr) {
    brass::HostValue val(v);
    val.update_gcref(new_ptr);
    return val.raw();
}

}
