#include <brass/embedding/brass_c_api.h>
#include <brass/embedding/embedding.hpp>
#include <brass/embedding/host_gc.hpp>
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

struct brass_gc_t {
    brass::HostGC gc;
    explicit brass_gc_t(size_t semispace_size) : gc(semispace_size) {}
};

extern "C" {

brass_engine_t* brass_engine_create(void) {
    try {
        return new brass_engine_t();
    } catch (...) {
        return nullptr;
    }
}

void brass_engine_destroy(brass_engine_t* engine) {
    delete engine;
}

void brass_engine_register_symbol(brass_engine_t* engine, const char* name, void* address) {
    if (!engine || !name) return;
    engine->engine.register_external_symbol(name, address);
}

void brass_engine_register_gc(brass_engine_t* engine, brass_gc_t* gc) {
    if (!engine) return;
    engine->engine.register_host_gc(gc ? &gc->gc : nullptr);
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
    delete module;
}

void brass_embed_module_add_external_symbol(brass_module_t* module, const char* name) {
    if (!module || !module->module || !name) return;
    module->module->add_external_symbol(name);
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
    delete module;
}

void* brass_embed_compiled_module_get_symbol(const brass_compiled_module_t* module, const char* name) {
    if (!module || !module->module || !name) return nullptr;
    return module->module->get_symbol_address(name);
}

int brass_compiled_module_patch_const32(brass_compiled_module_t* module, const char* site_name, int32_t new_val) {
    if (!module || !module->module || !site_name) return 0;
    return module->module->patch_constant(site_name, new_val) ? 1 : 0;
}

int brass_compiled_module_patch_const64(brass_compiled_module_t* module, const char* site_name, int64_t new_val) {
    if (!module || !module->module || !site_name) return 0;
    return module->module->patch_constant(site_name, new_val) ? 1 : 0;
}

int brass_compiled_module_patch_call(brass_compiled_module_t* module, const char* site_name, const void* new_target) {
    if (!module || !module->module || !site_name) return 0;
    return module->module->patch_call(site_name, new_target) ? 1 : 0;
}

int brass_compiled_module_patch_call_target(brass_compiled_module_t* module, const char* site_name, const char* new_target_fn) {
    if (!module || !module->module || !site_name || !new_target_fn) return 0;
    return module->module->patch_call(site_name, std::string_view(new_target_fn)) ? 1 : 0;
}

size_t brass_compiled_module_walk_stack(
    const brass_compiled_module_t* module,
    uintptr_t rbp,
    uintptr_t return_ip,
    brass_c_root_visitor_fn visitor,
    void* user_data
) {
    if (!module || !module->module || !visitor) return 0;
    return module->module->walk_stack(rbp, return_ip, visitor, user_data);
}

brass_gc_t* brass_host_gc_create(size_t semispace_size) {
    try {
        return new brass_gc_t(semispace_size > 0 ? semispace_size : brass::HostGC::DEFAULT_SEMISPACE_SIZE);
    } catch (...) {
        return nullptr;
    }
}

void brass_host_gc_destroy(brass_gc_t* gc) {
    delete gc;
}

uintptr_t brass_host_gc_allocate(brass_gc_t* gc, size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (!gc) return 0;
    try {
        return gc->gc.allocate(size, pointer_mask, type_tag);
    } catch (...) {
        return 0;
    }
}

brass_value_t brass_host_gc_allocate_value(brass_gc_t* gc, size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    if (!gc) return brass::HostValue::null_val().raw();
    try {
        return gc->gc.allocate_value(size, pointer_mask, type_tag).raw();
    } catch (...) {
        return brass::HostValue::null_val().raw();
    }
}

void brass_host_gc_collect(brass_gc_t* gc) {
    if (!gc) return;
    gc->gc.collect();
}

void brass_host_gc_safepoint(brass_gc_t* gc, uintptr_t rbp, uintptr_t return_ip) {
    if (!gc) return;
    gc->gc.safepoint(rbp, return_ip);
}

void brass_host_gc_set_stress_mode(brass_gc_t* gc, int enable) {
    if (!gc) return;
    gc->gc.set_stress_mode(enable != 0);
}

int brass_host_gc_get_stress_mode(const brass_gc_t* gc) {
    if (!gc) return 0;
    return gc->gc.stress_mode() ? 1 : 0;
}

size_t brass_host_gc_collection_count(const brass_gc_t* gc) {
    if (!gc) return 0;
    return gc->gc.collection_count();
}

void brass_host_gc_reset(brass_gc_t* gc) {
    if (!gc) return;
    gc->gc.reset();
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
