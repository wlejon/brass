#ifndef BRASS_C_API_H
#define BRASS_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct handles
typedef struct brass_engine_t brass_engine_t;
typedef struct brass_compiled_module_t brass_compiled_module_t;
typedef struct brass_module_t brass_module_t;
typedef struct brass_gc_t brass_gc_t;

// NaN-boxed 64-bit value representation
typedef uint64_t brass_value_t;

// Stack walk root visitor callback
typedef void (*brass_c_root_visitor_fn)(void** root_slot, void* user_data);

// Engine Lifecycle
brass_engine_t* brass_engine_create(void);
void brass_engine_destroy(brass_engine_t* engine);
void brass_engine_register_symbol(brass_engine_t* engine, const char* name, void* address);
void brass_engine_register_gc(brass_engine_t* engine, brass_gc_t* gc);

// Module Management
brass_module_t* brass_embed_module_create(const char* name);
void brass_embed_module_destroy(brass_module_t* module);
void brass_embed_module_add_external_symbol(brass_module_t* module, const char* name);

// Compilation & Lifecycle
brass_compiled_module_t* brass_engine_compile_module(brass_engine_t* engine, const brass_module_t* module);
void brass_embed_compiled_module_destroy(brass_compiled_module_t* module);

// Symbol Lookup & Entrypoint Retrieval
void* brass_embed_compiled_module_get_symbol(const brass_compiled_module_t* module, const char* name);

// Runtime Patching API
int brass_compiled_module_patch_const32(brass_compiled_module_t* module, const char* site_name, int32_t new_val);
int brass_compiled_module_patch_const64(brass_compiled_module_t* module, const char* site_name, int64_t new_val);
int brass_compiled_module_patch_call(brass_compiled_module_t* module, const char* site_name, const void* new_target);
int brass_compiled_module_patch_call_target(brass_compiled_module_t* module, const char* site_name, const char* new_target_fn);

// Stack Walking
size_t brass_compiled_module_walk_stack(
    const brass_compiled_module_t* module,
    uintptr_t rbp,
    uintptr_t return_ip,
    brass_c_root_visitor_fn visitor,
    void* user_data
);

// Host Cheney GC C Interface
brass_gc_t* brass_host_gc_create(size_t semispace_size);
void brass_host_gc_destroy(brass_gc_t* gc);
uintptr_t brass_host_gc_allocate(brass_gc_t* gc, size_t size, uint64_t pointer_mask, uint32_t type_tag);
brass_value_t brass_host_gc_allocate_value(brass_gc_t* gc, size_t size, uint64_t pointer_mask, uint32_t type_tag);
void brass_host_gc_collect(brass_gc_t* gc);
void brass_host_gc_safepoint(brass_gc_t* gc, uintptr_t rbp, uintptr_t return_ip);
void brass_host_gc_set_stress_mode(brass_gc_t* gc, int enable);
int brass_host_gc_get_stress_mode(const brass_gc_t* gc);
size_t brass_host_gc_collection_count(const brass_gc_t* gc);
void brass_host_gc_reset(brass_gc_t* gc);

// NaN-Box Value Construction
brass_value_t brass_value_from_f64(double d);
brass_value_t brass_value_from_i32(int32_t i);
brass_value_t brass_value_from_bool(int b);
brass_value_t brass_value_null(void);
brass_value_t brass_value_undefined(void);
brass_value_t brass_value_from_gcref(uintptr_t ptr);
brass_value_t brass_value_from_pointer(const void* ptr);
brass_value_t brass_value_from_raw(uint64_t raw);

// NaN-Box Value Type Checks
int brass_value_is_f64(brass_value_t v);
int brass_value_is_i32(brass_value_t v);
int brass_value_is_bool(brass_value_t v);
int brass_value_is_null(brass_value_t v);
int brass_value_is_undefined(brass_value_t v);
int brass_value_is_gcref(brass_value_t v);
int brass_value_is_pointer(brass_value_t v);

// NaN-Box Value Extraction
double brass_value_as_f64(brass_value_t v);
int32_t brass_value_as_i32(brass_value_t v);
int brass_value_as_bool(brass_value_t v);
uintptr_t brass_value_as_gcref(brass_value_t v);
void* brass_value_as_pointer(brass_value_t v);
uint64_t brass_value_raw(brass_value_t v);
brass_value_t brass_value_update_gcref(brass_value_t v, uintptr_t new_ptr);

#ifdef __cplusplus
}
#endif

#endif // BRASS_C_API_H
