#ifndef BRASS_C_API_H
#define BRASS_C_API_H

#include <stdint.h>
#include <stddef.h>

#include <brass/brass_export.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct handles
typedef struct brass_engine_t brass_engine_t;
typedef struct brass_compiled_module_t brass_compiled_module_t;
typedef struct brass_module_t brass_module_t;
typedef struct brass_heap_t brass_heap_t;

// NaN-boxed 64-bit value representation
typedef uint64_t brass_value_t;

// Stack walk root visitor callback
typedef void (*brass_c_root_visitor_fn)(void** root_slot, void* user_data);

// Engine Lifecycle
BRASS_API brass_engine_t* BRASS_CALL brass_engine_create(void);
BRASS_API void BRASS_CALL brass_engine_destroy(brass_engine_t* engine);
BRASS_API void BRASS_CALL brass_engine_register_symbol(brass_engine_t* engine, const char* name, void* address);

// Module Management
BRASS_API brass_module_t* BRASS_CALL brass_embed_module_create(const char* name);
BRASS_API void BRASS_CALL brass_embed_module_destroy(brass_module_t* module);
BRASS_API void BRASS_CALL brass_embed_module_add_external_symbol(brass_module_t* module, const char* name);

// Compilation & Lifecycle
BRASS_API brass_compiled_module_t* BRASS_CALL brass_engine_compile_module(brass_engine_t* engine, const brass_module_t* module);
BRASS_API void BRASS_CALL brass_embed_compiled_module_destroy(brass_compiled_module_t* module);

// Symbol Lookup & Entrypoint Retrieval
BRASS_API void* BRASS_CALL brass_embed_compiled_module_get_symbol(const brass_compiled_module_t* module, const char* name);

// Runtime Patching API
BRASS_API int BRASS_CALL brass_compiled_module_patch_const32(brass_compiled_module_t* module, const char* site_name, int32_t new_val);
BRASS_API int BRASS_CALL brass_compiled_module_patch_const64(brass_compiled_module_t* module, const char* site_name, int64_t new_val);
BRASS_API int BRASS_CALL brass_compiled_module_patch_call(brass_compiled_module_t* module, const char* site_name, const void* new_target);
BRASS_API int BRASS_CALL brass_compiled_module_patch_call_target(brass_compiled_module_t* module, const char* site_name, const char* new_target_fn);

// Stack Walking
BRASS_API size_t BRASS_CALL brass_compiled_module_walk_stack(
    const brass_compiled_module_t* module,
    uintptr_t rbp,
    uintptr_t return_ip,
    brass_c_root_visitor_fn visitor,
    void* user_data
);

// The garbage-collected heap (brass::gc::Heap; docs/gc_contract.md).
// Compiled code allocates from the calling thread's bound heap. A word an
// object's pointer mask marks is a reference when it is a raw address or a
// NaN-boxed gcref value (brass_value_from_gcref).
//
// A heap with an eden of `young_bytes` (0: the default size).
BRASS_API brass_heap_t* BRASS_CALL brass_heap_create(size_t young_bytes);
BRASS_API void BRASS_CALL brass_heap_destroy(brass_heap_t* heap);
// Binds `heap` (NULL: none) as the calling thread's heap, returning the one
// bound before. Compiled code called on the thread allocates from it.
BRASS_API brass_heap_t* BRASS_CALL brass_heap_bind(brass_heap_t* heap);
// A zeroed object; 0 when the heap cannot hold it. May collect: every
// reference the caller holds across the call must be a root.
BRASS_API uintptr_t BRASS_CALL brass_heap_allocate(brass_heap_t* heap, size_t size, uint64_t pointer_mask, uint32_t type_tag);
BRASS_API brass_value_t BRASS_CALL brass_heap_allocate_value(brass_heap_t* heap, size_t size, uint64_t pointer_mask, uint32_t type_tag);
// A full collection (full != 0) or a young-generation one.
BRASS_API void BRASS_CALL brass_heap_collect(brass_heap_t* heap, int full);
// A slot outside the heap whose word every collection visits and updates.
BRASS_API void BRASS_CALL brass_heap_add_root(brass_heap_t* heap, uint64_t* slot);
BRASS_API void BRASS_CALL brass_heap_remove_root(brass_heap_t* heap, uint64_t* slot);
// The barrier for a store of `value` into the object at `object`.
BRASS_API void BRASS_CALL brass_heap_write_barrier(brass_heap_t* heap, uintptr_t object, uint64_t value);
// Stress mode: 0 off, 1 a young collection at every allocation, 2 a full
// one, 3 alternating.
BRASS_API void BRASS_CALL brass_heap_set_stress(brass_heap_t* heap, int mode);
BRASS_API int BRASS_CALL brass_heap_get_stress(const brass_heap_t* heap);
BRASS_API size_t BRASS_CALL brass_heap_collection_count(const brass_heap_t* heap);

// NaN-Box Value Construction
BRASS_API brass_value_t BRASS_CALL brass_value_from_f64(double d);
BRASS_API brass_value_t BRASS_CALL brass_value_from_i32(int32_t i);
BRASS_API brass_value_t BRASS_CALL brass_value_from_bool(int b);
BRASS_API brass_value_t BRASS_CALL brass_value_null(void);
BRASS_API brass_value_t BRASS_CALL brass_value_undefined(void);
BRASS_API brass_value_t BRASS_CALL brass_value_from_gcref(uintptr_t ptr);
BRASS_API brass_value_t BRASS_CALL brass_value_from_pointer(const void* ptr);
BRASS_API brass_value_t BRASS_CALL brass_value_from_raw(uint64_t raw);

// NaN-Box Value Type Checks
BRASS_API int BRASS_CALL brass_value_is_f64(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_i32(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_bool(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_null(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_undefined(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_gcref(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_is_pointer(brass_value_t v);

// NaN-Box Value Extraction
BRASS_API double BRASS_CALL brass_value_as_f64(brass_value_t v);
BRASS_API int32_t BRASS_CALL brass_value_as_i32(brass_value_t v);
BRASS_API int BRASS_CALL brass_value_as_bool(brass_value_t v);
BRASS_API uintptr_t BRASS_CALL brass_value_as_gcref(brass_value_t v);
BRASS_API void* BRASS_CALL brass_value_as_pointer(brass_value_t v);
BRASS_API uint64_t BRASS_CALL brass_value_raw(brass_value_t v);
BRASS_API brass_value_t BRASS_CALL brass_value_update_gcref(brass_value_t v, uintptr_t new_ptr);

#ifdef __cplusplus
}
#endif

#endif // BRASS_C_API_H
