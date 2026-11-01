#ifndef BRASS_PUBLIC_C_API_H
#define BRASS_PUBLIC_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version numbers */
#define BRASS_VERSION_MAJOR 1
#define BRASS_VERSION_MINOR 0
#define BRASS_VERSION_PATCH 0

/* Platform export macros (BRASS_API, BRASS_CALL) */
#include <brass/brass_export.h>

/* Object lifetime
 *
 * Every object returned by a *_create function or by
 * brass_kernel_jit_compile* must be released with its matching
 * *_destroy function. Destroy calls may come in any order:
 *
 *  - brass_context_destroy releases the caller's reference to the context.
 *    The context itself stays alive until every module, builder, JIT engine,
 *    kernel JIT and kernel function created from it has been destroyed, so
 *    destroying those afterwards is safe. Do not pass the context to any
 *    other function after destroying it.
 *  - Function, block and value handles are owned by the context and are
 *    never destroyed individually. They die with their module: once
 *    brass_module_destroy runs, handles from that module (and builders
 *    working on it) are invalid, and every call using them fails with an
 *    error in brass_context_get_last_error. Handles from other modules are
 *    unaffected.
 *  - Compiled modules are owned by their JIT engine and freed by
 *    brass_jit_destroy. brass_compiled_module_destroy frees the module's
 *    code; later lookups through that handle fail with an error.
 *
 * Builder and JIT calls report misuse (NULL or invalidated handles, handles
 * from another module, an unpositioned builder) by returning NULL or an
 * error status and setting the context's last error. */

/* Status codes */
typedef enum BrassStatus {
    BRASS_OK                       =  0,
    BRASS_ERR_GENERIC             = -1,
    BRASS_ERR_INVALID_ARGUMENT    = -2,
    BRASS_ERR_OUT_OF_MEMORY       = -3,
    BRASS_ERR_VERIFICATION_FAILED = -4,
    BRASS_ERR_COMPILE_FAILED      = -5,
    BRASS_ERR_TRANSLATION_FAILED  = -6,
    BRASS_ERR_IO                  = -7,
    BRASS_ERR_NOT_FOUND           = -8
} BrassStatus;

/* Comparison operators */
typedef enum BrassCmpOp {
    BRASS_CMP_EQ  = 0,
    BRASS_CMP_NE  = 1,
    BRASS_CMP_SLT = 2,
    BRASS_CMP_SLE = 3,
    BRASS_CMP_SGT = 4,
    BRASS_CMP_SGE = 5,
    BRASS_CMP_ULT = 6,
    BRASS_CMP_ULE = 7,
    BRASS_CMP_UGT = 8,
    BRASS_CMP_UGE = 9
} BrassCmpOp;

/* Lane kinds for SIMD vectors */
typedef enum BrassLaneKind {
    BRASS_LANE_F32 = 0,
    BRASS_LANE_F64 = 1,
    BRASS_LANE_I32 = 2,
    BRASS_LANE_I64 = 3
} BrassLaneKind;

/* Object file formats for AOT emission */
typedef enum BrassObjectFormat {
    BRASS_OBJECT_AUTO          = 0,
    BRASS_OBJECT_COFF          = 1,
    BRASS_OBJECT_ELF           = 2,
    BRASS_OBJECT_MACHO         = 3,
    BRASS_OBJECT_COFF_AARCH64  = 4,
    BRASS_OBJECT_ELF_AARCH64   = 5,
    BRASS_OBJECT_MACHO_AARCH64 = 6,
    BRASS_OBJECT_COFF_X64      = 7,
    BRASS_OBJECT_ELF_X64       = 8,
    BRASS_OBJECT_MACHO_X64     = 9
} BrassObjectFormat;

/* Opaque handles */
typedef struct BrassContext_T* BrassContext;
typedef struct BrassModule_T* BrassModule;
typedef struct BrassFunction_T* BrassFunction;
typedef struct BrassBlock_T* BrassBlock;
typedef struct BrassBuilder_T* BrassBuilder;
typedef struct BrassValue_T* BrassValue;
typedef struct BrassType_T* BrassType;
typedef struct BrassJitEngine_T* BrassJitEngine;
typedef struct BrassCompiledModule_T* BrassCompiledModule;
typedef struct BrassKernelOptions_T* BrassKernelOptions;
typedef struct BrassKernelJit_T* BrassKernelJit;
typedef struct BrassKernelFunction_T* BrassKernelFunction;

/* Options configuration struct */
typedef struct BrassOptions {
    int enable_optimizations;
    int allow_fp_reassociation;
    int target_format; /* BrassObjectFormat: 0=Auto, 1=COFF, 2=ELF, 3=Mach-O */
    size_t vector_width;
} BrassOptions;

/* ========================================================================= */
/* Options API                                                               */
/* ========================================================================= */
BRASS_API BrassOptions* BRASS_CALL brass_options_create(void);
BRASS_API void BRASS_CALL brass_options_destroy(BrassOptions* opts);
BRASS_API void BRASS_CALL brass_options_set_optimize(BrassOptions* opts, int enable);
BRASS_API void BRASS_CALL brass_options_set_target_format(BrassOptions* opts, int format);

/* ========================================================================= */
/* Types API                                                                 */
/* ========================================================================= */
BRASS_API BrassType BRASS_CALL brass_type_void(void);
BRASS_API BrassType BRASS_CALL brass_type_i32(void);
BRASS_API BrassType BRASS_CALL brass_type_i64(void);
BRASS_API BrassType BRASS_CALL brass_type_f32(void);
BRASS_API BrassType BRASS_CALL brass_type_f64(void);
BRASS_API BrassType BRASS_CALL brass_type_bool(void);
BRASS_API BrassType BRASS_CALL brass_type_ptr(void);
BRASS_API BrassType BRASS_CALL brass_type_dynamic(void);
BRASS_API BrassType BRASS_CALL brass_type_v128(uint8_t lane_kind);
BRASS_API BrassType BRASS_CALL brass_type_v256(uint8_t lane_kind);

/* ========================================================================= */
/* Context & Error Management                                                */
/* ========================================================================= */
BRASS_API BrassContext BRASS_CALL brass_context_create(void);
BRASS_API void BRASS_CALL brass_context_destroy(BrassContext ctx);
BRASS_API const char* BRASS_CALL brass_context_get_last_error(BrassContext ctx);
BRASS_API void BRASS_CALL brass_context_set_error(BrassContext ctx, const char* msg);

/* ========================================================================= */
/* Module Management                                                         */
/* ========================================================================= */
BRASS_API BrassModule BRASS_CALL brass_module_create(BrassContext ctx, const char* name);
BRASS_API void BRASS_CALL brass_module_destroy(BrassModule mod);
BRASS_API void BRASS_CALL brass_module_add_external_symbol(BrassModule mod, const char* name);
BRASS_API BrassStatus BRASS_CALL brass_module_verify(BrassModule mod, char* err_buf, size_t err_buf_len);
BRASS_API BrassStatus BRASS_CALL brass_module_print_mir(BrassModule mod, char** out_str);

/* ========================================================================= */
/* Function & CFG                                                            */
/* ========================================================================= */
BRASS_API BrassFunction BRASS_CALL brass_function_create(BrassModule mod, const char* name, BrassType ret_type, const BrassType* param_types, size_t param_count);
BRASS_API BrassBlock BRASS_CALL brass_function_append_block(BrassFunction fn, const char* name);
BRASS_API BrassValue BRASS_CALL brass_block_add_param(BrassBlock blk, BrassType type);
BRASS_API BrassValue BRASS_CALL brass_block_get_param(BrassBlock blk, size_t index);
BRASS_API BrassValue BRASS_CALL brass_function_get_param(BrassFunction fn, size_t index);
/* Promises that pointer parameter `index` is the only way the function
 * reaches the object it points to (C `restrict`). Loop parallelization and
 * tiling rely on it to tell arrays apart. */
BRASS_API BrassStatus BRASS_CALL brass_function_set_param_noalias(BrassFunction fn, size_t index, int noalias);

/* ========================================================================= */
/* IR Builder                                                                */
/* ========================================================================= */
BRASS_API BrassBuilder BRASS_CALL brass_builder_create(BrassContext ctx, BrassFunction fn);
BRASS_API void BRASS_CALL brass_builder_destroy(BrassBuilder b);
BRASS_API void BRASS_CALL brass_builder_position_at_end(BrassBuilder b, BrassBlock blk);

/* Constants */
BRASS_API BrassValue BRASS_CALL brass_build_iconst_i32(BrassBuilder b, int32_t val);
BRASS_API BrassValue BRASS_CALL brass_build_iconst_i64(BrassBuilder b, int64_t val);
BRASS_API BrassValue BRASS_CALL brass_build_fconst_f32(BrassBuilder b, float val);
BRASS_API BrassValue BRASS_CALL brass_build_fconst_f64(BrassBuilder b, double val);
BRASS_API BrassValue BRASS_CALL brass_build_bconst(BrassBuilder b, int val);

/* Arithmetic & Bitwise */
BRASS_API BrassValue BRASS_CALL brass_build_add(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_sub(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_mul(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_sdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_udiv(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_fdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_and(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_or(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_xor(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_shl(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_shr(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_sar(BrassBuilder b, BrassValue lhs, BrassValue rhs);

/* Comparisons */
BRASS_API BrassValue BRASS_CALL brass_build_cmp(BrassBuilder b, BrassCmpOp op, BrassValue lhs, BrassValue rhs);

/* Control Flow */
BRASS_API BrassStatus BRASS_CALL brass_build_br(BrassBuilder b, BrassBlock target, const BrassValue* args, size_t arg_count);
BRASS_API BrassStatus BRASS_CALL brass_build_br_if(BrassBuilder b, BrassValue cond, BrassBlock true_target, const BrassValue* true_args, size_t true_arg_count, BrassBlock false_target, const BrassValue* false_args, size_t false_arg_count);
/* val == NULL emits `ret void` and is accepted only in a void function; a
 * non-NULL val must have the function's return type. */
BRASS_API BrassStatus BRASS_CALL brass_build_ret(BrassBuilder b, BrassValue val);
BRASS_API BrassStatus BRASS_CALL brass_build_unreachable(BrassBuilder b);

/* Calls & Function Pointers */
/* A call with a void return type is emitted and returns NULL. */
BRASS_API BrassValue BRASS_CALL brass_build_call(BrassBuilder b, const char* callee, BrassType return_type, const BrassValue* args, size_t arg_count);
BRASS_API BrassValue BRASS_CALL brass_build_func_addr(BrassBuilder b, const char* name);

/* Memory */
BRASS_API BrassValue BRASS_CALL brass_build_load(BrassBuilder b, BrassType type, BrassValue base, int32_t offset);
BRASS_API BrassStatus BRASS_CALL brass_build_store(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val);
BRASS_API BrassValue BRASS_CALL brass_build_load_indexed(BrassBuilder b, BrassType type, BrassValue base, BrassValue index, uint8_t scale, int32_t offset);
BRASS_API BrassStatus BRASS_CALL brass_build_store_indexed(BrassBuilder b, BrassType type, BrassValue base, BrassValue index, uint8_t scale, int32_t offset, BrassValue val);

/* Vector (SIMD) & FMA */
BRASS_API BrassValue BRASS_CALL brass_build_fma(BrassBuilder b, BrassValue a, BrassValue b_val, BrassValue c);
BRASS_API BrassValue BRASS_CALL brass_build_vfma(BrassBuilder b, BrassValue a, BrassValue b_val, BrassValue c);
BRASS_API BrassValue BRASS_CALL brass_build_vload(BrassBuilder b, BrassType type, BrassValue base, int32_t offset);
BRASS_API BrassStatus BRASS_CALL brass_build_vstore(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val);
BRASS_API BrassValue BRASS_CALL brass_build_vadd(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vsub(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vmul(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vmin(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vmax(BrassBuilder b, BrassValue lhs, BrassValue rhs);
BRASS_API BrassValue BRASS_CALL brass_build_vbroadcast(BrassBuilder b, BrassType vec_type, BrassValue scalar_val);
BRASS_API BrassValue BRASS_CALL brass_build_vextract_lane(BrassBuilder b, BrassValue vec_val, uint32_t lane);
BRASS_API BrassValue BRASS_CALL brass_build_vinsert_lane(BrassBuilder b, BrassValue vec_val, BrassValue scalar_val, uint32_t lane);
BRASS_API BrassValue BRASS_CALL brass_build_vzero(BrassBuilder b, BrassType vec_type);

/* ========================================================================= */
/* JIT Execution Engine                                                      */
/* ========================================================================= */
/* Each compiled module has its own code and symbols. A function name may be
 * defined by only one live compiled module of an engine: compiling a module
 * that defines a name already defined by a live compiled module (including
 * compiling the same module twice) fails. Destroying a compiled module
 * releases its names. brass_compiled_module_get_symbol looks only in its own
 * module; brass_jit_get_function_address searches all live modules.
 * Symbols registered with brass_jit_register_symbol apply to modules
 * compiled afterwards. */
BRASS_API BrassJitEngine BRASS_CALL brass_jit_create(BrassContext ctx);
BRASS_API void BRASS_CALL brass_jit_destroy(BrassJitEngine jit);
BRASS_API BrassStatus BRASS_CALL brass_jit_register_symbol(BrassJitEngine jit, const char* name, void* address);
BRASS_API BrassCompiledModule BRASS_CALL brass_jit_compile_module(BrassJitEngine jit, BrassModule mod);
BRASS_API void* BRASS_CALL brass_jit_get_function_address(BrassJitEngine jit, const char* name);
BRASS_API void BRASS_CALL brass_compiled_module_destroy(BrassCompiledModule mod);
BRASS_API void* BRASS_CALL brass_compiled_module_get_symbol(BrassCompiledModule mod, const char* name);

/* ========================================================================= */
/* Kernel JIT & Polyhedral Loop Parallelization                              */
/* ========================================================================= */
typedef struct BrassLoopAnalysis {
    int is_parallelizable;
    int is_doall;
    int is_reduction;
    int has_const_trip_count;
    uint64_t const_trip_count;
    char rejection_reason[256];
    size_t dependence_count;
} BrassLoopAnalysis;

/* Kernel Options */
BRASS_API BrassKernelOptions BRASS_CALL brass_kernel_options_create(void);
BRASS_API void BRASS_CALL brass_kernel_options_destroy(BrassKernelOptions opts);
BRASS_API void BRASS_CALL brass_kernel_options_set_optimize(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_avx2(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_fma(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_fp_reassociation(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_vectorize(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_parallel(BrassKernelOptions opts, int enable);
BRASS_API void BRASS_CALL brass_kernel_options_set_parallel_threshold(BrassKernelOptions opts, uint64_t threshold);
BRASS_API void BRASS_CALL brass_kernel_options_set_parallel_workers(BrassKernelOptions opts, uint32_t workers);
BRASS_API void BRASS_CALL brass_kernel_options_set_unroll_factor(BrassKernelOptions opts, size_t factor);

/* Kernel JIT Engine */
BRASS_API BrassKernelJit BRASS_CALL brass_kernel_jit_create(BrassContext ctx, const BrassKernelOptions opts);
BRASS_API void BRASS_CALL brass_kernel_jit_destroy(BrassKernelJit kj);
BRASS_API void BRASS_CALL brass_kernel_jit_set_parallel_workers(BrassKernelJit kj, uint32_t workers);
BRASS_API uint32_t BRASS_CALL brass_kernel_jit_get_parallel_workers(BrassKernelJit kj);
BRASS_API void BRASS_CALL brass_kernel_jit_set_parallel_threshold(BrassKernelJit kj, uint64_t threshold);
BRASS_API uint64_t BRASS_CALL brass_kernel_jit_get_parallel_threshold(BrassKernelJit kj);
BRASS_API BrassStatus BRASS_CALL brass_kernel_jit_register_symbol(BrassKernelJit kj, const char* name, void* address);
BRASS_API BrassKernelFunction BRASS_CALL brass_kernel_jit_compile(BrassKernelJit kj, BrassModule mod, const char* entry_name);
BRASS_API BrassKernelFunction BRASS_CALL brass_kernel_jit_compile_function(BrassKernelJit kj, BrassFunction fn);

/* Compiled Kernel Function */
BRASS_API void* BRASS_CALL brass_kernel_function_get_address(BrassKernelFunction kfn);
BRASS_API const char* BRASS_CALL brass_kernel_function_get_name(BrassKernelFunction kfn);
BRASS_API size_t BRASS_CALL brass_kernel_function_get_code_size(BrassKernelFunction kfn);
BRASS_API void BRASS_CALL brass_kernel_function_destroy(BrassKernelFunction kfn);

/* Polyhedral Loop Dependence Analysis */
BRASS_API BrassStatus BRASS_CALL brass_kernel_jit_analyze_loops(
    BrassKernelJit kj,
    BrassFunction fn,
    BrassLoopAnalysis* out_analyses,
    size_t max_analyses,
    size_t* out_analysis_count
);
BRASS_API BrassStatus BRASS_CALL brass_kernel_jit_auto_parallelize(
    BrassKernelJit kj,
    BrassFunction fn,
    int* out_changed
);

/* ========================================================================= */
/* PTX CUDA Target Emitter                                                   */
/* ========================================================================= */
BRASS_API BrassStatus BRASS_CALL brass_kernel_emit_ptx(
    BrassFunction fn,
    const char* sm_arch,
    char** out_ptx,
    size_t* out_len
);
BRASS_API void BRASS_CALL brass_free_string(char* str);

/* ========================================================================= */
/* GPU Execution (CUDA driver loaded dynamically at runtime)                  */
/* ========================================================================= */

typedef struct BrassGpuModule_T* BrassGpuModule;
typedef struct BrassGpuBuffer_T* BrassGpuBuffer;

/* Returns 1 when a CUDA driver + device + context is usable, else 0. */
BRASS_API int BRASS_CALL brass_gpu_available(void);
BRASS_API int BRASS_CALL brass_gpu_device_count(void);
BRASS_API BrassStatus BRASS_CALL brass_gpu_device_name(int index, char* out, size_t out_len);
BRASS_API BrassStatus BRASS_CALL brass_gpu_last_error(char* out, size_t out_len);

/* Device memory. Offsets are byte offsets into the allocation. */
BRASS_API BrassGpuBuffer BRASS_CALL brass_gpu_buffer_alloc(size_t bytes);
BRASS_API void BRASS_CALL brass_gpu_buffer_destroy(BrassGpuBuffer buf);
BRASS_API BrassStatus BRASS_CALL brass_gpu_buffer_upload(BrassGpuBuffer buf, const void* host, size_t bytes, size_t offset);
BRASS_API BrassStatus BRASS_CALL brass_gpu_buffer_download(BrassGpuBuffer buf, void* host, size_t bytes, size_t offset);
BRASS_API void* BRASS_CALL brass_gpu_buffer_device_ptr(BrassGpuBuffer buf);

/* PTX module JIT-compiled by the driver. */
BRASS_API BrassStatus BRASS_CALL brass_gpu_module_load(const char* ptx, BrassGpuModule* out_module);
BRASS_API void BRASS_CALL brass_gpu_module_destroy(BrassGpuModule mod);
BRASS_API int BRASS_CALL brass_gpu_module_has_function(BrassGpuModule mod, const char* entry);
BRASS_API BrassStatus BRASS_CALL brass_gpu_module_launch(
    BrassGpuModule mod,
    const char* entry,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    void** kernel_args, size_t shared_bytes);
BRASS_API BrassStatus BRASS_CALL brass_gpu_synchronize(void);

/* ========================================================================= */
/* AOT Binary Compilation                                                    */
/* ========================================================================= */
BRASS_API BrassStatus BRASS_CALL brass_compile_to_object(BrassModule mod, int target_format, void** out_bytes, size_t* out_size);
BRASS_API BrassStatus BRASS_CALL brass_compile_to_shared_lib(BrassModule mod, const char* output_path, const BrassOptions* opts);
BRASS_API void BRASS_CALL brass_free_buffer(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* BRASS_PUBLIC_C_API_H */
