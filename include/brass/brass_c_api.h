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

/* Platform export macros */
#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(BRASS_BUILD_SHARED)
    #define BRASS_API __declspec(dllexport)
  #elif defined(BRASS_SHARED)
    #define BRASS_API __declspec(dllimport)
  #else
    #define BRASS_API
  #endif
  #define BRASS_CALL __cdecl
#else
  #if defined(BRASS_BUILD_SHARED)
    #define BRASS_API __attribute__((visibility("default")))
  #else
    #define BRASS_API
  #endif
  #define BRASS_CALL
#endif

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
    BRASS_OBJECT_AUTO  = 0,
    BRASS_OBJECT_COFF  = 1,
    BRASS_OBJECT_ELF   = 2,
    BRASS_OBJECT_MACHO = 3
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
BRASS_API BrassStatus BRASS_CALL brass_build_ret(BrassBuilder b, BrassValue val);
BRASS_API BrassStatus BRASS_CALL brass_build_unreachable(BrassBuilder b);

/* Calls & Function Pointers */
BRASS_API BrassValue BRASS_CALL brass_build_call(BrassBuilder b, const char* callee, BrassType return_type, const BrassValue* args, size_t arg_count);
BRASS_API BrassValue BRASS_CALL brass_build_func_addr(BrassBuilder b, const char* name);

/* Memory */
BRASS_API BrassValue BRASS_CALL brass_build_load(BrassBuilder b, BrassType type, BrassValue base, int32_t offset);
BRASS_API BrassStatus BRASS_CALL brass_build_store(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val);

/* ========================================================================= */
/* Bronze IL Translation Bridge                                              */
/* ========================================================================= */
BRASS_API BrassStatus BRASS_CALL brass_translate_bronze_il(BrassContext ctx, const char* il_text, size_t len, const BrassOptions* opts, BrassModule* out_mod);

/* ========================================================================= */
/* JIT Execution Engine                                                      */
/* ========================================================================= */
BRASS_API BrassJitEngine BRASS_CALL brass_jit_create(BrassContext ctx);
BRASS_API void BRASS_CALL brass_jit_destroy(BrassJitEngine jit);
BRASS_API BrassStatus BRASS_CALL brass_jit_register_symbol(BrassJitEngine jit, const char* name, void* address);
BRASS_API BrassCompiledModule BRASS_CALL brass_jit_compile_module(BrassJitEngine jit, BrassModule mod);
BRASS_API void* BRASS_CALL brass_jit_get_function_address(BrassJitEngine jit, const char* name);
BRASS_API void BRASS_CALL brass_compiled_module_destroy(BrassCompiledModule mod);
BRASS_API void* BRASS_CALL brass_compiled_module_get_symbol(BrassCompiledModule mod, const char* name);

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
