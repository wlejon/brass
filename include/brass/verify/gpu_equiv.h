#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <brass/verify/gpu_equiv.hpp>
extern "C" {
#endif

struct BrassFunction_T;

int32_t brass_verify_spirv_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap);

int32_t brass_verify_ptx_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap);

int32_t brass_verify_spirv_ptx_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap);

#ifdef __cplusplus
}
#endif
