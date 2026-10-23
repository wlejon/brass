#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <brass/verify/gpu_kernel_prover.hpp>
extern "C" {
#endif

struct BrassFunction_T;

int32_t brass_verify_gpu_memory_safety(
    const struct BrassFunction_T* mir_fn,
    uint32_t local_size_x,
    uint32_t grid_size_x,
    uint64_t buffer0_static_bytes,
    char* err_buf,
    size_t err_buf_cap);

int32_t brass_verify_gpu_barrier_uniformity(
    const struct BrassFunction_T* mir_fn,
    uint32_t local_size_x,
    uint32_t subgroup_size,
    char* err_buf,
    size_t err_buf_cap);

int32_t brass_prove_tile_interior_bounds(
    uint32_t tile_base_min,
    uint32_t tile_base_max,
    uint32_t tile_extent,
    uint32_t dim_size,
    char* err_buf,
    size_t err_buf_cap);

#ifdef __cplusplus
}
#endif
