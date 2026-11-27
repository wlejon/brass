#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>

namespace brass {

struct GpuOptOptions {
    int opt_level = 2;
    bool enable_fold_constants = true;
    bool enable_cse = true;
    bool enable_gvn = true;
    bool enable_gvn_rle = true;
    bool enable_gvn_dse = true;
    bool enable_smt_guard_elim = true;
    bool fold_branches = false; // preserve uniform control flow for GPU barriers
    bool enable_superopt = true;
    bool enable_gpu_superopt = true;
    bool enable_licm = true;
    bool enable_fdiv_pow2 = true;
    bool enable_fma = false; // FP contraction (a*b+c -> fma): opt-in (changes IEEE-754 rounding)
    bool enable_small_loop_unroll = false;
    uint32_t max_unroll_trip_count = 8;
    bool enable_load_coalesce = false;
    bool assume_disjoint_buffers = true;
    bool enable_scalar_cleanup = true;
    uint32_t max_sat_conflicts = 1000;
    uint32_t gpu_max_ctaid = 65535;
    uint32_t gpu_max_global_tid = 0x3FFFFFFu;
    uint32_t gpu_subgroup_size = 32;
};

// Shared barrier-safe GPU MIR optimization pipeline (for SPIR-V / PTX kernels).
// Avoids CPU loop vectorization/unrolling/f64-demotion/auto-parallelization and
// defaults fold_branches=false so uniform barrier control flow stays intact.
bool optimize_gpu_function(Function& fn, Module& mod, const GpuOptOptions& opts = {});

/// Barrier-safe full unrolling of small constant-trip counted GPU loops.
bool unroll_small_gpu_loops(Function& fn, uint32_t max_trip_count = 8);

/// Coalesces 4 contiguous 32-bit scalar loads/stores into 128-bit `vload`/`vstore`.
bool coalesce_gpu_v4_accesses(Function& fn, bool assume_disjoint_buffers = true);

} // namespace brass
