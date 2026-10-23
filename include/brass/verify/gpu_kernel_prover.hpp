#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace brass::verify {

struct GpuBufferSpec {
    uint32_t param_index = 0;
    uint64_t static_byte_size = 0;      // > 0 when buffer has a known static byte capacity
    int32_t length_param_index = -1;    // >= 0 when capacity is param[length_param_index] * elem_bytes
    uint32_t elem_bytes = 4;            // element byte size for length_param_index scaling
    uint32_t base_alignment = 16;       // guaranteed alignment of the buffer base pointer
};

struct GpuProverOptions {
    uint32_t local_size_x = 256;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
    uint32_t grid_size_x = 256;
    uint32_t grid_size_y = 1;
    uint32_t grid_size_z = 1;
    uint32_t subgroup_size = 32;
    uint32_t loop_unroll_bound = 4;
    uint32_t max_sat_conflicts = 10000;
    std::vector<GpuBufferSpec> buffers;
};

struct GpuAccessViolation {
    bool is_shared = false;             // true = workgroup shared memory, false = global buffer
    bool is_alignment = false;          // true = natural alignment violation, false = out-of-bounds
    uint32_t region_index = 0;          // shared array index or kernel parameter index
    uint32_t tid_x = 0;
    uint32_t tid_y = 0;
    uint32_t tid_z = 0;
    uint32_t ctaid_x = 0;
    uint32_t ctaid_y = 0;
    uint32_t ctaid_z = 0;
    int64_t offending_index_or_offset = 0;
    uint64_t bound_limit = 0;
    std::string description;
};

struct GpuMemorySafetyResult {
    EquivStatus status = EquivStatus::ProvenEquivalent;
    uint32_t shared_accesses_proven = 0;
    uint32_t global_accesses_proven = 0;
    uint32_t alignments_proven = 0;
    std::vector<GpuAccessViolation> violations;
    std::string diagnostic;

    constexpr bool is_safe() const noexcept {
        return status == EquivStatus::ProvenEquivalent && violations.empty();
    }
    constexpr bool is_counterexample() const noexcept {
        return status == EquivStatus::Counterexample;
    }
};

struct BarrierDivergence {
    bool is_subgroup_collective = false; // false = workgroup barrier, true = subgroup shuffle/collective
    uint32_t inst_index = 0;
    uint32_t local_id_a[3] = {0, 0, 0};
    uint32_t local_id_b[3] = {0, 0, 0};
    std::string description;
};

struct BarrierUniformityResult {
    EquivStatus status = EquivStatus::ProvenEquivalent;
    uint32_t barriers_proven = 0;
    uint32_t collectives_proven = 0;
    std::vector<BarrierDivergence> divergences;
    std::string diagnostic;

    constexpr bool is_uniform() const noexcept {
        return status == EquivStatus::ProvenEquivalent && divergences.empty();
    }
    constexpr bool is_counterexample() const noexcept {
        return status == EquivStatus::Counterexample;
    }
};

struct TileBoundsProofResult {
    EquivStatus status = EquivStatus::ProvenEquivalent;
    uint32_t counterexample_tile_base = 0;
    uint32_t counterexample_local_offset = 0;
    std::string diagnostic;

    constexpr bool is_proven() const noexcept {
        return status == EquivStatus::ProvenEquivalent;
    }
};

/// Prove workgroup shared-memory bounds (`0 <= idx < count`), global buffer bounds
/// (`0 <= byte_offset` and `byte_offset + access_bytes <= buffer_byte_size`), and
/// natural alignment (`byte_offset % access_align == 0`) on a MIR GPU kernel.
GpuMemorySafetyResult verify_gpu_memory_safety(
    const Function& mir_fn,
    const GpuProverOptions& opts = {});

/// Prove workgroup shared-memory (`StorageClassWorkgroup`) and global buffer
/// (`PhysicalStorageBuffer`) bounds + natural alignment on a lowered `spirv::Module`.
GpuMemorySafetyResult verify_spirv_memory_safety(
    const spirv::Module& spv_mod,
    const GpuProverOptions& opts = {});

/// Prove workgroup shared-memory (`.shared`) and global buffer (`.global`)
/// bounds + natural alignment on a lowered `ptx::Function`.
GpuMemorySafetyResult verify_ptx_memory_safety(
    const ptx::Function& ptx_fn,
    const GpuProverOptions& opts = {});

/// Prove workgroup barrier (`bar.sync` / `OpControlBarrier`) and subgroup collective
/// (`shfl.sync` / `OpGroupNonUniform*`) control-flow uniformity using a 2-thread
/// relational encoding (`path_cond(local_id_A) != path_cond(local_id_B)`).
BarrierUniformityResult verify_gpu_barrier_uniformity(
    const Function& mir_fn,
    const GpuProverOptions& opts = {});

BarrierUniformityResult verify_spirv_barrier_uniformity(
    const spirv::Module& spv_mod,
    const GpuProverOptions& opts = {});

BarrierUniformityResult verify_ptx_barrier_uniformity(
    const ptx::Function& ptx_fn,
    const GpuProverOptions& opts = {});

/// Prove via `BVSolver` that for all `tile_base` in `[tile_base_min, tile_base_max]` and
/// all `local_offset` in `[0, tile_extent)`, `0 <= tile_base + local_offset < dim_size`
/// without 32-bit overflow.
TileBoundsProofResult prove_tile_interior_bounds(
    uint32_t tile_base_min,
    uint32_t tile_base_max,
    uint32_t tile_extent,
    uint32_t dim_size,
    std::string* diagnostic_out = nullptr);

/// Eliminate per-element `ult(tile_base + local_offset, dim_size)` / `slt` bounds checks
/// inside interior-tile blocks where `BVSolver` proves `tile_base + local_offset < dim_size`
/// under dominating tile-interior guards (`tile_base + tile_extent <= dim_size` and
/// `0 <= local_offset < tile_extent`). Returns the number of bounds comparisons eliminated.
uint32_t eliminate_interior_tile_bounds(
    Function& fn,
    Module& mod,
    uint32_t tile_extent,
    uint32_t dim_size);

} // namespace brass::verify

namespace brass {
using verify::BarrierDivergence;
using verify::BarrierUniformityResult;
using verify::GpuAccessViolation;
using verify::GpuBufferSpec;
using verify::GpuMemorySafetyResult;
using verify::GpuProverOptions;
using verify::TileBoundsProofResult;
using verify::eliminate_interior_tile_bounds;
using verify::prove_tile_interior_bounds;
using verify::verify_gpu_barrier_uniformity;
using verify::verify_gpu_memory_safety;
using verify::verify_ptx_barrier_uniformity;
using verify::verify_ptx_memory_safety;
using verify::verify_spirv_barrier_uniformity;
using verify::verify_spirv_memory_safety;
} // namespace brass
