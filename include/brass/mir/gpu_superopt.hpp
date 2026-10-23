#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>

namespace brass {

struct SuperoptOptions;

struct MagicU32 {
    uint32_t multiplier = 0;
    uint32_t pre_shift = 0;
    uint32_t post_shift = 0;
    bool needs_add = false;
};

/// Compute Hacker's Delight 32-bit unsigned magic multiplier and shifts for constant `divisor > 1`.
MagicU32 compute_magic_u32(uint32_t divisor) noexcept;

/// Prove bit-exactness of a candidate 32-bit magic multiplier `(M, pre_shift, post_shift, needs_add)`
/// against unsigned division `x / divisor` across all 2^32 inputs using `BVSolver`.
bool verify_magic_u32_with_solver(
    uint32_t divisor,
    const MagicU32& magic,
    uint32_t max_sat_conflicts = 2000);

/// Run solver-verified GPU MIR superoptimizations on a single function:
///   1. 32-bit constant `udiv`/`umod` (and proven non-negative `sdiv`/`smod`) magic-multiplier
///      strength reduction via `ptx_mul_hi_u32` + shifts/adds, verified bit-exact by `BVSolver`.
///   2. 32-bit index narrowing for 64-bit `add`/`sub`/`mul`/`shl` over extended 32-bit operands
///      when `BVSolver` proves no 31-bit overflow under dominating guards/branch conditions
///      and GPU thread coordinate bounds.
bool run_gpu_mir_superopt(
    Function& fn,
    Module& mod,
    const SuperoptOptions& opts);

bool run_gpu_mir_superopt(
    Function& fn,
    Module& mod);

bool run_gpu_mir_superopt(
    Module& mod,
    const SuperoptOptions& opts);

bool run_gpu_mir_superopt(
    Module& mod);

} // namespace brass
