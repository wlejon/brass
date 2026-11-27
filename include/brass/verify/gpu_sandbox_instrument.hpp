#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/mir/function.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <brass/verify/gpu_kernel_sandbox.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace brass::verify {

struct GpuClampOptions {
    bool force_clamp_all = false;
    bool zero_oob_loads = true;
    bool reject_untracked_pointers = true;
    const DebugContext* debug_ctx = nullptr;
};

struct GpuClampResult {
    bool ok = true;
    size_t clamped_global_loads = 0;
    size_t clamped_global_stores = 0;
    size_t untracked_accesses = 0;
    std::vector<std::string> violations;
    std::string diagnostic;
};

/// Guards global memory accesses in `fn` against the buffer capacities
/// declared in `prover_opts.buffers`. When `smt_result` proves full SMT
/// global memory safety and `!opts.force_clamp_all`, returns immediately with
/// zero overhead. Otherwise splits out-of-bounds loads (returning 0) and
/// stores (safe no-op) while rejecting untracked pointers if enabled.
GpuClampResult instrument_unproven_gpu_accesses(
    Function& fn,
    const GpuProverOptions& prover_opts,
    const GpuSandboxResult* smt_result = nullptr,
    const GpuClampOptions& opts = {});

struct GpuLoopFuelOptions {
    uint32_t max_loop_iterations = 1000000;
    bool reject_unbounded_divergent_barrier_loops = true;
    const DebugContext* debug_ctx = nullptr;
};

struct GpuLoopFuelResult {
    bool ok = true;
    size_t loops_statically_bounded = 0;
    size_t loops_fuel_instrumented = 0;
    std::vector<std::string> violations;
    std::string diagnostic;
};

/// Ensures every loop in `fn` terminates within `opts.max_loop_iterations`,
/// leaving small statically bounded affine loops untouched and inserting a
/// structured loop-fuel bound on unbounded loops. Rejects unbounded loops
/// containing barriers under divergent control flow when configured.
GpuLoopFuelResult instrument_gpu_loop_fuel(
    Function& fn,
    const GpuLoopFuelOptions& opts = {});

enum class HardeningTier : uint8_t {
    Rejected = 0,
    PureSmtZeroOverhead,
    HybridSmtClamped
};

struct GpuHardenOptions {
    GpuSandboxOptions sandbox_opts{};
    GpuClampOptions clamp_opts{};
    GpuLoopFuelOptions fuel_opts{};
    bool enable_loop_fuel = true;
    bool enable_auto_clamp = true;
};

struct GpuHardenResult {
    HardeningTier tier = HardeningTier::Rejected;
    GpuSandboxResult initial_sandbox;
    GpuSandboxResult final_sandbox;
    GpuClampResult clamp;
    GpuLoopFuelResult fuel;
    std::vector<std::string> violations;
    std::string diagnostic;

    bool passed() const noexcept { return tier != HardeningTier::Rejected && violations.empty(); }
};

/// Runs initial SMT verification, instruments unbounded loops with fuel, and
/// auto-clamps any global accesses not statically/SMT-proven before running
/// final portability/SPIR-V/barrier validation.
GpuHardenResult harden_and_verify_gpu_sandbox(
    Function& mir_fn,
    const GpuHardenOptions& opts = {});

} // namespace brass::verify

namespace brass {
using verify::GpuClampOptions;
using verify::GpuClampResult;
using verify::GpuHardenOptions;
using verify::GpuHardenResult;
using verify::GpuLoopFuelOptions;
using verify::GpuLoopFuelResult;
using verify::HardeningTier;
using verify::harden_and_verify_gpu_sandbox;
using verify::instrument_gpu_loop_fuel;
using verify::instrument_unproven_gpu_accesses;
} // namespace brass
