#pragma once

#include <brass/core/span.hpp>
#include <brass/debug/source_loc.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <brass/verify/smt_equiv.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace brass::verify {

enum class SandboxProofTier : uint8_t {
    None = 0,
    StaticOnly,              // Static SPIR-V audit only (SMT skipped or disabled)
    StaticFallbackLoopBound, // SMT hit loop_unroll_bound; fell back to static audit
    FullSmt                  // Full bounded-SMT memory + barrier uniformity proof
};

struct GpuSandboxOptions {
    GpuPortabilityOptions portability{};
    target::SpirvOptions spirv_opts{};
    target::PtxOptions ptx_opts{};
    target::MslOptions msl_opts{};
    GpuProverOptions prover_opts{};
    EquivOptions equiv_opts{};
    bool verify_spirv = true;
    bool verify_ptx_compile = false;
    bool verify_ptx_equiv = false;
    bool verify_msl_compile = false;
    bool run_smt_prover = true;
    bool require_full_smt = false;
    size_t max_smt_spirv_words = 3072;
    const Function* reference_mir = nullptr;
    const DebugContext* debug_ctx = nullptr;
};

struct GpuSandboxResult {
    bool portability_ok = false;
    bool spirv_compiled = false;
    bool spirv_parsed = false;
    bool ptx_compiled = false;
    bool ptx_equiv_ok = false;
    bool msl_compiled = false;
    bool memory_safe = false;
    bool barrier_uniform = false;
    bool global_bounds_proven = false;
    SandboxProofTier proof_tier = SandboxProofTier::None;
    bool equivalence_checked = false;
    bool equivalence_ok = false;
    size_t shared_accesses_proven = 0;
    size_t global_accesses_proven = 0;
    size_t barriers_proven = 0;
    size_t collectives_proven = 0;
    std::vector<std::string> violations;
    std::string diagnostic;

    bool passed() const noexcept {
        return portability_ok &&
               (!spirv_required_failed()) &&
               (!ptx_checked_ || ptx_compiled) &&
               (!ptx_equiv_checked_ || ptx_equiv_ok) &&
               (!msl_checked_ || msl_compiled) &&
               (!equivalence_checked || equivalence_ok) &&
               violations.empty();
    }

    bool spirv_checked() const noexcept { return spirv_checked_; }

private:
    bool spirv_required_failed() const noexcept {
        return spirv_checked_ && !(spirv_compiled && spirv_parsed && memory_safe && barrier_uniform);
    }
    bool spirv_checked_ = false;
    bool ptx_checked_ = false;
    bool ptx_equiv_checked_ = false;
    bool msl_checked_ = false;
    friend GpuSandboxResult verify_gpu_kernel_sandbox(const Function&, const GpuSandboxOptions&);
};

/// Unified pre-dispatch verification gate combining GPU portability rules,
/// SPIR-V compilation + static compute-shader safety audit, optional PTX/MSL
/// compilation and MIR<->PTX equivalence checking, bounded-SMT memory/barrier
/// uniformity proofs, and optional MIR equivalence checking.
GpuSandboxResult verify_gpu_kernel_sandbox(
    const Function& mir_fn,
    const GpuSandboxOptions& opts = {});

/// Runtime dispatch-time check enforcing the bounded-domain assumptions
/// (`kMaxProverDim{1,2,3}D` and `GpuProverOptions::param_constraints`)
/// under which `verify_gpu_kernel_sandbox` proved memory safety.
bool verify_gpu_dispatch_bounds(
    const GpuProverOptions& prover_opts,
    Span<const uint64_t> param_values,
    std::string* err_out = nullptr);

} // namespace brass::verify

namespace brass {
using verify::GpuSandboxOptions;
using verify::GpuSandboxResult;
using verify::SandboxProofTier;
using verify::verify_gpu_dispatch_bounds;
using verify::verify_gpu_kernel_sandbox;
} // namespace brass
