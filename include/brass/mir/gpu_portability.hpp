#pragma once

#include <brass/core/diagnostics.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>

#include <unordered_set>

namespace brass {

// Computes per-value thread/lane divergence and divergent control regions across
// a GPU MIR kernel. Workgroup-uniform values depend only on kernel parameters,
// constants, and workgroup coordinates (`gpu_ctaid_*` / `gpu_ntid_*` / `gpu_nctaid_*`);
// values depending on thread/lane IDs (`gpu_tid_*`, `gpu_lane_id`, shuffles,
// atomics, per-lane fragment extracts) are divergent.
class GpuDivergenceAnalysis {
public:
    explicit GpuDivergenceAnalysis(const Function& fn);

    bool is_divergent(const Value* v) const noexcept;
    bool is_divergent_branch(const BasicBlock* bb) const noexcept;
    bool is_under_divergent_control(const BasicBlock* bb) const noexcept;

private:
    void analyze(const Function& fn);

    std::unordered_set<const Value*> divergent_values_;
    std::unordered_set<const BasicBlock*> divergent_branches_;
    std::unordered_set<const BasicBlock*> under_divergent_control_;
};

struct GpuPortabilityOptions {
    bool check_ssa = true;
    bool reject_non_portable_intrinsics = true;
    bool allow_fp64 = false;
    bool allow_i64_atomics = false;
    bool check_shared_provenance = true;
    bool check_barrier_uniformity = true;
};

// Verifies that `fn` obeys Brass's single-source GPU portability contract
// (docs/gpu_portability.md) across PTX, SPIR-V, and Metal MSL.
bool verify_gpu_portability(const Function& fn,
                            const GpuPortabilityOptions& opts = {},
                            DiagnosticReporter* diag = nullptr);

} // namespace brass
