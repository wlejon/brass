#include <brass/verify/gpu_kernel_sandbox.hpp>

#include <brass/core/diagnostics.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>

#include <algorithm>
#include <sstream>

namespace brass::verify {

namespace {

void record_static_audit(const spirv::ComputeShaderAuditReport& audit, GpuSandboxResult& res) {
    const bool struct_ok = audit.parse_ok && audit.structural_ok && audit.dominance_ok;
    res.memory_safe = struct_ok && audit.shared_bounds_ok;
    res.barrier_uniform = struct_ok && audit.barriers_uniform;
    for (const auto& sa : audit.shared_accesses) {
        if (sa.proven_in_bounds) {
            ++res.shared_accesses_proven;
        } else {
            res.violations.push_back(sa.detail.empty() ? "shared memory safety violation" : sa.detail);
        }
    }
    for (const auto& b : audit.barriers) {
        if (b.uniform_control_flow) {
            ++res.barriers_proven;
        } else {
            res.violations.push_back(b.detail.empty() ? "barrier divergence" : b.detail);
        }
    }
    for (const auto& d : audit.verifier_diagnostics) {
        if (!d.message.empty()) {
            res.violations.push_back(d.message);
        }
    }
}

void build_diagnostic(GpuSandboxResult& res) {
    if (!res.violations.empty()) {
        std::ostringstream oss;
        for (size_t i = 0; i < res.violations.size(); ++i) {
            if (i > 0) oss << "; ";
            oss << res.violations[i];
        }
        res.diagnostic = oss.str();
        return;
    }
    if (res.passed()) {
        res.diagnostic = "sandbox verification passed";
    }
}

} // namespace

GpuSandboxResult verify_gpu_kernel_sandbox(const Function& mir_fn, const GpuSandboxOptions& opts) {
    GpuSandboxResult res;

    DiagnosticReporter diag;
    res.portability_ok = verify_gpu_portability(mir_fn, opts.portability, &diag);
    for (const auto& d : diag.diagnostics()) {
        if (d.severity == DiagnosticSeverity::Error || d.severity == DiagnosticSeverity::Fatal) {
            res.violations.push_back(d.message.empty() ? "gpu portability error" : d.message);
        }
    }

#if BRASS_WITH_SPIRV
    if (opts.verify_spirv && res.portability_ok) {
        res.spirv_checked_ = true;
        target::SpirvKernel spv;
        try {
            spv = target::SpirvTarget::compile(mir_fn, opts.spirv_opts);
            res.spirv_compiled = !spv.words.empty();
        } catch (const std::exception& ex) {
            res.violations.push_back(std::string("SPIR-V compile failed: ") + ex.what());
        }

        if (res.spirv_compiled) {
            spirv::Module spv_mod;
            std::string parse_err;
            if (!spirv::parse_binary(spv.words, spv_mod, &parse_err)) {
                res.violations.push_back(parse_err.empty() ? "failed to parse SPIR-V binary" : parse_err);
            } else {
                res.spirv_parsed = true;
                const auto audit = brass::verify::prove_spirv_shader_safety(spv_mod);
                const bool use_smt = opts.run_smt_prover && spv.words.size() <= opts.max_smt_spirv_words;
                if (!use_smt) {
                    record_static_audit(audit, res);
                } else {
                    const auto mem = verify_spirv_memory_safety(spv_mod, opts.prover_opts);
                    const auto bar = verify_spirv_barrier_uniformity(spv_mod, opts.prover_opts);
                    const bool loop_bound_unknown =
                        mem.status == EquivStatus::Unknown &&
                        mem.violations.empty() &&
                        mem.diagnostic.find("loop_unroll_bound") != std::string::npos &&
                        bar.is_uniform();
                    if (loop_bound_unknown) {
                        record_static_audit(audit, res);
                        res.shared_accesses_proven = std::max(res.shared_accesses_proven, size_t(mem.shared_accesses_proven));
                        res.global_accesses_proven = std::max(res.global_accesses_proven, size_t(mem.global_accesses_proven));
                        res.barriers_proven = std::max(res.barriers_proven, size_t(bar.barriers_proven));
                        res.collectives_proven = std::max(res.collectives_proven, size_t(bar.collectives_proven));
                    } else {
                        size_t audit_shared_proven = 0;
                        for (const auto& sa : audit.shared_accesses) {
                            if (sa.proven_in_bounds) ++audit_shared_proven;
                        }
                        size_t audit_bar_proven = 0;
                        for (const auto& b : audit.barriers) {
                            if (b.uniform_control_flow) ++audit_bar_proven;
                        }
                        res.memory_safe = mem.is_safe() && audit.parse_ok && audit.structural_ok &&
                                          audit.dominance_ok && audit.shared_bounds_ok;
                        res.barrier_uniform = bar.is_uniform() && audit.barriers_uniform;
                        res.shared_accesses_proven = std::max(size_t(mem.shared_accesses_proven), audit_shared_proven);
                        res.global_accesses_proven = mem.global_accesses_proven;
                        res.barriers_proven = std::max(size_t(bar.barriers_proven), audit_bar_proven);
                        res.collectives_proven = bar.collectives_proven;
                        for (const auto& v : mem.violations) {
                            res.violations.push_back(v.description.empty() ? "memory safety violation" : v.description);
                        }
                        if (!mem.is_safe() && mem.violations.empty() && !mem.diagnostic.empty()) {
                            res.violations.push_back(mem.diagnostic);
                        }
                        for (const auto& d : bar.divergences) {
                            res.violations.push_back(d.description.empty() ? "barrier divergence" : d.description);
                        }
                        if (!bar.is_uniform() && bar.divergences.empty() && !bar.diagnostic.empty()) {
                            res.violations.push_back(bar.diagnostic);
                        }
                        for (const auto& d : audit.verifier_diagnostics) {
                            if (!d.message.empty()) res.violations.push_back(d.message);
                        }
                    }
                }
            }
        }
    }
#else
    (void)opts;
#endif

    if (opts.reference_mir != nullptr) {
        res.equivalence_checked = true;
        const auto eq = verify_mir_equivalence(*opts.reference_mir, mir_fn, opts.equiv_opts);
        res.equivalence_ok = eq.is_equivalent();
        if (!res.equivalence_ok) {
            res.violations.push_back(eq.diagnostic.empty() ? "MIR equivalence check failed" : eq.diagnostic);
        }
    }

    build_diagnostic(res);
    return res;
}

} // namespace brass::verify
