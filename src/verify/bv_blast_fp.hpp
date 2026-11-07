#pragma once

#include "bv_cdcl_solver.hpp"
#include <brass/verify/bv_solver.hpp>
#include <vector>

namespace brass::verify {

/// Attempt exact IEEE-754 bit-blasting of a floating-point BVNode into `sat`.
/// Returns true if `n` was bit-blasted with 100% bit-accurate semantics into `out`,
/// or false if `n` requires uninterpreted fallback.
bool blast_fp_node(
    CDCLSolver& sat,
    const BVContext& ctx,
    const BVNode& n,
    const std::vector<std::vector<CDCLSolver::Lit>>& bits,
    std::vector<CDCLSolver::Lit>& out
);

} // namespace brass::verify
