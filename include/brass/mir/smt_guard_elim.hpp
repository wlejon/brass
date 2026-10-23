#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct SmtGuardElimStats {
    size_t guards_proven_eliminated = 0;
    size_t comparisons_proven_true = 0;
    size_t comparisons_proven_false = 0;
    size_t branches_folded = 0;
    size_t queries_solved = 0;

    void reset() noexcept {
        guards_proven_eliminated = 0;
        comparisons_proven_true = 0;
        comparisons_proven_false = 0;
        branches_folded = 0;
        queries_solved = 0;
    }
};

struct SmtGuardElimOptions {
    uint32_t max_sat_conflicts = 4000;
    uint32_t max_slice_depth = 16;
    bool fold_implied_comparisons = true;
    bool fold_branches = true;
    SmtGuardElimStats* stats = nullptr;
};

// Solver-backed relational guard and bounds-check elimination pass (#70).
// Discharges 2D strided bounds, tiled loop bounds, clamped indices,
// power-of-two bit-masked indices, and relational comparisons using
// the bit-exact QF_BV solver (#65), recording GuardElimCerts (#68).
bool run_smt_guard_elimination(
    Function& fn,
    Module& mod,
    const SmtGuardElimOptions& opts = {}
);

bool run_smt_guard_elimination(
    Module& mod,
    const SmtGuardElimOptions& opts = {}
);

} // namespace brass
