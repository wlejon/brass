#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/range_analysis.hpp>
#include <cstddef>
#include <iosfwd>
#include <sstream>
#include <string>

namespace brass {

struct RangeAnalysisStats {
    size_t bounds_checks_eliminated = 0;
    size_t checks_implied = 0;
    size_t guards_eliminated = 0;
    size_t branches_folded = 0;

    std::string format_report() const {
        std::ostringstream ss;
        ss << "=== Range Analysis & BCE Statistics ===\n"
           << "  Bounds Checks Eliminated: " << bounds_checks_eliminated << "\n"
           << "  Checks Implied:           " << checks_implied << "\n"
           << "  Guards Eliminated:        " << guards_eliminated << "\n"
           << "  Branches Folded:          " << branches_folded << "\n";
        return ss.str();
    }

    void dump(std::ostream& os) const {
        os << format_report();
    }
};

struct RangeAnalysisOptions {
    // Fold comparisons, guards and branches whose outcome value ranges decide.
    bool enable_bce = true;
    // Fold comparisons decided by a dominating comparison of the same two
    // values, such as a loop body's `i < n` re-testing its own exit condition.
    bool enable_implied_checks = true;
    bool dump_stats = false;
    RangeAnalysisStats* stats = nullptr;
};

// Bounds-check elimination on a single function. Every fold is proven from
// the function itself; nothing is speculated behind a guard or deopt exit, so
// the result is valid for ahead-of-time code too.
bool run_bounds_check_elimination(Function& fn, Module& mod, const RangeAnalysisOptions& opts = {});

// Bounds Check Elimination across an entire module
bool run_bounds_check_elimination(Module& mod, const RangeAnalysisOptions& opts = {});

} // namespace brass
