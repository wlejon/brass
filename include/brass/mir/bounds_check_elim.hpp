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
    size_t bounds_checks_hoisted = 0;
    size_t guards_eliminated = 0;
    size_t branches_folded = 0;

    std::string format_report() const {
        std::ostringstream ss;
        ss << "=== Range Analysis & BCE Statistics ===\n"
           << "  Bounds Checks Eliminated: " << bounds_checks_eliminated << "\n"
           << "  Bounds Checks Hoisted:    " << bounds_checks_hoisted << "\n"
           << "  Guards Eliminated:        " << guards_eliminated << "\n"
           << "  Branches Folded:          " << branches_folded << "\n";
        return ss.str();
    }

    void dump(std::ostream& os) const {
        os << format_report();
    }
};

struct RangeAnalysisOptions {
    bool enable_bce = true;
    bool enable_hoisting = true;
    bool dump_stats = false;
    RangeAnalysisStats* stats = nullptr;
};

// Bounds Check Elimination (local dominator-based + loop hoisting) on a single function
bool run_bounds_check_elimination(Function& fn, Module& mod, const RangeAnalysisOptions& opts = {});

// Bounds Check Elimination across an entire module
bool run_bounds_check_elimination(Module& mod, const RangeAnalysisOptions& opts = {});

} // namespace brass
