#pragma once

#include "opt_cli.hpp"
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_manager.hpp>
#include <iosfwd>

namespace brass::opt {

// Statistics sinks the selected passes report into.
struct OptStats {
    PartialEscapeStats pea;
    LoopOptStats loops;
    GvnPreStats pre;
    RangeAnalysisStats range;
    FmaOptStats fma;
};

// The pipeline brass-opt's transform flags select, as one declared list;
// passes report into `stats` and WBE statistics go to `report`.
Pipeline build_pipeline(const OptCli& cli, OptStats& stats, std::ostream& report);

// Prints the statistics the --dump-*-stats flags asked for.
void dump_stats(const OptCli& cli, const OptStats& stats, std::ostream& os);

} // namespace brass::opt
