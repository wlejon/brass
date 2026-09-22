#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/loop_opt.hpp>
#include <functional>
#include <string_view>

namespace brass {

struct GvnPreStats;
struct RangeAnalysisStats;

// The production module optimization sequence (the one the Bronze IL
// translator runs): SROA -> optional IPO -> GVN -> GVN-PRE -> SCCP ->
// CFG simplify -> loop unswitch -> jump threading -> CFG simplify -> BCE ->
// loop pipeline -> write-barrier elimination. Keeping it in one place lets
// tools (the differential fuzzer) run exactly what embedders ship.
struct PassPipelineOptions {
    bool enable_sroa = false;
    bool enable_inlining = false;
    bool enable_speculative_inlining = false;
    bool inline_leaf_only = false;
    bool enable_gvn = true;
    bool enable_gvn_pre = true;
    bool enable_sccp = true;
    bool enable_guard_elim = true;
    bool enable_cfg_simplify = true;
    bool enable_loop_unswitch = true;
    bool enable_jump_threading = true;
    bool enable_bce = true;
    bool enable_wbe = true;
    bool dump_wbe_stats = false;
    bool dump_range_stats = false;
    GvnPreStats* pre_stats = nullptr;
    RangeAnalysisStats* range_stats = nullptr;
    // Options for the loop pipeline stage (optimize_module_loops) and the
    // loop half of IPO.
    LoopOptOptions loop;
};

struct PassPipelineHooks {
    // Called with the pass name just before it runs.
    std::function<void(std::string_view)> before_pass;
    // Called with the pass name after it ran; returning false stops the
    // pipeline (run_pass_pipeline then returns false).
    std::function<bool(std::string_view)> after_pass;
};

// Runs the sequence on `mod`. Returns false only when a hook stopped it.
bool run_pass_pipeline(Module& mod, const PassPipelineOptions& options,
                       const PassPipelineHooks& hooks = {});

} // namespace brass
