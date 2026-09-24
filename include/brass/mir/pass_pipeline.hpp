#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_manager.hpp>

namespace brass {

struct GvnPreStats;
struct RangeAnalysisStats;

// The production module optimization sequence (the one a front end runs on
// the modules it ships): SROA -> optional IPO (speculative devirtualization,
// inlining, SROA again) -> GVN -> GVN-PRE -> SCCP ->
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

// The sequence as a declared pipeline. Its loop stage is loop_pipeline()
// without the loop-stage SROA (SROA already ran, and runs again after
// inlining); BCE runs twice by design, "bce" before the loop stage and
// "bce 2" inside it.
Pipeline pass_pipeline(const PassPipelineOptions& options);

// The fully optimizing configuration of the sequence, for the host CPU:
// every function and loop transform a shipping front end turns on, with the
// vector width and FMA the machine running it supports. It leaves inlining,
// parallel loops and every stats collector off. Tools (the differential
// fuzzer) use it to exercise what production code goes through.
PassPipelineOptions production_pass_pipeline_options();

// Runs pass_pipeline(options) on `mod`. Returns false only when a hook
// stopped it.
bool run_pass_pipeline(Module& mod, const PassPipelineOptions& options,
                       const PassPipelineHooks& hooks = {});

} // namespace brass
