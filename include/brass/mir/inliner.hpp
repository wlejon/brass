#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/call_graph.hpp>
#include <brass/mir/loop_opt.hpp>
#include <cstddef>

namespace brass {

struct InlinerOptions {
    size_t max_inline_depth = 4;
    size_t leaf_instruction_threshold = 30;
    double max_caller_growth_factor = 2.5;
    double loop_call_priority_bonus = 4.0;
    size_t max_callee_instruction_count = 120;
    size_t max_total_caller_instructions = 1000;
    bool enable_devirtualization = true;
    bool enable_loop_priority = true;
    bool enable_sroa = true;
    bool enable_gvn = true;
};

// Profitability decision for inlining a specific call site
bool should_inline_call(
    const Function& caller,
    const CallSite& site,
    const Function& callee,
    const CallGraph& cg,
    size_t current_depth,
    size_t loop_depth,
    size_t baseline_caller_size,
    size_t current_caller_size,
    const InlinerOptions& options
);

// Inlines call sites in a single function
bool inline_function(Function& fn, Module& mod);
bool inline_function(Function& fn, Module& mod, const InlinerOptions& options);

// Inlines call sites across an entire module in bottom-up leaf-first order
bool inline_module(Module& mod);
bool inline_module(Module& mod, const InlinerOptions& options);

// Interprocedural Optimization (IPO) Pipeline:
// 1. Devirtualization (monomorphic patchable_call -> direct call)
// 2. Inlining pass in bottom-up leaf-first order
// 3. Constant folding, CSE, DCE
// 4. Loop optimization & f64 demotion re-run
bool optimize_module_ipo(Module& mod);
bool optimize_module_ipo(Module& mod, const InlinerOptions& inline_opts, const LoopOptOptions& loop_opts);

} // namespace brass
