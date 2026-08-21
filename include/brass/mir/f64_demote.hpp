#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/dominators.hpp>

namespace brass {

struct F64DemoteOptions {
    bool enable_exact_div = true;
    bool enable_entry_param_demote = true;
};

// Analyzes and demotes exact-integer f64 SSA values, block parameters, and instructions
// within a single function to i64, inserting boundary conversions where necessary.
bool f64_demote_pass(Function& fn, const F64DemoteOptions& options = {});

// Runs the f64 demotion pass across all functions in a module.
bool f64_demote_module_pass(Module& mod, const F64DemoteOptions& options = {});

} // namespace brass
