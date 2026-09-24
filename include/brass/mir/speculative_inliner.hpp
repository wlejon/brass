#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <string>
#include <cstddef>

namespace brass {

struct SpeculativeInlinerOptions {
    bool enable_inlining = true;
    bool enable_polymorphic = true;
    size_t max_polymorphic_degree = 2;
    size_t max_callee_instruction_count = 120;
    size_t min_invocations = 1;
    std::string deopt_stub_prefix = "@deopt_slow_call";
};

// Speculatively devirtualizes and optionally inlines indirect/dynamic calls within a single function
// based on recorded runtime type feedback.
bool run_speculative_devirtualization(
    Function& fn,
    Module& mod,
    const runtime::TypeFeedbackVector* tfv,
    const SpeculativeInlinerOptions& opts = {}
);

// Speculatively devirtualizes indirect/dynamic calls across an entire module using FeedbackRegistry.
bool run_speculative_devirtualization(
    Module& mod,
    const runtime::FeedbackRegistry& registry,
    const SpeculativeInlinerOptions& opts = {}
);

// Speculatively devirtualizes indirect/dynamic calls across an entire module using the global FeedbackRegistry.
bool run_speculative_devirtualization(
    Module& mod,
    const SpeculativeInlinerOptions& opts = {}
);

// Rewrites every guard of `fn` with resume id `resume_id` as the branch its
// failure takes in the interpreters: `br_if %cond, <rest of its block>,
// target(state values...)`, where target is the id's resume_table block of
// `fn` (parameter i takes state value i), then drops that resume_table
// entry. For guards that resume in their own function (the speculative
// inliner's), whose failure needs no deoptimization. Returns false, changing
// nothing, when a guard with the id has an exit stub or the id names no
// resume_table block; the caller verifies the result (the resume block's
// code must now be dominated by what it reads).
bool lower_guards_to_local_branch(Function& fn, Module& mod, uint32_t resume_id);

} // namespace brass
