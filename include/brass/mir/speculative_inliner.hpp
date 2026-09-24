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
};

// Speculatively devirtualizes and optionally inlines indirect/dynamic calls within a single function
// based on recorded runtime type feedback. A monomorphic site becomes
// `br_if (callee == func_addr @target), fast, slow`: fast makes the direct
// (or inlined) call, slow the original call_indirect, so a mismatch is an
// ordinary branch, never a deoptimization.
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

} // namespace brass
