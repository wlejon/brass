#pragma once

#include <brass/mir/pass_manager.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/inliner.hpp>
#include <cstddef>
#include <iosfwd>
#include <string>

// The pass catalog: one PassStep factory per optimizer pass, each naming its
// step and wiring its options, statistics and follow-up cleanup. Every
// pipeline (pass_pipeline.hpp, loop_opt.hpp, inliner.hpp, the JIT tiers,
// brass-opt and the fuzzer) is a list of these, so a pass is configured the
// same way wherever it runs.
//
// Factories that take LoopOptOptions read the fields for their pass (and its
// statistics sink); the enable_* switches are the pipeline's business, not
// the factory's. A step whose options ask for cleanup (enable_dce) runs
// scalar_cleanup or eliminate_dead_code on each function it changed.
namespace brass {

struct GvnPreStats;
struct RangeAnalysisStats;
struct PartialEscapeStats;
struct DemoteStats;

namespace passes {

// True when `forced` or `fn` or its module allows FP reassociation.
bool fp_reassociation_allowed(const Function& fn, bool forced) noexcept;

// --- Scalar and CFG passes ---------------------------------------------
PassStep sroa(std::string name = "sroa", FunctionFilter filter = FunctionFilter::All);
PassStep gvn(FunctionFilter filter = FunctionFilter::NonWrapper);
PassStep gvn_pre(GvnPreStats* stats = nullptr, FunctionFilter filter = FunctionFilter::NonWrapper);
PassStep sccp(bool guard_elim = true, FunctionFilter filter = FunctionFilter::NonWrapper);
PassStep cfg_simplify(std::string name = "cfg_simplify", FunctionFilter filter = FunctionFilter::NonWrapper);
// Gated by options.profile_data (hot loops only) when a profile is given;
// `cfg_follow_up` runs CFG simplification on each function it changed.
PassStep loop_unswitch(const LoopOptOptions& options, bool cfg_follow_up);
PassStep jump_threading(const LoopOptOptions& options, bool cfg_follow_up);
// Range-analysis bounds-check elimination with implied-check folding.
PassStep bce(std::string name, bool dump_stats, RangeAnalysisStats* stats, bool cleanup_follow_up,
             FunctionFilter filter = FunctionFilter::All);
PassStep allocation_sinking(PartialEscapeStats* stats, FunctionFilter filter, bool cfg_follow_up);
PassStep dce(std::string name = "dce", FunctionFilter filter = FunctionFilter::LoopEligible);
PassStep split_critical_edges();

// --- Loop passes (FunctionFilter::LoopEligible) -------------------------
PassStep f64_demote(DemoteStats* stats);
PassStep select_opt(std::string name = "select_opt");
// LICM and the scalar cleanups iterated to a fixed point.
PassStep loop_cleanup(const LoopOptOptions& options);
PassStep loop_tile(const LoopOptOptions& options);
PassStep loop_distribution(const LoopOptOptions& options);
PassStep loop_fusion(const LoopOptOptions& options);
PassStep array_contraction(const LoopOptOptions& options);
PassStep parallel_loops(const LoopOptOptions& options);
PassStep slp_vectorize(const LoopOptOptions& options);
PassStep loop_vectorize(const LoopOptOptions& options);
PassStep fma(const LoopOptOptions& options);
// Gated by options.profile_data (loops hot enough to unroll) when given.
PassStep loop_unroll(const LoopOptOptions& options);
// Induction-variable strength reduction. It rewrites indexed accesses into
// a form the vectorizer does not match, so pipelines run it after the
// vectorizer and the unroller.
PassStep ivsr(const LoopOptOptions& options);

// --- Module passes ------------------------------------------------------
PassStep write_barrier_elim(bool dump_stats = false, std::ostream* report = nullptr);
// Rewrites every patchable_call to a direct call (see InlinerOptions).
PassStep devirtualize();
// Feedback-driven devirtualization and inlining from the global
// FeedbackRegistry.
PassStep speculative_devirtualization(size_t max_callee_instruction_count = 120);
// Bottom-up inlining. Never devirtualizes: a pipeline that wants that runs
// devirtualize() as its own step.
PassStep inline_calls(const InlinerOptions& options);

} // namespace passes
} // namespace brass
