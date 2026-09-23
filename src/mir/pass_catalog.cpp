#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/critical_edge.hpp>
#include <brass/mir/devirtualize.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/select_opt.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/speculative_inliner.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/pgo/profile_data.hpp>
#include <ostream>
#include <utility>

namespace brass::passes {

namespace {

PassStep function_step(std::string name, FunctionFilter filter, std::function<bool(Function&)> run,
                       std::function<bool(Function&)> follow_up = {}) {
    PassStep s;
    s.name = std::move(name);
    s.scope = PassScope::Function;
    s.filter = filter;
    s.run_function = std::move(run);
    s.follow_up = std::move(follow_up);
    return s;
}

PassStep loop_step(std::string name, std::function<bool(Function&)> run,
                   std::function<bool(Function&)> follow_up = {}) {
    return function_step(std::move(name), FunctionFilter::LoopEligible, std::move(run), std::move(follow_up));
}

PassStep module_step(std::string name, std::function<bool(Module&)> run) {
    PassStep s;
    s.name = std::move(name);
    s.scope = PassScope::Module;
    s.run_module = std::move(run);
    return s;
}

// The cleanup the loop driver has always run after a transform: the full
// scalar cleanup, or DCE alone, and nothing when the options turn DCE off.
std::function<bool(Function&)> cleanup(const LoopOptOptions& o) {
    if (!o.enable_dce) return {};
    return [](Function& fn) { return scalar_cleanup(fn); };
}

std::function<bool(Function&)> dce_only(const LoopOptOptions& o) {
    if (!o.enable_dce) return {};
    return [](Function& fn) { return eliminate_dead_code(fn); };
}

std::function<bool(Function&)> cfg_cleanup(bool enabled) {
    if (!enabled) return {};
    return [](Function& fn) { return cfg_simplify_function(fn, CfgSimplifyOptions{}); };
}

// With a profile for `fn`, true when some loop header runs at least
// `min_header_count` times and (for `min_iterations` > 0) averages that many
// iterations per entry. Without one, true: no profile means no gate.
bool has_hot_loop(Function& fn, const pgo::ProfileData* profile, uint64_t min_header_count, uint64_t min_iterations) {
    if (!profile) return true;
    const auto* prof = profile->find_function(std::string(fn.name()));
    if (!prof) return true;
    mir::BranchProbabilityAnalysis bpa(fn, *prof);
    DominatorTree dom(fn);
    LoopAnalysis la(fn, dom);
    for (LoopInfo* loop : la.post_order_loops()) {
        if (!loop || !loop->header()) continue;
        const uint64_t hdr_cnt = bpa.block_frequency_info().get_block_count(loop->header());
        if (min_iterations == 0) {
            if (hdr_cnt >= min_header_count) return true;
            continue;
        }
        BasicBlock* ph = loop->preheader();
        const uint64_t ph_cnt = ph ? bpa.block_frequency_info().get_block_count(ph) : 0;
        const uint64_t iters = ph_cnt > 0 ? hdr_cnt / ph_cnt : hdr_cnt;
        if (iters >= min_iterations && hdr_cnt >= min_header_count) return true;
    }
    return false;
}

} // namespace

bool fp_reassociation_allowed(const Function& fn, bool forced) noexcept {
    return forced || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
}

PassStep sroa(std::string name, FunctionFilter filter) {
    return function_step(std::move(name), filter, [](Function& fn) { return sroa_function(fn, SroaOptions{}); });
}

PassStep gvn(FunctionFilter filter) {
    return function_step("gvn", filter, [](Function& fn) { return gvn_function(fn, GvnOptions{}); });
}

PassStep gvn_pre(GvnPreStats* stats, FunctionFilter filter) {
    return function_step("gvn_pre", filter, [stats](Function& fn) {
        GvnPreOptions o;
        o.stats = stats;
        return gvn_pre_function(fn, o);
    });
}

PassStep sccp(bool guard_elim, FunctionFilter filter) {
    return function_step("sccp", filter, [guard_elim](Function& fn) {
        SccpOptions o;
        o.enable_guard_elim = guard_elim;
        return sccp_function(fn, o);
    });
}

PassStep cfg_simplify(std::string name, FunctionFilter filter) {
    return function_step(std::move(name), filter,
                         [](Function& fn) { return cfg_simplify_function(fn, CfgSimplifyOptions{}); });
}

PassStep loop_unswitch(const LoopOptOptions& options, bool cfg_follow_up) {
    const LoopUnswitchOptions uo = options.unswitch_options;
    LoopUnswitchStats* stats = options.stats ? &options.stats->unswitch_stats : nullptr;
    const pgo::ProfileData* profile = options.profile_data;
    const uint64_t min_count = options.min_pgo_unswitch_count;
    return function_step("loop_unswitch", FunctionFilter::All, [=](Function& fn) {
        if (!has_hot_loop(fn, profile, min_count, 0)) return false;
        return unswitch_loops_in_function(fn, uo, stats);
    }, cfg_cleanup(cfg_follow_up));
}

PassStep jump_threading(const LoopOptOptions& options, bool cfg_follow_up) {
    const JumpThreadingOptions jo = options.jump_threading_options;
    JumpThreadingStats* stats = options.stats ? &options.stats->jump_threading_stats : nullptr;
    return function_step("jump_threading", FunctionFilter::All,
                         [=](Function& fn) { return run_jump_threading(fn, jo, stats); },
                         cfg_cleanup(cfg_follow_up));
}

PassStep bce(std::string name, bool dump_stats, RangeAnalysisStats* stats, bool cleanup_follow_up,
             FunctionFilter filter) {
    std::function<bool(Function&)> follow;
    if (cleanup_follow_up) follow = [](Function& fn) { return scalar_cleanup(fn); };
    return function_step(std::move(name), filter, [=](Function& fn) {
        if (!fn.parent()) return false;
        RangeAnalysisOptions o;
        o.enable_bce = true;
        o.enable_implied_checks = true;
        o.dump_stats = dump_stats;
        o.stats = stats;
        return run_bounds_check_elimination(fn, *fn.parent(), o);
    }, std::move(follow));
}

PassStep allocation_sinking(PartialEscapeStats* stats, FunctionFilter filter, bool cfg_follow_up) {
    return function_step("allocation_sinking", filter, [stats](Function& fn) {
        AllocationSinkingOptions o;
        o.stats = stats;
        return sink_allocations(fn, o);
    }, cfg_cleanup(cfg_follow_up));
}

PassStep dce(std::string name, FunctionFilter filter) {
    return function_step(std::move(name), filter, [](Function& fn) { return eliminate_dead_code(fn); });
}

PassStep split_critical_edges() {
    return function_step("split_critical_edges", FunctionFilter::All,
                         [](Function& fn) { return brass::split_critical_edges(fn); });
}

PassStep f64_demote(DemoteStats* stats) {
    return loop_step("f64_demote", [stats](Function& fn) {
        F64DemoteOptions o;
        o.stats = stats;
        return f64_demote_pass(fn, o);
    });
}

PassStep select_opt(std::string name) {
    return loop_step(std::move(name), [](Function& fn) { return simplify_cfg_diamonds(fn); });
}

PassStep loop_cleanup(const LoopOptOptions& options) {
    LoopCleanupOptions o;
    o.licm = options.enable_licm;
    o.fold_and_dce = options.enable_dce;
    o.max_iterations = options.max_iterations;
    return loop_step("loop_cleanup", [o](Function& fn) { return brass::loop_cleanup(fn, o); });
}

PassStep loop_tile(const LoopOptOptions& options) {
    LoopTileOptions o;
    o.tile_size_i = options.tile_size_i;
    o.tile_size_j = options.tile_size_j;
    return loop_step("loop_tile", [o](Function& fn) {
        DominatorTree dom(fn);
        return loop_tile_pass(fn, dom, o);
    }, cleanup(options));
}

PassStep loop_distribution(const LoopOptOptions& options) {
    LoopDistributionOptions o;
    if (options.stats) o.stats = &options.stats->distribution_stats;
    return loop_step("loop_distribution", [o](Function& fn) {
        DominatorTree dom(fn);
        return loop_distribution_pass(fn, dom, o);
    }, dce_only(options));
}

PassStep loop_fusion(const LoopOptOptions& options) {
    LoopFusionOptions o;
    if (options.stats) o.stats = &options.stats->fusion_stats;
    return loop_step("loop_fusion", [o](Function& fn) {
        DominatorTree dom(fn);
        return loop_fusion_pass(fn, dom, o);
    }, dce_only(options));
}

PassStep array_contraction(const LoopOptOptions& options) {
    ArrayContractionOptions o;
    o.fuse_loops_first = options.enable_loop_fusion;
    if (options.stats) o.stats = &options.stats->contraction_stats;
    return loop_step("array_contraction", [o](Function& fn) {
        DominatorTree dom(fn);
        return array_contraction_pass(fn, dom, o);
    }, dce_only(options));
}

PassStep parallel_loops(const LoopOptOptions& options) {
    ParallelLoopOptions base;
    base.parallel_threshold = options.parallel_threshold;
    base.parallel_workers = options.parallel_workers;
    base.stats = options.parallel_stats ? options.parallel_stats
                                        : (options.stats ? &options.stats->parallel_stats : nullptr);
    const bool forced = options.enable_fp_reassociation;
    return loop_step("parallel_loops", [base, forced](Function& fn) {
        ParallelLoopOptions o = base;
        o.allow_fp_reassociation = fp_reassociation_allowed(fn, forced);
        DominatorTree dom(fn);
        return auto_parallelize_function(fn, dom, o);
    }, dce_only(options));
}

PassStep slp_vectorize(const LoopOptOptions& options) {
    const bool forced = options.enable_fp_reassociation;
    return loop_step("slp_vectorize", [forced](Function& fn) {
        SlpOptions o;
        o.allow_fp_reassociation = fp_reassociation_allowed(fn, forced);
        return slp_vectorize_function(fn, o);
    }, dce_only(options));
}

PassStep loop_vectorize(const LoopOptOptions& options) {
    LoopVectorizeOptions base;
    base.enable_avx2 = options.enable_avx2;
    base.vector_width = options.vector_width;
    const bool forced = options.enable_fp_reassociation;
    return loop_step("loop_vectorize", [base, forced](Function& fn) {
        LoopVectorizeOptions o = base;
        o.allow_fp_reassociation = fp_reassociation_allowed(fn, forced);
        DominatorTree dom(fn);
        return loop_vectorize_pass(fn, dom, o);
    }, cleanup(options));
}

PassStep fma(const LoopOptOptions& options) {
    FmaOptOptions o;
    o.stats = options.fma_stats;
    return loop_step("fma", [o](Function& fn) { return fma_opt_pass(fn, o); }, dce_only(options));
}

PassStep loop_unroll(const LoopOptOptions& options) {
    const size_t factor = options.unroll_factor;
    const bool forced = options.enable_fp_reassociation;
    const pgo::ProfileData* profile = options.profile_data;
    const uint64_t min_iters = options.min_pgo_unroll_iterations;
    return loop_step("loop_unroll", [=](Function& fn) {
        if (!has_hot_loop(fn, profile, 10, min_iters)) return false;
        LoopUnrollOptions o;
        o.unroll_factor = factor;
        o.enable_reduction_jam = true;
        o.enable_fp_reduction_jam = fp_reassociation_allowed(fn, forced);
        o.enable_general_unroll = true;
        DominatorTree dom(fn);
        return loop_unroll_pass(fn, dom, o);
    }, [](Function& fn) { return scalar_cleanup(fn); });
}

PassStep ivsr(const LoopOptOptions& options) {
    std::function<bool(Function&)> follow;
    if (options.enable_dce) {
        // The original induction variable often dies once its uses moved
        // to the scaled one.
        follow = [](Function& fn) {
            LoopCleanupOptions o;
            o.licm = false;
            return brass::loop_cleanup(fn, o);
        };
    }
    return loop_step("ivsr", [](Function& fn) { return strength_reduce_induction_variables(fn); }, std::move(follow));
}

PassStep write_barrier_elim(bool dump_stats, std::ostream* report) {
    return module_step("wbe", [dump_stats, report](Module& mod) {
        WriteBarrierElimination wbe(dump_stats);
        const bool changed = wbe.run_on_module(mod);
        if (report) wbe.dump_stats(*report);
        return changed;
    });
}

PassStep devirtualize() {
    return module_step("devirtualize", [](Module& mod) { return devirtualize_module(mod); });
}

PassStep speculative_devirtualization(size_t max_callee_instruction_count) {
    return module_step("speculative_devirt", [max_callee_instruction_count](Module& mod) {
        SpeculativeInlinerOptions o;
        o.enable_inlining = true;
        o.enable_polymorphic = true;
        o.max_callee_instruction_count = max_callee_instruction_count;
        return run_speculative_devirtualization(mod, o);
    });
}

PassStep inline_calls(const InlinerOptions& options) {
    InlinerOptions o = options;
    o.enable_devirtualization = false;
    return module_step("inline", [o](Module& mod) { return inline_module(mod, o); });
}

} // namespace brass::passes
