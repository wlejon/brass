#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_catalog.hpp>

namespace brass {

namespace {

PartialEscapeStats* pea_stats_of(const LoopOptOptions& o) {
    if (o.pea_stats) return o.pea_stats;
    return o.stats ? &o.stats->pea_stats : nullptr;
}

RangeAnalysisStats* range_stats_of(const LoopOptOptions& o) {
    if (o.range_stats) return o.range_stats;
    return o.stats ? &o.stats->bce_stats : nullptr;
}

} // namespace

Pipeline loop_pipeline(const LoopOptOptions& o) {
    Pipeline p;
    p.set_marks_loop_optimized(true);
    if (o.enable_f64_demote) p.add(passes::f64_demote(o.demote_stats));
    if (o.enable_diamond_select) p.add(passes::select_opt());
    if (o.enable_sroa) p.add(passes::sroa("sroa", FunctionFilter::LoopEligible));
    if (o.enable_allocation_sinking || o.enable_partial_escape) {
        p.add(passes::allocation_sinking(pea_stats_of(o), FunctionFilter::LoopEligible, false));
    }
    if (o.enable_licm || o.enable_dce) p.add(passes::loop_cleanup(o));
    if (o.enable_loop_tile) p.add(passes::loop_tile(o));
    if (o.enable_loop_distribution) p.add(passes::loop_distribution(o));
    if (o.enable_loop_fusion) p.add(passes::loop_fusion(o));
    if (o.enable_array_contraction) p.add(passes::array_contraction(o));
    if (o.enable_parallel_loops) p.add(passes::parallel_loops(o));
    // The late BCE run: after f64 demotion and loop restructuring have
    // exposed checks the early (pre-loop) run could not decide. Named as a
    // repeat so pipelines that also run the early one list both.
    if (o.enable_bce) {
        p.add(passes::bce("bce 2", o.dump_range_stats, range_stats_of(o), o.enable_dce, FunctionFilter::LoopEligible));
    }
    if (o.enable_slp) p.add(passes::slp_vectorize(o));
    if (o.enable_vectorize) p.add(passes::loop_vectorize(o));
    if (o.enable_fma) p.add(passes::fma(o));
    if (o.enable_unroll) p.add(passes::loop_unroll(o));
    // After the vectorizer and unroller: IVSR turns `load_indexed a, i, 8`
    // into a byte-offset form neither matches.
    if (o.enable_ivsr) p.add(passes::ivsr(o));
    if (o.enable_dce) p.add(passes::dce());
    if (o.enable_diamond_select) p.add(passes::select_opt("select_opt 2"));
    return p;
}

Pipeline function_pipeline(const LoopOptOptions& o) {
    Pipeline p;
    if (o.enable_sroa) p.add(passes::sroa());
    if (o.enable_gvn) p.add(passes::gvn(FunctionFilter::All));
    if (o.enable_sccp) p.add(passes::sccp(o.enable_guard_elim, FunctionFilter::All));
    if (o.enable_cfg_simplify) p.add(passes::cfg_simplify("cfg_simplify", FunctionFilter::All));
    if (o.enable_loop_unswitch) p.add(passes::loop_unswitch(o, o.enable_cfg_simplify));
    if (o.enable_jump_threading) p.add(passes::jump_threading(o, o.enable_cfg_simplify));
    if (o.enable_allocation_sinking || o.enable_partial_escape) {
        p.add(passes::allocation_sinking(pea_stats_of(o), FunctionFilter::All, o.enable_cfg_simplify));
    }
    // SROA and allocation sinking already ran above, on every function.
    LoopOptOptions loops = o;
    loops.enable_sroa = false;
    loops.enable_allocation_sinking = false;
    loops.enable_partial_escape = false;
    p.append(loop_pipeline(loops));
    return p;
}

bool optimize_function_loops(Function& fn, const LoopOptOptions& options) {
    return run_pipeline(fn, loop_pipeline(options)).changed;
}

bool optimize_module_loops(Module& mod, const LoopOptOptions& options) {
    return run_pipeline(mod, loop_pipeline(options)).changed;
}

bool optimize_function(Function& fn) {
    return optimize_function(fn, LoopOptOptions{});
}

bool optimize_function(Function& fn, const LoopOptOptions& options) {
    return run_pipeline(fn, function_pipeline(options)).changed;
}

bool optimize_module(Module& mod) {
    return optimize_module(mod, LoopOptOptions{});
}

bool optimize_module(Module& mod, const LoopOptOptions& options) {
    return run_pipeline(mod, function_pipeline(options)).changed;
}

} // namespace brass
