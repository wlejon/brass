#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/pass_catalog.hpp>

namespace brass {

Pipeline pass_pipeline(const PassPipelineOptions& options) {
    Pipeline p;
    if (options.enable_sroa) p.add(passes::sroa());
    if (options.enable_inlining) {
        if (options.enable_speculative_inlining) p.add(passes::speculative_devirtualization());
        InlinerOptions inliner;
        inliner.only_inline_leaf_functions = options.inline_leaf_only;
        p.add(passes::inline_calls(inliner));
        // Inlined bodies bring their allocations into the caller.
        if (options.enable_sroa) p.add(passes::sroa("sroa 2"));
    }
    if (options.enable_gvn) p.add(passes::gvn());
    if (options.enable_gvn_pre) p.add(passes::gvn_pre(options.pre_stats));
    if (options.enable_sccp) p.add(passes::sccp(options.enable_guard_elim));
    if (options.enable_cfg_simplify) p.add(passes::cfg_simplify());
    if (options.enable_loop_unswitch) p.add(passes::loop_unswitch(options.loop, false));
    if (options.enable_jump_threading) p.add(passes::jump_threading(options.loop, false));
    if ((options.enable_loop_unswitch || options.enable_jump_threading) && options.enable_cfg_simplify) {
        p.add(passes::cfg_simplify("cfg_simplify 2"));
    }
    // The early BCE run, before loop restructuring; the loop stage runs the
    // late one ("bce 2").
    if (options.enable_bce) {
        p.add(passes::bce("bce", options.dump_range_stats, options.range_stats, false));
    }
    LoopOptOptions loops = options.loop;
    loops.enable_sroa = false;
    p.append(loop_pipeline(loops));
    if (options.enable_wbe) p.add(passes::write_barrier_elim(options.dump_wbe_stats));
    return p;
}

bool run_pass_pipeline(Module& mod, const PassPipelineOptions& options,
                       const PassPipelineHooks& hooks) {
    return run_pipeline(mod, pass_pipeline(options), hooks).completed;
}

} // namespace brass
