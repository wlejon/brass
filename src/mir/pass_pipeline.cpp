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

PassPipelineOptions production_pass_pipeline_options() {
    PassPipelineOptions p;
    p.enable_sroa = true;
    p.enable_inlining = false;
    p.enable_speculative_inlining = false;
    p.inline_leaf_only = true;
    p.enable_gvn = true;
    p.enable_gvn_pre = true;
    p.enable_sccp = true;
    p.enable_guard_elim = true;
    p.enable_cfg_simplify = true;
    p.enable_loop_unswitch = true;
    p.enable_jump_threading = true;
    p.enable_bce = true;
    p.enable_wbe = true;

    LoopOptOptions& l = p.loop;
    l.enable_fp_reassociation = false;
    l.enable_f64_demote = true;
    l.enable_vectorize = true;
    l.enable_slp = true;
    l.enable_loop_tile = true;
    l.tile_size_i = 16;
    l.tile_size_j = 16;
    l.enable_sroa = true;
    l.enable_gvn = true;
    l.enable_sccp = true;
    l.enable_guard_elim = true;
    l.enable_cfg_simplify = true;
    l.enable_loop_unswitch = true;
    l.enable_jump_threading = true;
    l.enable_trace_layout = true;
    l.enable_partial_escape = true;
    l.enable_allocation_sinking = true;
    l.enable_loop_fusion = true;
    l.enable_loop_distribution = true;
    l.enable_array_contraction = true;
    l.enable_parallel_loops = false;
    l.parallel_threshold = 1000;
    l.parallel_workers = 0;
    l.enable_bce = true;
    // The vector width follows what the CPU running it supports.
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx2")) {
        l.enable_avx2 = true;
        l.vector_width = 256;
    }
    l.enable_fma = true;
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    // NEON is 128 bits wide and FMA is part of the base ISA.
    l.vector_width = 128;
    l.enable_fma = true;
#endif
    return p;
}

} // namespace brass
