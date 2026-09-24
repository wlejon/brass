#include <brass/il_translator/il_pipeline.hpp>

namespace brass::il {

PassPipelineOptions pass_pipeline_options(const TranslatorOptions& options) {
    PassPipelineOptions p;
    p.enable_sroa = options.enable_sroa;
    p.enable_inlining = options.enable_inlining;
    p.enable_speculative_inlining = options.enable_speculative_inlining;
    p.inline_leaf_only = options.inline_leaf_only;
    p.enable_gvn = options.enable_gvn;
    p.enable_gvn_pre = options.enable_gvn_pre;
    p.enable_sccp = options.enable_sccp;
    p.enable_guard_elim = options.enable_guard_elim;
    p.enable_cfg_simplify = options.enable_cfg_simplify;
    p.enable_loop_unswitch = options.enable_loop_unswitch;
    p.enable_jump_threading = options.enable_jump_threading;
    p.enable_bce = options.enable_bce;
    p.enable_wbe = options.enable_wbe;
    p.dump_wbe_stats = options.dump_wbe_stats;
    p.dump_range_stats = options.dump_range_stats;
    p.pre_stats = options.pre_stats_collector;
    p.range_stats = options.range_stats_collector;

    LoopOptOptions& l = p.loop;
    l.enable_fp_reassociation = options.allow_fp_reassociation;
    l.enable_f64_demote = options.enable_f64_demote;
    l.enable_vectorize = options.enable_vectorize;
    l.enable_slp = options.enable_slp;
    l.enable_loop_tile = options.enable_loop_tile;
    l.tile_size_i = options.tile_size;
    l.tile_size_j = options.tile_size;
    l.enable_sroa = options.enable_sroa;
    l.enable_gvn = options.enable_gvn;
    l.enable_sccp = options.enable_sccp;
    l.enable_guard_elim = options.enable_guard_elim;
    l.enable_cfg_simplify = options.enable_cfg_simplify;
    l.enable_loop_unswitch = options.enable_loop_unswitch;
    l.enable_jump_threading = options.enable_jump_threading;
    l.enable_trace_layout = options.enable_trace_layout;
    l.enable_partial_escape = options.enable_partial_escape;
    l.enable_allocation_sinking = options.enable_allocation_sinking;
    l.enable_loop_fusion = options.enable_loop_fusion;
    l.enable_loop_distribution = options.enable_loop_distribution;
    l.enable_array_contraction = options.enable_array_contraction;
    l.dump_loop_transform_stats = options.dump_loop_transform_stats;
    l.stats = options.loop_transform_stats_collector;
    l.pea_stats = options.pea_stats_collector;
    l.demote_stats = options.demote_stats_collector;
    l.enable_avx2 = options.enable_avx2;
    l.enable_fma = options.enable_fma;
    l.vector_width = options.vector_width;
    l.dump_fma_stats = options.dump_fma_stats;
    l.fma_stats = options.fma_stats_collector;
    l.enable_parallel_loops = options.enable_parallel_loops;
    l.parallel_threshold = options.parallel_threshold;
    l.parallel_workers = options.parallel_workers;
    l.dump_parallel_stats = options.dump_parallel_stats;
    l.parallel_stats = options.parallel_stats_collector;
    l.enable_bce = options.enable_bce;
    l.dump_range_stats = options.dump_range_stats;
    l.range_stats = options.range_stats_collector;
    return p;
}

TranslatorOptions bronze_translator_options() {
    TranslatorOptions options;
    options.enable_optimizations = true;
    options.enable_inlining = false;
    options.enable_speculative_inlining = false;
    options.enable_sroa = true;
    options.enable_gvn = true;
    options.enable_sccp = true;
    options.enable_guard_elim = true;
    options.enable_cfg_simplify = true;
    options.enable_loop_unswitch = true;
    options.enable_jump_threading = true;
    options.enable_trace_layout = true;
    options.enable_f64_demote = true;
    options.enable_bce = true;
    options.enable_wbe = true;
    options.enable_gvn_pre = true;
    options.enable_loop_fusion = true;
    options.enable_loop_distribution = true;
    options.enable_array_contraction = true;
    options.enable_partial_escape = true;
    options.enable_allocation_sinking = true;
    options.enable_tlab = true;
    options.use_bronze_tlab = true;
    options.pin_tls_register = true;
    // Bronze compiles for the host; the vector width follows what the CPU
    // running it supports.
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx2")) {
        options.enable_avx2 = true;
        options.vector_width = 256;
    }
    options.enable_fma = true;
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    // NEON is 128 bits wide and FMA is part of the base ISA (as bronze's
    // brass_backend.cpp sets for an AArch64 target).
    options.vector_width = 128;
    options.enable_fma = true;
#endif
    return options;
}

} // namespace brass::il
