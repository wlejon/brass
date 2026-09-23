#include "opt_pipeline.hpp"
#include <brass/mir/pass_catalog.hpp>
#include <ostream>

namespace brass::opt {

namespace {

LoopOptOptions loop_options(const OptCli& cli, OptStats& stats) {
    LoopOptOptions o;
    o.enable_vectorize = cli.enable_vectorize;
    o.enable_slp = cli.enable_slp;
    o.enable_loop_tile = cli.enable_loop_tile;
    o.enable_loop_fusion = cli.enable_loop_fusion;
    o.enable_loop_distribution = cli.enable_loop_distribution;
    o.enable_array_contraction = cli.enable_array_contraction;
    o.stats = &stats.loops;
    o.tile_size_i = cli.tile_size;
    o.tile_size_j = cli.tile_size;
    o.enable_avx2 = cli.enable_avx2;
    o.enable_fma = cli.enable_fma;
    o.vector_width = cli.vector_width;
    o.dump_fma_stats = cli.dump_fma_stats;
    o.fma_stats = &stats.fma;
    o.enable_parallel_loops = cli.enable_parallel_loops;
    o.parallel_threshold = cli.parallel_threshold;
    o.parallel_workers = cli.parallel_workers;
    o.dump_parallel_stats = cli.dump_parallel_stats;
    o.enable_bce = cli.enable_bce;
    o.dump_range_stats = cli.dump_range_stats;
    o.range_stats = &stats.range;
    o.enable_sroa = false;
    return o;
}

} // namespace

Pipeline build_pipeline(const OptCli& cli, OptStats& stats, std::ostream& report) {
    const LoopOptOptions loops = loop_options(cli, stats);
    Pipeline p;
    if (cli.enable_allocation_sinking) p.add(passes::allocation_sinking(&stats.pea, FunctionFilter::All, false));
    if (cli.enable_speculative_inlining) p.add(passes::speculative_devirtualization());
    if (cli.enable_inlining) p.add(passes::inline_calls(InlinerOptions{}));
    // Inlining brings callee allocations into callers, so it always runs SROA.
    if (cli.enable_sroa || cli.enable_inlining) p.add(passes::sroa());
    if (cli.enable_gvn) p.add(passes::gvn());
    if (cli.enable_gvn_pre) p.add(passes::gvn_pre(&stats.pre));
    if (cli.enable_sccp) p.add(passes::sccp(cli.enable_guard_elim));
    if (cli.enable_cfg_simplify) p.add(passes::cfg_simplify());
    if (cli.enable_loop_unswitch) p.add(passes::loop_unswitch(loops, false));
    if (cli.enable_jump_threading) p.add(passes::jump_threading(loops, false));
    if (cli.enable_bce) p.add(passes::bce("bce", cli.dump_range_stats, &stats.range, false));
    if (cli.any_loop_transform()) p.append(loop_pipeline(loops));
    if (cli.enable_wbe) p.add(passes::write_barrier_elim(cli.dump_wbe_stats, cli.dump_wbe_stats ? &report : nullptr));
    return p;
}

void dump_stats(const OptCli& cli, const OptStats& stats, std::ostream& os) {
    if (cli.dump_pea_stats) os << stats.pea.format_report() << "\n";
    if (cli.dump_pre_stats) stats.pre.dump(os);
    if (cli.dump_loop_transform_stats) {
        os << stats.loops.fusion_stats.format_report() << stats.loops.distribution_stats.format_report()
           << stats.loops.contraction_stats.format_report();
    }
    if (cli.dump_fma_stats) os << stats.fma.format_report() << "\n";
    if (cli.dump_parallel_stats) os << stats.loops.parallel_stats.format_report();
    if (cli.dump_range_stats) stats.range.dump(os);
}

} // namespace brass::opt
