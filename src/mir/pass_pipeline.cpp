#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/write_barrier_elim.hpp>

namespace brass {

bool run_pass_pipeline(Module& mod, const PassPipelineOptions& options,
                       const PassPipelineHooks& hooks) {
    auto run = [&](std::string_view name, const std::function<void()>& pass) {
        if (hooks.before_pass) hooks.before_pass(name);
        pass();
        return !hooks.after_pass || hooks.after_pass(name);
    };

    if (options.enable_sroa) {
        if (!run("sroa", [&] { sroa_module(mod); })) return false;
    }

    if (options.enable_inlining) {
        const bool ok = run("ipo", [&] {
            InlinerOptions inliner_opts;
            inliner_opts.enable_sroa = options.enable_sroa;
            // GVN runs module-wide right after, so the inliner's own GVN
            // would only repeat it.
            inliner_opts.enable_gvn = false;
            inliner_opts.enable_speculative_devirtualization = options.enable_speculative_inlining;
            inliner_opts.only_inline_leaf_functions = options.inline_leaf_only;
            optimize_module_ipo(mod, inliner_opts, options.loop);
        });
        if (!ok) return false;
    }

    if (options.enable_gvn) {
        if (!run("gvn", [&] { gvn_module(mod); })) return false;
    }
    if (options.enable_gvn_pre) {
        const bool ok = run("gvn_pre", [&] {
            GvnPreOptions pre_opts;
            pre_opts.stats = options.pre_stats;
            gvn_pre_module(mod, pre_opts);
        });
        if (!ok) return false;
    }
    if (options.enable_sccp) {
        const bool ok = run("sccp", [&] {
            SccpOptions sccp_opts;
            sccp_opts.enable_guard_elim = options.enable_guard_elim;
            sccp_module(mod, sccp_opts);
        });
        if (!ok) return false;
    }
    if (options.enable_cfg_simplify) {
        if (!run("cfg_simplify 1", [&] { cfg_simplify_module(mod); })) return false;
    }
    if (options.enable_loop_unswitch) {
        if (!run("loop_unswitch", [&] { unswitch_loops_in_module(mod); })) return false;
    }
    if (options.enable_jump_threading) {
        if (!run("jump_threading", [&] { jump_thread_module(mod); })) return false;
    }
    if ((options.enable_loop_unswitch || options.enable_jump_threading) && options.enable_cfg_simplify) {
        if (!run("cfg_simplify 2", [&] { cfg_simplify_module(mod); })) return false;
    }
    if (options.enable_bce) {
        const bool ok = run("bce", [&] {
            RangeAnalysisOptions bce_opts;
            bce_opts.enable_bce = true;
            bce_opts.enable_implied_checks = true;
            bce_opts.dump_stats = options.dump_range_stats;
            bce_opts.stats = options.range_stats;
            run_bounds_check_elimination(mod, bce_opts);
        });
        if (!ok) return false;
    }
    if (!run("loops", [&] { optimize_module_loops(mod, options.loop); })) return false;

    if (options.enable_wbe) {
        const bool ok = run("wbe", [&] {
            WriteBarrierElimination wbe(options.dump_wbe_stats);
            wbe.run_on_module(mod);
        });
        if (!ok) return false;
    }
    return true;
}

} // namespace brass
