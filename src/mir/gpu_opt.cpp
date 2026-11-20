#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>

namespace brass {

bool optimize_gpu_function(Function& fn, Module& mod, const GpuOptOptions& opts) {
    if (opts.opt_level <= 0 || fn.blocks().empty()) return false;

    bool changed = false;
    fn.rebuild_cfg_predecessors();

    if (opts.enable_fold_constants) {
        changed |= fold_constants(fn);
    }
    if (opts.enable_cse) {
        changed |= dominator_cse(fn);
    }
    if (opts.enable_gvn) {
        GvnOptions gvn_opts;
        gvn_opts.enable_cse = true;
        gvn_opts.enable_rle = opts.enable_gvn_rle;
        gvn_opts.enable_dse = opts.enable_gvn_dse;
        changed |= gvn_function(fn, gvn_opts);
    }
    if (opts.opt_level >= 2) {
        if (opts.enable_smt_guard_elim) {
            SmtGuardElimOptions ge_opts;
            ge_opts.fold_branches = opts.fold_branches;
            ge_opts.max_sat_conflicts = opts.max_sat_conflicts;
            changed |= run_smt_guard_elimination(fn, mod, ge_opts);
        }
        if (opts.enable_superopt) {
            SuperoptOptions sopt_opts;
            sopt_opts.max_sat_conflicts = opts.max_sat_conflicts;
            sopt_opts.gpu_max_ctaid = opts.gpu_max_ctaid;
            sopt_opts.gpu_max_global_tid = opts.gpu_max_global_tid;
            sopt_opts.gpu_subgroup_size = opts.gpu_subgroup_size;
            changed |= run_mir_superopt(fn, mod, sopt_opts);
        }
    }
    if (opts.enable_scalar_cleanup) {
        changed |= scalar_cleanup(fn);
    }

    fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass
