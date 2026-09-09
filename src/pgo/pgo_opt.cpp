#include <brass/pgo/pgo_opt.hpp>

namespace brass::pgo {

bool optimize_function_pgo(
    Function& fn,
    const ProfileData& profile,
    const LoopOptOptions& loop_opts,
    const InlinerOptions& inline_opts
) {
    (void)inline_opts;
    LoopOptOptions opts = loop_opts;
    opts.profile_data = &profile;
    return optimize_function(fn, opts);
}

bool optimize_function_pgo(Function& fn, const ProfileData& profile) {
    LoopOptOptions loop_opts;
    InlinerOptions inline_opts;
    return optimize_function_pgo(fn, profile, loop_opts, inline_opts);
}

bool optimize_module_pgo(
    Module& mod,
    const ProfileData& profile,
    const LoopOptOptions& loop_opts,
    const InlinerOptions& inline_opts
) {
    bool changed = false;

    // 1. Profile-Guided Inlining
    InlinerOptions in_opts = inline_opts;
    in_opts.profile_data = &profile;
    changed |= inline_module(mod, in_opts);

    // 2. Profile-Guided Function Optimization (Loop opts, unrolling, unswitching)
    LoopOptOptions l_opts = loop_opts;
    l_opts.profile_data = &profile;
    changed |= optimize_module(mod, l_opts);

    return changed;
}

bool optimize_module_pgo(Module& mod, const ProfileData& profile) {
    LoopOptOptions loop_opts;
    InlinerOptions inline_opts;
    return optimize_module_pgo(mod, profile, loop_opts, inline_opts);
}

void optimize_block_layout_pgo(
    codegen::LirFunction& lir_fn,
    const Function& mir_fn,
    const ProfileData& profile
) {
    const auto* fp = profile.find_function(std::string(mir_fn.name()));
    if (fp) {
        mir::BranchProbabilityAnalysis bpa(mir_fn, *fp);
        codegen::BlockLayoutOptions opts;
        opts.cold_block_at_end = true;
        opts.branch_prob = &bpa.branch_probability_info();
        opts.block_freq = &bpa.block_frequency_info();
        opts.mir_function = &mir_fn;
        codegen::optimize_block_layout(lir_fn, opts);
    } else {
        codegen::optimize_block_layout(lir_fn);
    }
}

} // namespace brass::pgo
