#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/pgo/profile_data.hpp>

namespace brass::pgo {

// PGO optimization pipeline for a single function
bool optimize_function_pgo(
    Function& fn,
    const ProfileData& profile,
    const LoopOptOptions& loop_opts,
    const InlinerOptions& inline_opts
);
bool optimize_function_pgo(Function& fn, const ProfileData& profile);

// PGO optimization pipeline for an entire module:
// 1. Profile-guided interprocedural inlining
// 2. Profile-guided loop optimization (unrolling and unswitching on hot loops)
// 3. Constant propagation and dead code elimination
bool optimize_module_pgo(
    Module& mod,
    const ProfileData& profile,
    const LoopOptOptions& loop_opts,
    const InlinerOptions& inline_opts
);
bool optimize_module_pgo(Module& mod, const ProfileData& profile);

// Profile-guided trace block layout for LIR
void optimize_block_layout_pgo(
    codegen::LirFunction& lir_fn,
    const Function& mir_fn,
    const ProfileData& profile
);

} // namespace brass::pgo
