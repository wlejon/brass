#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/pass_manager.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/partial_escape.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/bounds_check_elim.hpp>

namespace brass {

struct DemoteStats;
struct FmaOptStats;

struct LoopOptStats {
    LoopUnswitchStats unswitch_stats;
    JumpThreadingStats jump_threading_stats;
    PartialEscapeStats pea_stats;
    LoopFusionStats fusion_stats;
    LoopDistributionStats distribution_stats;
    ArrayContractionStats contraction_stats;
    ParallelLoopStats parallel_stats;
    RangeAnalysisStats bce_stats;
};

namespace pgo {
class ProfileData;
}

struct LoopOptOptions {
    bool enable_licm = true;
    bool enable_ivsr = true;
    bool enable_dce = true;
    bool enable_diamond_select = true;
    bool enable_unroll = true;
    bool enable_f64_demote = true;
    bool enable_slp = true;
    bool enable_vectorize = true;
    size_t unroll_factor = 4;
    size_t max_iterations = 8;
    bool enable_fp_reassociation = false; // Opt-in FP reassociation (default OFF / IEEE-strict)
    bool enable_sroa = false;
    bool enable_gvn = true;
    bool enable_sccp = true;
    bool enable_guard_elim = true;
    bool enable_cfg_simplify = true;
    bool enable_loop_unswitch = false;
    bool enable_jump_threading = false;
    bool enable_trace_layout = false;
    bool enable_loop_tile = false;
    bool enable_loop_fusion = false;
    bool enable_loop_distribution = false;
    bool enable_array_contraction = false;
    bool dump_loop_transform_stats = false;
    size_t tile_size_i = 16;
    size_t tile_size_j = 16;
    LoopUnswitchOptions unswitch_options;
    JumpThreadingOptions jump_threading_options;
    DemoteStats* demote_stats = nullptr;
    LoopOptStats* stats = nullptr;
    PartialEscapeStats* pea_stats = nullptr;
    bool enable_partial_escape = false;
    bool enable_allocation_sinking = false;
    const pgo::ProfileData* profile_data = nullptr;
    uint64_t min_pgo_unroll_iterations = 4;
    uint64_t min_pgo_unswitch_count = 10;
    bool enable_avx2 = false;
    bool enable_fma = false;
    uint32_t vector_width = 0;
    bool dump_fma_stats = false;
    FmaOptStats* fma_stats = nullptr;
    bool enable_parallel_loops = false;
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0;
    bool dump_parallel_stats = false;
    ParallelLoopStats* parallel_stats = nullptr;
    bool enable_bce = false;
    bool dump_range_stats = false;
    RangeAnalysisStats* range_stats = nullptr;
};

// The loop stage, as a declared pipeline of the passes `options` enables, all
// restricted to loop-eligible functions: f64 demotion, select formation,
// [SROA, allocation sinking,] LICM + scalar cleanup, tiling, distribution,
// fusion, array contraction, auto-parallelization, late BCE ("bce 2"), SLP,
// loop vectorization, FMA, unrolling, IVSR, DCE, select formation again.
Pipeline loop_pipeline(const LoopOptOptions& options);

// The general function pipeline: SROA, GVN, SCCP, CFG simplification, loop
// unswitching, jump threading and allocation sinking on every function, then
// loop_pipeline (without a second SROA or sinking run).
Pipeline function_pipeline(const LoopOptOptions& options);

// Run function_pipeline on one function / every function of a module.
bool optimize_function(Function& fn);
bool optimize_function(Function& fn, const LoopOptOptions& options);

bool optimize_module(Module& mod);
bool optimize_module(Module& mod, const LoopOptOptions& options);

// Run loop_pipeline on one function / every function of a module.
bool optimize_function_loops(Function& fn, const LoopOptOptions& options = {});
bool optimize_module_loops(Module& mod, const LoopOptOptions& options = {});

// Helper to clone a function into a target module
Function* clone_function(const Function& src, Module& dst_mod);

// Helper to clone an entire module
std::unique_ptr<Module> clone_module(const Module& src);

// The module a per-function tier-up compiles: src's declarations, `fn_name`
// with the bodies of the module functions it names directly and of those
// listed in `also` (its speculated call targets, say) for the optimizer to
// inline, and every other module function those bodies name as an external
// symbol, linked by name. Null when src has no such function.
std::unique_ptr<Module> clone_function_module(const Module& src, std::string_view fn_name,
                                              const std::vector<std::string>& also = {});

} // namespace brass
