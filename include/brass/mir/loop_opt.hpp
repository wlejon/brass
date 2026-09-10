#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <memory>

#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/partial_escape.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/array_contraction.hpp>

namespace brass {

struct DemoteStats;

struct LoopOptStats {
    LoopUnswitchStats unswitch_stats;
    JumpThreadingStats jump_threading_stats;
    PartialEscapeStats pea_stats;
    LoopFusionStats fusion_stats;
    LoopDistributionStats distribution_stats;
    ArrayContractionStats contraction_stats;
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
    size_t tile_size_k = 16;
    bool enable_loop_interchange = true;
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
};

// General optimization pipeline: SROA -> GVN (CSE + RLE + DSE) -> Loop Opt -> SLP Vectorizer
bool optimize_function(Function& fn);
bool optimize_function(Function& fn, const LoopOptOptions& options);

bool optimize_module(Module& mod);
bool optimize_module(Module& mod, const LoopOptOptions& options);

// Optimize loops in a single function (LICM, IVSR, Constant Folding, DCE)
bool optimize_function_loops(Function& fn, const LoopOptOptions& options = {});

// Optimize loops in an entire module
bool optimize_module_loops(Module& mod, const LoopOptOptions& options = {});

// Helper to clone a function into a target module
Function* clone_function(const Function& src, Module& dst_mod);

// Helper to clone an entire module
std::unique_ptr<Module> clone_module(const Module& src);

} // namespace brass
