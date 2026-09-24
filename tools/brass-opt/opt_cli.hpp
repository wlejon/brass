#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace brass::opt {

// Everything brass-opt's command line can ask for.
struct OptCli {
    std::string input_file;
    std::string output_file;
    std::string run_fn;
    std::string obj_format;
    std::string shared_output_file;
    std::vector<std::string> run_arg_strings;

    // Modes
    bool verify_only = false;
    bool print_canonical = false;
    bool check_roundtrip = false;
    bool print_pipeline = false;
    bool compile_object = false;
    bool emit_shared = false;
    bool use_jit = false;
    bool use_baseline_jit = false;
    bool gc_stress = false;

    // Analyses (reports only)
    bool run_escape_analysis = false;
    bool run_alias_analysis = false;
    bool enable_partial_escape = false;  // without sinking: report candidates

    // Transforms
    bool enable_inlining = false;
    bool enable_speculative_inlining = false;
    bool enable_sroa = false;
    bool enable_allocation_sinking = false;
    bool enable_gvn = false;
    bool enable_gvn_pre = false;
    bool enable_sccp = false;
    bool enable_guard_elim = false;
    bool enable_cfg_simplify = false;
    bool enable_loop_unswitch = false;
    bool enable_jump_threading = false;
    bool enable_bce = false;
    bool enable_wbe = false;
    bool enable_vectorize = false;
    bool enable_slp = false;
    bool enable_loop_tile = false;
    size_t tile_size = 16;
    bool enable_loop_fusion = false;
    bool enable_loop_distribution = false;
    bool enable_array_contraction = false;
    bool enable_parallel_loops = false;
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0;
    bool enable_avx2 = false;
    bool enable_fma = false;
    uint32_t vector_width = 0;

    // Statistics
    bool dump_tfv_stats = false;
    bool dump_pea_stats = false;
    bool dump_pre_stats = false;
    bool dump_range_stats = false;
    bool dump_wbe_stats = false;
    bool dump_loop_transform_stats = false;
    bool dump_parallel_stats = false;
    bool dump_fma_stats = false;

    // Backend and runtime
    bool enable_trace_layout = true;
    bool enable_schedule_insns = true;
    bool enable_software_pipeline = false;
    bool enable_pic = true;
    bool enable_osr = false;
    uint64_t osr_threshold = 0;
    bool dump_tiering_stats = false;
    bool enable_background_compile = false;
    size_t jit_threads = 2;
    bool dump_jit_thread_stats = false;

    // PGO
    bool enable_pgo_instrument = false;
    std::string pgo_use_file;
    bool dump_branch_probabilities = false;

    // Debug info
    bool debug_info = false;
    bool dump_debug_lines = false;
    std::string emit_source_map_file;
    std::string symbolize_offset_arg;

    // True when a loop-stage transform was asked for.
    bool any_loop_transform() const noexcept {
        return enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion ||
               enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops;
    }
};

void print_usage(std::ostream& os, const char* prog);

enum class ParseOutcome { Run, ExitOk, ExitError };

// Parses argv into `cli`, reporting errors on `err`. ExitOk after --help (or
// no arguments), ExitError on a bad command line.
ParseOutcome parse_command_line(int argc, char** argv, OptCli& cli, std::ostream& out, std::ostream& err);

} // namespace brass::opt
