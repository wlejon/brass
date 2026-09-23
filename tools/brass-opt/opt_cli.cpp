#include "opt_cli.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/tiering.hpp>
#include <ostream>
#include <stdexcept>
#include <string_view>

namespace brass::opt {

void print_usage(std::ostream& os, const char* prog) {
    os << "brass-opt - Brass MIR Optimizer and Compiler Tool (v" << brass::version_string() << ")\n"
       << "Usage: " << prog << " [options] <input-file>\n\n"
       << "Options:\n"
       << "  -h, --help            Show this help message\n"
       << "  -v, --verify          Parse and verify MIR module\n"
       << "  -p, --print           Parse, verify, and print canonical MIR\n"
       << "  --check-roundtrip     Assert byte-identical parse(print(x)) roundtrip\n"
       << "  --print-pipeline      Print the optimization steps the other flags select, then exit\n"
       << "  -c, --compile         Compile MIR module to relocatable object file (.obj/.o)\n"
       << "  --format <coff|elf|macho> Object file format for --compile (default: host format;\n"
       << "                        --emit-shared supports coff and elf only)\n"
       << "  --emit-shared <file>  Compile MIR module directly to shared library (.dll/.so)\n"
       << "  -shared               Compile MIR module directly to shared library (.dll/.so)\n"
       << "  --jit                 Use in-memory JIT execution engine for --run\n"
       << "  --baseline-jit        Use fast Tier-1 Baseline JIT execution engine for --run\n"
       << "  -r, --run <fn>        Execute function <fn>\n"
       << "  --args <a1> <a2>...   Arguments to pass to the function executed with --run\n"
       << "  --gc-stress           Enable moving GC stress mode (collects at every allocation/safepoint)\n"
       << "  --inline              Run interprocedural inlining (then SROA)\n"
       << "  --speculative-inlining Run feedback-driven speculative devirtualization and inlining\n"
       << "  --dump-tfv-stats      Dump Type Feedback Vector (TFV) statistics\n"
       << "  --sroa                Run Scalar Replacement of Aggregates (SROA)\n"
       << "  --escape-analysis     Report Escape Analysis on module functions\n"
       << "  --partial-escape      Report Partial Escape Analysis candidates\n"
       << "  --sink-allocations    Run Allocation Sinking and Hot Path Scalarization\n"
       << "  --dump-pea-stats      Dump Partial Escape Analysis and Allocation Sinking statistics\n"
       << "  --gvn                 Run Global Value Numbering (GVN), RLE & DSE\n"
       << "  --enable-pre, --enable-gvn-pre Run Global Value Numbering with PRE (GVN-PRE)\n"
       << "  --dump-pre-stats      Dump GVN-PRE statistics\n"
       << "  --alias-analysis      Report Alias Analysis on module functions\n"
       << "  --vectorize           Run loop vectorization on countable loops\n"
       << "  --slp                 Run SLP straight-line vectorization\n"
       << "  --loop-tile           Run loop tiling / cache blocking on nested loops\n"
       << "  --tile-size <N>       Tile size for loop tiling (default: 16)\n"
       << "  --enable-loop-fusion  Run loop fusion (loop jamming)\n"
       << "  --enable-loop-distribution Run loop distribution (loop fission)\n"
       << "  --enable-array-contraction Run array contraction (buffer elimination)\n"
       << "  --dump-loop-transform-stats Dump loop transformation statistics\n"
       << "  --enable-parallel-loops Run polyhedral loop dependence & auto-parallelization\n"
       << "  --parallel-threshold=<N> Cost threshold for parallelization (default: 1000)\n"
       << "  --parallel-workers=<N> Number of parallel worker threads\n"
       << "  --dump-parallel-stats Dump parallel loop statistics\n"
       << "  --enable-avx2         Vectorize for AVX2\n"
       << "  --enable-fma, --fma   Run FMA contraction\n"
       << "  --vector-width=<N>    Vector width in bits for the loop vectorizer\n"
       << "  --dump-fma-stats      Dump FMA contraction statistics\n"
       << "  --sccp                Run Sparse Conditional Constant Propagation (SCCP)\n"
       << "  --guard-elim          Run Speculation Guard Elimination (with SCCP)\n"
       << "  --bce                 Run Value Range Analysis & Bounds Check Elimination\n"
       << "  --dump-range-stats    Dump Range Analysis & Bounds Check Elimination statistics\n"
       << "  --cfg-simplify        Run CFG Simplification & Dead Block Compaction\n"
       << "  --loop-unswitch       Run Loop Unswitching on candidate loops\n"
       << "  --jump-threading      Run SSA Jump Threading\n"
       << "  --trace-layout        Run LIR Trace Scheduling & Fall-Through Block Layout\n"
       << "  --schedule-insns      Run Machine Instruction Scheduling\n"
       << "  --software-pipeline   Run Loop Modulo Scheduling & Software Pipelining\n"
       << "  --wbe, --enable-wbe   Run Write Barrier Elimination (WBE)\n"
       << "  --dump-wbe-stats      Dump Write Barrier Elimination statistics\n"
       << "  --pgo-instrument      Instrument module with Knuth-Stevenson minimal edge counters\n"
       << "  --pgo-use=<file>      Load profile data (.bprof) for profile-guided optimization\n"
       << "  --dump-branch-probabilities Dump block frequencies and edge branch probabilities\n"
       << "  --enable-pic          Enable Polymorphic Inline Caching for dynamic property accesses\n"
       << "  --dump-ic-stats       Dump Inline Cache hit/miss and state statistics\n"
       << "  --enable-osr          Enable On-Stack Replacement (OSR) in interpreter\n"
       << "  --osr-threshold=<N>   Loop backedge threshold for OSR migration (default: 100)\n"
       << "  --dump-tiering-stats  Dump tiering feedback and OSR statistics\n"
       << "  --enable-background-compile Enable background JIT compiler worker threads\n"
       << "  --jit-threads=<N>     Number of background JIT worker threads (default: 2)\n"
       << "  --dump-jit-thread-stats Dump background JIT worker thread pool statistics\n"
       << "  -g, --debug-info, --emit-debug-info Preserve and emit debug information (DWARF/CodeView)\n"
       << "  --dump-debug-lines    Dump decoded source line mappings from debug table\n"
       << "  --emit-source-map=<file.map> Emit standard JSON Source Map V3 to <file.map>\n"
       << "  --symbolize-offset=<fn,offset> Symbolize function offset to source location\n"
       << "  -o <file>             Write output to <file> instead of stdout\n"
       << "\nThe selected transforms run as one pipeline, in this order: allocation sinking,\n"
       << "speculative devirtualization, inlining, SROA, GVN, GVN-PRE, SCCP, CFG simplification,\n"
       << "loop unswitching, jump threading, BCE, the loop stage, WBE. The module is verified\n"
       << "after every step.\n";
}

namespace {

struct Flag {
    std::string_view spellings[3];  // unused entries empty
    bool OptCli::*sets[2];          // unused entries null
    bool value = true;
};

// Flags that only switch booleans.
bool apply_flag(std::string_view arg, OptCli& cli) {
    static const Flag flags[] = {
        {{"-v", "--verify"}, {&OptCli::verify_only}},
        {{"-p", "--print"}, {&OptCli::print_canonical}},
        {{"--check-roundtrip"}, {&OptCli::check_roundtrip}},
        {{"--print-pipeline"}, {&OptCli::print_pipeline}},
        {{"--gc-stress"}, {&OptCli::gc_stress}},
        {{"--inline"}, {&OptCli::enable_inlining}},
        {{"--speculative-inlining", "--enable-speculative-inlining"}, {&OptCli::enable_speculative_inlining}},
        {{"--dump-tfv-stats"}, {&OptCli::dump_tfv_stats}},
        {{"--sroa"}, {&OptCli::enable_sroa}},
        {{"--escape-analysis"}, {&OptCli::run_escape_analysis}},
        {{"--partial-escape"}, {&OptCli::enable_partial_escape}},
        {{"--sink-allocations"}, {&OptCli::enable_allocation_sinking, &OptCli::enable_partial_escape}},
        {{"--dump-pea-stats"}, {&OptCli::dump_pea_stats}},
        {{"--gvn"}, {&OptCli::enable_gvn}},
        {{"--enable-pre", "--enable-gvn-pre"}, {&OptCli::enable_gvn_pre}},
        {{"--dump-pre-stats"}, {&OptCli::dump_pre_stats, &OptCli::enable_gvn_pre}},
        {{"--sccp"}, {&OptCli::enable_sccp}},
        {{"--guard-elim"}, {&OptCli::enable_guard_elim, &OptCli::enable_sccp}},
        {{"--bce"}, {&OptCli::enable_bce}},
        {{"--dump-range-stats"}, {&OptCli::dump_range_stats, &OptCli::enable_bce}},
        {{"--cfg-simplify"}, {&OptCli::enable_cfg_simplify}},
        {{"--loop-unswitch"}, {&OptCli::enable_loop_unswitch}},
        {{"--jump-threading"}, {&OptCli::enable_jump_threading}},
        {{"--trace-layout"}, {&OptCli::enable_trace_layout}},
        {{"--no-trace-layout"}, {&OptCli::enable_trace_layout}, false},
        {{"--schedule-insns"}, {&OptCli::enable_schedule_insns}},
        {{"--no-schedule-insns"}, {&OptCli::enable_schedule_insns}, false},
        {{"--software-pipeline"}, {&OptCli::enable_software_pipeline}},
        {{"--no-software-pipeline"}, {&OptCli::enable_software_pipeline}, false},
        {{"--wbe", "--enable-wbe"}, {&OptCli::enable_wbe}},
        {{"--dump-wbe-stats"}, {&OptCli::dump_wbe_stats, &OptCli::enable_wbe}},
        {{"--enable-osr"}, {&OptCli::enable_osr}},
        {{"--dump-tiering-stats"}, {&OptCli::dump_tiering_stats}},
        {{"--enable-background-compile"}, {&OptCli::enable_background_compile}},
        {{"--dump-jit-thread-stats"}, {&OptCli::dump_jit_thread_stats}},
        {{"--pgo-instrument"}, {&OptCli::enable_pgo_instrument}},
        {{"--dump-branch-probabilities"}, {&OptCli::dump_branch_probabilities}},
        {{"--alias-analysis"}, {&OptCli::run_alias_analysis}},
        {{"--vectorize"}, {&OptCli::enable_vectorize}},
        {{"--enable-avx2"}, {&OptCli::enable_avx2}},
        {{"--enable-fma", "--fma"}, {&OptCli::enable_fma}},
        {{"--dump-fma-stats"}, {&OptCli::dump_fma_stats}},
        {{"--slp"}, {&OptCli::enable_slp}},
        {{"--loop-tile"}, {&OptCli::enable_loop_tile}},
        {{"--enable-loop-fusion"}, {&OptCli::enable_loop_fusion}},
        {{"--enable-loop-distribution"}, {&OptCli::enable_loop_distribution}},
        {{"--enable-array-contraction"}, {&OptCli::enable_array_contraction}},
        {{"--dump-loop-transform-stats"}, {&OptCli::dump_loop_transform_stats}},
        {{"--enable-parallel-loops"}, {&OptCli::enable_parallel_loops}},
        {{"--dump-parallel-stats"}, {&OptCli::dump_parallel_stats}},
        {{"--enable-pic"}, {&OptCli::enable_pic}},
        {{"--no-pic"}, {&OptCli::enable_pic}, false},
        {{"--dump-ic-stats"}, {&OptCli::dump_ic_stats}},
        {{"-c", "--compile"}, {&OptCli::compile_object}},
        {{"-shared"}, {&OptCli::emit_shared}},
        {{"--jit"}, {&OptCli::use_jit}},
        {{"--baseline-jit"}, {&OptCli::use_baseline_jit}},
        {{"-g", "--debug-info", "--emit-debug-info"}, {&OptCli::debug_info}},
        {{"--dump-debug-lines"}, {&OptCli::dump_debug_lines}},
    };
    for (const Flag& f : flags) {
        for (std::string_view s : f.spellings) {
            if (s.empty() || s != arg) continue;
            for (bool OptCli::*member : f.sets) {
                if (member) cli.*member = f.value;
            }
            return true;
        }
    }
    return false;
}

// "--name=value" or "--name value": stores the value in `value` (advancing
// `i` for the second form) and returns true; false when `arg` is neither.
bool take_value(std::string_view name, const std::string& arg, int argc, char** argv, int& i, std::string& value) {
    const std::string eq = std::string(name) + "=";
    if (arg.rfind(eq, 0) == 0) {
        value = arg.substr(eq.size());
        return true;
    }
    if (arg == name && i + 1 < argc) {
        value = argv[++i];
        return true;
    }
    return false;
}

} // namespace

ParseOutcome parse_command_line(int argc, char** argv, OptCli& cli, std::ostream& out, std::ostream& err) {
    cli.osr_threshold = runtime::BACKEDGE_OSR_THRESHOLD;
    if (argc < 2) {
        print_usage(out, argv[0]);
        return ParseOutcome::ExitOk;
    }
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            std::string v;
            if (arg == "-h" || arg == "--help") {
                print_usage(out, argv[0]);
                return ParseOutcome::ExitOk;
            } else if (apply_flag(arg, cli)) {
                continue;
            } else if (take_value("--osr-threshold", arg, argc, argv, i, v)) {
                cli.enable_osr = true;
                cli.osr_threshold = std::stoull(v);
            } else if (take_value("--jit-threads", arg, argc, argv, i, v)) {
                cli.jit_threads = static_cast<size_t>(std::stoul(v));
            } else if (take_value("--parallel-threshold", arg, argc, argv, i, v)) {
                cli.parallel_threshold = std::stoull(v);
            } else if (take_value("--parallel-workers", arg, argc, argv, i, v)) {
                cli.parallel_workers = static_cast<uint32_t>(std::stoul(v));
            } else if (take_value("--tile-size", arg, argc, argv, i, v)) {
                cli.tile_size = static_cast<size_t>(std::stoul(v));
            } else if (arg == "--vector-width-256") {
                cli.vector_width = 256;
            } else if (arg.rfind("--vector-width=", 0) == 0) {
                cli.vector_width = static_cast<uint32_t>(std::stoul(arg.substr(15)));
            } else if (take_value("--pgo-use", arg, argc, argv, i, v)) {
                cli.pgo_use_file = v;
            } else if (take_value("--format", arg, argc, argv, i, v)) {
                if (v != "coff" && v != "elf" && v != "macho") {
                    err << "Error: unknown --format '" << v << "' (expected coff, elf or macho)\n";
                    return ParseOutcome::ExitError;
                }
                cli.obj_format = v;
            } else if (take_value("--emit-source-map", arg, argc, argv, i, v)) {
                cli.emit_source_map_file = v;
            } else if (take_value("--symbolize-offset", arg, argc, argv, i, v)) {
                cli.symbolize_offset_arg = v;
            } else if (arg == "-r" || arg == "--run") {
                if (i + 1 >= argc) {
                    err << "Error: --run requires a function name argument\n";
                    return ParseOutcome::ExitError;
                }
                cli.run_fn = argv[++i];
            } else if (arg == "--emit-shared") {
                cli.emit_shared = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') cli.shared_output_file = argv[++i];
            } else if (arg == "--args") {
                while (i + 1 < argc && argv[i + 1][0] != '-') cli.run_arg_strings.push_back(argv[++i]);
            } else if (arg == "-o") {
                if (i + 1 >= argc) {
                    err << "Error: -o requires an output file argument\n";
                    return ParseOutcome::ExitError;
                }
                cli.output_file = argv[++i];
            } else if (arg.starts_with("-o")) {
                cli.output_file = arg.substr(2);
            } else if (arg == "-") {
                cli.input_file = "-";
            } else if (!arg.empty() && arg[0] == '-') {
                err << "Error: Unknown option (or option missing its value) '" << arg << "'\n";
                return ParseOutcome::ExitError;
            } else {
                cli.input_file = arg;
            }
        }
    } catch (const std::exception& e) {
        err << "Error: bad numeric argument (" << e.what() << ")\n";
        return ParseOutcome::ExitError;
    }
    return ParseOutcome::Run;
}

} // namespace brass::opt
