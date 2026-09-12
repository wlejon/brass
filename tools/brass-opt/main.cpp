#include <brass/brass.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/mir/speculative_inliner.hpp>
#include "opt_actions.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "brass-opt - Brass MIR Optimizer and Compiler Tool (v" << brass::version_string() << ")\n"
              << "Usage: " << prog << " [options] <input-file>\n\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  -v, --verify          Parse and verify MIR module\n"
              << "  -p, --print           Parse, verify, and print canonical MIR\n"
              << "  --check-roundtrip     Assert byte-identical parse(print(x)) roundtrip\n"
              << "  -c, --compile         Compile MIR module to relocatable object file (.obj/.o)\n"
              << "  --format <coff|elf>   Object file format for --compile (default: host format)\n"
              << "  --emit-shared <file>  Compile MIR module directly to shared library (.dll/.so)\n"
              << "  -shared               Compile MIR module directly to shared library (.dll/.so)\n"
              << "  --jit                 Use in-memory JIT execution engine for --run\n"
              << "  -r, --run <fn>        Execute function <fn>\n"
              << "  --args <a1> <a2>...   Arguments to pass to the function executed with --run\n"
              << "  --gc-stress           Enable moving GC stress mode (collects at every allocation/safepoint)\n"
              << "  --inline              Run interprocedural function inlining and IPO optimization pipeline\n"
              << "  --speculative-inlining Run feedback-driven speculative devirtualization and inlining\n"
              << "  --dump-tfv-stats      Dump Type Feedback Vector (TFV) statistics\n"
              << "  --sroa                Run Scalar Replacement of Aggregates (SROA)\n"
              << "  --escape-analysis     Run Escape Analysis on module functions\n"
              << "  --partial-escape      Run Partial Escape Analysis on module functions\n"
              << "  --sink-allocations    Run Allocation Sinking and Hot Path Scalarization\n"
              << "  --dump-pea-stats      Dump Partial Escape Analysis and Allocation Sinking statistics\n"
              << "  --gvn                 Run Global Value Numbering (GVN), RLE & DSE\n"
              << "  --enable-pre, --enable-gvn-pre Run Global Value Numbering with PRE (GVN-PRE)\n"
              << "  --dump-pre-stats      Dump GVN-PRE statistics\n"
              << "  --alias-analysis      Run Alias Analysis on module functions\n"
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
              << "  --sccp                Run Sparse Conditional Constant Propagation (SCCP)\n"
              << "  --guard-elim          Run Speculation Guard Elimination\n"
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
              << "  -o <file>             Write output to <file> instead of stdout\n";
}

bool read_file(const std::string& path, std::string& content) {
    if (path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        content = ss.str();
        return true;
    }
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
    return true;
}

bool write_file(const std::string& path, const std::string& content) {
    if (path.empty() || path == "-") {
        std::cout << content;
        return true;
    }
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    std::string input_file;
    std::string output_file;
    std::string run_fn;
    std::string obj_format;
    std::string shared_output_file;
    std::vector<std::string> run_arg_strings;
    bool verify_only = false;
    bool print_canonical = false;
    bool check_roundtrip = false;
    bool gc_stress = false;
    bool compile_object = false;
    bool emit_shared = false;
    bool use_jit = false;
    bool enable_inlining = false;
    bool enable_speculative_inlining = false;
    bool dump_tfv_stats = false;
    bool enable_sroa = false;
    bool enable_gvn = false;
    bool enable_gvn_pre = false;
    bool dump_pre_stats = false;
    brass::GvnPreStats pre_stats;
    bool run_escape_analysis = false;
    bool enable_partial_escape = false;
    bool enable_allocation_sinking = false;
    bool dump_pea_stats = false;
    bool run_alias_analysis = false;
    bool enable_vectorize = false;
    bool enable_slp = false;
    bool enable_loop_tile = false;
    size_t tile_size = 16;
    bool enable_loop_fusion = false;
    bool enable_loop_distribution = false;
    bool enable_array_contraction = false;
    bool dump_loop_transform_stats = false;
    bool enable_parallel_loops = false;
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0;
    bool dump_parallel_stats = false;
    bool enable_sccp = false;
    bool enable_guard_elim = false;
    bool enable_bce = false;
    bool dump_range_stats = false;
    brass::RangeAnalysisStats range_stats;
    bool enable_cfg_simplify = false;
    bool enable_loop_unswitch = false;
    bool enable_jump_threading = false;
    bool enable_trace_layout = true;
    bool enable_schedule_insns = true;
    bool enable_software_pipeline = false;
    bool enable_pgo_instrument = false;
    std::string pgo_use_file;
    bool dump_branch_probabilities = false;
    bool debug_info = false;
    bool dump_debug_lines = false;
    std::string emit_source_map_file;
    std::string symbolize_offset_arg;
    bool enable_pic = true;
    bool dump_ic_stats = false;
    bool enable_wbe = false;
    bool dump_wbe_stats = false;
    bool enable_avx2 = false;
    bool enable_fma = false;
    uint32_t vector_width = 0;
    bool dump_fma_stats = false;
    brass::FmaOptStats fma_stats;
    bool enable_osr = false;
    uint64_t osr_threshold = brass::runtime::BACKEDGE_OSR_THRESHOLD;
    bool dump_tiering_stats = false;
    bool enable_background_compile = false;
    size_t jit_threads = 2;
    bool dump_jit_thread_stats = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verify") {
            verify_only = true;
        } else if (arg == "-p" || arg == "--print") {
            print_canonical = true;
        } else if (arg == "--check-roundtrip") {
            check_roundtrip = true;
        } else if (arg == "--gc-stress") {
            gc_stress = true;
        } else if (arg == "--inline") {
            enable_inlining = true;
        } else if (arg == "--speculative-inlining" || arg == "--enable-speculative-inlining") {
            enable_speculative_inlining = true;
        } else if (arg == "--dump-tfv-stats") {
            dump_tfv_stats = true;
        } else if (arg == "--sroa") {
            enable_sroa = true;
        } else if (arg == "--escape-analysis") {
            run_escape_analysis = true;
        } else if (arg == "--partial-escape") {
            enable_partial_escape = true;
        } else if (arg == "--sink-allocations") {
            enable_allocation_sinking = true;
            enable_partial_escape = true;
        } else if (arg == "--dump-pea-stats") {
            dump_pea_stats = true;
        } else if (arg == "--gvn") {
            enable_gvn = true;
        } else if (arg == "--enable-pre" || arg == "--enable-gvn-pre") {
            enable_gvn_pre = true;
        } else if (arg == "--dump-pre-stats") {
            dump_pre_stats = true;
            enable_gvn_pre = true;
        } else if (arg == "--sccp") {
            enable_sccp = true;
        } else if (arg == "--guard-elim") {
            enable_guard_elim = true;
            enable_sccp = true;
        } else if (arg == "--bce") {
            enable_bce = true;
        } else if (arg == "--dump-range-stats") {
            dump_range_stats = true;
            enable_bce = true;
        } else if (arg == "--cfg-simplify") {
            enable_cfg_simplify = true;
        } else if (arg == "--loop-unswitch") {
            enable_loop_unswitch = true;
        } else if (arg == "--jump-threading") {
            enable_jump_threading = true;
        } else if (arg == "--trace-layout") {
            enable_trace_layout = true;
        } else if (arg == "--no-trace-layout") {
            enable_trace_layout = false;
        } else if (arg == "--schedule-insns") {
            enable_schedule_insns = true;
        } else if (arg == "--no-schedule-insns") {
            enable_schedule_insns = false;
        } else if (arg == "--software-pipeline") {
            enable_software_pipeline = true;
        } else if (arg == "--no-software-pipeline") {
            enable_software_pipeline = false;
        } else if (arg == "--wbe" || arg == "--enable-wbe") {
            enable_wbe = true;
        } else if (arg == "--dump-wbe-stats") {
            dump_wbe_stats = true;
            enable_wbe = true;
        } else if (arg == "--enable-osr") {
            enable_osr = true;
        } else if (arg.rfind("--osr-threshold=", 0) == 0) {
            enable_osr = true;
            osr_threshold = std::stoull(arg.substr(16));
        } else if (arg == "--osr-threshold" && i + 1 < argc) {
            enable_osr = true;
            osr_threshold = std::stoull(argv[++i]);
        } else if (arg == "--dump-tiering-stats") {
            dump_tiering_stats = true;
        } else if (arg == "--enable-background-compile") {
            enable_background_compile = true;
        } else if (arg.rfind("--jit-threads=", 0) == 0) {
            jit_threads = static_cast<size_t>(std::stoul(arg.substr(14)));
        } else if (arg == "--jit-threads" && i + 1 < argc) {
            jit_threads = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--dump-jit-thread-stats") {
            dump_jit_thread_stats = true;
        } else if (arg == "--pgo-instrument") {
            enable_pgo_instrument = true;
        } else if (arg.rfind("--pgo-use=", 0) == 0) {
            pgo_use_file = arg.substr(10);
        } else if (arg == "--pgo-use") {
            if (i + 1 < argc) {
                pgo_use_file = argv[++i];
            } else {
                std::cerr << "Error: --pgo-use requires a file path\n";
                return 1;
            }
        } else if (arg == "--dump-branch-probabilities") {
            dump_branch_probabilities = true;
        } else if (arg == "--alias-analysis") {
            run_alias_analysis = true;
        } else if (arg == "--vectorize") {
            enable_vectorize = true;
        } else if (arg == "--enable-avx2") {
            enable_avx2 = true;
        } else if (arg == "--enable-fma" || arg == "--fma") {
            enable_fma = true;
        } else if (arg == "--vector-width=256" || arg == "--vector-width-256") {
            vector_width = 256;
        } else if (arg.rfind("--vector-width=", 0) == 0) {
            vector_width = static_cast<uint32_t>(std::stoul(arg.substr(15)));
        } else if (arg == "--dump-fma-stats") {
            dump_fma_stats = true;
        } else if (arg == "--slp") {
            enable_slp = true;
        } else if (arg == "--loop-tile") {
            enable_loop_tile = true;
        } else if (arg == "--enable-loop-fusion") {
            enable_loop_fusion = true;
        } else if (arg == "--enable-loop-distribution") {
            enable_loop_distribution = true;
        } else if (arg == "--enable-array-contraction") {
            enable_array_contraction = true;
        } else if (arg == "--dump-loop-transform-stats") {
            dump_loop_transform_stats = true;
        } else if (arg == "--enable-parallel-loops") {
            enable_parallel_loops = true;
        } else if (arg.rfind("--parallel-threshold=", 0) == 0) {
            parallel_threshold = std::stoull(arg.substr(21));
        } else if (arg == "--parallel-threshold" && i + 1 < argc) {
            parallel_threshold = std::stoull(argv[++i]);
        } else if (arg.rfind("--parallel-workers=", 0) == 0) {
            parallel_workers = static_cast<uint32_t>(std::stoul(arg.substr(19)));
        } else if (arg == "--parallel-workers" && i + 1 < argc) {
            parallel_workers = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--dump-parallel-stats") {
            dump_parallel_stats = true;
        } else if (arg == "--tile-size") {
            if (i + 1 < argc) {
                tile_size = static_cast<size_t>(std::stoul(argv[++i]));
            } else {
                std::cerr << "Error: --tile-size requires an integer argument\n";
                return 1;
            }
        } else if (arg.rfind("--tile-size=", 0) == 0) {
            tile_size = static_cast<size_t>(std::stoul(arg.substr(12)));
        } else if (arg == "--enable-pic") {
            enable_pic = true;
        } else if (arg == "--no-pic") {
            enable_pic = false;
        } else if (arg == "--dump-ic-stats") {
            dump_ic_stats = true;
        } else if (arg == "-c" || arg == "--compile") {
            compile_object = true;
        } else if (arg == "--emit-shared") {
            emit_shared = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                shared_output_file = argv[++i];
            }
        } else if (arg == "-shared") {
            emit_shared = true;
        } else if (arg == "--jit") {
            use_jit = true;
        } else if (arg == "--format") {
            if (i + 1 < argc) {
                obj_format = argv[++i];
            } else {
                std::cerr << "Error: --format requires an argument (coff or elf)\n";
                return 1;
            }
        } else if (arg == "-r" || arg == "--run") {
            if (i + 1 < argc) {
                run_fn = argv[++i];
            } else {
                std::cerr << "Error: --run requires a function name argument\n";
                return 1;
            }
        } else if (arg == "--args") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                run_arg_strings.push_back(argv[++i]);
            }
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: -o requires an output file argument\n";
                return 1;
            }
        } else if (arg.starts_with("-o")) {
            output_file = arg.substr(2);
        } else if (arg == "-g" || arg == "--debug-info" || arg == "--emit-debug-info") {
            debug_info = true;
        } else if (arg == "--dump-debug-lines") {
            dump_debug_lines = true;
        } else if (arg.starts_with("--emit-source-map=")) {
            emit_source_map_file = arg.substr(18);
        } else if (arg == "--emit-source-map") {
            if (i + 1 < argc) {
                emit_source_map_file = argv[++i];
            } else {
                std::cerr << "Error: --emit-source-map requires a file argument\n";
                return 1;
            }
        } else if (arg.starts_with("--symbolize-offset=")) {
            symbolize_offset_arg = arg.substr(19);
        } else if (arg == "--symbolize-offset") {
            if (i + 1 < argc) {
                symbolize_offset_arg = argv[++i];
            } else {
                std::cerr << "Error: --symbolize-offset requires <fn,offset> argument\n";
                return 1;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            if (arg == "-") {
                input_file = "-";
            } else {
                std::cerr << "Error: Unknown option '" << arg << "'\n";
                return 1;
            }
        } else {
            input_file = arg;
        }
    }

    (void)enable_trace_layout;
    (void)enable_pic;

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified.\n";
        return 1;
    }

    std::string source_text;
    if (!read_file(input_file, source_text)) {
        std::cerr << "Error: Could not read input file '" << input_file << "'\n";
        return 1;
    }

    brass::DiagnosticReporter diag;
    auto mod = brass::parse_module(source_text, &diag, input_file);
    if (!mod || diag.has_errors()) {
        std::cerr << diag.format_all();
        return 1;
    }

    bool ok = brass::verify_module(*mod, &diag);
    if (!ok || diag.has_errors()) {
        std::cerr << diag.format_all();
        return 1;
    }

    std::unique_ptr<brass::pgo::ProfileData> pgo_profile;
    if (!pgo_use_file.empty()) {
        std::string pgo_err;
        pgo_profile = brass::pgo::ProfileData::read_from_file(pgo_use_file, &pgo_err);
        if (!pgo_profile) {
            std::cerr << "Error: Could not load profile from '" << pgo_use_file << "': " << pgo_err << "\n";
            return 1;
        }
        if (!dump_branch_probabilities) {
            brass::pgo::optimize_module_pgo(*mod, *pgo_profile);
            brass::DiagnosticReporter pgo_diag;
            if (!brass::verify_module(*mod, &pgo_diag) || pgo_diag.has_errors()) {
                std::cerr << "Verification failed after PGO optimization:\n" << pgo_diag.format_all();
                return 1;
            }
        }
    }

    if (dump_branch_probabilities) {
        if (!pgo_profile) {
            std::cerr << "Error: --dump-branch-probabilities requires --pgo-use=<file.bprof>\n";
            return 1;
        }
        for (const brass::Function* fn : mod->functions()) {
            if (!fn) continue;
            const auto* fp = pgo_profile->find_function(std::string(fn->name()));
            if (!fp) {
                std::cout << "Function '" << fn->name() << "': No profile available\n";
                continue;
            }
            brass::mir::BranchProbabilityAnalysis bpa(*fn, *fp);
            const auto& bfi = bpa.block_frequency_info();
            const auto& bpi = bpa.branch_probability_info();
            std::cout << "Function '" << fn->name() << "' (entry count=" << bfi.entry_count() << "):\n";
            for (const brass::BasicBlock* bb : fn->blocks()) {
                if (!bb) continue;
                std::cout << "  block " << bb->name() << ": count=" << bfi.get_block_count(bb)
                          << ", freq=" << bfi.get_block_frequency(bb)
                          << (bfi.is_hot_block(bb) ? " [HOT]" : "")
                          << (bfi.is_cold_block(bb) ? " [COLD]" : "") << "\n";
                for (const brass::BasicBlock* succ : bb->successors()) {
                    if (!succ) continue;
                    std::cout << "    edge -> " << succ->name()
                              << ": count=" << bpi.get_edge_count(bb, succ)
                              << ", prob=" << bpi.get_edge_probability(bb, succ) << "\n";
                }
            }
        }
    }

    if (enable_pgo_instrument) {
        brass::pgo::instrument_module(*mod);
        brass::DiagnosticReporter pgo_diag;
        if (!brass::verify_module(*mod, &pgo_diag) || pgo_diag.has_errors()) {
            std::cerr << "Verification failed after PGO instrumentation:\n" << pgo_diag.format_all();
            return 1;
        }
    }

    if (run_escape_analysis) {
        for (const brass::Function* fn : mod->functions()) {
            if (!fn) continue;
            brass::EscapeAnalysis ea(*fn);
            std::cout << "Function '" << fn->name() << "': "
                      << ea.allocations().size() << " allocations ("
                      << ea.non_escaping_allocations().size() << " non-escaping)\n";
            for (const brass::Value* alloc_val : ea.allocations()) {
                std::cout << "  alloc %" << alloc_val->id() << ": "
                          << brass::escape_state_name(ea.get_escape_state(alloc_val)) << "\n";
            }
        }
    }

    brass::PartialEscapeStats pea_stats;
    brass::LoopOptStats loop_stats;
    if (enable_partial_escape && !enable_allocation_sinking) {
        for (const brass::Function* fn : mod->functions()) {
            if (!fn) continue;
            brass::PartialEscapeAnalysis pea(*fn);
            std::cout << "Function '" << fn->name() << "': "
                      << pea.candidate_allocations().size() << " candidate allocations\n";
            for (const brass::Value* alloc_val : pea.candidate_allocations()) {
                auto frontier = pea.get_materialization_frontier(alloc_val);
                std::cout << "  alloc %" << alloc_val->id() << ": frontier " << frontier.size() << " edges\n";
                for (const auto& e : frontier) {
                    std::cout << "    edge " << (e.from ? e.from->name() : "<null>")
                              << " -> " << (e.to ? e.to->name() : "<null>") << "\n";
                }
            }
        }
    }

    if (enable_allocation_sinking) {
        brass::AllocationSinkingOptions sink_opts;
        sink_opts.stats = &pea_stats;
        brass::sink_allocations(*mod, sink_opts);
        brass::DiagnosticReporter pea_diag;
        if (!brass::verify_module(*mod, &pea_diag) || pea_diag.has_errors()) {
            std::cerr << "Verification failed after Allocation Sinking:\n" << pea_diag.format_all();
            return 1;
        }
    }

    if (run_alias_analysis) {
        for (const brass::Function* fn : mod->functions()) {
            if (!fn) continue;
            brass::AliasAnalysis aa(*fn);
            std::cout << "Alias Analysis for Function '" << fn->name() << "':\n";
            std::vector<const brass::Instruction*> mem_insts;
            for (const brass::BasicBlock* bb : fn->blocks()) {
                if (!bb) continue;
                for (const brass::Instruction* inst : *bb) {
                    if (inst && brass::is_memory(inst->opcode())) {
                        mem_insts.push_back(inst);
                    }
                }
            }
            std::cout << "  " << mem_insts.size() << " memory instructions\n";
            for (size_t i = 0; i < mem_insts.size(); ++i) {
                for (size_t j = i + 1; j < mem_insts.size(); ++j) {
                    const brass::Instruction* m1 = mem_insts[i];
                    const brass::Instruction* m2 = mem_insts[j];
                    if (m1->operand_count() > 0 && m2->operand_count() > 0) {
                        brass::AliasResult res = aa.alias(m1->operand(0), m1->offset(), m1->memory_type(),
                                                          m2->operand(0), m2->offset(), m2->memory_type());
                        std::cout << "  " << brass::opcode_name(m1->opcode()) << " (off " << m1->offset() << ") vs "
                                  << brass::opcode_name(m2->opcode()) << " (off " << m2->offset() << "): "
                                  << brass::alias_result_name(res) << "\n";
                    }
                }
            }
        }
    }

    auto init_loop_opts = [&](brass::LoopOptOptions& loop_opts) {
        loop_opts.enable_vectorize = enable_vectorize;
        loop_opts.enable_slp = enable_slp;
        loop_opts.enable_loop_tile = enable_loop_tile;
        loop_opts.enable_loop_fusion = enable_loop_fusion;
        loop_opts.enable_loop_distribution = enable_loop_distribution;
        loop_opts.enable_array_contraction = enable_array_contraction;
        loop_opts.stats = &loop_stats;
        loop_opts.tile_size_i = tile_size;
        loop_opts.tile_size_j = tile_size;
        loop_opts.tile_size_k = tile_size;
        loop_opts.enable_avx2 = enable_avx2;
        loop_opts.enable_fma = enable_fma;
        loop_opts.vector_width = vector_width;
        loop_opts.dump_fma_stats = dump_fma_stats;
        loop_opts.fma_stats = &fma_stats;
        loop_opts.enable_parallel_loops = enable_parallel_loops;
        loop_opts.parallel_threshold = parallel_threshold;
        loop_opts.parallel_workers = parallel_workers;
        loop_opts.dump_parallel_stats = dump_parallel_stats;
        loop_opts.enable_bce = enable_bce;
        loop_opts.dump_range_stats = dump_range_stats;
        loop_opts.range_stats = &range_stats;
    };

    if (enable_inlining) {
        brass::InlinerOptions inliner_opts;
        inliner_opts.enable_gvn = enable_gvn;
        inliner_opts.enable_speculative_devirtualization = enable_speculative_inlining;
        brass::LoopOptOptions loop_opts;
        init_loop_opts(loop_opts);
        loop_opts.enable_gvn = enable_gvn;
        loop_opts.enable_sccp = enable_sccp;
        loop_opts.enable_guard_elim = enable_guard_elim;
        loop_opts.enable_cfg_simplify = enable_cfg_simplify;
        loop_opts.enable_loop_unswitch = enable_loop_unswitch;
        loop_opts.enable_jump_threading = enable_jump_threading;
        loop_opts.enable_trace_layout = enable_trace_layout;
        brass::optimize_module_ipo(*mod, inliner_opts, loop_opts);
        brass::DiagnosticReporter inlining_diag;
        if (!brass::verify_module(*mod, &inlining_diag) || inlining_diag.has_errors()) {
            std::cerr << "Verification failed after inlining:\n" << inlining_diag.format_all();
            return 1;
        }
    } else if (enable_speculative_inlining) {
        brass::SpeculativeInlinerOptions spec_opts;
        spec_opts.enable_inlining = true;
        spec_opts.enable_polymorphic = true;
        brass::run_speculative_devirtualization(*mod, spec_opts);
        brass::DiagnosticReporter spec_diag;
        if (!brass::verify_module(*mod, &spec_diag) || spec_diag.has_errors()) {
            std::cerr << "Verification failed after speculative inlining:\n" << spec_diag.format_all();
            return 1;
        }
    } else if (enable_sroa) {
        brass::sroa_module(*mod);
        brass::DiagnosticReporter sroa_diag;
        if (!brass::verify_module(*mod, &sroa_diag) || sroa_diag.has_errors()) {
            std::cerr << "Verification failed after SROA:\n" << sroa_diag.format_all();
            return 1;
        }
        if (enable_gvn) {
            brass::gvn_module(*mod);
            brass::DiagnosticReporter gvn_diag;
            if (!brass::verify_module(*mod, &gvn_diag) || gvn_diag.has_errors()) {
                std::cerr << "Verification failed after GVN:\n" << gvn_diag.format_all();
                return 1;
            }
        }
        if (enable_gvn_pre) {
            brass::GvnPreOptions pre_opts;
            pre_opts.stats = &pre_stats;
            brass::gvn_pre_module(*mod, pre_opts);
            brass::DiagnosticReporter pre_diag;
            if (!brass::verify_module(*mod, &pre_diag) || pre_diag.has_errors()) {
                std::cerr << "Verification failed after GVN-PRE:\n" << pre_diag.format_all();
                return 1;
            }
        }
        if (enable_sccp) {
            brass::SccpOptions sccp_opts;
            sccp_opts.enable_guard_elim = enable_guard_elim;
            brass::sccp_module(*mod, sccp_opts);
            brass::DiagnosticReporter sccp_diag;
            if (!brass::verify_module(*mod, &sccp_diag) || sccp_diag.has_errors()) {
                std::cerr << "Verification failed after SCCP:\n" << sccp_diag.format_all();
                return 1;
            }
        }
        if (enable_cfg_simplify) {
            brass::cfg_simplify_module(*mod);
            brass::DiagnosticReporter cfg_diag;
            if (!brass::verify_module(*mod, &cfg_diag) || cfg_diag.has_errors()) {
                std::cerr << "Verification failed after CFG Simplification:\n" << cfg_diag.format_all();
                return 1;
            }
        }
        bool any_loop_opt = enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion || enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops;
        if (any_loop_opt) {
            brass::LoopOptOptions loop_opts;
            init_loop_opts(loop_opts);
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_gvn || enable_gvn_pre) {
        if (enable_gvn) {
            brass::gvn_module(*mod);
            brass::DiagnosticReporter gvn_diag;
            if (!brass::verify_module(*mod, &gvn_diag) || gvn_diag.has_errors()) {
                std::cerr << "Verification failed after GVN:\n" << gvn_diag.format_all();
                return 1;
            }
        }
        if (enable_gvn_pre) {
            brass::GvnPreOptions pre_opts;
            pre_opts.stats = &pre_stats;
            brass::gvn_pre_module(*mod, pre_opts);
            brass::DiagnosticReporter pre_diag;
            if (!brass::verify_module(*mod, &pre_diag) || pre_diag.has_errors()) {
                std::cerr << "Verification failed after GVN-PRE:\n" << pre_diag.format_all();
                return 1;
            }
        }
        if (enable_sccp) {
            brass::SccpOptions sccp_opts;
            sccp_opts.enable_guard_elim = enable_guard_elim;
            brass::sccp_module(*mod, sccp_opts);
            brass::DiagnosticReporter sccp_diag;
            if (!brass::verify_module(*mod, &sccp_diag) || sccp_diag.has_errors()) {
                std::cerr << "Verification failed after SCCP:\n" << sccp_diag.format_all();
                return 1;
            }
        }
        if (enable_cfg_simplify) {
            brass::cfg_simplify_module(*mod);
            brass::DiagnosticReporter cfg_diag;
            if (!brass::verify_module(*mod, &cfg_diag) || cfg_diag.has_errors()) {
                std::cerr << "Verification failed after CFG Simplification:\n" << cfg_diag.format_all();
                return 1;
            }
        }
        bool any_loop_opt = enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion || enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops;
        if (any_loop_opt) {
            brass::LoopOptOptions loop_opts;
            init_loop_opts(loop_opts);
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_sccp) {
        brass::SccpOptions sccp_opts;
        sccp_opts.enable_guard_elim = enable_guard_elim;
        brass::sccp_module(*mod, sccp_opts);
        brass::DiagnosticReporter sccp_diag;
        if (!brass::verify_module(*mod, &sccp_diag) || sccp_diag.has_errors()) {
            std::cerr << "Verification failed after SCCP:\n" << sccp_diag.format_all();
            return 1;
        }
        if (enable_cfg_simplify) {
            brass::cfg_simplify_module(*mod);
            brass::DiagnosticReporter cfg_diag;
            if (!brass::verify_module(*mod, &cfg_diag) || cfg_diag.has_errors()) {
                std::cerr << "Verification failed after CFG Simplification:\n" << cfg_diag.format_all();
                return 1;
            }
        }
        bool any_loop_opt = enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion || enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops;
        if (any_loop_opt) {
            brass::LoopOptOptions loop_opts;
            init_loop_opts(loop_opts);
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_cfg_simplify) {
        brass::cfg_simplify_module(*mod);
        brass::DiagnosticReporter cfg_diag;
        if (!brass::verify_module(*mod, &cfg_diag) || cfg_diag.has_errors()) {
            std::cerr << "Verification failed after CFG Simplification:\n" << cfg_diag.format_all();
            return 1;
        }
        bool any_loop_opt = enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion || enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops;
        if (any_loop_opt) {
            brass::LoopOptOptions loop_opts;
            init_loop_opts(loop_opts);
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_bce || enable_loop_tile || enable_vectorize || enable_slp || enable_loop_fusion || enable_loop_distribution || enable_array_contraction || enable_fma || enable_parallel_loops) {
        if (enable_bce) {
            brass::RangeAnalysisOptions bce_opts;
            bce_opts.enable_bce = true;
            bce_opts.enable_hoisting = true;
            bce_opts.dump_stats = dump_range_stats;
            bce_opts.stats = &range_stats;
            brass::run_bounds_check_elimination(*mod, bce_opts);
            brass::DiagnosticReporter bce_diag;
            if (!brass::verify_module(*mod, &bce_diag) || bce_diag.has_errors()) {
                std::cerr << "Verification failed after BCE:\n" << bce_diag.format_all();
                return 1;
            }
        }
        brass::LoopOptOptions loop_opts;
        init_loop_opts(loop_opts);
        brass::optimize_module_loops(*mod, loop_opts);
        brass::DiagnosticReporter opt_diag;
        if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
            std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
            return 1;
        }
    }

    if (enable_loop_unswitch) {
        brass::unswitch_loops_in_module(*mod);
        brass::DiagnosticReporter unsw_diag;
        if (!brass::verify_module(*mod, &unsw_diag) || unsw_diag.has_errors()) {
            std::cerr << "Verification failed after Loop Unswitching:\n" << unsw_diag.format_all();
            return 1;
        }
    }

    if (enable_jump_threading) {
        brass::jump_thread_module(*mod);
        brass::DiagnosticReporter jt_diag;
        if (!brass::verify_module(*mod, &jt_diag) || jt_diag.has_errors()) {
            std::cerr << "Verification failed after Jump Threading:\n" << jt_diag.format_all();
            return 1;
        }
    }

    if (enable_wbe) {
        brass::WriteBarrierElimination wbe(dump_wbe_stats);
        wbe.run_on_module(*mod);
        brass::DiagnosticReporter wbe_diag;
        if (!brass::verify_module(*mod, &wbe_diag) || wbe_diag.has_errors()) {
            std::cerr << "Verification failed after Write Barrier Elimination:\n" << wbe_diag.format_all();
            return 1;
        }
        if (dump_wbe_stats) {
            wbe.dump_stats(std::cout);
        }
    }

    if (dump_pea_stats) {
        std::cout << pea_stats.format_report() << "\n";
    }

    if (dump_pre_stats) {
        pre_stats.dump(std::cout);
    }

    if (dump_loop_transform_stats) {
        std::cout << loop_stats.fusion_stats.format_report()
                  << loop_stats.distribution_stats.format_report()
                  << loop_stats.contraction_stats.format_report();
    }

    if (dump_fma_stats) {
        std::cout << fma_stats.format_report() << "\n";
    }

    if (dump_parallel_stats) {
        std::cout << loop_stats.parallel_stats.format_report();
    }

    if (dump_range_stats) {
        range_stats.dump(std::cout);
    }

    (void)debug_info;

    if (dump_debug_lines) {
        brass::execute_dump_debug_lines(*mod);
        if (!compile_object && !emit_shared && run_fn.empty()) {
            return 0;
        }
    }

    if (!symbolize_offset_arg.empty()) {
        return brass::execute_symbolize_offset(*mod, symbolize_offset_arg) ? 0 : 1;
    }

    if (!emit_source_map_file.empty()) {
        if (!brass::execute_emit_source_map(*mod, emit_source_map_file)) {
            return 1;
        }
        if (!compile_object && !emit_shared && run_fn.empty()) {
            return 0;
        }
    }

    if (compile_object) {
        brass::CompileObjectOptions c_opts;
        c_opts.obj_format = obj_format;
        c_opts.enable_schedule_insns = enable_schedule_insns;
        c_opts.enable_software_pipeline = enable_software_pipeline;
        c_opts.output_file = output_file;
        c_opts.input_file = input_file;
        return brass::execute_compile_object(*mod, c_opts) ? 0 : 1;
    }

    if (emit_shared) {
        brass::EmitSharedOptions s_opts;
        s_opts.obj_format = obj_format;
        s_opts.shared_output_file = shared_output_file;
        s_opts.output_file = output_file;
        s_opts.input_file = input_file;
        return brass::execute_emit_shared(*mod, s_opts) ? 0 : 1;
    }

    if (!run_fn.empty()) {
        brass::RunFunctionOptions r_opts;
        r_opts.run_fn = run_fn;
        r_opts.run_arg_strings = run_arg_strings;
        r_opts.use_jit = use_jit;
        r_opts.gc_stress = gc_stress;
        r_opts.enable_osr = enable_osr;
        r_opts.osr_threshold = osr_threshold;
        r_opts.enable_schedule_insns = enable_schedule_insns;
        r_opts.enable_software_pipeline = enable_software_pipeline;
        r_opts.dump_ic_stats = dump_ic_stats;
        r_opts.dump_tiering_stats = dump_tiering_stats;
        r_opts.enable_background_compile = enable_background_compile;
        r_opts.jit_threads = jit_threads;
        r_opts.dump_jit_thread_stats = dump_jit_thread_stats;
        return brass::execute_run_function(*mod, r_opts) ? 0 : 1;
    }

    if (check_roundtrip) {
        std::string canonical1 = brass::to_string(*mod);
        brass::DiagnosticReporter rt_diag;
        auto mod2 = brass::parse_module(canonical1, &rt_diag, "<canonical-roundtrip>");
        if (!mod2 || rt_diag.has_errors()) {
            std::cerr << "Roundtrip parse failed:\n" << rt_diag.format_all() << "\n";
            return 1;
        }

        bool rt_ok = brass::verify_module(*mod2, &rt_diag);
        if (!rt_ok || rt_diag.has_errors()) {
            std::cerr << "Roundtrip verify failed:\n" << rt_diag.format_all() << "\n";
            return 1;
        }

        std::string canonical2 = brass::to_string(*mod2);
        if (canonical1 != canonical2) {
            std::cerr << "Roundtrip mismatch: canonical representation is not byte-identical!\n";
            std::cerr << "--- First Canon ---\n" << canonical1
                      << "--- Second Canon ---\n" << canonical2 << "\n";
            return 1;
        }

        std::cout << "Roundtrip verified: byte-identical canonical representation ("
                  << canonical1.size() << " bytes)\n";
        return 0;
    }

    if (verify_only && !print_canonical) {
        std::cout << "Module '" << mod->name() << "' verified successfully ("
                  << mod->function_count() << " functions).\n";
        return 0;
    }

    std::string canonical = brass::to_string(*mod);
    if (!write_file(output_file, canonical)) {
        std::cerr << "Error: Could not write output file '" << output_file << "'\n";
        return 1;
    }

    if (dump_tfv_stats) {
        brass::runtime::FeedbackRegistry::instance().dump_stats(std::cout);
    }

    return 0;
}
