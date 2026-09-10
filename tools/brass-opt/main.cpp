#include <brass/brass.hpp>
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
              << "  --sroa                Run Scalar Replacement of Aggregates (SROA)\n"
              << "  --escape-analysis     Run Escape Analysis on module functions\n"
              << "  --gvn                 Run Global Value Numbering (GVN), RLE & DSE\n"
              << "  --alias-analysis      Run Alias Analysis on module functions\n"
              << "  --vectorize           Run loop vectorization on countable loops\n"
              << "  --slp                 Run SLP straight-line vectorization\n"
              << "  --loop-tile           Run loop tiling / cache blocking on nested loops\n"
              << "  --tile-size <N>       Tile size for loop tiling (default: 16)\n"
              << "  --sccp                Run Sparse Conditional Constant Propagation (SCCP)\n"
              << "  --guard-elim          Run Speculation Guard Elimination\n"
              << "  --cfg-simplify        Run CFG Simplification & Dead Block Compaction\n"
              << "  --loop-unswitch       Run Loop Unswitching on candidate loops\n"
              << "  --jump-threading      Run SSA Jump Threading\n"
              << "  --trace-layout        Run LIR Trace Scheduling & Fall-Through Block Layout\n"
              << "  --schedule-insns      Run Machine Instruction Scheduling\n"
              << "  --software-pipeline   Run Loop Modulo Scheduling & Software Pipelining\n"
              << "  --pgo-instrument      Instrument module with Knuth-Stevenson minimal edge counters\n"
              << "  --pgo-use=<file>      Load profile data (.bprof) for profile-guided optimization\n"
              << "  --dump-branch-probabilities Dump block frequencies and edge branch probabilities\n"
              << "  --enable-pic          Enable Polymorphic Inline Caching for dynamic property accesses\n"
              << "  --dump-ic-stats       Dump Inline Cache hit/miss and state statistics\n"
              << "  -g, --debug-info      Preserve and emit debug information and .brass_dbg section\n"
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

bool write_binary_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return true;
}

brass::RuntimeValue parse_arg_for_type(brass::Type type, const std::string& arg_str) {
    switch (type.kind()) {
        case brass::TypeKind::I32:
            return brass::RuntimeValue::from_i32(static_cast<int32_t>(std::stol(arg_str, nullptr, 0)));
        case brass::TypeKind::I64:
            return brass::RuntimeValue::from_i64(std::stoll(arg_str, nullptr, 0));
        case brass::TypeKind::F32:
            return brass::RuntimeValue::from_f32(std::stof(arg_str));
        case brass::TypeKind::F64:
            return brass::RuntimeValue::from_f64(std::stod(arg_str));
        case brass::TypeKind::Ptr:
            return brass::RuntimeValue::from_ptr(static_cast<uintptr_t>(std::stoull(arg_str, nullptr, 0)));
        case brass::TypeKind::GCRef:
            return brass::RuntimeValue::from_gcref(static_cast<uintptr_t>(std::stoull(arg_str, nullptr, 0)));
        case brass::TypeKind::Void:
            return brass::RuntimeValue::from_void();
        default:
            break;
    }
    return brass::RuntimeValue::from_i64(std::stoll(arg_str, nullptr, 0));
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
    bool enable_sroa = false;
    bool enable_gvn = false;
    bool run_escape_analysis = false;
    bool run_alias_analysis = false;
    bool enable_vectorize = false;
    bool enable_slp = false;
    bool enable_loop_tile = false;
    size_t tile_size = 16;
    bool enable_sccp = false;
    bool enable_guard_elim = false;
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
    std::string emit_source_map_file;
    std::string symbolize_offset_arg;
    bool enable_pic = true;
    bool dump_ic_stats = false;

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
        } else if (arg == "--sroa") {
            enable_sroa = true;
        } else if (arg == "--escape-analysis") {
            run_escape_analysis = true;
        } else if (arg == "--gvn") {
            enable_gvn = true;
        } else if (arg == "--sccp") {
            enable_sccp = true;
        } else if (arg == "--guard-elim") {
            enable_guard_elim = true;
            enable_sccp = true;
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
        } else if (arg == "--slp") {
            enable_slp = true;
        } else if (arg == "--loop-tile") {
            enable_loop_tile = true;
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
        } else if (arg == "-g" || arg == "--debug-info") {
            debug_info = true;
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

    if (enable_inlining) {
        brass::InlinerOptions inliner_opts;
        inliner_opts.enable_gvn = enable_gvn;
        brass::LoopOptOptions loop_opts;
        loop_opts.enable_vectorize = enable_vectorize;
        loop_opts.enable_slp = enable_slp;
        loop_opts.enable_gvn = enable_gvn;
        loop_opts.enable_sccp = enable_sccp;
        loop_opts.enable_guard_elim = enable_guard_elim;
        loop_opts.enable_cfg_simplify = enable_cfg_simplify;
        loop_opts.enable_loop_tile = enable_loop_tile;
        loop_opts.enable_loop_unswitch = enable_loop_unswitch;
        loop_opts.enable_jump_threading = enable_jump_threading;
        loop_opts.enable_trace_layout = enable_trace_layout;
        loop_opts.tile_size_i = tile_size;
        loop_opts.tile_size_j = tile_size;
        loop_opts.tile_size_k = tile_size;
        brass::optimize_module_ipo(*mod, inliner_opts, loop_opts);
        brass::DiagnosticReporter inlining_diag;
        if (!brass::verify_module(*mod, &inlining_diag) || inlining_diag.has_errors()) {
            std::cerr << "Verification failed after inlining:\n" << inlining_diag.format_all();
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
        if (enable_loop_tile || enable_vectorize || enable_slp) {
            brass::LoopOptOptions loop_opts;
            loop_opts.enable_vectorize = enable_vectorize;
            loop_opts.enable_slp = enable_slp;
            loop_opts.enable_loop_tile = enable_loop_tile;
            loop_opts.tile_size_i = tile_size;
            loop_opts.tile_size_j = tile_size;
            loop_opts.tile_size_k = tile_size;
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_gvn) {
        brass::gvn_module(*mod);
        brass::DiagnosticReporter gvn_diag;
        if (!brass::verify_module(*mod, &gvn_diag) || gvn_diag.has_errors()) {
            std::cerr << "Verification failed after GVN:\n" << gvn_diag.format_all();
            return 1;
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
        if (enable_loop_tile || enable_vectorize || enable_slp) {
            brass::LoopOptOptions loop_opts;
            loop_opts.enable_vectorize = enable_vectorize;
            loop_opts.enable_slp = enable_slp;
            loop_opts.enable_loop_tile = enable_loop_tile;
            loop_opts.tile_size_i = tile_size;
            loop_opts.tile_size_j = tile_size;
            loop_opts.tile_size_k = tile_size;
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
        if (enable_loop_tile || enable_vectorize || enable_slp) {
            brass::LoopOptOptions loop_opts;
            loop_opts.enable_vectorize = enable_vectorize;
            loop_opts.enable_slp = enable_slp;
            loop_opts.enable_loop_tile = enable_loop_tile;
            loop_opts.tile_size_i = tile_size;
            loop_opts.tile_size_j = tile_size;
            loop_opts.tile_size_k = tile_size;
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
        if (enable_loop_tile || enable_vectorize || enable_slp) {
            brass::LoopOptOptions loop_opts;
            loop_opts.enable_vectorize = enable_vectorize;
            loop_opts.enable_slp = enable_slp;
            loop_opts.enable_loop_tile = enable_loop_tile;
            loop_opts.tile_size_i = tile_size;
            loop_opts.tile_size_j = tile_size;
            loop_opts.tile_size_k = tile_size;
            brass::optimize_module_loops(*mod, loop_opts);
            brass::DiagnosticReporter opt_diag;
            if (!brass::verify_module(*mod, &opt_diag) || opt_diag.has_errors()) {
                std::cerr << "Verification failed after loop optimization/vectorization:\n" << opt_diag.format_all();
                return 1;
            }
        }
    } else if (enable_loop_tile || enable_vectorize || enable_slp) {
        brass::LoopOptOptions loop_opts;
        loop_opts.enable_vectorize = enable_vectorize;
        loop_opts.enable_slp = enable_slp;
        loop_opts.enable_loop_tile = enable_loop_tile;
        loop_opts.tile_size_i = tile_size;
        loop_opts.tile_size_j = tile_size;
        loop_opts.tile_size_k = tile_size;
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

    (void)debug_info;

    if (!symbolize_offset_arg.empty()) {
        size_t comma = symbolize_offset_arg.find(',');
        if (comma == std::string::npos) {
            std::cerr << "Error: Invalid --symbolize-offset format, expected <fn,offset>\n";
            return 1;
        }
        std::string fn_name = symbolize_offset_arg.substr(0, comma);
        uint32_t offset = static_cast<uint32_t>(std::stoul(symbolize_offset_arg.substr(comma + 1), nullptr, 0));

        brass::Target target = brass::Target::host();
        brass::codegen::SchedOptions sched_opts;
        auto obj = brass::object::compile_module_to_object(*mod, target, sched_opts);
        brass::Symbolicator symbolicator(mod->debug_context(), obj.debug_tables);
        brass::StackTrace trace = symbolicator.symbolize_offset(fn_name, offset);
        std::cout << trace.format();
        return 0;
    }

    if (!emit_source_map_file.empty()) {
        brass::Target target = brass::Target::host();
        brass::codegen::SchedOptions sched_opts;
        auto obj = brass::object::compile_module_to_object(*mod, target, sched_opts);
        brass::SourceMap combined_sm(mod->name().empty() ? "output" : std::string(mod->name()));
        for (const auto& f_path : mod->debug_context().files()) {
            combined_sm.add_source(f_path);
        }
        for (const auto& table : obj.debug_tables) {
            for (const auto& entry : table.line_entries()) {
                combined_sm.add_mapping(entry.code_offset, entry.loc);
            }
        }
        std::string sm_err;
        if (!combined_sm.write_file(emit_source_map_file, &sm_err)) {
            std::cerr << "Error writing source map: " << sm_err << "\n";
            return 1;
        }
        if (!compile_object && !emit_shared && run_fn.empty()) {
            std::cout << "Successfully emitted source map to '" << emit_source_map_file << "'\n";
            return 0;
        }
    }

    if (compile_object) {
        brass::Target target = brass::Target::host();
        if (obj_format == "coff") {
            target = brass::Target::x64_windows();
        } else if (obj_format == "elf") {
            target = brass::Target::x64_linux();
        }

        brass::codegen::SchedOptions sched_opts;
        sched_opts.enable_pre_ra = enable_schedule_insns;
        sched_opts.enable_post_ra = enable_schedule_insns;
        sched_opts.enable_software_pipelining = enable_software_pipeline;

        auto obj = brass::object::compile_module_to_object(*mod, target, sched_opts);
        std::vector<uint8_t> binary_data;
        if (target.is_windows() || obj_format == "coff") {
            binary_data = brass::object::emit_coff_object(obj);
        } else {
            binary_data = brass::object::emit_elf_object(obj);
        }

        if (output_file.empty()) {
            if (input_file != "-") {
                size_t dot_pos = input_file.find_last_of('.');
                std::string base = (dot_pos != std::string::npos) ? input_file.substr(0, dot_pos) : input_file;
                output_file = base + (target.is_windows() ? ".obj" : ".o");
            } else {
                output_file = target.is_windows() ? "out.obj" : "out.o";
            }
        }

        if (!write_binary_file(output_file, binary_data)) {
            std::cerr << "Error: Could not write object file to '" << output_file << "'\n";
            return 1;
        }

        std::cout << "Successfully emitted object file '" << output_file << "' ("
                  << binary_data.size() << " bytes, " << (target.is_windows() ? "COFF" : "ELF64") << ")\n";
        return 0;
    }

    if (emit_shared) {
        brass::Target target = brass::Target::host();
        brass::target::OutputFormat fmt = brass::target::OutputFormat::Auto;
        if (obj_format == "coff") {
            target = brass::Target::x64_windows();
            fmt = brass::target::OutputFormat::WindowsPeDll;
        } else if (obj_format == "elf") {
            target = brass::Target::x64_linux();
            fmt = brass::target::OutputFormat::LinuxElfSo;
        }

        std::string final_output = shared_output_file.empty() ? output_file : shared_output_file;
        if (final_output.empty()) {
            if (input_file != "-") {
                size_t dot_pos = input_file.find_last_of('.');
                std::string base = (dot_pos != std::string::npos) ? input_file.substr(0, dot_pos) : input_file;
                final_output = base + (target.is_windows() ? ".dll" : ".so");
            } else {
                final_output = target.is_windows() ? "out.dll" : "out.so";
            }
        }

        brass::target::LinkerOptions link_opts;
        link_opts.format = fmt;
        link_opts.export_all_functions = true;

        if (!brass::target::AotLinker::link_to_file(*mod, final_output, target, link_opts)) {
            std::cerr << "Error: Could not link shared library to '" << final_output << "'\n";
            return 1;
        }

        std::cout << "Successfully emitted shared library '" << final_output << "' ("
                  << (target.is_windows() ? "PE32+ DLL" : "ELF64 SO") << ")\n";
        return 0;
    }

    if (!run_fn.empty()) {
        const brass::Function* fn = mod->get_function(run_fn);
        if (!fn) {
            std::cerr << "Error: Function '" << run_fn << "' not found in module '" << mod->name() << "'\n";
            return 1;
        }

        std::vector<brass::RuntimeValue> run_args;
        for (size_t i = 0; i < fn->param_count(); ++i) {
            if (i < run_arg_strings.size()) {
                try {
                    run_args.push_back(parse_arg_for_type(fn->param_type(i), run_arg_strings[i]));
                } catch (const std::exception& ex) {
                    std::cerr << "Error: Could not parse argument " << i << " ('" << run_arg_strings[i]
                              << "') for parameter type " << fn->param_type(i) << ": " << ex.what() << "\n";
                    return 1;
                }
            } else {
                run_args.push_back(brass::RuntimeValue::from_i64(0));
            }
        }

        if (use_jit) {
            brass::codegen::JitExecutionEngine jit(brass::Target::host());
            brass::codegen::SchedOptions sched_opts;
            sched_opts.enable_pre_ra = enable_schedule_insns;
            sched_opts.enable_post_ra = enable_schedule_insns;
            sched_opts.enable_software_pipelining = enable_software_pipeline;
            jit.set_sched_options(sched_opts);
            jit.register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
            if (!jit.compile_and_load(*mod)) {
                std::cerr << "Error: JIT compilation/loading failed for module '" << mod->name() << "'\n";
                return 1;
            }

            try {
                brass::RuntimeValue result = jit.invoke(run_fn, run_args);
                if (!fn->return_type().is_void()) {
                    std::cout << result << "\n";
                }
                if (dump_ic_stats) {
                    brass::runtime::ICRegistry::global().dump_stats(std::cout);
                }
            } catch (const std::exception& ex) {
                std::cerr << "JIT Execution error: " << ex.what() << "\n";
                return 1;
            }
            return 0;
        }

        brass::Interpreter interp;
        interp.register_external_function("brass_pgo_inc", [](brass::Interpreter&, const std::vector<brass::RuntimeValue>& args) {
            if (!args.empty()) {
                uint32_t idx = args[0].is_i32() ? args[0].as_u32() : static_cast<uint32_t>(args[0].as_u64());
                brass_pgo_inc(idx);
            }
            return brass::RuntimeValue::from_void();
        });
        if (gc_stress) {
            interp.gc().set_stress_mode(true);
        }

        try {
            brass::RuntimeValue result = interp.run(*mod, run_fn, run_args);
            if (!fn->return_type().is_void()) {
                std::cout << result << "\n";
            }
        } catch (const std::exception& ex) {
            std::cerr << "Runtime error during execution: " << ex.what() << "\n";
            return 1;
        }

        return 0;
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

    return 0;
}
