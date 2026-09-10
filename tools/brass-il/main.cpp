#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/mir/f64_demote.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace brass;
using namespace brass::il;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: brass-il <input.il> [--run] [--emit-mir] [--emit-shared <output.dll/so>] [-shared] [--inline] [--sroa] [--escape-analysis] [--partial-escape] [--sink-allocations] [--dump-pea-stats] [--gvn] [--no-gvn] [--sccp] [--no-sccp] [--guard-elim] [--no-guard-elim] [--cfg-simplify] [--no-cfg-simplify] [--loop-unswitch] [--no-loop-unswitch] [--jump-threading] [--no-jump-threading] [--trace-layout] [--no-trace-layout] [--schedule-insns] [--no-schedule-insns] [--software-pipeline] [--alias-analysis] [--vectorize] [--slp] [--loop-tile] [--tile-size <N>] [--enable-pic] [--dump-ic-stats] [--demote-stats] [--pgo-instrument] [--pgo-use <file>] [--dump-branch-probabilities] [-o <output.obj>] [--no-opt] [--no-demote] [--reassoc] [--timed <N>]\n";
        return 1;
    }

    std::string input_file;
    std::string output_obj;
    std::string output_shared;
    bool emit_shared = false;
    bool run_jit = false;
    bool emit_mir = false;
    bool raw_output = false;
    bool show_demote_stats = false;
    int timed_iterations = 0;
    bool enable_pgo_instrument = false;
    std::string pgo_use_file;
    bool dump_branch_probabilities = false;
    bool enable_schedule_insns = false;
    bool enable_software_pipeline = false;
    bool debug_info = false;
    std::string emit_source_map_file;
    std::string symbolize_offset_arg;
    TranslatorOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: brass-il <input.il> [options]\n\n"
                      << "Options:\n"
                      << "  -h, --help            Show this help message\n"
                      << "  --run                 Execute using JIT engine\n"
                      << "  --emit-mir            Print generated MIR\n"
                      << "  --emit-shared <file>  Compile directly to shared library (.dll/.so)\n"
                      << "  -shared               Compile directly to shared library\n"
                      << "  -g, --debug-info      Preserve and emit debug information and .brass_dbg section\n"
                      << "  --emit-source-map=<f> Emit standard JSON Source Map V3 to <f>\n"
                      << "  --symbolize-offset=<fn,off> Symbolize function offset to source location\n"
                      << "  -o <file>             Write object file to <file>\n";
            return 0;
        } else if (arg == "--run") {
            run_jit = true;
        } else if (arg == "--emit-mir") {
            emit_mir = true;
        } else if (arg == "--emit-shared") {
            emit_shared = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                output_shared = argv[++i];
            }
        } else if (arg == "-shared") {
            emit_shared = true;
        } else if (arg == "--inline") {
            options.enable_inlining = true;
        } else if (arg == "--sroa") {
            options.enable_sroa = true;
        } else if (arg == "--escape-analysis") {
            options.run_escape_analysis = true;
        } else if (arg == "--partial-escape") {
            options.enable_partial_escape = true;
        } else if (arg == "--sink-allocations") {
            options.enable_allocation_sinking = true;
            options.enable_partial_escape = true;
        } else if (arg == "--dump-pea-stats") {
            options.dump_pea_stats = true;
        } else if (arg == "--gvn") {
            options.enable_gvn = true;
        } else if (arg == "--no-gvn") {
            options.enable_gvn = false;
        } else if (arg == "--sccp") {
            options.enable_sccp = true;
        } else if (arg == "--no-sccp") {
            options.enable_sccp = false;
        } else if (arg == "--guard-elim") {
            options.enable_guard_elim = true;
        } else if (arg == "--no-guard-elim") {
            options.enable_guard_elim = false;
        } else if (arg == "--cfg-simplify") {
            options.enable_cfg_simplify = true;
        } else if (arg == "--no-cfg-simplify") {
            options.enable_cfg_simplify = false;
        } else if (arg == "--loop-unswitch") {
            options.enable_loop_unswitch = true;
        } else if (arg == "--no-loop-unswitch") {
            options.enable_loop_unswitch = false;
        } else if (arg == "--jump-threading") {
            options.enable_jump_threading = true;
        } else if (arg == "--no-jump-threading") {
            options.enable_jump_threading = false;
        } else if (arg == "--trace-layout") {
            options.enable_trace_layout = true;
        } else if (arg == "--no-trace-layout") {
            options.enable_trace_layout = false;
        } else if (arg == "--schedule-insns") {
            enable_schedule_insns = true;
        } else if (arg == "--no-schedule-insns") {
            enable_schedule_insns = false;
        } else if (arg == "--software-pipeline") {
            enable_software_pipeline = true;
        } else if (arg == "--no-software-pipeline") {
            enable_software_pipeline = false;
        } else if (arg == "--alias-analysis") {
            options.run_alias_analysis = true;
        } else if (arg == "--vectorize") {
            options.enable_vectorize = true;
        } else if (arg == "--no-vectorize") {
            options.enable_vectorize = false;
        } else if (arg == "--slp") {
            options.enable_slp = true;
        } else if (arg == "--no-slp") {
            options.enable_slp = false;
        } else if (arg == "--loop-tile") {
            options.enable_loop_tile = true;
        } else if (arg == "--no-loop-tile") {
            options.enable_loop_tile = false;
        } else if (arg == "--tile-size" && i + 1 < argc) {
            options.tile_size = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg.rfind("--tile-size=", 0) == 0) {
            options.tile_size = static_cast<size_t>(std::stoul(arg.substr(12)));
        } else if (arg == "--enable-pic") {
            options.enable_pic = true;
        } else if (arg == "--no-pic") {
            options.enable_pic = false;
        } else if (arg == "--dump-ic-stats") {
            options.dump_ic_stats = true;
        } else if (arg == "--wbe" || arg == "--enable-wbe") {
            options.enable_wbe = true;
        } else if (arg == "--no-wbe") {
            options.enable_wbe = false;
        } else if (arg == "--dump-wbe-stats") {
            options.dump_wbe_stats = true;
        } else if (arg == "--demote-stats") {
            show_demote_stats = true;
        } else if (arg == "--raw-output") {
            raw_output = true;
        } else if (arg == "--no-opt") {
            options.enable_optimizations = false;
        } else if (arg == "--no-demote") {
            options.enable_f64_demote = false;
        } else if (arg == "--reassoc") {
            options.allow_fp_reassociation = true;
        } else if (arg == "--timed" && i + 1 < argc) {
            timed_iterations = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--pgo-instrument") {
            enable_pgo_instrument = true;
        } else if (arg.rfind("--pgo-use=", 0) == 0) {
            pgo_use_file = arg.substr(10);
        } else if (arg == "--pgo-use" && i + 1 < argc) {
            pgo_use_file = argv[++i];
        } else if (arg == "--dump-branch-probabilities") {
            dump_branch_probabilities = true;
        } else if (arg == "-o" && i + 1 < argc) {
            output_obj = argv[++i];
        } else if (arg.rfind("-o", 0) == 0 && arg.size() > 2) {
            output_obj = arg.substr(2);
        } else if (arg == "-g" || arg == "--debug-info") {
            debug_info = true;
        } else if (arg.rfind("--emit-source-map=", 0) == 0) {
            emit_source_map_file = arg.substr(18);
        } else if (arg == "--emit-source-map" && i + 1 < argc) {
            emit_source_map_file = argv[++i];
        } else if (arg.rfind("--symbolize-offset=", 0) == 0) {
            symbolize_offset_arg = arg.substr(19);
        } else if (arg == "--symbolize-offset" && i + 1 < argc) {
            symbolize_offset_arg = argv[++i];
        } else if (arg[0] != '-') {
            input_file = arg;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input .il file specified.\n";
        return 1;
    }

    std::ifstream ifs(input_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Cannot open input file: " << input_file << "\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string il_source = buffer.str();

    DemoteStats demote_stats;
    if (show_demote_stats) {
        options.demote_stats = true;
        options.demote_stats_collector = &demote_stats;
    }

    PartialEscapeStats pea_stats;
    if (options.dump_pea_stats) {
        options.pea_stats_collector = &pea_stats;
    }

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, options, &diag);
    if (!res.success || !res.module) {
        std::cerr << "Bronze IL Translation Failed:\n";
        std::cerr << diag.format_all() << "\n";
        if (!res.error_message.empty()) {
            std::cerr << res.error_message << "\n";
        }
        return 1;
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
        auto obj = brass::object::compile_module_to_object(*res.module, target, sched_opts);
        brass::Symbolicator symbolicator(res.module->debug_context(), obj.debug_tables);
        brass::StackTrace trace = symbolicator.symbolize_offset(fn_name, offset);
        std::cout << trace.format();
        return 0;
    }

    if (!emit_source_map_file.empty()) {
        brass::Target target = brass::Target::host();
        brass::codegen::SchedOptions sched_opts;
        auto obj = brass::object::compile_module_to_object(*res.module, target, sched_opts);
        brass::SourceMap combined_sm(res.module->name().empty() ? "output" : std::string(res.module->name()));
        for (const auto& f_path : res.module->debug_context().files()) {
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
        if (output_obj.empty() && !emit_shared && !run_jit && !emit_mir) {
            std::cout << "Successfully emitted source map to '" << emit_source_map_file << "'\n";
            return 0;
        }
    }

    if (show_demote_stats) {
        demote_stats.module_name = res.module->name();
        std::cout << demote_stats.format_report() << "\n";
    }

    if (options.dump_pea_stats) {
        std::cout << pea_stats.format_report() << "\n";
    }

    if (options.enable_partial_escape && !options.enable_allocation_sinking) {
        for (const Function* fn : res.module->functions()) {
            if (!fn) continue;
            PartialEscapeAnalysis pea(*fn);
            std::cout << "Partial Escape Analysis for Function '" << fn->name() << "': "
                      << pea.candidate_allocations().size() << " candidate allocations\n";
            for (const Value* alloc_val : pea.candidate_allocations()) {
                auto frontier = pea.get_materialization_frontier(alloc_val);
                std::cout << "  alloc %" << alloc_val->id() << ": frontier " << frontier.size() << " edges\n";
            }
        }
    }

    if (options.run_escape_analysis) {
        for (const Function* fn : res.module->functions()) {
            if (!fn) continue;
            EscapeAnalysis ea(*fn);
            std::cout << "Escape Analysis for Function '" << fn->name() << "': "
                      << ea.allocations().size() << " allocations ("
                      << ea.non_escaping_allocations().size() << " non-escaping)\n";
            for (const Value* alloc_val : ea.allocations()) {
                std::cout << "  alloc %" << alloc_val->id() << ": "
                          << escape_state_name(ea.get_escape_state(alloc_val)) << "\n";
            }
        }
    }

    if (options.run_alias_analysis) {
        for (const Function* fn : res.module->functions()) {
            if (!fn) continue;
            AliasAnalysis aa(*fn);
            std::cout << "Alias Analysis for Function '" << fn->name() << "':\n";
            std::vector<const Instruction*> mem_insts;
            for (const BasicBlock* bb : fn->blocks()) {
                if (!bb) continue;
                for (const Instruction* inst : *bb) {
                    if (inst && is_memory(inst->opcode())) {
                        mem_insts.push_back(inst);
                    }
                }
            }
            std::cout << "  " << mem_insts.size() << " memory instructions\n";
            for (size_t i = 0; i < mem_insts.size(); ++i) {
                for (size_t j = i + 1; j < mem_insts.size(); ++j) {
                    const Instruction* m1 = mem_insts[i];
                    const Instruction* m2 = mem_insts[j];
                    if (m1->operand_count() > 0 && m2->operand_count() > 0) {
                        AliasResult a_res = aa.alias(m1->operand(0), m1->offset(), m1->memory_type(),
                                                     m2->operand(0), m2->offset(), m2->memory_type());
                        std::cout << "  " << opcode_name(m1->opcode()) << " (off " << m1->offset() << ") vs "
                                  << opcode_name(m2->opcode()) << " (off " << m2->offset() << "): "
                                  << alias_result_name(a_res) << "\n";
                    }
                }
            }
        }
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
            brass::pgo::optimize_module_pgo(*res.module, *pgo_profile);
            DiagnosticReporter pgo_diag;
            if (!verify_module(*res.module, &pgo_diag) || pgo_diag.has_errors()) {
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
        std::cout << "=== Branch Probability Analysis ===\n";
        for (const Function* fn : res.module->functions()) {
            if (!fn) continue;
            const auto* fp = pgo_profile->find_function(std::string(fn->name()));
            if (!fp) {
                std::cout << "Function '" << fn->name() << "': No profile available\n";
                continue;
            }
            mir::BranchProbabilityAnalysis bpa(*fn, *fp);
            const auto& bfi = bpa.block_frequency_info();
            const auto& bpi = bpa.branch_probability_info();
            std::cout << "Function '" << fn->name() << "' (entry count=" << bfi.entry_count() << "):\n";
            for (const BasicBlock* bb : fn->blocks()) {
                if (!bb) continue;
                std::cout << "  block " << bb->name() << ": count=" << bfi.get_block_count(bb)
                          << ", freq=" << bfi.get_block_frequency(bb)
                          << (bfi.is_hot_block(bb) ? " [HOT]" : "")
                          << (bfi.is_cold_block(bb) ? " [COLD]" : "") << "\n";
                for (const BasicBlock* succ : bb->successors()) {
                    if (!succ) continue;
                    std::cout << "    edge -> " << succ->name()
                              << ": count=" << bpi.get_edge_count(bb, succ)
                              << ", prob=" << bpi.get_edge_probability(bb, succ) << "\n";
                }
            }
        }
    }

    if (enable_pgo_instrument) {
        brass::pgo::instrument_module(*res.module);
        DiagnosticReporter pgo_diag;
        if (!verify_module(*res.module, &pgo_diag) || pgo_diag.has_errors()) {
            std::cerr << "Verification failed after PGO instrumentation:\n" << pgo_diag.format_all();
            return 1;
        }
    }

    if (emit_mir) {
        std::cout << to_string(*res.module) << "\n";
    }

    if (!output_obj.empty()) {
        HostEngine engine;
        if (!engine.compile_to_object(*res.module, output_obj)) {
            std::cerr << "Error: Failed to compile module to object format: " << output_obj << "\n";
            return 1;
        }
        std::cout << "[brass-il] Successfully wrote object to: " << output_obj << "\n";
    }

    if (!output_shared.empty() || emit_shared) {
        std::string out_path = output_shared;
        if (out_path.empty()) {
            size_t dot_pos = input_file.find_last_of('.');
            std::string base = (dot_pos != std::string::npos) ? input_file.substr(0, dot_pos) : input_file;
            out_path = base + (Target::host().is_windows() ? ".dll" : ".so");
        }
        target::LinkerOptions link_opts;
        link_opts.export_all_functions = true;
        if (!target::AotLinker::link_to_file(*res.module, out_path, Target::host(), link_opts)) {
            std::cerr << "Error: Failed to link shared library: " << out_path << "\n";
            return 1;
        }
        std::cout << "[brass-il] Successfully wrote shared library to: " << out_path << "\n";
    }

    if (run_jit || (output_obj.empty() && output_shared.empty() && !emit_shared)) {
        codegen::JitExecutionEngine jit;
        codegen::SchedOptions sched_opts;
        sched_opts.enable_pre_ra = enable_schedule_insns;
        sched_opts.enable_post_ra = enable_schedule_insns;
        sched_opts.enable_software_pipelining = enable_software_pipeline;
        jit.set_sched_options(sched_opts);
        register_bronze_runtime_symbols(&jit);
        jit.register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));

        if (!jit.compile_and_load(*res.module)) {
            std::cerr << "Error: JIT compilation failed.\n";
            return 1;
        }

        Function* main_fn = res.module->get_function("main");
        if (main_fn) {
            auto run_main = [&]() {
                if (main_fn->return_type() == Type::f64()) {
                    auto fn_ptr = jit.get_function_ptr<double(*)()>("main");
                    if (fn_ptr) {
                        double r = fn_ptr();
                        if (!raw_output && timed_iterations == 0) {
                            std::cout << "[brass-il] main() returned f64: " << r << "\n";
                        }
                    }
                } else if (main_fn->return_type() == Type::i32()) {
                    auto fn_ptr = jit.get_function_ptr<int32_t(*)()>("main");
                    if (fn_ptr) {
                        int32_t r = fn_ptr();
                        if (!raw_output && timed_iterations == 0) {
                            std::cout << "[brass-il] main() returned i32: " << r << "\n";
                        }
                    }
                } else if (main_fn->return_type() == Type::i64()) {
                    auto fn_ptr = jit.get_function_ptr<int64_t(*)()>("main");
                    if (fn_ptr) {
                        int64_t r = fn_ptr();
                        if (!raw_output && timed_iterations == 0) {
                            std::cout << "[brass-il] main() returned i64: " << r << "\n";
                        }
                    }
                } else {
                    auto fn_ptr = jit.get_function_ptr<void(*)()>("main");
                    if (fn_ptr) {
                        fn_ptr();
                        if (!raw_output && timed_iterations == 0) {
                            std::cout << "[brass-il] main() executed (void).\n";
                        }
                    }
                }
            };

            if (timed_iterations > 0) {
                // First run executes for verification
                run_main();

                // Silence print statements during timing repetitions
                bronze_set_print_enabled(false);

                // Warmup
                run_main();

                // Timed runs
                std::vector<double> samples;
                samples.reserve(timed_iterations);
                for (int iter = 0; iter < timed_iterations; ++iter) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    run_main();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    samples.push_back(elapsed_ms);
                }

                bronze_set_print_enabled(true);

                std::sort(samples.begin(), samples.end());
                double med = samples[samples.size() / 2];
                double min_v = samples.front();
                double max_v = samples.back();
                double spread = (max_v - min_v) / 2.0;
                std::cout << "\n[brass-il-timed: " << std::fixed << std::setprecision(4)
                          << med << " +/- " << spread << " (min: " << min_v << ", max: " << max_v << ")]\n";
            } else {
                run_main();
            }

            if (options.dump_ic_stats) {
                brass::runtime::ICRegistry::global().dump_stats(std::cout);
            }
        }
    }

    return 0;
}
