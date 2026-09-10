#include <brass/brass.hpp>
#include <brass/fuzz/ir_mutator.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/fuzz/delta_reducer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <iomanip>

using namespace brass;
using namespace brass::fuzz;

static void print_usage(const char* prog) {
    std::cout << "brass-fuzz - Brass Differential Fuzzing & Hardening Tool (v" << brass::version_string() << ")\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  --iterations=<N>      Number of test iterations to generate and run (default: 50)\n"
              << "  --seed=<N>            64-bit initial RNG seed (default: current epoch time)\n"
              << "  --timeout-ms=<N>      Execution timeout per test in milliseconds (default: 500)\n"
              << "  --minimize=<path>     Run Delta-Reducer on reproducer MIR at <path>\n"
              << "  --repro=<path>        Execute differential verification on reproducer MIR at <path>\n"
              << "  --repro-dir=<dir>     Directory to save reproducer files on failure (default: .)\n"
              << "  --sandboxed           Enable OS process isolation & memory quota (Windows Job Objects)\n";
}

int main(int argc, char** argv) {
    uint32_t iterations = 50;
    uint64_t initial_seed = 0;
    uint32_t timeout_ms = 500;
    std::string minimize_path;
    std::string repro_path;
    std::string repro_dir = ".";
    bool sandboxed = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.rfind("--iterations=", 0) == 0) {
            iterations = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        } else if (arg.rfind("-n=", 0) == 0) {
            iterations = static_cast<uint32_t>(std::stoul(arg.substr(3)));
        } else if (arg.rfind("--seed=", 0) == 0) {
            initial_seed = std::stoull(arg.substr(7));
        } else if (arg.rfind("-s=", 0) == 0) {
            initial_seed = std::stoull(arg.substr(3));
        } else if (arg.rfind("--timeout-ms=", 0) == 0) {
            timeout_ms = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        } else if (arg.rfind("--minimize=", 0) == 0) {
            minimize_path = arg.substr(11);
        } else if (arg.rfind("--repro=", 0) == 0) {
            repro_path = arg.substr(8);
        } else if (arg.rfind("--repro-dir=", 0) == 0) {
            repro_dir = arg.substr(12);
        } else if (arg == "--sandboxed") {
            sandboxed = true;
        }
    }

    if (initial_seed == 0) {
        initial_seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
    }

    // Mode 1: Reproduce an existing failing test case
    if (!repro_path.empty()) {
        std::ifstream file(repro_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open reproducer file '" << repro_path << "'\n";
            return 1;
        }
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        DiagnosticReporter diag;
        auto mod = parse_module(source, &diag);
        if (!mod) {
            std::cerr << "Error parsing reproducer: " << diag.format_all() << "\n";
            return 1;
        }

        std::string fn_name = "fuzz_fn";
        if (mod->get_function("fuzz_fn")) {
            fn_name = "fuzz_fn";
        } else if (mod->function_count() > 0 && mod->functions()[0]) {
            fn_name = std::string(mod->functions()[0]->name());
        }

        DiffFuzzerOptions diff_opts;
        diff_opts.timeout_ms = timeout_ms;
        diff_opts.reproducer_dir = repro_dir;
        diff_opts.sandbox_process = sandboxed;
        DiffFuzzer fuzzer(diff_opts);

        std::cout << "Replaying reproducer '" << repro_path << "' on function '" << fn_name << "'...\n";
        DiffResult res = fuzzer.run_test(*mod, fn_name, {RuntimeValue::from_i64(1), RuntimeValue::from_i64(2)}, 0);
        if (res.passed) {
            std::cout << "Result: PASSED (Tiers matched: Interp=" << res.tier0_interp.value
                      << ", JIT unopt=" << res.tier1_jit_unopt.value
                      << ", JIT opt=" << res.tier2_jit_opt.value << ")\n";
            return 0;
        } else {
            std::cout << "Result: FAILED / REPRODUCED: " << res.mismatch_reason << "\n";
            return 2;
        }
    }

    // Mode 2: Delta Minimizer
    if (!minimize_path.empty()) {
        std::ifstream file(minimize_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file to minimize: '" << minimize_path << "'\n";
            return 1;
        }
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        DiagnosticReporter diag;
        auto mod = parse_module(source, &diag);
        if (!mod) {
            std::cerr << "Error parsing module to minimize: " << diag.format_all() << "\n";
            return 1;
        }

        std::string fn_name = "fuzz_fn";
        if (mod->get_function("fuzz_fn")) {
            fn_name = "fuzz_fn";
        } else if (mod->function_count() > 0 && mod->functions()[0]) {
            fn_name = std::string(mod->functions()[0]->name());
        }

        std::cout << "Running automated delta-reducer on '" << minimize_path << "' (" << fn_name << ")...\n";

        DiffFuzzerOptions diff_opts;
        diff_opts.timeout_ms = timeout_ms;
        DiffFuzzer fuzzer(diff_opts);

        auto oracle = [&](const Module& m, std::string_view name) -> bool {
            DiffResult r = fuzzer.run_test(m, name, {RuntimeValue::from_i64(1), RuntimeValue::from_i64(2)}, 0);
            return !r.passed;
        };

        DeltaReducer reducer;
        ReductionResult res = reducer.reduce(*mod, fn_name, oracle);

        if (!res.success) {
            std::cout << "Could not reduce: initial test did not fail oracle or error occurred.\n";
            return 1;
        }

        std::cout << "Reduction complete:\n"
                  << "  Instructions: " << res.initial_instructions << " -> " << res.final_instructions << "\n"
                  << "  Blocks:       " << res.initial_blocks << " -> " << res.final_blocks << "\n";

        std::string out_mir = minimize_path + ".min.mir";
        std::string out_il = minimize_path + ".min.il";
        if (res.minimized_module) {
            DeltaReducer::export_mir(*res.minimized_module, out_mir);
            DeltaReducer::export_il(*res.minimized_module, out_il);
            std::cout << "Saved minimal reproducer to:\n"
                      << "  " << out_mir << "\n"
                      << "  " << out_il << "\n";
        }
        return 0;
    }

    // Mode 3: Continuous Differential Fuzzing
    std::cout << "Starting Brass Differential Fuzzer (Iterations: " << iterations
              << ", Seed: " << initial_seed << ", Timeout: " << timeout_ms << "ms)...\n";

    DiffFuzzerOptions diff_opts;
    diff_opts.timeout_ms = timeout_ms;
    diff_opts.reproducer_dir = repro_dir;
    diff_opts.sandbox_process = sandboxed;
    DiffFuzzer fuzzer(diff_opts);

    uint32_t pass_count = 0;
    uint32_t fail_count = 0;
    uint32_t timeout_count = 0;
    uint32_t fault_count = 0;

    std::unordered_set<uint64_t> covered_edges;
    std::unordered_set<uint32_t> covered_opcodes;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < iterations; ++i) {
        uint64_t seed = initial_seed + i;
        Module mod("fuzz_mod_" + std::to_string(seed));

        IrGeneratorOptions gen_opts;
        gen_opts.min_instructions = 10;
        gen_opts.max_instructions = 30;
        gen_opts.enable_vectors = true;
        gen_opts.enable_loops = true;
        gen_opts.enable_diamonds = true;
        gen_opts.enable_switches = true;
        gen_opts.enable_exceptions = true;

        IrGenerator generator(gen_opts);
        Function* fn = generator.generate(mod, "fuzz_fn", seed);
        if (!fn) {
            fail_count++;
            continue;
        }

        // Apply mutations
        FuzzRng mut_rng(seed ^ 0xDEADBEEFULL);
        IrMutator::mutate_function(*fn, mut_rng);

        // Record edge and opcode coverage
        for (BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (inst) covered_opcodes.insert(static_cast<uint32_t>(inst->opcode()));
            }
            Instruction* term = bb->terminator();
            if (term) {
                if (term->branch_target().block) {
                    uint64_t edge = (static_cast<uint64_t>(bb->id()) << 32) | term->branch_target().block->id();
                    covered_edges.insert(edge);
                }
                if (term->true_target().block) {
                    uint64_t edge = (static_cast<uint64_t>(bb->id()) << 32) | term->true_target().block->id();
                    covered_edges.insert(edge);
                }
                if (term->false_target().block) {
                    uint64_t edge = (static_cast<uint64_t>(bb->id()) << 32) | term->false_target().block->id();
                    covered_edges.insert(edge);
                }
            }
        }

        std::vector<RuntimeValue> args = {
            RuntimeValue::from_i64(static_cast<int64_t>(seed % 100)),
            RuntimeValue::from_i64(static_cast<int64_t>((seed >> 8) % 100))
        };

        DiffResult res = fuzzer.run_test(mod, "fuzz_fn", args, seed);
        if (res.passed) {
            pass_count++;
        } else {
            fail_count++;
            if (res.tier0_interp.status == ExecutionStatus::Timeout ||
                res.tier1_jit_unopt.status == ExecutionStatus::Timeout ||
                res.tier2_jit_opt.status == ExecutionStatus::Timeout) {
                timeout_count++;
            }
            if (res.tier0_interp.status == ExecutionStatus::CrashOrFault ||
                res.tier1_jit_unopt.status == ExecutionStatus::CrashOrFault ||
                res.tier2_jit_opt.status == ExecutionStatus::CrashOrFault) {
                fault_count++;
            }
            std::cout << "[FAIL] Seed " << seed << ": " << res.mismatch_reason << "\n";
            if (!res.reproducer_path.empty()) {
                std::cout << "       Saved reproducer: " << res.reproducer_path << "\n";
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    double rate = (elapsed_sec > 0.0) ? (static_cast<double>(iterations) / elapsed_sec) : 0.0;
    double pass_rate = (iterations > 0) ? (100.0 * pass_count / iterations) : 0.0;

    std::cout << "\n"
              << "============================================================\n"
              << "              Brass Differential Fuzzing Report             \n"
              << "============================================================\n"
              << " Total test cases   : " << iterations << "\n"
              << " Passed             : " << pass_count << " (" << std::fixed << std::setprecision(1) << pass_rate << "%)\n"
              << " Discrepancies      : " << fail_count << "\n"
              << " Timeouts           : " << timeout_count << "\n"
              << " Faults/Crashes     : " << fault_count << "\n"
              << " Total time elapsed : " << std::setprecision(2) << elapsed_sec << "s\n"
              << " Execution rate     : " << std::setprecision(1) << rate << " tests/sec\n"
              << " Edge coverage      : " << covered_edges.size() << " unique CFG edges\n"
              << " Opcode coverage    : " << covered_opcodes.size() << " unique MIR opcodes\n"
              << "============================================================\n";

    return (fail_count == 0) ? 0 : 1;
}
