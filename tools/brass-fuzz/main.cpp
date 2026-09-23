#include <brass/brass.hpp>
#include <brass/fuzz/ir_mutator.hpp>
#include <brass/fuzz/program_generator.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/fuzz/delta_reducer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <unordered_set>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <filesystem>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif

using namespace brass;
using namespace brass::fuzz;

static void print_usage(const char* prog) {
    std::cout << "brass-fuzz - Brass Differential Fuzzing & Hardening Tool (v" << brass::version_string() << ")\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help                 Show this help message\n"
              << "  --iterations=<N>           Number of programs to generate and run (default: 50)\n"
              << "  --seed=<N>                 First seed; program i uses seed+i (default: current time)\n"
              << "  --timeout-ms=<N>           Execution timeout per tier in milliseconds (default: 500)\n"
              << "  --pipeline=all|bronze|legacy\n"
              << "                             Optimizer run by the optimized tiers (default: all)\n"
              << "  --generator=structured|legacy\n"
              << "                             Program generator (default: structured)\n"
              << "  --skip-pass=<a,b,...>      Leave these pipeline steps out (e.g. jump_threading)\n"
              << "  --max-statements=<N>       Statements per generated program (default: 36); small\n"
              << "                             values re-find a failure class as a small program\n"
              << "  --no-div-overflow          Never generate INT_MIN / -1\n"
              << "  --no-bisect                Do not search for the first pass that changes the answer\n"
              << "  --max-repros-per-class=<N> Reproducer files saved per failure class (default: 3)\n"
              << "  --minimize=<path>          Run the delta reducer on a reproducer, keeping its failure class\n"
              << "  --repro=<path>             Re-run a reproducer (reads its ARGS, PIPELINE and SKIP lines);\n"
              << "                             exits 0 on pass, 2 on failure, 3 on a failure other than\n"
              << "                             --expect-class=<class> when that is given\n"
              << "  --dump-opt=<path>          With --repro: write the module after the pipeline as a\n"
              << "                             reproducer and exit (its JIT-unopt tier runs that code)\n"
              << "  --repro-dir=<dir>         Directory for reproducer files (default: .)\n"
              << "  --chunk-size=<N>           Run the campaign as child processes of N programs each, so\n"
              << "                             a fault that kills the process loses one chunk only\n"
              << "  --jit-only                 Compare only the JIT tiers against the interpreter (the\n"
              << "                             interpreter-on-optimized and bytecode tiers are skipped)\n"
              << "  --unopt-only               Compare only the unoptimized JIT against the interpreter;\n"
              << "                             no pass runs, so a backend bug minimizes on its own\n"
              << "  --no-fast-interp           Skip the bytecode (FastInterpreter) tiers\n"
              << "  --sandboxed                Enable OS process isolation & memory quota (Windows Job Objects)\n";
}

namespace {

struct ReproHeader {
    std::vector<RuntimeValue> args;
    std::string pipeline;
    std::string failure_class;
    std::vector<std::string> skip;
    bool has_skip = false;
};

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ',' || c == ' ') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

ReproHeader read_repro_header(const std::string& source) {
    ReproHeader h;
    std::istringstream in(source);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("; ARGS:", 0) == 0) {
            std::istringstream vals(line.substr(7));
            long long v = 0;
            while (vals >> v) h.args.push_back(RuntimeValue::from_i64(static_cast<int64_t>(v)));
        } else if (line.rfind("; PIPELINE: ", 0) == 0) {
            h.pipeline = line.substr(12);
        } else if (line.rfind("; CLASS: ", 0) == 0) {
            h.failure_class = line.substr(9);
        } else if (line.rfind("; SKIP:", 0) == 0) {
            h.skip = split_list(line.substr(7));
            h.has_skip = true;
        }
    }
    if (h.args.empty()) h.args = {RuntimeValue::from_i64(1), RuntimeValue::from_i64(2)};
    return h;
}

std::string entry_name(const Module& mod) {
    if (mod.get_function("fuzz_fn")) return "fuzz_fn";
    if (mod.function_count() > 0 && mod.functions()[0]) return std::string(mod.functions()[0]->name());
    return "fuzz_fn";
}

// std::system() returns the exit code on Windows and a wait status elsewhere.
int system_exit_code(int rc) {
#if defined(_WIN32)
    return rc;
#else
    return (rc != -1 && WIFEXITED(rc)) ? WEXITSTATUS(rc) : -1;
#endif
}

const char* null_device() {
#if defined(_WIN32)
    return "NUL";
#else
    return "/dev/null";
#endif
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    out.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return true;
}

// A std::system() line that re-runs this tool with `args`, stdout and
// stderr going to `out` (a quoted path; empty discards them).
std::string self_command(const std::string& self, const std::string& args, const std::string& out) {
    const std::string child = "\"" + self + "\"" + args + " >" + (out.empty() ? null_device() : out) + " 2>&1";
#if defined(_WIN32)
    // cmd.exe strips the outer quote pair of a /c line that starts with one.
    return "\"" + child + "\"";
#else
    // A cross build runs under an emulator (qemu-user without binfmt, see
    // scripts/linux-tests.sh); the child has to be started through it too.
    const char* exec_prefix = std::getenv("BRASS_FUZZ_EXEC_PREFIX");
    return (exec_prefix && *exec_prefix ? std::string(exec_prefix) + " " : std::string()) + child;
#endif
}

// Runs the campaign as child processes of `chunk` programs each. A codegen
// bug can corrupt the fuzzer beyond what the fault handler recovers, and
// in one process that ends the whole campaign; here it costs one chunk.
int run_chunked(int argc, char** argv, uint64_t first_seed, uint32_t iterations, uint32_t chunk,
                const std::string& repro_dir) {
    std::string passthrough;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--iterations=", 0) == 0 || arg.rfind("-n=", 0) == 0 || arg.rfind("--seed=", 0) == 0 ||
            arg.rfind("-s=", 0) == 0 || arg.rfind("--chunk-size=", 0) == 0) {
            continue;
        }
        passthrough += " \"" + arg + "\"";
    }
    struct Agg {
        uint64_t count = 0;
        uint64_t first_seed = UINT64_MAX;
        std::string first_line;
    };
    std::map<std::string, Agg> classes;
    uint64_t passed = 0, failed = 0, lost = 0;
    std::vector<std::string> crashed;
    std::filesystem::create_directories(repro_dir);

    for (uint32_t done = 0; done < iterations; done += chunk) {
        const uint32_t n = std::min(chunk, iterations - done);
        const uint64_t seed = first_seed + done;
        const std::string log = repro_dir + "/chunk_" + std::to_string(seed) + ".log";
        const std::string cmd = self_command(argv[0], passthrough + " --seed=" + std::to_string(seed) +
                                                          " --iterations=" + std::to_string(n),
                                            "\"" + log + "\"");
        const int rc = std::system(cmd.c_str());
        std::string text;
        read_file(log, text);
        std::istringstream in(text);
        std::string line;
        bool in_table = false;
        bool finished = false;
        std::map<std::string, Agg> partial;
        while (std::getline(in, line)) {
            if (line.rfind(" Passed             : ", 0) == 0) {
                passed += std::stoull(line.substr(22));
                finished = true;
            } else if (line.rfind(" Failures           : ", 0) == 0) {
                failed += std::stoull(line.substr(22));
            } else if (line.rfind(" Failure classes", 0) == 0) {
                in_table = true;
            } else if (in_table && line.rfind("   ", 0) == 0) {
                // "   COUNT  CLASS  (first: ...)"; class names may contain spaces.
                const size_t num = line.find_first_not_of(' ');
                const size_t gap = line.find("  ", num);
                const size_t tail = line.find("  (first:", gap);
                if (num == std::string::npos || gap == std::string::npos || tail == std::string::npos) continue;
                classes[line.substr(gap + 2, tail - gap - 2)].count += std::stoull(line.substr(num, gap - num));
            } else if (line.rfind("[FAIL] Seed ", 0) == 0) {
                // First occurrence of a class in this chunk: remember the seed.
                const size_t open = line.find(" [");
                const size_t close = line.find("]:", open);
                if (open == std::string::npos || close == std::string::npos) continue;
                const uint64_t s = std::stoull(line.substr(12, open - 12));
                Agg& a = partial[line.substr(open + 2, close - open - 2)];
                if (s < a.first_seed) {
                    a.first_seed = s;
                    a.first_line = line.substr(0, 200);
                }
            }
        }
        for (const auto& [cls, a] : partial) {
            Agg& g = classes[cls];
            if (!finished) g.count += 1;
            if (a.first_seed < g.first_seed) {
                g.first_seed = a.first_seed;
                g.first_line = a.first_line;
            }
        }
        if (!finished) {
            ++lost;
            crashed.push_back("seeds " + std::to_string(seed) + ".." + std::to_string(seed + n - 1) +
                              " (exit " + std::to_string(rc) + ", " + log + ")");
        }
        std::cout << "chunk " << seed << " +" << n << (finished ? " done" : " CRASHED") << std::endl;
    }

    std::cout << "\n============================================================\n"
              << " Chunked campaign: " << iterations << " programs from seed " << first_seed << " in chunks of "
              << chunk << "\n Passed: " << passed << "  Failed: " << failed << "  (in finished chunks)\n"
              << " Crashed chunks: " << lost << "\n";
    for (const std::string& c : crashed) std::cout << "   " << c << "\n";
    std::cout << " Failure classes (a crashed chunk counts each class it reached once):\n";
    for (const auto& [cls, a] : classes) {
        std::cout << "   " << std::setw(6) << a.count << "  " << cls << "  (first: seed " << a.first_seed << ")\n";
    }
    std::cout << "============================================================\n";
    return (failed == 0 && lost == 0) ? 0 : 1;
}

// A --skip-pass name (or a reproducer's skip header) that names no step of
// the pipeline is a typo or a stale name, not something to ignore.
bool skips_are_known(FuzzPipeline pipeline, const std::vector<std::string>& skip) {
    const std::vector<std::string> unknown = unknown_skip_names(pipeline, skip);
    for (const std::string& name : unknown) {
        std::cerr << "Error: pipeline '" << pipeline_name(pipeline) << "' has no step named '" << name << "'\n";
    }
    if (!unknown.empty()) {
        std::cerr << "Steps:";
        for (const std::string& step : fuzz_pipeline(pipeline).names()) std::cerr << " [" << step << "]";
        std::cerr << "\n";
    }
    return unknown.empty();
}

} // namespace

int main(int argc, char** argv) {
    uint32_t iterations = 50;
    uint64_t initial_seed = 0;
    uint32_t timeout_ms = 500;
    std::string minimize_path;
    std::string repro_path;
    std::string repro_dir = ".";
    bool sandboxed = false;
    bool pipeline_given = false;
    FuzzPipeline pipeline = FuzzPipeline::AllPasses;
    bool legacy_generator = false;
    bool div_overflow = true;
    bool bisect = true;
    uint32_t max_repros_per_class = 3;
    uint32_t max_statements = ProgramGeneratorOptions{}.max_statements;
    std::string expect_class;
    bool skip_given = false;
    std::vector<std::string> skip_passes;
    uint32_t chunk_size = 0;
    bool jit_only = false;
    bool unopt_only = false;
    bool no_fast_interp = false;
    std::string dump_opt_path;

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
        } else if (arg.rfind("--pipeline=", 0) == 0) {
            if (!parse_pipeline(arg.substr(11), pipeline)) {
                std::cerr << "Unknown pipeline '" << arg.substr(11) << "' (expected all, bronze or legacy)\n";
                return 1;
            }
            pipeline_given = true;
        } else if (arg.rfind("--generator=", 0) == 0) {
            const std::string g = arg.substr(12);
            if (g != "structured" && g != "legacy") {
                std::cerr << "Unknown generator '" << g << "' (expected structured or legacy)\n";
                return 1;
            }
            legacy_generator = g == "legacy";
        } else if (arg.rfind("--skip-pass=", 0) == 0) {
            for (const std::string& s : split_list(arg.substr(12))) skip_passes.push_back(s);
            skip_given = true;
        } else if (arg.rfind("--expect-class=", 0) == 0) {
            expect_class = arg.substr(15);
        } else if (arg.rfind("--max-statements=", 0) == 0) {
            max_statements = static_cast<uint32_t>(std::stoul(arg.substr(17)));
        } else if (arg == "--no-div-overflow") {
            div_overflow = false;
        } else if (arg == "--no-bisect") {
            bisect = false;
        } else if (arg.rfind("--max-repros-per-class=", 0) == 0) {
            max_repros_per_class = static_cast<uint32_t>(std::stoul(arg.substr(23)));
        } else if (arg.rfind("--minimize=", 0) == 0) {
            minimize_path = arg.substr(11);
        } else if (arg.rfind("--repro=", 0) == 0) {
            repro_path = arg.substr(8);
        } else if (arg.rfind("--dump-opt=", 0) == 0) {
            dump_opt_path = arg.substr(11);
        } else if (arg.rfind("--repro-dir=", 0) == 0) {
            repro_dir = arg.substr(12);
        } else if (arg == "--sandboxed") {
            sandboxed = true;
        } else if (arg == "--jit-only") {
            jit_only = true;
        } else if (arg == "--unopt-only") {
            unopt_only = true;
        } else if (arg == "--no-fast-interp") {
            no_fast_interp = true;
        } else if (arg.rfind("--chunk-size=", 0) == 0) {
            chunk_size = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            return 1;
        }
    }

    if (initial_seed == 0) {
        initial_seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
    }

    DiffFuzzerOptions diff_opts;
    diff_opts.timeout_ms = timeout_ms;
    diff_opts.reproducer_dir = repro_dir;
    diff_opts.sandbox_process = sandboxed;
    diff_opts.pipeline = pipeline;
    diff_opts.bisect = bisect;
    diff_opts.skip_passes = skip_passes;
    if (jit_only || no_fast_interp) {
        diff_opts.tier4_fast_interp = false;
        diff_opts.tier5_fast_interp_opt = false;
    }
    if (jit_only) {
        // The interpreter tiers are target-independent; an emulated target
        // (scripts/linux-tests.sh runs aarch64 under qemu) spends its time
        // on the tiers that run its own machine code.
        diff_opts.tier3_interp_opt = false;
        diff_opts.tier4_fast_interp = false;
        diff_opts.tier5_fast_interp_opt = false;
    }
    if (unopt_only) {
        diff_opts.tier2_jit_opt = false;
        diff_opts.tier3_interp_opt = false;
        diff_opts.tier4_fast_interp = false;
        diff_opts.tier5_fast_interp_opt = false;
    }

    // Modes 1 and 2 work on an existing reproducer.
    const std::string& input_path = !repro_path.empty() ? repro_path : minimize_path;
    if (!input_path.empty()) {
        std::string source;
        if (!read_file(input_path, source)) {
            std::cerr << "Error: Could not open '" << input_path << "'\n";
            return 1;
        }
        DiagnosticReporter diag;
        auto mod = parse_module(source, &diag);
        if (!mod) {
            std::cerr << "Error parsing '" << input_path << "': " << diag.format_all() << "\n";
            return 1;
        }
        const ReproHeader header = read_repro_header(source);
        if (!pipeline_given && !header.pipeline.empty()) {
            parse_pipeline(header.pipeline, diff_opts.pipeline);
        }
        if (!skip_given && header.has_skip) diff_opts.skip_passes = header.skip;
        diff_opts.save_reproducers = false;
        if (!skips_are_known(diff_opts.pipeline, diff_opts.skip_passes)) return 1;
        DiffFuzzer fuzzer(diff_opts);
        const std::string fn_name = entry_name(*mod);

        if (!repro_path.empty() && !dump_opt_path.empty()) {
            // The module after the pipeline, as a reproducer of its own: its
            // unoptimized JIT tier then runs exactly the code the optimized
            // tier ran, and a codegen bug minimizes without the passes.
            OptimizeOutcome opt = fuzzer.optimize(*mod);
            if (!opt.module) {
                std::cerr << "Error: the pipeline did not produce a module (" << opt.pass << "): " << opt.message
                          << "\n";
                return 1;
            }
            std::ofstream out(dump_opt_path, std::ios::binary);
            std::istringstream header_lines(source);
            std::string line;
            while (std::getline(header_lines, line)) {
                if (line.rfind("; ARGS:", 0) == 0) out << line << "\n";
            }
            out << "; PIPELINE: " << pipeline_name(diff_opts.pipeline) << "\n"
                << "; optimized by brass-fuzz --dump-opt from " << repro_path << "\n\n"
                << to_string(*opt.module);
            if (!out) {
                std::cerr << "Error: could not write '" << dump_opt_path << "'\n";
                return 1;
            }
            std::cout << "Wrote the optimized module to '" << dump_opt_path << "'\n";
            return 0;
        }
        if (!repro_path.empty()) {
            std::cout << "Replaying '" << repro_path << "' on '" << fn_name << "' with pipeline "
                      << pipeline_name(diff_opts.pipeline) << "...\n";
            DiffResult res = fuzzer.run_test(*mod, fn_name, header.args, 0);
            if (res.passed) {
                std::cout << "Result: PASSED (Interp=" << res.tier0_interp.value
                          << ", JIT unopt=" << res.tier1_jit_unopt.value
                          << ", JIT opt=" << res.tier2_jit_opt.value
                          << ", Interp opt=" << res.tier3_interp_opt.value << ")\n";
                return 0;
            }
            std::cout << "Result: FAILED [" << res.failure_class << "]: " << res.mismatch_reason << "\n";
            return (expect_class.empty() || res.failure_class == expect_class) ? 2 : 3;
        }

        std::string target_class = header.failure_class;
        if (target_class.empty()) {
            DiffResult first = fuzzer.run_test(*mod, fn_name, header.args, 0);
            if (first.passed) {
                std::cout << "Could not reduce: the reproducer passes.\n";
                return 1;
            }
            target_class = first.failure_class;
        }
        std::cout << "Reducing '" << minimize_path << "' (" << fn_name << "), keeping class "
                  << target_class << "..." << std::endl;

        // Candidates run in a child process: a program that trips a codegen
        // bug can corrupt the process that runs it.
        auto write_candidate = [&](const Module& m, const std::string& path) {
            std::ofstream os(path);
            os << "; CLASS: " << target_class << "\n; PIPELINE: " << pipeline_name(diff_opts.pipeline) << "\n";
            if (!diff_opts.skip_passes.empty()) {
                os << "; SKIP:";
                for (const std::string& s : diff_opts.skip_passes) os << ' ' << s;
                os << "\n";
            }
            os << "; ARGS:";
            for (const RuntimeValue& a : header.args) os << ' ' << static_cast<int64_t>(a.raw_bits());
            os << "\n\n";
            print_module(m, os);
        };
        const std::string cand_path = minimize_path + ".cand.mir";
        const std::string self = argv[0];
        auto oracle = [&](const Module& m, std::string_view) -> bool {
            write_candidate(m, cand_path);
            const std::string cmd = self_command(self, " --repro=\"" + cand_path + "\" --expect-class=" +
                                                           target_class + " --timeout-ms=" +
                                                           std::to_string(timeout_ms) +
                                                           (bisect ? "" : " --no-bisect") +
                                                           (jit_only ? " --jit-only" : "") +
                                                           (unopt_only ? " --unopt-only" : "") +
                                                           (no_fast_interp ? " --no-fast-interp" : ""),
                                                 "");
            return system_exit_code(std::system(cmd.c_str())) == 2;
        };
        if (!oracle(*mod, fn_name)) {
            std::cout << "Could not reduce: the reproducer does not fail with " << target_class << ".\n";
            return 1;
        }
        DeltaReducerOptions ropts;
        ropts.max_passes = 40;
        DeltaReducer reducer(ropts);
        ReductionResult res = reducer.reduce(*mod, fn_name, oracle);
        std::remove(cand_path.c_str());
        if (!res.success || !res.minimized_module) {
            std::cout << "Could not reduce.\n";
            return 1;
        }
        std::cout << "Reduction complete:\n"
                  << "  Instructions: " << res.initial_instructions << " -> " << res.final_instructions << "\n"
                  << "  Blocks:       " << res.initial_blocks << " -> " << res.final_blocks << "\n";
        const std::string out_mir = minimize_path + ".min.mir";
        write_candidate(*res.minimized_module, out_mir);
        std::cout << "Saved minimal reproducer to " << out_mir << "\n";
        return 0;
    }

    if (!skips_are_known(pipeline, skip_passes)) return 1;
    if (chunk_size > 0) return run_chunked(argc, argv, initial_seed, iterations, chunk_size, repro_dir);

    // Mode 3: continuous differential fuzzing.
    std::cout << "Starting Brass Differential Fuzzer (Iterations: " << iterations
              << ", Seed: " << initial_seed << ", Timeout: " << timeout_ms << "ms, Pipeline: "
              << pipeline_name(pipeline) << ", Generator: " << (legacy_generator ? "legacy" : "structured")
              << ")...\n";

    diff_opts.save_reproducers = false;
    DiffFuzzer fuzzer(diff_opts);

    uint32_t pass_count = 0;
    uint32_t fail_count = 0;
    struct ClassInfo {
        uint32_t count = 0;
        uint32_t saved = 0;
        uint64_t first_seed = 0;
        std::string first_path;
    };
    std::map<std::string, ClassInfo> classes;

    std::unordered_set<uint64_t> covered_edges;
    std::unordered_set<uint32_t> covered_opcodes;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < iterations; ++i) {
        uint64_t seed = initial_seed + i;
        Module mod("fuzz_mod_" + std::to_string(seed));

        Function* fn = nullptr;
        if (legacy_generator) {
            IrGeneratorOptions gen_opts;
            gen_opts.min_instructions = 10;
            gen_opts.max_instructions = 30;
            IrGenerator generator(gen_opts);
            fn = generator.generate(mod, "fuzz_fn", seed);
            FuzzRng mut_rng(seed ^ 0xDEADBEEFULL);
            if (fn) IrMutator::mutate_function(*fn, mut_rng);
        } else {
            ProgramGeneratorOptions gen_opts;
            gen_opts.enable_div_overflow = div_overflow;
            gen_opts.max_statements = max_statements;
            ProgramGenerator generator(gen_opts);
            fn = generator.generate(mod, "fuzz_fn", seed);
        }
        if (!fn) {
            fail_count++;
            continue;
        }

        for (BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (inst) covered_opcodes.insert(static_cast<uint32_t>(inst->opcode()));
            }
            Instruction* term = bb->terminator();
            if (!term) continue;
            auto edge = [&](const BasicBlock* to) {
                if (to) covered_edges.insert((static_cast<uint64_t>(bb->id()) << 32) | to->id());
            };
            edge(term->branch_target().block);
            edge(term->true_target().block);
            edge(term->false_target().block);
        }

        std::vector<RuntimeValue> args = {
            RuntimeValue::from_i64(static_cast<int64_t>(seed % 100)),
            RuntimeValue::from_i64(static_cast<int64_t>((seed >> 8) % 100))
        };

        DiffResult res = fuzzer.run_test(mod, "fuzz_fn", args, seed);
        if (res.passed) {
            pass_count++;
            continue;
        }
        fail_count++;
        ClassInfo& info = classes[res.failure_class];
        info.count++;
        std::string path;
        if (info.saved < max_repros_per_class) {
            path = DiffFuzzer::save_reproducer(mod, seed, res.reproducer_note, repro_dir);
            info.saved++;
            if (path.empty()) {
                std::cout << "[WARN] Could not write a reproducer into " << repro_dir << "\n";
            }
        }
        if (!path.empty()) {
            // A reproducer is only useful if the text form is the same program.
            std::string text;
            DiagnosticReporter pdiag;
            std::unique_ptr<Module> back = read_file(path, text) ? parse_module(text, &pdiag) : nullptr;
            if (!back || !verify_module(*back, &pdiag)) {
                std::cout << "[WARN] Reproducer " << path << " does not parse back cleanly: "
                          << pdiag.format_all().substr(0, 300) << "\n";
            }
            if (info.first_path.empty()) {
                info.first_path = path;
                info.first_seed = seed;
            }
        }
        if (info.count == 1) {
            std::cout << "[FAIL] Seed " << seed << " [" << res.failure_class << "]: "
                      << res.mismatch_reason.substr(0, 300) << "\n";
            if (!path.empty()) std::cout << "       Saved reproducer: " << path << "\n";
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
              << " Pipeline           : " << pipeline_name(pipeline) << "\n"
              << " Total test cases   : " << iterations << "\n"
              << " Passed             : " << pass_count << " (" << std::fixed << std::setprecision(1) << pass_rate << "%)\n"
              << " Failures           : " << fail_count << "\n"
              << " Total time elapsed : " << std::setprecision(2) << elapsed_sec << "s\n"
              << " Execution rate     : " << std::setprecision(1) << rate << " tests/sec\n"
              << " Edge coverage      : " << covered_edges.size() << " unique CFG edges\n"
              << " Opcode coverage    : " << covered_opcodes.size() << " unique MIR opcodes\n";
    if (!classes.empty()) {
        std::cout << " Failure classes    :\n";
        for (const auto& [name, info] : classes) {
            std::cout << "   " << std::setw(6) << info.count << "  " << name
                      << "  (first: seed " << info.first_seed << ", " << info.first_path << ")\n";
        }
    }
    std::cout << "============================================================\n";

    return (fail_count == 0) ? 0 : 1;
}
