#include "bench_utils.hpp"
#include "bench_load.hpp"
#include "bench_numeric.hpp"
#include "bench_js_shapes.hpp"
#include "bench_gc.hpp"
#include "bench_compile_speed.hpp"
#include "bench_simd_math.hpp"
#include "bench_interpreter.hpp"
#include <cstdlib>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace brass::bench;

namespace {

// Default logical CPU the benchmark thread is pinned to on Windows (see
// pin_current_thread). CPU 2 is a core on the first CCD / first cluster,
// away from CPU 0, which takes most interrupts. Elsewhere pinning is opt-in
// (--cpu), because Linux threads inherit the creator's mask and brass's
// background threads would then share the one core.
#if defined(_WIN32)
constexpr int kDefaultBenchCpu = 2;
#else
constexpr int kDefaultBenchCpu = -1;
#endif

// Suite runs whose per-key medians become the goldens on --update-ratchet.
// Enough that a key whose number is bimodal on a host shows both modes, so
// its spread is recorded (RatchetManager::record_margin).
constexpr int kDefaultRecordRuns = 5;

bool read_env(const char* name, std::string& out) {
#if defined(_MSC_VER)
    // MSVC deprecates getenv (C4996, an error under /WX).
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return false;
    out = owned;
    std::free(owned);
    return true;
#else
    const char* v = std::getenv(name);
    if (!v) return false;
    out = v;
    return true;
#endif
}

bool read_env_int(const char* name, int& out) {
    std::string v;
    if (!read_env(name, v)) return false;
    out = std::atoi(v.c_str());
    return true;
}

void run_suite(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet) {
    // Warm up CPU clock frequency and OS memory subsystem
    {
        volatile double dummy = 0.0;
        for (int w = 0; w < 2000000; ++w) {
            dummy = dummy + 1.0 / (w + 1.0);
        }
        (void)dummy;
        brass_zeroupper();
    }

    // Numeric & Algorithmic Microbenchmarks
    run_numeric_benchmarks(results, ratchet);

    // SIMD Vector Math Microbenchmarks
    run_simd_math_benchmarks(results, ratchet);

    // JS-Shaped Benchmarks (NaN-boxing, shape guards, patchable IC, Cheney GC alloc)
    run_js_shapes_benchmarks(results, ratchet);

    // GC Model Comparison (Brass Stack-Maps vs Shadow-Stack)
    run_gc_benchmark(results);

    // Compile-Speed Benchmark (Parse, Verify, ISEL, RegAlloc, Codegen)
    run_compile_speed_benchmark(results);

    // Fast Bytecode Interpreter Benchmarks (FastInterpreter vs native; oracle and baseline JIT for context)
    run_interpreter_benchmarks(results, ratchet);
}

std::string quote_arg(const std::string& s) { return "\"" + s + "\""; }

int record_in_processes(const std::string& self, int runs, int cpu, bool force, const std::string& host,
                        const std::string& ratchet_path, RatchetManager& ratchet) {
    std::map<std::string, std::vector<double>> per_key;
    for (int run = 0; run < runs; ++run) {
        const std::string scratch = ratchet_path + ".run" + std::to_string(run) + ".tmp";
        std::error_code ec;
        std::filesystem::remove(scratch, ec);
        std::string cmd = quote_arg(self) + " --update-ratchet --runs=1 --cpu=" + std::to_string(cpu) +
                          " --host=" + quote_arg(host) + " --ratchet-file=" + quote_arg(scratch);
        if (force) cmd += " --force";
#if defined(_WIN32)
        cmd = "\"" + cmd + "\"";  // cmd /c strips one pair of outer quotes
#endif
        std::cout << "\n[RUN] " << (run + 1) << " of " << runs << " (own process)\n" << std::flush;
        const int rc = std::system(cmd.c_str());
        RatchetManager one;
        const bool ok = rc == 0 && one.load(scratch);
        std::filesystem::remove(scratch, ec);
        if (!ok) {
            std::cerr << "[RATCHET] ERROR: recording run " << (run + 1) << " failed (exit " << rc
                      << "; a busy machine is refused, see its [LOAD] lines); nothing recorded.\n";
            return 1;
        }
        for (const auto& [key, val] : one.ratios()) per_key[key].push_back(val);
    }

    std::cout << "\n[RATCHET] goldens for host " << host << ": each key's median over " << runs
              << " processes\n";
    for (const auto& [key, vals] : per_key) {
        const double golden = RatchetManager::median(vals);
        if (golden <= 0.0) continue;
        ratchet.set_ratio(key, golden);
        const double m = ratchet.record_margin(key, vals, golden);
        std::ostringstream line;
        line << std::defaultfloat << std::setprecision(4) << "  " << std::left << std::setw(30) << key << golden;
        if (m > 0.0) {
            line << "  margin " << static_cast<int>(m * 100.0 + 0.5) << "% on this host (runs:";
            for (double v : vals) line << " " << v;
            line << ")";
        }
        std::cout << line.str() << "\n";
    }
    if (!ratchet.save(ratchet_path)) {
        std::cerr << "[RATCHET] ERROR: Failed to write goldens to: " << ratchet_path << "\n";
        return 1;
    }
    std::cout << "[RATCHET] Wrote " << per_key.size() << " goldens for host " << host << " to " << ratchet_path
              << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool check_ratchet = false;
    bool update_ratchet = false;
    bool force = false;
    int runs = 0;  // 0: one run, or kDefaultRecordRuns when recording
    std::string custom_ratchet_file;
    std::string host;
    read_env("BRASS_BENCH_HOST", host);
    int cpu = kDefaultBenchCpu;
    read_env_int("BRASS_BENCH_CPU", cpu);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--check-ratchet") {
            check_ratchet = true;
        } else if (arg == "--update-ratchet") {
            update_ratchet = true;
        } else if (arg == "--force") {
            force = true;
        } else if (arg.rfind("--runs=", 0) == 0) {
            runs = std::atoi(arg.c_str() + 7);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (arg.rfind("--host=", 0) == 0) {
            host = arg.substr(7);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg.rfind("--ratchet-file=", 0) == 0) {
            custom_ratchet_file = arg.substr(15);
        } else if (arg == "--ratchet-file" && i + 1 < argc) {
            custom_ratchet_file = argv[++i];
        } else if (arg.rfind("--cpu=", 0) == 0) {
            cpu = std::atoi(arg.c_str() + 6);
        } else if (arg == "--cpu" && i + 1 < argc) {
            cpu = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Brass Performance Benchmark Suite\n"
                      << "Usage: brass_benchmarks [options]\n"
                      << "Options:\n"
                      << "  --check-ratchet           Fail (exit 1) when a key regresses past its golden by more\n"
                      << "                            than its margin. Goldens are per machine:\n"
                      << "                            bench/hosts/<host>.json; a host without one reports and passes\n"
                      << "  --update-ratchet          Record this machine's goldens: the per-key median over\n"
                      << "                            --runs suite runs (default " << kDefaultRecordRuns << "). Refused on a busy\n"
                      << "                            machine unless --force\n"
                      << "  --runs <n>                Run the suite n times; every key's number is its median\n"
                      << "  --host <name>             Use <name>'s goldens (default: this machine's host name;\n"
                      << "                            env BRASS_BENCH_HOST)\n"
                      << "  --ratchet-file <path>     Use this goldens file instead of the host's\n"
                      << "  --force                   Record goldens even when the machine was busy\n"
                      << "  --cpu <n>                 Pin the benchmark thread to logical CPU n; -1 = no pinning\n"
                      << "                            (default " << kDefaultBenchCpu << "; env BRASS_BENCH_CPU)\n"
                      << "  -h, --help                Show this help message\n";
            return 0;
        }
    }
    if (runs <= 0) runs = update_ratchet ? kDefaultRecordRuns : 1;

    // 1. This machine's goldens
    host = RatchetManager::normalize_host(host.empty() ? machine_host_name() : host);
    const std::string ratchet_path =
        custom_ratchet_file.empty() ? RatchetManager::host_ratchet_path(host) : custom_ratchet_file;
    RatchetManager ratchet;
    const bool has_goldens = ratchet.load(ratchet_path);
    if (has_goldens) {
        std::cout << "[RATCHET] host " << host << ": goldens from " << ratchet_path << "\n";
    } else {
        std::cout << "[RATCHET] host " << host << ": no goldens at " << ratchet_path
                  << "; numbers are reported, not checked (--update-ratchet records them).\n";
    }
    RatchetManager::active() = &ratchet;

    // Recording over several runs: each run is its own process, because a
    // host's bimodal key picks its mode per process (halo: twelve runs in one
    // process all read the same mode of a key that differs ~23% between
    // processes), and the margin must see both. Each child is an ordinary
    // one-run --update-ratchet into a scratch file, with its own load guard:
    // a child that saw a busy machine exits non-zero and nothing is recorded.
    if (update_ratchet && runs > 1) {
        return record_in_processes(argv[0], runs, cpu, force, host, ratchet_path, ratchet);
    }

    // 2. Emit Run Provenance Header, then the machine-load guard
    BenchmarkReporter::print_header("Brass JIT vs Native C++ (-O3) & GC Performance");
    const double load_before = report_load_before_run(1.0);
    if (cpu >= 0) {
        if (pin_current_thread(cpu)) {
            std::cout << "[PIN] benchmark thread pinned to logical CPU " << cpu << "\n";
        } else {
            std::cout << "[PIN] could not pin to logical CPU " << cpu << "; running unpinned\n";
        }
    }
    const CpuLoadSample run_start = sample_cpu_load();

    // 3. The suite, `runs` times; each key's number is its median over them.
    std::vector<BenchmarkResult> results;
    std::map<std::string, std::vector<double>> per_key;
    for (int run = 0; run < runs; ++run) {
        if (runs > 1) std::cout << "\n[RUN] " << (run + 1) << " of " << runs << "\n";
        results.clear();
        run_suite(results, ratchet);
        for (const auto& r : results) {
            if (!r.key.empty() && r.ratio > 0.0) per_key[r.key].push_back(r.ratio);
        }
    }
    if (runs > 1) {
        for (auto& r : results) {
            auto it = per_key.find(r.key);
            if (it != per_key.end()) r.ratio = RatchetManager::median(it->second);
        }
        std::cout << "\n[RUN] numbers below are each key's median over " << runs << " runs.\n";
    }

    const double load_during = report_load_during_run(run_start);
    const bool machine_busy = load_before > kBusyMachinePercent || load_during > kBusyMachinePercent;

    // 4. Record goldens if requested
    if (update_ratchet) {
        BenchmarkReporter::print_ratchet_summary(results, ratchet);
        if (machine_busy && !force) {
            std::cerr << "[RATCHET] ERROR: the machine was busy (see [LOAD] above); not recording goldens "
                         "from this run. Re-run on a quiet machine, or pass --force.\n";
            return 1;
        }
        if (machine_busy) {
            std::cerr << "[RATCHET] WARNING: --force: writing goldens from a run on a busy machine.\n";
        }
        if (is_debug_build()) {
            std::cerr << "[RATCHET] ERROR: goldens come from optimized builds only.\n";
            return 1;
        }
        for (const auto& r : results) {
            if (r.key.empty() || r.ratio <= 0.0) continue;
            ratchet.set_ratio(r.key, r.ratio);
            const double m = ratchet.record_margin(r.key, per_key[r.key], r.ratio);
            if (m > 0.0) {
                std::cout << "[RATCHET] " << r.key << ": runs on this host spread past half its margin; margin "
                          << static_cast<int>(m * 100.0 + 0.5) << "% here (runs:";
                for (double v : per_key[r.key]) std::cout << " " << v;
                std::cout << ")\n";
            }
        }
        if (!ratchet.save(ratchet_path)) {
            std::cerr << "[RATCHET] ERROR: Failed to write goldens to: " << ratchet_path << "\n";
            return 1;
        }
        std::cout << "[RATCHET] Wrote " << per_key.size() << " goldens for host " << host << " to "
                  << ratchet_path << "\n";
        return 0;
    }

    // 5. The one verdict: regression against this machine's goldens
    BenchmarkReporter::print_ratchet_summary(results, ratchet);
    if (!has_goldens) {
        std::cout << "[RATCHET] PASS: host " << host << " has no goldens; " << results.size()
                  << " keys reported, none checked.\n\n";
        return 0;
    }
    std::vector<std::string> ratchet_failures;
    const bool ratchet_ok = ratchet.check_ratchet(results, ratchet_failures);
    size_t new_keys = 0;
    for (const auto& r : results) {
        if (!r.key.empty() && !ratchet.has_ratio(r.key)) ++new_keys;
    }
    if (new_keys > 0) {
        std::cout << "[RATCHET] " << new_keys << " key(s) have no golden yet ([NEW]); --update-ratchet records them.\n";
    }
    if (machine_busy) {
        std::cout << "[LOAD] WARNING: the machine was busy during this run (see [LOAD] above); "
                     "treat the verdict below as untrustworthy.\n";
    }

    if (ratchet_ok) {
        std::cout << "[RATCHET] PASS: no key regressed past its margin (" << results.size() << " keys).\n\n";
        return 0;
    }

    // Plain runs (no --check-ratchet) report but never fail: they are for
    // reading numbers. --check-ratchet is the gate.
    const bool gate = check_ratchet && !is_debug_build();
    std::ostream& out = gate ? std::cerr : std::cout;
    out << "\n[RATCHET] " << (gate ? "FAIL" : "INFO") << ": " << ratchet_failures.size()
        << " key(s) regressed past their margin:\n";
    for (const auto& f : ratchet_failures) out << "  - " << f << "\n";
    out << "\n";
    return gate ? 1 : 0;
}
