#include "bench_utils.hpp"
#include "bench_load.hpp"
#include "bench_numeric.hpp"
#include "bench_js_shapes.hpp"
#include "bench_gc.hpp"
#include "bench_compile_speed.hpp"
#include "bench_simd_math.hpp"
#include "bench_interpreter.hpp"
#include <cstdlib>
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

bool read_env_int(const char* name, int& out) {
#if defined(_MSC_VER)
    // MSVC deprecates getenv (C4996, an error under /WX).
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return false;
    out = std::atoi(owned);
    std::free(owned);
    return true;
#else
    const char* v = std::getenv(name);
    if (!v) return false;
    out = std::atoi(v);
    return true;
#endif
}

} // namespace

int main(int argc, char** argv) {
    bool check_ratchet = false;
    bool update_ratchet = false;
    std::string custom_ratchet_file;
    int cpu = kDefaultBenchCpu;
    read_env_int("BRASS_BENCH_CPU", cpu);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--check-ratchet") {
            check_ratchet = true;
        } else if (arg == "--update-ratchet") {
            update_ratchet = true;
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
                      << "                            than its margin (bench/ratchet.json + platform overlay)\n"
                      << "  --update-ratchet          Write this run's medians as the new goldens\n"
                      << "  --ratchet-file <path>     Specify custom path to ratchet.json\n"
                      << "  --cpu <n>                 Pin the benchmark thread to logical CPU n; -1 = no pinning\n"
                      << "                            (default " << kDefaultBenchCpu << "; env BRASS_BENCH_CPU)\n"
                      << "  -h, --help                Show this help message\n";
            return 0;
        }
    }

    // 1. Locate and load golden ratchet values
    std::string ratchet_path = RatchetManager::find_ratchet_file(custom_ratchet_file);
    RatchetManager ratchet;
    if (!ratchet.load(ratchet_path)) {
        std::cout << "[RATCHET] No goldens at " << ratchet_path << "; every key reports as new.\n";
    }
    const std::string overlay_path = RatchetManager::platform_overlay_path(ratchet_path);
    if (ratchet.load_overlay(overlay_path)) {
        std::cout << "[RATCHET] " << RatchetManager::platform_name() << " goldens from " << overlay_path << "\n";
    }
    RatchetManager::active() = &ratchet;

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

    // Warm up CPU clock frequency and OS memory subsystem
    {
        volatile double dummy = 0.0;
        for (int w = 0; w < 2000000; ++w) {
            dummy = dummy + 1.0 / (w + 1.0);
        }
        (void)dummy;
        brass_zeroupper();
    }

    std::vector<BenchmarkResult> results;

    // 3. Numeric & Algorithmic Microbenchmarks
    run_numeric_benchmarks(results, ratchet);

    // 3b. SIMD Vector Math Microbenchmarks
    run_simd_math_benchmarks(results, ratchet);

    // 4. JS-Shaped Benchmarks (NaN-boxing, shape guards, patchable IC, Cheney GC alloc)
    run_js_shapes_benchmarks(results, ratchet);

    // 5. GC Model Comparison (Brass Stack-Maps vs Shadow-Stack)
    run_gc_benchmark(results);

    // 6. Compile-Speed Benchmark (Parse, Verify, ISEL, RegAlloc, Codegen)
    run_compile_speed_benchmark(results);

    // 6b. Fast Bytecode Interpreter Benchmarks (FastInterpreter vs native; oracle and baseline JIT for context)
    run_interpreter_benchmarks(results, ratchet);

    const double load_during = report_load_during_run(run_start);
    const bool machine_busy = load_before > kBusyMachinePercent || load_during > kBusyMachinePercent;

    // 7. Update ratchet if requested
    if (update_ratchet) {
        if (machine_busy) {
            std::cerr << "[RATCHET] WARNING: writing goldens from a run on a busy machine.\n";
        }
        ratchet.update_from_results(results);
        const bool to_overlay = ratchet.has_overlay();
        const std::string& target = to_overlay ? overlay_path : ratchet_path;
        const bool ok = to_overlay ? ratchet.save_overlay(overlay_path) : ratchet.save(ratchet_path);
        if (!ok) {
            std::cerr << "[RATCHET] ERROR: Failed to write updated ratchet to: " << target << "\n";
            return 1;
        }
        std::cout << "[RATCHET] Wrote " << results.size() << " goldens to " << target << "\n";
        return 0;
    }

    // 8. The one verdict: regression against the goldens
    BenchmarkReporter::print_ratchet_summary(results, ratchet);
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
