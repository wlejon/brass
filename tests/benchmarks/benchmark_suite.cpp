#include "bench_utils.hpp"
#include "bench_numeric.hpp"
#include "bench_js_shapes.hpp"
#include "bench_gc.hpp"
#include "bench_compile_speed.hpp"
#include <vector>
#include <string>
#include <iostream>

using namespace brass::bench;

int main(int argc, char** argv) {
    bool check_ratchet = false;
    bool update_ratchet = false;
    std::string custom_ratchet_file;

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
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Brass Performance Benchmark Suite\n"
                      << "Usage: brass_benchmarks [options]\n"
                      << "Options:\n"
                      << "  --check-ratchet           Verify measured ratios against golden ratchet (< 10% regression)\n"
                      << "  --update-ratchet          Update ratchet.json with current measured ratios\n"
                      << "  --ratchet-file <path>     Specify custom path to ratchet.json\n"
                      << "  -h, --help                Show this help message\n";
            return 0;
        }
    }

    // 1. Locate and load golden ratchet ratios
    std::string ratchet_path = RatchetManager::find_ratchet_file(custom_ratchet_file);
    RatchetManager ratchet = RatchetManager::defaults();
    bool loaded_from_disk = ratchet.load(ratchet_path);
    (void)loaded_from_disk;

    // 2. Emit Run Provenance Header
    BenchmarkReporter::print_header("Brass JIT vs Native C++ (-O3) & GC Performance");

    // Warm up CPU clock frequency and OS memory subsystem
    {
        volatile double dummy = 0.0;
        for (int w = 0; w < 2000000; ++w) {
            dummy += 1.0 / (w + 1.0);
        }
        (void)dummy;
        brass_zeroupper();
    }

    std::vector<BenchmarkResult> results;

    // 3. Numeric & Algorithmic Microbenchmarks
    run_numeric_benchmarks(results, ratchet);

    // 4. JS-Shaped Benchmarks (NaN-boxing, shape guards, patchable IC, Cheney GC alloc)
    run_js_shapes_benchmarks(results, ratchet);

    // 5. GC Model Comparison (Brass Stack-Maps vs Shadow-Stack)
    run_gc_benchmark(results);

    // 6. Compile-Speed Benchmark (Parse, Verify, ISEL, RegAlloc, Codegen)
    run_compile_speed_benchmark(results);

    // 7. Update ratchet if requested
    if (update_ratchet) {
        ratchet.update_from_results(results, false);
        if (ratchet.save(ratchet_path)) {
            std::cout << "[RATCHET] Successfully updated golden ratchet ratios at: " << ratchet_path << "\n";
        } else {
            std::cerr << "[RATCHET] ERROR: Failed to write updated ratchet to: " << ratchet_path << "\n";
            return 1;
        }
    }

    // 8. Check ratchet if requested
    bool any_ratchet_failed = false;
    std::vector<std::string> ratchet_failures;
    if (check_ratchet) {
        BenchmarkReporter::print_ratchet_summary(results, ratchet, 1.10);
        bool ratchet_ok = ratchet.check_ratchet(results, 1.10, ratchet_failures);
        if (!ratchet_ok) {
            any_ratchet_failed = true;
        } else {
            std::cout << "[RATCHET] All " << results.size() << " benchmarks passed performance ratchet verification (< 10% regression margin).\n\n";
        }
    }

    // 9. Check if any benchmark failed its target bar
    bool any_benchmark_failed = false;
    std::vector<std::string> benchmark_failures;
    for (const auto& res : results) {
        if (!res.passes_bar) {
            any_benchmark_failed = true;
            std::ostringstream oss;
            if (res.key == "compile_speed") {
                oss << "Benchmark '" << res.name << "': measured "
                    << std::fixed << std::setprecision(2) << res.ratio << " ms exceeds target "
                    << res.target_ratio << " ms";
            } else if (res.key == "gc_model_speedup") {
                oss << "Benchmark '" << res.name << "': measured "
                    << std::fixed << std::setprecision(2) << res.ratio << "x is below required "
                    << res.target_ratio << "x speedup";
            } else {
                oss << "Benchmark '" << res.name << "' (key: " << res.key << "): measured "
                    << std::fixed << std::setprecision(2) << res.ratio << "x exceeds target "
                    << res.target_ratio << "x";
            }
            benchmark_failures.push_back(oss.str());
        }
    }

    // 10. Exit code handling: non-zero in Release if any benchmark or ratchet failed
    if (any_benchmark_failed || any_ratchet_failed) {
        if (is_debug_build()) {
            std::cout << "[INFO] Benchmark targets / ratchet check reported issues (informational in Debug build):\n";
            for (const auto& f : benchmark_failures) {
                std::cout << "  - [INFO] Target failure: " << f << "\n";
            }
            for (const auto& f : ratchet_failures) {
                std::cout << "  - [INFO] Ratchet failure: " << f << "\n";
            }
            std::cout << "\n";
            return 0;
        } else {
            std::cerr << "\n[BENCHMARK] FATAL: One or more benchmarks failed in Release build!\n";
            for (const auto& f : benchmark_failures) {
                std::cerr << "  - Target failure: " << f << "\n";
            }
            for (const auto& f : ratchet_failures) {
                std::cerr << "  - Ratchet regression: " << f << "\n";
            }
            std::cerr << "\n";
            return 1;
        }
    }

    return 0;
}
