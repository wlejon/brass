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
    if (check_ratchet) {
        BenchmarkReporter::print_ratchet_summary(results, ratchet, 1.10);
        std::vector<std::string> failures;
        bool ratchet_ok = ratchet.check_ratchet(results, 1.10, failures);
        if (!ratchet_ok) {
            std::cerr << "[RATCHET] FATAL: Performance ratchet check failed! Regressions detected:\n";
            for (const auto& f : failures) {
                std::cerr << "  - " << f << "\n";
            }
            return 1;
        }
        std::cout << "[RATCHET] All " << results.size() << " benchmarks passed performance ratchet verification (< 10% regression margin).\n\n";
    }

    return 0;
}
