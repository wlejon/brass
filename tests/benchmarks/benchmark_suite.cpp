#include "bench_utils.hpp"
#include "bench_numeric.hpp"
#include "bench_gc.hpp"
#include "bench_compile_speed.hpp"
#include <vector>

using namespace brass::bench;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    BenchmarkReporter::print_header("Brass JIT vs Native C++ (-O3) & GC Performance");

    std::vector<BenchmarkResult> results;

    // 1. Numeric & Algorithmic Microbenchmarks
    run_numeric_benchmarks(results);

    // 2. GC Model Comparison (Brass Stack-Maps vs Shadow-Stack)
    run_gc_benchmark(results);

    // 3. Compile-Speed Benchmark (Parse, Verify, ISEL, RegAlloc, Codegen)
    run_compile_speed_benchmark(results);

    return 0;
}
