#pragma once

#include "bench_utils.hpp"
#include <vector>

namespace brass::bench {

void run_simd_math_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet = RatchetManager::defaults());

} // namespace brass::bench
