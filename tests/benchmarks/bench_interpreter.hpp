#pragma once

#include "bench_utils.hpp"
#include <vector>

namespace brass::bench {

void run_interpreter_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet = RatchetManager::defaults());

} // namespace brass::bench
