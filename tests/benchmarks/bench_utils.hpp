#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <ctime>
#include <cctype>

#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace brass::bench {

// ============================================================================
// Compiler Barriers & Anti-DCE Utilities
// ============================================================================

template <typename T>
inline void DoNotOptimize(T& val) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(val) : : "memory");
#elif defined(__GNUC__)
    asm volatile("" : "+m"(val) : : "memory");
#elif defined(_MSC_VER)
    #if defined(_M_X64) || defined(_M_IX86)
    _ReadWriteBarrier();
    #endif
    *reinterpret_cast<char volatile*>(&reinterpret_cast<char&>(val)) =
        *reinterpret_cast<char const volatile*>(&reinterpret_cast<char const&>(val));
#else
    char volatile* p = reinterpret_cast<char volatile*>(&val);
    *p = *p;
#endif
}

template <typename T>
inline void DoNotOptimize(const T& val) {
#if defined(__clang__)
    asm volatile("" : : "r,m"(val) : "memory");
#elif defined(__GNUC__)
    asm volatile("" : : "m"(val) : "memory");
#elif defined(_MSC_VER)
    #if defined(_M_X64) || defined(_M_IX86)
    _ReadWriteBarrier();
    #endif
    char volatile v = *reinterpret_cast<char const volatile*>(&reinterpret_cast<char const&>(val));
    (void)v;
#else
    char const volatile* p = reinterpret_cast<char const volatile*>(&val);
    (void)*p;
#endif
}

inline void ClobberMemory() {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#elif defined(_MSC_VER)
    #if defined(_M_X64) || defined(_M_IX86)
    _ReadWriteBarrier();
    #endif
#endif
}

// ============================================================================
// Build Type & Environment Utilities
// ============================================================================

inline bool is_debug_build() {
#if defined(BRASS_BUILD_TYPE)
    std::string_view bt = BRASS_BUILD_TYPE;
    if (bt == "Debug" || bt == "debug" || bt == "DEBUG") {
        return true;
    }
    if (bt == "Release" || bt == "release" || bt == "RELEASE" || bt == "RelWithDebInfo" || bt == "MinSizeRel") {
        return false;
    }
#endif
#if !defined(NDEBUG)
    return true;
#else
    return false;
#endif
}


// ============================================================================
// Stopwatch & Benchmark Timing
// ============================================================================

class Stopwatch {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// Timing statistics across multi-repetition measurements
struct TimingStats {
    std::vector<double> samples;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;

    TimingStats() = default;

    explicit TimingStats(std::vector<double> s) {
        samples = std::move(s);
        if (samples.empty()) return;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        min = sorted.front();
        max = sorted.back();
        size_t n = sorted.size();
        if (n % 2 == 1) {
            median = sorted[n / 2];
        } else {
            median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }
    }

    TimingStats(std::initializer_list<double> list)
        : TimingStats(std::vector<double>(list)) {}
};

struct BenchmarkResult {
    std::string key; // Benchmark identifier, e.g. "fib", "matmul_i64_32_naive"
    std::string name;
    size_t iterations = 0;
    size_t repetitions = 5;
    double native_ms = 0.0;
    double native_min_ms = 0.0;
    double native_max_ms = 0.0;
    double native_scalar_ms = 0.0;
    double native_scalar_min_ms = 0.0;
    double native_scalar_max_ms = 0.0;
    double brass_ms = 0.0;
    double brass_min_ms = 0.0;
    double brass_max_ms = 0.0;
    double ratio = 0.0; // vs scalar or vs native (median)
    double ratio_min = 0.0;
    double ratio_max = 0.0;
    double ratio_vec = 0.0; // vs vectorized native (median)
    double ratio_vec_min = 0.0;
    double ratio_vec_max = 0.0;
    double target_ratio = 0.0; // Golden ratio from ratchet
    bool passes_bar = true;
    std::string notes;
};

inline size_t default_bench_placements() noexcept {
    return is_debug_build() ? 1 : 5;
}

inline size_t default_bench_repetitions() noexcept {
    return is_debug_build() ? 1 : 5;
}

constexpr size_t DEFAULT_BENCH_REPETITIONS = 5;

template <typename F>
inline TimingStats measure_repetitions(size_t repetitions, F&& func) {
    if (is_debug_build()) {
        repetitions = std::min(repetitions, default_bench_repetitions());
    }
    // Warmup pass to prime instruction caches and CPU frequency
    func();

    std::vector<double> samples;
    samples.reserve(repetitions);
    Stopwatch sw;
    for (size_t r = 0; r < repetitions; ++r) {
        sw.start();
        func();
        samples.push_back(sw.stop_ms());
    }
    return TimingStats(std::move(samples));
}

struct PairedRepetitionResult {
    TimingStats native_stats;
    TimingStats brass_stats;
    TimingStats ratio_stats;
};

struct TripletRepetitionResult {
    TimingStats native_stats;       // Vectorized native
    TimingStats scalar_stats;       // Scalar native
    TimingStats brass_stats;        // Brass JIT
    TimingStats ratio_scalar_stats; // Brass / Scalar
    TimingStats ratio_vec_stats;    // Brass / Vectorized
};

inline void brass_zeroupper() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("vzeroupper" ::: "memory");
#endif
#endif
}

constexpr size_t DEFAULT_BENCH_PLACEMENTS = 5;
constexpr size_t kPlacementPaddings[5] = { 0, 64, 128, 192, 256 };

template <typename FNative, typename FJitFactory>
inline PairedRepetitionResult measure_paired_multi_placement(
    size_t repetitions,
    FNative&& fn_native,
    FJitFactory&& jit_factory
) {
    if (is_debug_build()) {
        repetitions = std::min(repetitions, default_bench_repetitions());
    }
    fn_native();
    brass_zeroupper();

    const size_t num_placements = default_bench_placements();
    using RunnerType = std::decay_t<decltype(jit_factory(0))>;
    std::vector<RunnerType> jit_runners;
    jit_runners.reserve(num_placements);
    for (size_t p = 0; p < num_placements; ++p) {
        auto r = jit_factory(kPlacementPaddings[p]);
        r();
        brass_zeroupper();
        jit_runners.push_back(std::move(r));
    }

    std::vector<double> nat_samples, jit_samples, ratio_samples;
    nat_samples.reserve(repetitions);
    jit_samples.reserve(repetitions);
    ratio_samples.reserve(repetitions);
    Stopwatch sw;

    for (size_t r = 0; r < repetitions; ++r) {
        std::vector<double> p_nat_samples, p_jit_samples, p_rat_samples;
        p_nat_samples.reserve(num_placements);
        p_jit_samples.reserve(num_placements);
        p_rat_samples.reserve(num_placements);

        for (auto& runner : jit_runners) {
            sw.start();
            fn_native();
            double n_ms = sw.stop_ms();
            brass_zeroupper();

            sw.start();
            runner();
            double j_ms = sw.stop_ms();
            brass_zeroupper();

            p_nat_samples.push_back(n_ms);
            p_jit_samples.push_back(j_ms);
            p_rat_samples.push_back((n_ms > 0.0) ? (j_ms / n_ms) : 1.0);
        }

        std::sort(p_nat_samples.begin(), p_nat_samples.end());
        std::sort(p_jit_samples.begin(), p_jit_samples.end());
        std::sort(p_rat_samples.begin(), p_rat_samples.end());

        nat_samples.push_back(p_nat_samples[num_placements / 2]);
        jit_samples.push_back(p_jit_samples[num_placements / 2]);
        ratio_samples.push_back(p_rat_samples[num_placements / 2]);
    }

    PairedRepetitionResult res;
    res.native_stats = TimingStats(std::move(nat_samples));
    res.brass_stats = TimingStats(std::move(jit_samples));
    res.ratio_stats = TimingStats(std::move(ratio_samples));
    return res;
}

template <typename F1, typename F2, typename FJitFactory>
inline TripletRepetitionResult measure_triplet_multi_placement(
    size_t repetitions,
    F1&& fn_vec,
    F2&& fn_scalar,
    FJitFactory&& jit_factory
) {
    if (is_debug_build()) {
        repetitions = std::min(repetitions, default_bench_repetitions());
    }
    fn_vec();
    brass_zeroupper();
    fn_scalar();
    brass_zeroupper();

    const size_t num_placements = default_bench_placements();
    using RunnerType = std::decay_t<decltype(jit_factory(0))>;
    std::vector<RunnerType> jit_runners;
    jit_runners.reserve(num_placements);
    for (size_t p = 0; p < num_placements; ++p) {
        auto r = jit_factory(kPlacementPaddings[p]);
        r();
        brass_zeroupper();
        jit_runners.push_back(std::move(r));
    }

    std::vector<double> native_samples, scalar_samples, brass_samples;
    std::vector<double> ratio_scalar_samples, ratio_vec_samples;
    native_samples.reserve(repetitions);
    scalar_samples.reserve(repetitions);
    brass_samples.reserve(repetitions);
    ratio_scalar_samples.reserve(repetitions);
    ratio_vec_samples.reserve(repetitions);

    Stopwatch sw;
    for (size_t r = 0; r < repetitions; ++r) {
        std::vector<double> p_vec_samples, p_scalar_samples, p_jit_samples;
        std::vector<double> p_r_scalar_samples, p_r_vec_samples;
        p_vec_samples.reserve(num_placements);
        p_scalar_samples.reserve(num_placements);
        p_jit_samples.reserve(num_placements);
        p_r_scalar_samples.reserve(num_placements);
        p_r_vec_samples.reserve(num_placements);

        for (auto& runner : jit_runners) {
            sw.start();
            fn_vec();
            double n_ms = sw.stop_ms();
            brass_zeroupper();

            sw.start();
            fn_scalar();
            double s_ms = sw.stop_ms();
            brass_zeroupper();

            sw.start();
            runner();
            double j_ms = sw.stop_ms();
            brass_zeroupper();

            p_vec_samples.push_back(n_ms);
            p_scalar_samples.push_back(s_ms);
            p_jit_samples.push_back(j_ms);
            p_r_scalar_samples.push_back((s_ms > 0.0) ? (j_ms / s_ms) : 1.0);
            p_r_vec_samples.push_back((n_ms > 0.0) ? (j_ms / n_ms) : 1.0);
        }

        std::sort(p_vec_samples.begin(), p_vec_samples.end());
        std::sort(p_scalar_samples.begin(), p_scalar_samples.end());
        std::sort(p_jit_samples.begin(), p_jit_samples.end());
        std::sort(p_r_scalar_samples.begin(), p_r_scalar_samples.end());
        std::sort(p_r_vec_samples.begin(), p_r_vec_samples.end());

        native_samples.push_back(p_vec_samples[num_placements / 2]);
        scalar_samples.push_back(p_scalar_samples[num_placements / 2]);
        brass_samples.push_back(p_jit_samples[num_placements / 2]);
        ratio_scalar_samples.push_back(p_r_scalar_samples[num_placements / 2]);
        ratio_vec_samples.push_back(p_r_vec_samples[num_placements / 2]);
    }

    return TripletRepetitionResult{
        TimingStats(std::move(native_samples)),
        TimingStats(std::move(scalar_samples)),
        TimingStats(std::move(brass_samples)),
        TimingStats(std::move(ratio_scalar_samples)),
        TimingStats(std::move(ratio_vec_samples))
    };
}

template <typename FNative, typename FJit>
inline PairedRepetitionResult measure_paired_repetitions(size_t repetitions, FNative&& fn_native, FJit&& fn_jit) {
    return measure_paired_multi_placement(repetitions, std::forward<FNative>(fn_native), [&](size_t) { return fn_jit; });
}

template <typename F1, typename F2, typename F3>
inline TripletRepetitionResult measure_triplet_repetitions(size_t repetitions, F1&& fn_vec, F2&& fn_scalar, F3&& fn_jit) {
    return measure_triplet_multi_placement(repetitions, std::forward<F1>(fn_vec), std::forward<F2>(fn_scalar), [&](size_t) { return fn_jit; });
}

class RatchetManager;

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
    double target_ratio,
    const std::string& notes = ""
);

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
    const RatchetManager& ratchet,
    double default_target = 1.30,
    const std::string& notes = ""
);

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
    double target_ratio,
    const std::string& notes = ""
);

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
    const RatchetManager& ratchet,
    double default_target = 1.30,
    const std::string& notes = ""
);

// ============================================================================
// Performance Ratchet Manager
// ============================================================================

class RatchetManager {
public:
    static RatchetManager*& active() {
        static RatchetManager* s_active = nullptr;
        return s_active;
    }

    static RatchetManager defaults() {
        RatchetManager rm;
        rm.ratios_ = {
            {"cheney_gc", 1.35},
            {"collatz", 1.20},
            {"compile_speed", 1000.00},
            // FastInterpreter time over Oracle time (lower is better).
            {"fast_interp_collatz_1000", 0.08},
            {"fast_interp_fib_iter_40", 0.12},
            {"fast_interp_fib_rec_22", 0.22},
            {"fast_interp_matmul_32x32", 0.12},
            {"fast_interp_prime_sieve_100k", 0.10},
            {"fib", 2.50},
            {"gc_model_speedup", 1.25},
            {"icache", 0.80},
            {"linked_list", 0.95},
            {"matmul_f64_32_reassoc_naive", 1.25},
            {"matmul_f64_32_reassoc_preopt", 1.25},
            {"matmul_f64_32_strict_naive", 3.30},
            {"matmul_f64_32_strict_preopt", 3.30},
            {"matmul_f64_64_reassoc_naive", 1.05},
            {"matmul_f64_64_reassoc_preopt", 1.05},
            {"matmul_f64_64_strict_naive", 2.50},
            {"matmul_f64_64_strict_preopt", 2.50},
            {"matmul_i64_32_naive", 1.45},
            {"matmul_i64_32_preopt", 1.45},
            {"matmul_i64_64_naive", 1.45},
            {"matmul_i64_64_preopt", 1.45},
            {"nanbox", 1.15},
            {"shapes", 1.35},
            {"sieve", 1.35},
            {"simd_dot4", 0.50},
            {"simd_matmul4x4", 0.50},
            {"simd_vec3_math", 2.20}
        };
        return rm;
    }

    bool has_ratio(const std::string& key) const {
        return ratios_.find(key) != ratios_.end();
    }

    static std::string find_ratchet_file(const std::string& explicit_path = "") {
        if (!explicit_path.empty()) {
            std::ifstream f(explicit_path);
            if (f.good()) return explicit_path;
        }
#if defined(BRASS_BENCH_RATCHET_PATH)
        {
            std::ifstream f(BRASS_BENCH_RATCHET_PATH);
            if (f.good()) return BRASS_BENCH_RATCHET_PATH;
        }
#endif
        const std::vector<std::string> candidates = {
            "bench/ratchet.json",
            "../bench/ratchet.json",
            "../../bench/ratchet.json"
        };
        for (const auto& path : candidates) {
            std::ifstream f(path);
            if (f.good()) return path;
        }
        return "bench/ratchet.json";
    }

    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        size_t pos = 0;
        while (pos < content.size()) {
            size_t quote_start = content.find('"', pos);
            if (quote_start == std::string::npos) break;
            size_t quote_end = content.find('"', quote_start + 1);
            if (quote_end == std::string::npos) break;
            std::string key = content.substr(quote_start + 1, quote_end - quote_start - 1);

            size_t colon_pos = content.find(':', quote_end);
            if (colon_pos == std::string::npos) break;

            size_t val_start = colon_pos + 1;
            while (val_start < content.size() && (std::isspace(static_cast<unsigned char>(content[val_start])) || content[val_start] == '\r' || content[val_start] == '\n')) {
                val_start++;
            }

            size_t val_end = val_start;
            while (val_end < content.size() && (std::isdigit(static_cast<unsigned char>(content[val_end])) || content[val_end] == '.' || content[val_end] == '-' || content[val_end] == '+' || content[val_end] == 'e' || content[val_end] == 'E')) {
                val_end++;
            }

            if (val_end > val_start) {
                std::string num_str = content.substr(val_start, val_end - val_start);
                try {
                    double val = std::stod(num_str);
                    ratios_[key] = val;
                } catch (...) {}
            }
            pos = val_end;
        }
        return !ratios_.empty();
    }

    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        file << "{\n";
        size_t idx = 0;
        for (auto it = ratios_.begin(); it != ratios_.end(); ++it, ++idx) {
            file << "  \"" << it->first << "\": " << std::fixed << std::setprecision(2) << it->second;
            if (idx + 1 < ratios_.size()) {
                file << ",";
            }
            file << "\n";
        }
        file << "}\n";
        return true;
    }

    double get_ratio(const std::string& key, double default_val = 1.30) const {
        auto it = ratios_.find(key);
        if (it != ratios_.end()) {
            return it->second;
        }
        return default_val;
    }

    void set_ratio(const std::string& key, double val) {
        ratios_[key] = val;
    }

    const std::map<std::string, double>& ratios() const {
        return ratios_;
    }

    bool check_ratchet(const std::vector<BenchmarkResult>& results, double max_regression_factor, std::vector<std::string>& out_failures) const {
        bool all_passed = true;
        for (const auto& res : results) {
            if (res.key.empty()) continue;
            if (res.key == "gc_model_speedup") {
                double golden = get_ratio(res.key, 1.25);
                double min_allowed = golden * (2.0 - max_regression_factor);
                if (res.ratio < min_allowed) {
                    all_passed = false;
                    std::ostringstream oss;
                    oss << "Benchmark '" << res.name << "' (key: " << res.key << "): measured "
                        << std::fixed << std::setprecision(2) << res.ratio << "x is below ratchet speedup "
                        << min_allowed << "x";
                    out_failures.push_back(oss.str());
                }
            } else {
                double golden = get_ratio(res.key, 1.30);
                double max_allowed = golden * max_regression_factor;
                if (res.ratio > max_allowed) {
                    all_passed = false;
                    std::ostringstream oss;
                    if (res.key == "compile_speed") {
                        oss << "Benchmark '" << res.name << "' (key: " << res.key << "): measured "
                            << std::fixed << std::setprecision(2) << res.ratio << " ms exceeds ratchet "
                            << golden << " ms * " << max_regression_factor << " (" << max_allowed << " ms)";
                    } else {
                        oss << "Benchmark '" << res.name << "' (key: " << res.key << "): measured "
                            << std::fixed << std::setprecision(2) << res.ratio << "x exceeds ratchet "
                            << golden << "x * " << max_regression_factor << " (" << max_allowed << "x)";
                    }
                    out_failures.push_back(oss.str());
                }
            }
        }
        return all_passed;
    }

    void update_from_results(const std::vector<BenchmarkResult>& results, bool only_if_improved = false) {
        for (const auto& res : results) {
            if (res.key.empty() || res.ratio <= 0.0) continue;
            auto it = ratios_.find(res.key);
            if (it == ratios_.end()) {
                ratios_[res.key] = res.ratio;
            } else if (res.key == "gc_model_speedup") {
                if (!only_if_improved || res.ratio > it->second) {
                    it->second = res.ratio;
                }
            } else if (!only_if_improved || res.ratio < it->second) {
                it->second = res.ratio;
            }
        }
    }

private:
    std::map<std::string, double> ratios_;
};

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
    double target_ratio,
    const std::string& notes
) {
    if (RatchetManager::active() && RatchetManager::active()->has_ratio(key)) {
        target_ratio = RatchetManager::active()->get_ratio(key, target_ratio);
    }
    BenchmarkResult r;
    r.key = key;
    r.name = name;
    r.iterations = iterations;
    r.repetitions = paired.ratio_stats.samples.size();
    r.native_ms = paired.native_stats.median;
    r.native_min_ms = paired.native_stats.min;
    r.native_max_ms = paired.native_stats.max;
    r.native_scalar_ms = 0.0;
    r.brass_ms = paired.brass_stats.median;
    r.brass_min_ms = paired.brass_stats.min;
    r.brass_max_ms = paired.brass_stats.max;
    r.ratio = paired.ratio_stats.median;
    r.ratio_min = paired.ratio_stats.min;
    r.ratio_max = paired.ratio_stats.max;
    r.ratio_vec = 0.0;
    r.target_ratio = target_ratio;
    r.passes_bar = (target_ratio > 0.0) ? (r.ratio <= target_ratio) : true;
    if (!notes.empty()) {
        r.notes = notes;
    } else if (target_ratio > 0.0) {
        std::ostringstream oss;
        oss << "<= " << std::fixed << std::setprecision(2) << target_ratio << "x baseline";
        r.notes = oss.str();
    }
    return r;
}

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
    const RatchetManager& ratchet,
    double default_target,
    const std::string& notes
) {
    double target = ratchet.has_ratio(key) ? ratchet.get_ratio(key, default_target) : default_target;
    return make_paired_result(key, name, iterations, paired, target, notes);
}

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
    double target_ratio,
    const std::string& notes
) {
    if (RatchetManager::active() && RatchetManager::active()->has_ratio(key)) {
        target_ratio = RatchetManager::active()->get_ratio(key, target_ratio);
    }
    BenchmarkResult r;
    r.key = key;
    r.name = name;
    r.iterations = iterations;
    r.repetitions = triplet.ratio_scalar_stats.samples.size();
    r.native_ms = triplet.native_stats.median;
    r.native_min_ms = triplet.native_stats.min;
    r.native_max_ms = triplet.native_stats.max;
    r.native_scalar_ms = triplet.scalar_stats.median;
    r.native_scalar_min_ms = triplet.scalar_stats.min;
    r.native_scalar_max_ms = triplet.scalar_stats.max;
    r.brass_ms = triplet.brass_stats.median;
    r.brass_min_ms = triplet.brass_stats.min;
    r.brass_max_ms = triplet.brass_stats.max;
    r.ratio = triplet.ratio_scalar_stats.median;
    r.ratio_min = triplet.ratio_scalar_stats.min;
    r.ratio_max = triplet.ratio_scalar_stats.max;
    r.ratio_vec = triplet.ratio_vec_stats.median;
    r.ratio_vec_min = triplet.ratio_vec_stats.min;
    r.ratio_vec_max = triplet.ratio_vec_stats.max;
    r.target_ratio = target_ratio;
    r.passes_bar = (target_ratio > 0.0) ? (r.ratio <= target_ratio) : true;
    if (!notes.empty()) {
        r.notes = notes;
    } else {
        std::ostringstream oss;
        oss << "<= " << std::fixed << std::setprecision(2) << target_ratio << "x scalar (vec: "
            << std::fixed << std::setprecision(2) << r.ratio_vec << "x)";
        r.notes = oss.str();
    }
    return r;
}

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
    const RatchetManager& ratchet,
    double default_target,
    const std::string& notes
) {
    double target = ratchet.has_ratio(key) ? ratchet.get_ratio(key, default_target) : default_target;
    return make_triplet_result(key, name, iterations, triplet, target, notes);
}

// ============================================================================
// Shadow-stack frame definition for GC comparison
// ============================================================================

struct ShadowStackFrame {
    ShadowStackFrame* prev = nullptr;
    uint32_t count = 0;
    void* roots[8] = {nullptr};
};

struct ThreadShadowStack {
    ShadowStackFrame* top = nullptr;

    inline void push(ShadowStackFrame* frame, uint32_t count) noexcept {
        frame->prev = top;
        frame->count = count;
        top = frame;
        DoNotOptimize(top);
    }

    inline void pop() noexcept {
        if (top) {
            top = top->prev;
            DoNotOptimize(top);
        }
    }
};

} // namespace brass::bench

// ============================================================================
// Reporter (extracted to bench_reporter.hpp)
// ============================================================================

#include "bench_reporter.hpp"
