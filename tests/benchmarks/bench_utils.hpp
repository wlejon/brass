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

inline std::string get_provenance_git_commit() {
    auto run_cmd = [](const char* cmd) -> std::string {
        std::string out;
#if defined(_WIN32)
        FILE* pipe = _popen(cmd, "r");
#else
        FILE* pipe = popen(cmd, "r");
#endif
        if (!pipe) return "";
        char buf[256];
        while (fgets(buf, sizeof(buf), pipe) != nullptr) {
            out += buf;
        }
#if defined(_WIN32)
        int rc = _pclose(pipe);
#else
        int rc = pclose(pipe);
#endif
        if (rc != 0) return "";
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '\t')) {
            out.pop_back();
        }
        return out;
    };

    std::string sha = run_cmd("git rev-parse --short HEAD");
    if (!sha.empty()) {
        std::string status = run_cmd("git status --porcelain");
        if (!status.empty()) {
            sha += "-dirty";
        }
        return sha;
    }
#if defined(BRASS_GIT_COMMIT)
    return BRASS_GIT_COMMIT;
#else
    return "unknown";
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

constexpr size_t DEFAULT_BENCH_REPETITIONS = 5;

template <typename F>
inline TimingStats measure_repetitions(size_t repetitions, F&& func) {
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

template <typename FNative, typename FJit>
inline PairedRepetitionResult measure_paired_repetitions(size_t repetitions, FNative&& fn_native, FJit&& fn_jit) {
    // Warmup pass to prime instruction caches and CPU frequency
    fn_native();
    fn_jit();

    PairedRepetitionResult res;
    std::vector<double> nat_samples, jit_samples, ratio_samples;
    nat_samples.reserve(repetitions);
    jit_samples.reserve(repetitions);
    ratio_samples.reserve(repetitions);
    Stopwatch sw;
    for (size_t r = 0; r < repetitions; ++r) {
        sw.start();
        fn_native();
        double n_ms = sw.stop_ms();

        sw.start();
        fn_jit();
        double j_ms = sw.stop_ms();

        nat_samples.push_back(n_ms);
        jit_samples.push_back(j_ms);
        double rat = (n_ms > 0.0) ? (j_ms / n_ms) : 1.0;
        ratio_samples.push_back(rat);
    }
    res.native_stats = TimingStats(std::move(nat_samples));
    res.brass_stats = TimingStats(std::move(jit_samples));
    res.ratio_stats = TimingStats(std::move(ratio_samples));
    return res;
}

struct TripletRepetitionResult {
    TimingStats native_stats;       // Vectorized native
    TimingStats scalar_stats;       // Scalar native
    TimingStats brass_stats;        // Brass JIT
    TimingStats ratio_scalar_stats; // Brass / Scalar
    TimingStats ratio_vec_stats;    // Brass / Vectorized
};

template <typename F1, typename F2, typename F3>
inline TripletRepetitionResult measure_triplet_repetitions(
    size_t repetitions,
    F1&& fn_vec,
    F2&& fn_scalar,
    F3&& fn_jit
) {
    // Warmup pass to prime instruction caches and CPU frequency
    fn_vec();
    fn_scalar();
    fn_jit();

    std::vector<double> native_samples, scalar_samples, brass_samples;
    std::vector<double> ratio_scalar_samples, ratio_vec_samples;
    native_samples.reserve(repetitions);
    scalar_samples.reserve(repetitions);
    brass_samples.reserve(repetitions);
    ratio_scalar_samples.reserve(repetitions);
    ratio_vec_samples.reserve(repetitions);

    Stopwatch sw;
    for (size_t r = 0; r < repetitions; ++r) {
        sw.start();
        fn_vec();
        double n_ms = sw.stop_ms();

        sw.start();
        fn_scalar();
        double s_ms = sw.stop_ms();

        sw.start();
        fn_jit();
        double j_ms = sw.stop_ms();

        native_samples.push_back(n_ms);
        scalar_samples.push_back(s_ms);
        brass_samples.push_back(j_ms);

        double r_scalar = (s_ms > 0.0) ? (j_ms / s_ms) : 1.0;
        double r_vec = (n_ms > 0.0) ? (j_ms / n_ms) : 1.0;
        ratio_scalar_samples.push_back(r_scalar);
        ratio_vec_samples.push_back(r_vec);
    }

    return TripletRepetitionResult{
        TimingStats(std::move(native_samples)),
        TimingStats(std::move(scalar_samples)),
        TimingStats(std::move(brass_samples)),
        TimingStats(std::move(ratio_scalar_samples)),
        TimingStats(std::move(ratio_vec_samples))
    };
}

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
    double target_ratio,
    const std::string& notes = ""
) {
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

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
    double target_ratio,
    const std::string& notes = ""
) {
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

// ============================================================================
// Performance Ratchet Manager
// ============================================================================

class RatchetManager {
public:
    static RatchetManager defaults() {
        RatchetManager rm;
        rm.ratios_ = {
            {"cheney_gc", 1.40},
            {"collatz", 1.10},
            {"compile_speed", 1000.00},
            {"fib", 1.30},
            {"icache", 1.10},
            {"linked_list", 1.05},
            {"matmul_f64_32_reassoc_naive", 1.05},
            {"matmul_f64_32_reassoc_preopt", 1.05},
            {"matmul_f64_32_strict_naive", 2.15},
            {"matmul_f64_32_strict_preopt", 2.10},
            {"matmul_f64_64_reassoc_naive", 0.90},
            {"matmul_f64_64_reassoc_preopt", 0.90},
            {"matmul_f64_64_strict_naive", 1.80},
            {"matmul_f64_64_strict_preopt", 1.80},
            {"matmul_i64_32_naive", 1.50},
            {"matmul_i64_32_preopt", 1.50},
            {"matmul_i64_64_naive", 1.55},
            {"matmul_i64_64_preopt", 1.55},
            {"nanbox", 2.20},
            {"shapes", 1.30},
            {"sieve", 1.40}
        };
        return rm;
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

// ============================================================================
// Reporter
// ============================================================================

class BenchmarkReporter {
public:
    static void print_header(std::string_view title) {
        // 1. Timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &now_c);
#else
        localtime_r(&now_c, &tm_buf);
#endif
        char time_str[64];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

        // 2. Git SHA (runtime resolution falling back to compile-time definition)
        std::string git_sha = get_provenance_git_commit();

        // 3. Build Type
#if defined(BRASS_BUILD_TYPE)
        std::string build_type = BRASS_BUILD_TYPE;
#elif defined(NDEBUG)
        std::string build_type = "Release";
#else
        std::string build_type = "Debug";
#endif

        // 4. Compiler Info
        std::string compiler_info;
#if defined(_MSC_VER)
        compiler_info = "MSVC " + std::to_string(_MSC_VER);
#if defined(_MSC_FULL_VER)
        compiler_info += " (" + std::to_string(_MSC_FULL_VER) + ")";
#endif
#elif defined(__clang__)
        compiler_info = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
        compiler_info = std::string("GCC ") + __VERSION__;
#else
        compiler_info = "Unknown C++ Compiler";
#endif

        std::cout << "\n==================================================================================================================================\n";
        std::cout << "  BRASS PERFORMANCE BENCHMARK SUITE: " << title << "\n";
        std::cout << "==================================================================================================================================\n";
        std::cout << "  Run Provenance:\n";
        std::cout << "  - Timestamp:    " << time_str << "\n";
        std::cout << "  - Git Commit:   " << git_sha << "\n";
        std::cout << "  - Build Type:   " << build_type << "\n";
        std::cout << "  - Compiler:     " << compiler_info << "\n";
        std::cout << "==================================================================================================================================\n";
        std::cout << std::left
                  << std::setw(34) << "Benchmark"
                  << std::setw(11) << "Iterations"
                  << std::setw(12) << "Native -O3"
                  << std::setw(12) << "Scalar -O3"
                  << std::setw(12) << "Brass (ms)"
                  << std::setw(22) << "vs Scalar"
                  << std::setw(22) << "vs -O3"
                  << std::setw(9)  << "Status"
                  << "\n";
        std::cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
    }

    static void print_row(const BenchmarkResult& res) {
        std::ostringstream s_ratio, s_vec;
        if (res.ratio > 0.0) {
            s_ratio << std::fixed << std::setprecision(2) << res.ratio << "x";
            if (res.ratio_min > 0.0 && res.ratio_max > 0.0) {
                s_ratio << " [" << res.ratio_min << "-" << res.ratio_max << "x]";
            }
        } else {
            s_ratio << "-";
        }

        if (res.ratio_vec > 0.0) {
            s_vec << std::fixed << std::setprecision(2) << res.ratio_vec << "x";
            if (res.ratio_vec_min > 0.0 && res.ratio_vec_max > 0.0) {
                s_vec << " [" << res.ratio_vec_min << "-" << res.ratio_vec_max << "x]";
            }
        } else {
            s_vec << "-";
        }

        // Status is strictly derived from exact ratio vs target ratio in Release, informational in Debug
        bool pass = (res.target_ratio > 0.0) ? (res.ratio <= res.target_ratio) : res.passes_bar;
        std::string status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");

        std::cout << std::left
                  << std::setw(34) << res.name
                  << std::setw(11) << res.iterations
                  << std::fixed << std::setprecision(2);
        if (res.native_ms > 0.0) {
            std::cout << std::setw(12) << res.native_ms;
        } else {
            std::cout << std::setw(12) << "-";
        }

        if (res.native_scalar_ms > 0.0) {
            std::cout << std::setw(12) << res.native_scalar_ms;
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << std::setw(12) << res.brass_ms
                  << std::setw(22) << s_ratio.str()
                  << std::setw(22) << s_vec.str()
                  << std::setw(9)  << status_str;

        if (!res.notes.empty()) {
            std::cout << " (" << res.notes << ")";
        }
        std::cout << "\n";
    }

    static void print_ratchet_summary(const std::vector<BenchmarkResult>& results, const RatchetManager& rm, double regression_factor = 1.10) {
        std::cout << "\n==================================================================================================================================\n";
        std::cout << "  PERFORMANCE RATCHET VERIFICATION (Regression Margin: " << static_cast<int>((regression_factor - 1.0) * 100.0 + 0.5) << "%)\n";
        std::cout << "==================================================================================================================================\n";
        std::cout << std::left
                  << std::setw(28) << "Key"
                  << std::setw(34) << "Benchmark"
                  << std::setw(16) << "Golden Ratchet"
                  << std::setw(16) << "Max Allowed"
                  << std::setw(26) << "Measured Ratio"
                  << std::setw(10) << "Status"
                  << "\n";
        std::cout << "----------------------------------------------------------------------------------------------------------------------------------\n";

        for (const auto& res : results) {
            if (res.key.empty()) continue;
            std::ostringstream s_golden, s_max, s_measured;
            std::string status_str;

            if (res.key == "gc_model_speedup") {
                double golden = rm.get_ratio(res.key, 1.25);
                double min_allowed = golden * (2.0 - regression_factor);
                bool pass = (res.ratio >= min_allowed);
                status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");
                s_golden << std::fixed << std::setprecision(2) << golden << "x (min)";
                s_max << std::fixed << std::setprecision(2) << min_allowed << "x (min)";
                s_measured << std::fixed << std::setprecision(2) << res.ratio << "x";
                if (res.ratio_min > 0.0 && res.ratio_max > 0.0) {
                    s_measured << " [" << res.ratio_min << "-" << res.ratio_max << "x]";
                }
            } else if (res.key == "compile_speed") {
                double golden = rm.get_ratio(res.key, 1000.00);
                double max_allowed = golden * regression_factor;
                bool pass = (res.ratio <= max_allowed);
                status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");
                s_golden << std::fixed << std::setprecision(2) << golden << " ms";
                s_max << std::fixed << std::setprecision(2) << max_allowed << " ms";
                s_measured << std::fixed << std::setprecision(2) << res.ratio << " ms";
                if (res.ratio_min > 0.0 && res.ratio_max > 0.0) {
                    s_measured << " [" << res.ratio_min << "-" << res.ratio_max << " ms]";
                }
            } else {
                double golden = rm.get_ratio(res.key, 1.30);
                double max_allowed = golden * regression_factor;
                bool pass = (res.ratio <= max_allowed);
                status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");
                s_golden << std::fixed << std::setprecision(2) << golden << "x";
                s_max << std::fixed << std::setprecision(2) << max_allowed << "x";
                s_measured << std::fixed << std::setprecision(2) << res.ratio << "x";
                if (res.ratio_min > 0.0 && res.ratio_max > 0.0) {
                    s_measured << " [" << res.ratio_min << "-" << res.ratio_max << "x]";
                }
            }

            std::cout << std::left
                      << std::setw(28) << res.key
                      << std::setw(34) << res.name
                      << std::setw(16) << s_golden.str()
                      << std::setw(16) << s_max.str()
                      << std::setw(26) << s_measured.str()
                      << std::setw(10) << status_str
                      << "\n";
        }
        std::cout << "==================================================================================================================================\n\n";
    }

    static void print_gc_comparison(
        const TimingStats& shadow_stats,
        const TimingStats& brass_stats,
        const TimingStats& speedup_stats,
        bool passed
    ) {
        std::string status_str;
        if (is_debug_build()) {
            status_str = "[INFO] Debug Build (>= 1.25x bar informational)";
        } else {
            status_str = passed ? "[PASS] Verified >= 1.25x Speedup" : "[FAIL] Below 1.25x Bar";
        }
        std::cout << "\n----------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "  GC MODEL COMPARISON (Live GC references across subroutine calls, " << speedup_stats.samples.size() << " repetitions):\n";
        std::cout << "  - (a) Shadow-Stack Model:      " << std::fixed << std::setprecision(2) << shadow_stats.median << " ms ["
                  << shadow_stats.min << "-" << shadow_stats.max << " ms]\n";
        std::cout << "  - (b) Brass Stack-Map Model:   " << std::fixed << std::setprecision(2) << brass_stats.median << " ms ["
                  << brass_stats.min << "-" << brass_stats.max << " ms]\n";
        std::cout << "  - Speedup Ratio:               " << std::fixed << std::setprecision(2) << speedup_stats.median << "x ["
                  << speedup_stats.min << "-" << speedup_stats.max << "x] faster (Required: >= 1.25x)\n";
        std::cout << "  - Status:                      " << status_str << "\n";
        std::cout << "==================================================================================================================================\n\n";
    }

    static void print_compile_speed(
        size_t function_count,
        size_t instruction_count,
        size_t mir_bytes,
        size_t machine_bytes,
        size_t peak_rss_bytes,
        const TimingStats& parse_stats,
        const TimingStats& verify_stats,
        const TimingStats& codegen_stats,
        const TimingStats& total_stats,
        bool deterministic,
        bool passed
    ) {
        double total_ms = total_stats.median;
        double fn_per_sec = (total_ms > 0.0) ? (static_cast<double>(function_count) / (total_ms / 1000.0)) : 0.0;
        double kb_per_sec = (total_ms > 0.0) ? (static_cast<double>(mir_bytes) / 1024.0 / (total_ms / 1000.0)) : 0.0;

        std::string status_str;
        if (is_debug_build()) {
            status_str = passed ? "[INFO] Sub-2s Target Met (Debug)" : "[INFO] Exceeded 2s Limit (Informational in Debug)";
        } else {
            status_str = passed ? "[PASS] Sub-2s Target Met" : "[FAIL] Exceeded 2s Limit";
        }

        std::cout << "\n==================================================================================================================================\n";
        std::cout << "  BRASS COMPILE-SPEED BENCHMARK (" << total_stats.samples.size() << " Repetitions):\n";
        std::cout << "==================================================================================================================================\n";
        std::cout << "  - Functions Compiled:          " << function_count << "\n";
        std::cout << "  - MIR Instructions:            " << instruction_count << "\n";
        std::cout << "  - Textual MIR Source Size:     " << std::fixed << std::setprecision(1) << (static_cast<double>(mir_bytes) / 1024.0) << " KB ("
                  << std::fixed << std::setprecision(2) << (static_cast<double>(mir_bytes) / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "  - Emitted Machine Code Size:   " << std::fixed << std::setprecision(1) << (static_cast<double>(machine_bytes) / 1024.0) << " KB\n";
        if (peak_rss_bytes > 0) {
            std::cout << "  - Peak Working Set (Memory):   " << std::fixed << std::setprecision(2) << (static_cast<double>(peak_rss_bytes) / 1024.0 / 1024.0) << " MB\n";
        }
        std::cout << "  - Breakdown (Median [min-max]):\n";
        std::cout << "      * MIR Parse:               " << std::fixed << std::setprecision(2) << parse_stats.median << " ms ["
                  << parse_stats.min << "-" << parse_stats.max << " ms]\n";
        std::cout << "      * MIR Verification:        " << std::fixed << std::setprecision(2) << verify_stats.median << " ms ["
                  << verify_stats.min << "-" << verify_stats.max << " ms]\n";
        std::cout << "      * ISEL, RegAlloc, Codegen: " << std::fixed << std::setprecision(2) << codegen_stats.median << " ms ["
                  << codegen_stats.min << "-" << codegen_stats.max << " ms]\n";
        std::cout << "  - Total End-to-End Time:       " << std::fixed << std::setprecision(2) << total_stats.median << " ms ["
                  << total_stats.min << "-" << total_stats.max << " ms] (Target: < 2000.0 ms)\n";
        std::cout << "  - Determinism Verification:    " << (deterministic ? "[PASS] Verified 100% Byte-for-Byte Deterministic" : "[FAIL] Determinism mismatch") << "\n";
        std::cout << "  - Throughput:                  " << std::fixed << std::setprecision(0) << fn_per_sec << " functions/sec | "
                  << std::fixed << std::setprecision(1) << kb_per_sec << " KB/sec\n";
        std::cout << "  - Status:                      " << status_str << "\n";
        std::cout << "==================================================================================================================================\n\n";
    }
};

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
