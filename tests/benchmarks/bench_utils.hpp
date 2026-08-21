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

struct BenchmarkResult {
    std::string key; // Benchmark identifier, e.g. "fib", "matmul_i64_32_naive"
    std::string name;
    size_t iterations = 0;
    double native_ms = 0.0;
    double native_scalar_ms = 0.0;
    double brass_ms = 0.0;
    double ratio = 0.0; // vs scalar or vs native
    double ratio_vec = 0.0; // vs vectorized native
    double target_ratio = 0.0; // Golden ratio from ratchet
    bool passes_bar = true;
    std::string notes;
};

// ============================================================================
// Performance Ratchet Manager
// ============================================================================

class RatchetManager {
public:
    static RatchetManager defaults() {
        RatchetManager rm;
        rm.ratios_ = {
            {"cheney_gc", 1.35},
            {"collatz", 1.05},
            {"compile_speed", 1000.00},
            {"fib", 1.25},
            {"icache", 0.75},
            {"linked_list", 0.90},
            {"matmul_f64_32_naive", 1.00},
            {"matmul_f64_32_preopt", 1.00},
            {"matmul_f64_64_naive", 0.85},
            {"matmul_f64_64_preopt", 0.85},
            {"matmul_i64_32_naive", 1.50},
            {"matmul_i64_32_preopt", 1.50},
            {"matmul_i64_64_naive", 1.45},
            {"matmul_i64_64_preopt", 1.45},
            {"nanbox", 1.15},
            {"shapes", 1.10},
            {"sieve", 1.25}
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
        return all_passed;
    }

    void update_from_results(const std::vector<BenchmarkResult>& results, bool only_if_improved = false) {
        for (const auto& res : results) {
            if (res.key.empty() || res.ratio <= 0.0) continue;
            auto it = ratios_.find(res.key);
            if (it == ratios_.end()) {
                ratios_[res.key] = res.ratio;
            } else if (!only_if_improved || res.ratio < it->second) {
                it->second = res.ratio;
            }
        }
    }

private:
    std::map<std::string, double> ratios_;
};

// ============================================================================
// Benchmark Reporter with Run Provenance & Honest Status Derivation
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

        std::cout << "\n========================================================================================================================\n";
        std::cout << "  BRASS PERFORMANCE BENCHMARK SUITE: " << title << "\n";
        std::cout << "========================================================================================================================\n";
        std::cout << "  Run Provenance:\n";
        std::cout << "  - Timestamp:    " << time_str << "\n";
        std::cout << "  - Git Commit:   " << git_sha << "\n";
        std::cout << "  - Build Type:   " << build_type << "\n";
        std::cout << "  - Compiler:     " << compiler_info << "\n";
        std::cout << "========================================================================================================================\n";
        std::cout << std::left
                  << std::setw(34) << "Benchmark"
                  << std::setw(11) << "Iterations"
                  << std::setw(14) << "Native -O3"
                  << std::setw(14) << "Scalar -O3"
                  << std::setw(13) << "Brass (ms)"
                  << std::setw(11) << "vs Scalar"
                  << std::setw(10) << "vs -O3"
                  << std::setw(9)  << "Status"
                  << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    }

    static void print_row(const BenchmarkResult& res) {
        std::ostringstream s_ratio, s_vec;
        s_ratio << std::fixed << std::setprecision(2) << res.ratio << "x";
        if (res.ratio_vec > 0.0) {
            s_vec << std::fixed << std::setprecision(2) << res.ratio_vec << "x";
        } else {
            s_vec << "-";
        }

        // Status is strictly derived from exact ratio vs target ratio in Release, informational in Debug
        bool pass = (res.target_ratio > 0.0) ? (res.ratio <= res.target_ratio) : res.passes_bar;
        std::string status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");

        std::cout << std::left
                  << std::setw(34) << res.name
                  << std::setw(11) << res.iterations
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << res.native_ms;
        if (res.native_scalar_ms > 0.0) {
            std::cout << std::setw(14) << res.native_scalar_ms;
        } else {
            std::cout << std::setw(14) << "-";
        }
        std::cout << std::setw(13) << res.brass_ms
                  << std::setw(11) << s_ratio.str()
                  << std::setw(10) << s_vec.str()
                  << std::setw(9)  << status_str;
        if (!res.notes.empty()) {
            std::cout << " (" << res.notes << ")";
        }
        std::cout << "\n";
    }

    static void print_ratchet_summary(const std::vector<BenchmarkResult>& results, const RatchetManager& rm, double regression_factor = 1.10) {
        std::cout << "\n========================================================================================================================\n";
        std::cout << "  PERFORMANCE RATCHET VERIFICATION (Regression Margin: " << std::fixed << std::setprecision(0) << ((regression_factor - 1.0) * 100.0) << "%)\n";
        std::cout << "========================================================================================================================\n";
        std::cout << std::left
                  << std::setw(24) << "Key"
                  << std::setw(34) << "Benchmark"
                  << std::setw(16) << "Golden Ratchet"
                  << std::setw(16) << "Max Allowed"
                  << std::setw(16) << "Measured Ratio"
                  << std::setw(10) << "Status"
                  << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        for (const auto& res : results) {
            if (res.key.empty()) continue;
            double golden = rm.get_ratio(res.key, 1.30);
            double max_allowed = golden * regression_factor;
            bool pass = (res.ratio <= max_allowed);
            std::string status_str = is_debug_build() ? "[INFO]" : (pass ? "[PASS]" : "[FAIL]");

            std::ostringstream s_golden, s_max, s_measured;
            if (res.key == "compile_speed") {
                s_golden << std::fixed << std::setprecision(2) << golden << " ms";
                s_max << std::fixed << std::setprecision(2) << max_allowed << " ms";
                s_measured << std::fixed << std::setprecision(2) << res.ratio << " ms";
            } else {
                s_golden << std::fixed << std::setprecision(2) << golden << "x";
                s_max << std::fixed << std::setprecision(2) << max_allowed << "x";
                s_measured << std::fixed << std::setprecision(2) << res.ratio << "x";
            }

            std::cout << std::left
                      << std::setw(24) << res.key
                      << std::setw(34) << res.name
                      << std::setw(16) << s_golden.str()
                      << std::setw(16) << s_max.str()
                      << std::setw(16) << s_measured.str()
                      << std::setw(10) << status_str
                      << "\n";
        }
        std::cout << "========================================================================================================================\n\n";
    }

    static void print_gc_comparison(double shadow_stack_ms, double brass_stack_map_ms, double speedup, bool passed) {
        std::string status_str;
        if (is_debug_build()) {
            status_str = "[INFO] Debug Build (>= 1.5x bar informational)";
        } else {
            status_str = passed ? "[PASS] Verified >= 1.5x Speedup" : "[FAIL] Below 1.5x Bar";
        }
        std::cout << "\n------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "  GC MODEL COMPARISON (Live GC references across subroutine calls):\n";
        std::cout << "  - (a) Shadow-Stack Model:      " << std::fixed << std::setprecision(2) << shadow_stack_ms << " ms\n";
        std::cout << "  - (b) Brass Stack-Map Model:   " << std::fixed << std::setprecision(2) << brass_stack_map_ms << " ms\n";
        std::cout << "  - Speedup Ratio:               " << std::fixed << std::setprecision(2) << speedup << "x faster (Required: >= 1.5x)\n";
        std::cout << "  - Status:                      " << status_str << "\n";
        std::cout << "========================================================================================================================\n\n";
    }

    static void print_compile_speed(
        size_t function_count,
        size_t instruction_count,
        size_t mir_bytes,
        size_t machine_bytes,
        size_t peak_rss_bytes,
        double parse_ms,
        double verify_ms,
        double codegen_ms,
        double total_ms,
        bool deterministic,
        bool passed
    ) {
        double fn_per_sec = (total_ms > 0.0) ? (static_cast<double>(function_count) / (total_ms / 1000.0)) : 0.0;
        double kb_per_sec = (total_ms > 0.0) ? (static_cast<double>(mir_bytes) / 1024.0 / (total_ms / 1000.0)) : 0.0;

        std::string status_str;
        if (is_debug_build()) {
            status_str = passed ? "[INFO] Sub-2s Target Met (Debug)" : "[INFO] Exceeded 2s Limit (Informational in Debug)";
        } else {
            status_str = passed ? "[PASS] Sub-2s Target Met" : "[FAIL] Exceeded 2s Limit";
        }

        std::cout << "\n========================================================================================================================\n";
        std::cout << "  BRASS COMPILE-SPEED BENCHMARK:\n";
        std::cout << "========================================================================================================================\n";
        std::cout << "  - Functions Compiled:          " << function_count << "\n";
        std::cout << "  - MIR Instructions:            " << instruction_count << "\n";
        std::cout << "  - Textual MIR Source Size:     " << std::fixed << std::setprecision(1) << (static_cast<double>(mir_bytes) / 1024.0) << " KB ("
                  << std::fixed << std::setprecision(2) << (static_cast<double>(mir_bytes) / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "  - Emitted Machine Code Size:   " << std::fixed << std::setprecision(1) << (static_cast<double>(machine_bytes) / 1024.0) << " KB\n";
        if (peak_rss_bytes > 0) {
            std::cout << "  - Peak Working Set (Memory):   " << std::fixed << std::setprecision(2) << (static_cast<double>(peak_rss_bytes) / 1024.0 / 1024.0) << " MB\n";
        }
        std::cout << "  - Breakdown:\n";
        std::cout << "      * MIR Parse:               " << std::fixed << std::setprecision(2) << parse_ms << " ms\n";
        std::cout << "      * MIR Verification:        " << std::fixed << std::setprecision(2) << verify_ms << " ms\n";
        std::cout << "      * ISEL, RegAlloc, Codegen: " << std::fixed << std::setprecision(2) << codegen_ms << " ms\n";
        std::cout << "  - Total End-to-End Time:       " << std::fixed << std::setprecision(2) << total_ms << " ms (Target: < 2000.0 ms)\n";
        std::cout << "  - Determinism Verification:    " << (deterministic ? "[PASS] Verified 100% Byte-for-Byte Deterministic" : "[FAIL] Determinism mismatch") << "\n";
        std::cout << "  - Throughput:                  " << std::fixed << std::setprecision(0) << fn_per_sec << " functions/sec | "
                  << std::fixed << std::setprecision(1) << kb_per_sec << " KB/sec\n";
        std::cout << "  - Status:                      " << status_str << "\n";
        std::cout << "========================================================================================================================\n\n";
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
