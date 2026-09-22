#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <ctime>
#include <cctype>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "bench_utils.hpp"

namespace brass::bench {

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

} // namespace brass::bench
