#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cassert>

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
    std::string name;
    size_t iterations = 0;
    double native_ms = 0.0;
    double brass_ms = 0.0;
    double ratio = 0.0; // brass / native
    bool passes_bar = true;
    std::string notes;
};

class BenchmarkReporter {
public:
    static void print_header(std::string_view title) {
        std::cout << "\n========================================================================================\n";
        std::cout << "  BRASS PERFORMANCE BENCHMARK SUITE: " << title << "\n";
        std::cout << "========================================================================================\n";
        std::cout << std::left
                  << std::setw(32) << "Benchmark"
                  << std::setw(12) << "Iterations"
                  << std::setw(14) << "Native (ms)"
                  << std::setw(14) << "Brass (ms)"
                  << std::setw(10) << "Ratio"
                  << std::setw(10) << "Status"
                  << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
    }

    static void print_row(const BenchmarkResult& res) {
        std::cout << std::left
                  << std::setw(32) << res.name
                  << std::setw(12) << res.iterations
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << res.native_ms
                  << std::setw(14) << res.brass_ms
                  << std::setprecision(2)
                  << std::setw(10) << res.ratio
                  << std::setw(10) << (res.passes_bar ? "[PASS]" : "[FAIL]");
        if (!res.notes.empty()) {
            std::cout << " (" << res.notes << ")";
        }
        std::cout << "\n";
    }

    static void print_gc_comparison(double shadow_stack_ms, double brass_stack_map_ms, double speedup, bool passed) {
        std::cout << "\n----------------------------------------------------------------------------------------\n";
        std::cout << "  GC MODEL COMPARISON (Live GC references across subroutine calls):\n";
        std::cout << "  - (a) Shadow-Stack Model:      " << std::fixed << std::setprecision(2) << shadow_stack_ms << " ms\n";
        std::cout << "  - (b) Brass Stack-Map Model:   " << std::fixed << std::setprecision(2) << brass_stack_map_ms << " ms\n";
        std::cout << "  - Speedup Ratio:               " << std::fixed << std::setprecision(2) << speedup << "x faster (Required: >= 1.5x)\n";
        std::cout << "  - Status:                      " << (passed ? "[PASS] Verified >= 1.5x Speedup" : "[FAIL] Below 1.5x Bar") << "\n";
        std::cout << "========================================================================================\n\n";
    }

    static void print_compile_speed(
        size_t function_count,
        size_t instruction_count,
        size_t mir_bytes,
        size_t machine_bytes,
        double parse_ms,
        double verify_ms,
        double codegen_ms,
        double total_ms,
        bool passed
    ) {
        double fn_per_sec = (total_ms > 0.0) ? (static_cast<double>(function_count) / (total_ms / 1000.0)) : 0.0;
        double kb_per_sec = (total_ms > 0.0) ? (static_cast<double>(mir_bytes) / 1024.0 / (total_ms / 1000.0)) : 0.0;

        std::cout << "\n========================================================================================\n";
        std::cout << "  BRASS COMPILE-SPEED BENCHMARK:\n";
        std::cout << "========================================================================================\n";
        std::cout << "  - Functions Compiled:          " << function_count << "\n";
        std::cout << "  - MIR Instructions:            " << instruction_count << "\n";
        std::cout << "  - Textual MIR Source Size:     " << std::fixed << std::setprecision(1) << (static_cast<double>(mir_bytes) / 1024.0) << " KB\n";
        std::cout << "  - Emitted Machine Code Size:   " << std::fixed << std::setprecision(1) << (static_cast<double>(machine_bytes) / 1024.0) << " KB\n";
        std::cout << "  - Breakdown:\n";
        std::cout << "      * MIR Parse:               " << std::fixed << std::setprecision(2) << parse_ms << " ms\n";
        std::cout << "      * MIR Verification:        " << std::fixed << std::setprecision(2) << verify_ms << " ms\n";
        std::cout << "      * ISEL, RegAlloc, Codegen: " << std::fixed << std::setprecision(2) << codegen_ms << " ms\n";
        std::cout << "  - Total End-to-End Time:       " << std::fixed << std::setprecision(2) << total_ms << " ms (Target: < 2000.0 ms)\n";
        std::cout << "  - Throughput:                  " << std::fixed << std::setprecision(0) << fn_per_sec << " functions/sec | "
                  << std::fixed << std::setprecision(1) << kb_per_sec << " KB/sec\n";
        std::cout << "  - Status:                      " << (passed ? "[PASS] Sub-2s Target Met" : "[FAIL] Exceeded 2s Limit") << "\n";
        std::cout << "========================================================================================\n\n";
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
