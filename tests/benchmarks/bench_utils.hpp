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

namespace brass::bench {

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
};

// Shadow-stack frame definition for GC comparison
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
    }

    inline void pop() noexcept {
        if (top) top = top->prev;
    }
};

} // namespace brass::bench
