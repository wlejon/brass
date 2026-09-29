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
#include <cstring>

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
// Fixed placement for native baselines
// ============================================================================

// A native baseline's loop runs at whatever alignment the linker gives it,
// and an edit to any code linked before it moves that: a relink that changed
// only a header shifted matmul_i64_32 by 14-16%. The baselines are therefore
// out-of-line and, under MSVC, in their own grouped section, which the
// linker sorts ahead of ordinary code (.text$mn) at the page-aligned start of
// .text, so their addresses depend only on the baselines themselves. GCC and
// Clang get 64-byte function alignment instead. (The JIT side needs none of
// this: it is measured at several code placements and the median taken.)
#if defined(_MSC_VER)
#define BRASS_BENCH_SECTION __declspec(code_seg(".text$bnat"))
#define BRASS_BENCH_NATIVE __declspec(noinline) BRASS_BENCH_SECTION
#else
#define BRASS_BENCH_SECTION __attribute__((aligned(64)))
#define BRASS_BENCH_NATIVE __attribute__((noinline)) BRASS_BENCH_SECTION
#endif

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

// One benchmark's outcome. `ratio` is the number the ratchet gates: a median
// over repetitions, lower-is-better unless the key's RatchetPolicy says
// otherwise (see bench_ratchet.hpp).
struct BenchmarkResult {
    std::string key; // Benchmark identifier, e.g. "fib", "matmul_i64_32_naive"
    std::string name;
    size_t iterations = 0;
    size_t repetitions = 0;
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
    std::string notes;
};

inline size_t default_bench_placements() noexcept {
    return is_debug_build() ? 1 : 5;
}

// Repetitions per benchmark. Each repetition is itself the median over the
// code placements, so the reported ratio is a median of medians; 7 keeps a
// single slow repetition (a context switch, a frequency dip) from moving it.
inline size_t default_bench_repetitions() noexcept {
    return is_debug_build() ? 1 : 7;
}

constexpr size_t DEFAULT_BENCH_REPETITIONS = 7;

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

inline BenchmarkResult make_paired_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const PairedRepetitionResult& paired,
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
    r.brass_ms = paired.brass_stats.median;
    r.brass_min_ms = paired.brass_stats.min;
    r.brass_max_ms = paired.brass_stats.max;
    r.ratio = paired.ratio_stats.median;
    r.ratio_min = paired.ratio_stats.min;
    r.ratio_max = paired.ratio_stats.max;
    r.notes = notes;
    return r;
}

inline BenchmarkResult make_triplet_result(
    const std::string& key,
    const std::string& name,
    size_t iterations,
    const TripletRepetitionResult& triplet,
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
    if (!notes.empty()) {
        r.notes = notes;
    } else {
        std::ostringstream oss;
        oss << "vec: " << std::fixed << std::setprecision(2) << r.ratio_vec << "x";
        r.notes = oss.str();
    }
    return r;
}

// ============================================================================
// Deterministic data placement
// ============================================================================

// Benchmark buffers whose relative placement is the same in every process.
// A plain std::vector lands wherever the heap puts it, so two runs of the
// same binary can differ in how the matrices of one benchmark alias in L1
// (4K aliasing): a whole run then reads ~13% slower on one key. Each
// allocation here starts at a page boundary plus an offset that steps by a
// cache line per allocation, in allocation order, which the program fixes.
inline void* placed_alloc(size_t bytes) {
    static unsigned counter = 0;
    const size_t shift = 64 * (1 + counter++ % 15);
    void* raw = ::operator new(bytes + 4096 + shift);
    const uintptr_t page = (reinterpret_cast<uintptr_t>(raw) + 4095) & ~uintptr_t{4095};
    auto* p = reinterpret_cast<unsigned char*>(page + shift);
    std::memcpy(p - sizeof(void*), &raw, sizeof(void*));
    return p;
}

inline void placed_free(void* p) noexcept {
    void* raw = nullptr;
    std::memcpy(&raw, static_cast<unsigned char*>(p) - sizeof(void*), sizeof(void*));
    ::operator delete(raw);
}

template <typename T>
struct PlacedAllocator {
    using value_type = T;
    PlacedAllocator() = default;
    template <typename U>
    PlacedAllocator(const PlacedAllocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(placed_alloc(n * sizeof(T))); }
    void deallocate(T* p, size_t) noexcept { placed_free(p); }
    template <typename U>
    bool operator==(const PlacedAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const PlacedAllocator<U>&) const noexcept { return false; }
};

template <typename T>
using PlacedVector = std::vector<T, PlacedAllocator<T>>;

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
// Ratchet and reporter
// ============================================================================

#include "bench_ratchet.hpp"
#include "bench_reporter.hpp"
