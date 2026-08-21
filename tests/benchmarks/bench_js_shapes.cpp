#include "bench_js_shapes.hpp"
#include "bench_js_shapes_modules.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

extern "C" {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t target_stub_1(int64_t x) {
    DoNotOptimize(x);
    int64_t res = (x + 3) & 0x7FFFFFFF;
    DoNotOptimize(res);
    return res;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t target_stub_2(int64_t x) {
    DoNotOptimize(x);
    int64_t res = ((x ^ 0x5555) + 1) & 0x7FFFFFFF;
    DoNotOptimize(res);
    return res;
}

} // extern "C"

namespace {

// ============================================================================
// 1. NaN-Box Tag-Test Loop Baseline
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
double native_nanbox_tag_test(const uint64_t* data, int64_t len) {
    double acc = 0.0;
    for (int64_t i = 0; i < len; ++i) {
        uint64_t v = data[i];
        if (v < 0xFFF8000000000000ULL) {
            double d;
            std::memcpy(&d, &v, sizeof(double));
            acc += d;
        } else if ((v >> 32) == 0xFFF90000U) {
            int32_t iv = static_cast<int32_t>(v & 0xFFFFFFFFU);
            acc += static_cast<double>(iv);
        } else {
            acc += 1.0;
        }
    }
    return acc;
}

// ============================================================================
// 2. Shape-Guarded Field Load Baseline
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t native_shape_guard(const uintptr_t* objs, int64_t len) {
    int64_t sum = 0;
    for (int64_t i = 0; i < len; ++i) {
        const uint64_t* obj = reinterpret_cast<const uint64_t*>(objs[i]);
        uint64_t shape = obj[0];
        if (shape == 0xAA01) {
            sum += obj[1];
        } else if (shape == 0xBB02) {
            sum += obj[2];
        } else {
            sum += obj[3] * 2;
        }
    }
    return sum;
}

// ============================================================================
// 3. Patchable Call Baseline (Phase 1 -> Patch -> Phase 2)
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t native_patchable_ic_loop(int64_t iters_phase1, int64_t iters_phase2) {
    int64_t acc = 0;
    for (int64_t i = 0; i < iters_phase1; ++i) {
        int64_t sub = target_stub_1(acc);
        acc += sub;
    }
    for (int64_t i = 0; i < iters_phase2; ++i) {
        int64_t sub = target_stub_2(acc);
        acc += sub;
    }
    return acc;
}

// ============================================================================
// 4. GC Allocation Loop Shadow-Stack Model
// ============================================================================
struct ShadowLinkedNode {
    int64_t val;
    ShadowLinkedNode* next;
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t shadow_stack_alloc_loop(int64_t num_nodes, ThreadShadowStack& ss) {
    ShadowStackFrame frame;
    frame.roots[0] = nullptr;
    ss.push(&frame, 1);

    auto* gc = brass::brass_get_active_gc();
    std::vector<uintptr_t*> roots(1);

    for (int64_t i = 0; i < num_nodes; ++i) {
        roots[0] = reinterpret_cast<uintptr_t*>(&frame.roots[0]);
        ClobberMemory();
        uintptr_t node = gc->allocate(16, 2, 1, roots);
        int64_t* val_ptr = reinterpret_cast<int64_t*>(node);
        *val_ptr = i + 1;
        uintptr_t* next_ptr = reinterpret_cast<uintptr_t*>(node + 8);
        *next_ptr = reinterpret_cast<uintptr_t>(frame.roots[0]);
        frame.roots[0] = reinterpret_cast<void*>(node);
        DoNotOptimize(frame.roots[0]);
        ClobberMemory();
    }

    int64_t sum = 0;
    uintptr_t cur = reinterpret_cast<uintptr_t>(frame.roots[0]);
    while (cur != 0) {
        sum += *reinterpret_cast<int64_t*>(cur);
        cur = *reinterpret_cast<uintptr_t*>(cur + 8);
    }

    ss.pop();
    return sum;
}

} // namespace

namespace brass::bench {

void run_js_shapes_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet) {
    Stopwatch sw;

    // ------------------------------------------------------------------------
    // (a) NaN-box tag-test loops
    // ------------------------------------------------------------------------
    {
        constexpr size_t N = 50000;
        constexpr size_t iters = 200;
        std::vector<uint64_t> nanbox_data(N);

        // Populate dataset: 70% doubles, 20% boxed int32s, 10% fallbacks
        for (size_t i = 0; i < N; ++i) {
            if (i % 10 < 7) {
                double d = static_cast<double>(i % 100) + 0.5;
                uint64_t raw;
                std::memcpy(&raw, &d, sizeof(double));
                nanbox_data[i] = raw;
            } else if (i % 10 < 9) {
                int32_t iv = static_cast<int32_t>((i % 50) + 1);
                uint64_t raw = (0xFFF90000ULL << 32) | static_cast<uint32_t>(iv);
                nanbox_data[i] = raw;
            } else {
                nanbox_data[i] = 0xFFFA000000000000ULL; // Other tag
            }
        }

        // Native baseline
        sw.start();
        double native_res = 0.0;
        for (size_t k = 0; k < iters; ++k) {
            native_res = native_nanbox_tag_test(nanbox_data.data(), static_cast<int64_t>(N));
            DoNotOptimize(native_res);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_nanbox_tag_test_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto fn = jit.get_function_ptr<double(*)(const uint64_t*, int64_t)>("nanbox_tag_test");
        if (!fn) {
            std::cerr << "FATAL: nanbox_tag_test function pointer is null!\n";
            std::abort();
        }

        sw.start();
        double jit_res = 0.0;
        for (size_t k = 0; k < iters; ++k) {
            jit_res = fn(nanbox_data.data(), static_cast<int64_t>(N));
            DoNotOptimize(jit_res);
        }
        double jit_ms = sw.stop_ms();

        if (std::abs(native_res - jit_res) > 1e-4) {
            std::cerr << "FATAL: NaN-Box tag-test mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        double ratio = (native_ms > 0.0) ? (jit_ms / native_ms) : 1.0;
        double target = ratchet.get_ratio("nanbox", 1.15);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        BenchmarkResult r{"nanbox", "NaN-Box Tag-Test Loop", iters, native_ms, -1.0, jit_ms, ratio, -1.0, target, ratio <= target, notes.str()};
        BenchmarkReporter::print_row(r);
        results.push_back(r);
    }

    // ------------------------------------------------------------------------
    // (b) Shape-guarded field loads
    // ------------------------------------------------------------------------
    {
        constexpr size_t N = 20000;
        constexpr size_t iters = 200;

        struct TestObj {
            uint64_t shape;
            int64_t slot0;
            int64_t slot1;
            int64_t slot2;
        };

        std::vector<TestObj> storage(N);
        std::vector<uintptr_t> obj_ptrs(N);

        for (size_t i = 0; i < N; ++i) {
            if (i % 20 < 16) {
                storage[i] = {0xAA01, static_cast<int64_t>(i + 1), 0, 0};
            } else if (i % 20 < 19) {
                storage[i] = {0xBB02, 0, static_cast<int64_t>((i + 1) * 3), 0};
            } else {
                storage[i] = {0xCC03, 0, 0, static_cast<int64_t>(i + 1)};
            }
            obj_ptrs[i] = reinterpret_cast<uintptr_t>(&storage[i]);
        }

        // Native baseline
        sw.start();
        int64_t native_res = 0;
        for (size_t k = 0; k < iters; ++k) {
            native_res = native_shape_guard(obj_ptrs.data(), static_cast<int64_t>(N));
            DoNotOptimize(native_res);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_shape_guard_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto fn = jit.get_function_ptr<int64_t(*)(const uintptr_t*, int64_t)>("shape_guard");
        if (!fn) {
            std::cerr << "FATAL: shape_guard function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_res = 0;
        for (size_t k = 0; k < iters; ++k) {
            jit_res = fn(obj_ptrs.data(), static_cast<int64_t>(N));
            DoNotOptimize(jit_res);
        }
        double jit_ms = sw.stop_ms();

        if (native_res != jit_res) {
            std::cerr << "FATAL: Shape guard result mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        double ratio = (native_ms > 0.0) ? (jit_ms / native_ms) : 1.0;
        double target = ratchet.get_ratio("shapes", 1.10);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        BenchmarkResult r{"shapes", "Shape-Guarded Field Loads", iters, native_ms, -1.0, jit_ms, ratio, -1.0, target, ratio <= target, notes.str()};
        BenchmarkReporter::print_row(r);
        results.push_back(r);
    }

    // ------------------------------------------------------------------------
    // (c) Patchable-call inline-cache loop
    // ------------------------------------------------------------------------
    {
        constexpr int64_t phase1_iters = 25000;
        constexpr int64_t phase2_iters = 25000;
        constexpr size_t outer_rounds = 50;

        // Native baseline
        sw.start();
        int64_t native_res = 0;
        for (size_t r = 0; r < outer_rounds; ++r) {
            native_res = native_patchable_ic_loop(phase1_iters, phase2_iters);
            DoNotOptimize(native_res);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_patchable_ic_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto fn = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("patchable_ic_runner");
        if (!fn) {
            std::cerr << "FATAL: patchable_ic_runner function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_res = 0;
        for (size_t r = 0; r < outer_rounds; ++r) {
            // Reset to target 1
            jit.patch_call("ic_site_bench", "target_stub_1");
            int64_t mid = fn(0, phase1_iters);
            // Patch to target 2
            jit.patch_call("ic_site_bench", "target_stub_2");
            jit_res = fn(mid, phase2_iters);
            DoNotOptimize(jit_res);
        }
        double jit_ms = sw.stop_ms();

        if (native_res != jit_res) {
            std::cerr << "FATAL: Patchable IC result mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        double ratio = (native_ms > 0.0) ? (jit_ms / native_ms) : 1.0;
        double target = ratchet.get_ratio("icache", 0.65);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        BenchmarkResult r{"icache", "Patchable-Call Inline Cache", outer_rounds, native_ms, -1.0, jit_ms, ratio, -1.0, target, ratio <= target, notes.str()};
        BenchmarkReporter::print_row(r);
        results.push_back(r);
    }

    // ------------------------------------------------------------------------
    // (d) Allocation loop building small linked objects under mini-Cheney GC
    // ------------------------------------------------------------------------
    {
        constexpr int64_t num_nodes = 20000;
        constexpr size_t gc_rounds = 20;

        // (a) Shadow-Stack Model
        ThreadShadowStack ss;
        MiniCheneyGC ss_gc(4 * 1024 * 1024);
        brass_set_active_gc(&ss_gc);
        sw.start();
        int64_t shadow_res = 0;
        for (size_t r = 0; r < gc_rounds; ++r) {
            ss_gc.reset();
            shadow_res = shadow_stack_alloc_loop(num_nodes, ss);
            DoNotOptimize(shadow_res);
        }
        double shadow_ms = sw.stop_ms();

        // (b) Brass Stack-Map Model
        MiniCheneyGC gc(4 * 1024 * 1024); // 4 MB semi-space
        brass_set_active_gc(&gc);

        auto mod = build_gc_alloc_loop_module();
        JitExecutionEngine jit;
        jit.register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
        jit.register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
        jit.compile_and_load(*mod);
        brass_set_active_stack_maps(&jit.stack_maps());

        auto fn = jit.get_function_ptr<int64_t(*)(int64_t)>("gc_alloc_runner");
        if (!fn) {
            std::cerr << "FATAL: gc_alloc_runner function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_res = 0;
        for (size_t r = 0; r < gc_rounds; ++r) {
            gc.reset();
            jit_res = fn(num_nodes);
            DoNotOptimize(jit_res);
        }
        double jit_ms = sw.stop_ms();

        int64_t expected_sum = num_nodes * (num_nodes + 1) / 2;
        if (jit_res != expected_sum || shadow_res != expected_sum) {
            std::cerr << "FATAL: GC Linked Node Alloc result mismatch: expected=" << expected_sum
                      << ", shadow=" << shadow_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        double ratio = (shadow_ms > 0.0) ? (jit_ms / shadow_ms) : 1.0;
        double target = ratchet.get_ratio("cheney_gc", 1.10);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        BenchmarkResult r{"cheney_gc", "Linked Node Alloc (Cheney GC)", gc_rounds, shadow_ms, -1.0, jit_ms, ratio, -1.0, target, ratio <= target, notes.str()};
        BenchmarkReporter::print_row(r);
        results.push_back(r);

        brass_set_active_gc(nullptr);
        brass_set_active_stack_maps(nullptr);
    }
}

} // namespace brass::bench
