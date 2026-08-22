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
    // ------------------------------------------------------------------------
    // (a) NaN-box tag-test loops
    // ------------------------------------------------------------------------
    {
        constexpr size_t N = 100000;
        constexpr size_t iters = 100;
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

        auto run_native = [data = nanbox_data.data(), N, iters]() {
            double res = 0.0;
            for (size_t k = 0; k < iters; ++k) {
                res = native_nanbox_tag_test(data, static_cast<int64_t>(N));
                DoNotOptimize(res);
            }
            return res;
        };

        // Brass JIT
        auto mod = build_nanbox_tag_test_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto fn = jit->get_function_ptr<double(*)(const uint64_t*, int64_t)>("nanbox_tag_test");
            if (!fn) {
                std::cerr << "FATAL: nanbox_tag_test function pointer is null!\n";
                std::abort();
            }
            return [jit, fn, data = nanbox_data.data(), N, iters]() {
                double res = 0.0;
                for (size_t k = 0; k < iters; ++k) {
                    res = fn(data, static_cast<int64_t>(N));
                    DoNotOptimize(res);
                }
                return res;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        double native_res = run_native();
        auto test_jit = make_jit_runner(0);
        double jit_res = test_jit();

        if (std::abs(native_res - jit_res) > 1e-4) {
            std::cerr << "FATAL: NaN-Box tag-test mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("nanbox", "NaN-Box Tag-Test Loop", iters, paired, ratchet.get_ratio("nanbox", 1.15)));
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // (b) Shape-guarded field loads
    // ------------------------------------------------------------------------
    {
        constexpr size_t N = 100000;
        constexpr size_t iters = 100;

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

        auto run_native = [ptrs = obj_ptrs.data(), N, iters]() {
            int64_t res = 0;
            for (size_t k = 0; k < iters; ++k) {
                res = native_shape_guard(ptrs, static_cast<int64_t>(N));
                DoNotOptimize(res);
            }
            return res;
        };

        // Brass JIT
        auto mod = build_shape_guard_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto fn = jit->get_function_ptr<int64_t(*)(const uintptr_t*, int64_t)>("shape_guard");
            if (!fn) {
                std::cerr << "FATAL: shape_guard function pointer is null!\n";
                std::abort();
            }
            return [jit, fn, ptrs = obj_ptrs.data(), N, iters]() {
                int64_t res = 0;
                for (size_t k = 0; k < iters; ++k) {
                    res = fn(ptrs, static_cast<int64_t>(N));
                    DoNotOptimize(res);
                }
                return res;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        int64_t native_res = run_native();
        auto test_jit = make_jit_runner(0);
        int64_t jit_res = test_jit();

        if (native_res != jit_res) {
            std::cerr << "FATAL: Shape guard result mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("shapes", "Shape-Guarded Field Loads", iters, paired, ratchet.get_ratio("shapes", 1.20)));
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // (c) Patchable-call inline-cache loop
    // ------------------------------------------------------------------------
    {
        constexpr int64_t phase1_iters = 1000000;
        constexpr int64_t phase2_iters = 1000000;
        constexpr size_t outer_rounds = 5;

        auto run_native = [phase1_iters, phase2_iters, outer_rounds]() {
            int64_t res = 0;
            for (size_t r = 0; r < outer_rounds; ++r) {
                res = native_patchable_ic_loop(phase1_iters, phase2_iters);
                DoNotOptimize(res);
            }
            return res;
        };

        // Brass JIT
        auto mod = build_patchable_ic_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto fn = jit->get_function_ptr<int64_t(*)(int64_t, int64_t)>("patchable_ic_runner");
            if (!fn) {
                std::cerr << "FATAL: patchable_ic_runner function pointer is null!\n";
                std::abort();
            }
            return [jit, fn, phase1_iters, phase2_iters, outer_rounds]() {
                int64_t res = 0;
                for (size_t r = 0; r < outer_rounds; ++r) {
                    // Reset to target 1
                    jit->patch_call("ic_site_bench", "target_stub_1");
                    int64_t mid = fn(0, phase1_iters);
                    // Patch to target 2
                    jit->patch_call("ic_site_bench", "target_stub_2");
                    res = fn(mid, phase2_iters);
                    DoNotOptimize(res);
                }
                return res;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        int64_t native_res = run_native();
        auto test_jit = make_jit_runner(0);
        int64_t jit_res = test_jit();

        if (native_res != jit_res) {
            std::cerr << "FATAL: Patchable IC result mismatch: native=" << native_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("icache", "Patchable-Call Inline Cache", outer_rounds, paired, ratchet.get_ratio("icache", 0.80)));
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // (d) Allocation loop building small linked objects under mini-Cheney GC
    // ------------------------------------------------------------------------
    {
        constexpr int64_t num_nodes = 50000;
        constexpr size_t gc_rounds = 30;

        // (a) Shadow-Stack Model
        ThreadShadowStack ss;
        MiniCheneyGC ss_gc(8 * 1024 * 1024);

        // (b) Brass Stack-Map Model
        auto mod = build_gc_alloc_loop_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
            jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
            jit->compile_and_load(*mod, padding);
            auto fn = jit->get_function_ptr<int64_t(*)(int64_t)>("gc_alloc_runner");
            if (!fn) {
                std::cerr << "FATAL: gc_alloc_runner function pointer is null!\n";
                std::abort();
            }
            auto gc = std::make_shared<MiniCheneyGC>(8 * 1024 * 1024);
            return [jit, gc, fn, num_nodes, gc_rounds]() {
                brass_set_active_gc(gc.get());
                brass_set_active_stack_maps(&jit->stack_maps());
                int64_t res = 0;
                for (size_t r = 0; r < gc_rounds; ++r) {
                    gc->reset();
                    res = fn(num_nodes);
                    DoNotOptimize(res);
                }
                brass_set_active_gc(nullptr);
                brass_set_active_stack_maps(nullptr);
                return res;
            };
        };

        auto run_shadow = [&ss, &ss_gc, num_nodes, gc_rounds]() {
            brass_set_active_gc(&ss_gc);
            brass_set_active_stack_maps(nullptr);
            int64_t res = 0;
            for (size_t r = 0; r < gc_rounds; ++r) {
                ss_gc.reset();
                res = shadow_stack_alloc_loop(num_nodes, ss);
                DoNotOptimize(res);
            }
            return res;
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_shadow, make_jit_runner);

        int64_t shadow_res = run_shadow();
        auto test_jit = make_jit_runner(0);
        int64_t jit_res = test_jit();
        int64_t expected_sum = num_nodes * (num_nodes + 1) / 2;
        if (jit_res != expected_sum || shadow_res != expected_sum) {
            std::cerr << "FATAL: GC Linked Node Alloc result mismatch: expected=" << expected_sum
                      << ", shadow=" << shadow_res << ", jit=" << jit_res << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("cheney_gc", "Linked Node Alloc (Cheney GC)", gc_rounds, paired, ratchet.get_ratio("cheney_gc", 1.35)));
        BenchmarkReporter::print_row(results.back());

        brass_set_active_gc(nullptr);
        brass_set_active_stack_maps(nullptr);
    }
}

} // namespace brass::bench
