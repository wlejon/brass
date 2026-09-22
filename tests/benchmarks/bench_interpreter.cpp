#include "bench_interpreter.hpp"
#include "bench_numeric_modules.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

// ============================================================================
// 1. Native C++ Baseline Implementations
// ============================================================================

uint64_t native_fib_iter(uint64_t n) {
    if (n < 2) return n;
    uint64_t a = 0, b = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        uint64_t c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int64_t native_collatz_sum(int64_t max_n) {
    int64_t total_steps = 0;
    for (int64_t i = 1; i <= max_n; ++i) {
        int64_t n = i;
        int64_t steps = 0;
        while (n > 1) {
            if ((n & 1) == 0) {
                n = n >> 1;
            } else {
                n = 3 * n + 1;
            }
            steps++;
        }
        total_steps += steps;
    }
    return total_steps;
}

int64_t native_prime_sieve(int64_t* is_prime, int64_t limit) {
    std::fill(is_prime, is_prime + limit, int64_t(1));
    is_prime[0] = 0;
    is_prime[1] = 0;
    for (int64_t p = 2; p * p < limit; ++p) {
        if (is_prime[p]) {
            for (int64_t i = p * p; i < limit; i += p) {
                is_prime[i] = 0;
            }
        }
    }
    int64_t count = 0;
    for (int64_t i = 2; i < limit; ++i) {
        if (is_prime[i]) count++;
    }
    return count;
}

int64_t native_fib_rec(int64_t n) {
    if (n < 2) return n;
    return native_fib_rec(n - 1) + native_fib_rec(n - 2);
}

void native_matmul_i64(const int64_t* A, const int64_t* B, int64_t* C, int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (int64_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// 2. Recursive Fibonacci Module Builder
// ============================================================================

std::unique_ptr<Module> build_fib_rec_module() {
    auto mod = std::make_unique<Module>("bench_fib_rec");
    Function* fn = mod->create_function("fib_rec", Type::i64(), {Type::i64()});
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_case = b.create_block("base_case");
    BasicBlock* rec_case = b.create_block("rec_case");
    fn->append_block(base_case);
    fn->append_block(rec_case);

    Value* n = b.add_block_param(entry, Type::i64());
    Value* two = b.build_iconst_i64(2);
    Value* cond = b.build_slt(n, two);
    b.build_br_if(cond, base_case, rec_case);

    b.position_at_end(base_case);
    b.build_ret(n);

    b.position_at_end(rec_case);
    Value* one = b.build_iconst_i64(1);
    Value* n_minus_1 = b.build_sub(n, one);
    Value* n_minus_2 = b.build_sub(n, two);
    Value* r1 = b.build_call("fib_rec", Type::i64(), {n_minus_1});
    Value* r2 = b.build_call("fib_rec", Type::i64(), {n_minus_2});
    Value* sum = b.build_add(r1, r2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    verify_function(*fn);
    return mod;
}

// ============================================================================
// 3. Benchmark Metric Record
// ============================================================================

struct InterpBenchmarkMetric {
    std::string key;
    std::string name;
    size_t iterations = 0;
    double oracle_ms = 0.0;
    double fast_ms = 0.0;
    double jit_ms = 0.0;
    double native_ms = 0.0;
    double oracle_ns_op = 0.0;
    double fast_ns_op = 0.0;
    double jit_ns_op = 0.0;
    double native_ns_op = 0.0;
    double fast_vs_oracle_speedup = 0.0;
    double jit_vs_fast_speedup = 0.0;
};

void print_interpreter_benchmark_header() {
    std::cout << "\n";
    std::cout << "========================================================================================================================\n";
    std::cout << " Interpreter Performance: Oracle vs FastInterpreter vs Baseline JIT vs Native C++\n";
    std::cout << "========================================================================================================================\n";
    std::cout << std::left  << std::setw(32) << " Benchmark"
              << std::right << std::setw(15) << "Oracle (ns/op)"
              << std::right << std::setw(15) << "Fast (ns/op)"
              << std::right << std::setw(15) << "JIT (ns/op)"
              << std::right << std::setw(15) << "Native (ns/op)"
              << std::right << std::setw(14) << "Fast/Oracle"
              << std::right << std::setw(14) << "JIT/Fast"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
}

void print_interpreter_benchmark_row(const InterpBenchmarkMetric& m) {
    std::cout << std::left  << " " << std::setw(31) << m.name
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << m.oracle_ns_op
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << m.fast_ns_op
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << m.jit_ns_op
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << m.native_ns_op
              << std::right << std::setw(13) << std::fixed << std::setprecision(2) << m.fast_vs_oracle_speedup << "x"
              << std::right << std::setw(13) << std::fixed << std::setprecision(2) << m.jit_vs_fast_speedup << "x"
              << "\n";
}

void print_interpreter_benchmark_summary(const std::vector<InterpBenchmarkMetric>& metrics) {
    if (metrics.empty()) return;
    double log_sum_fo = 0.0;
    double log_sum_jf = 0.0;
    for (const auto& m : metrics) {
        log_sum_fo += std::log(std::max(m.fast_vs_oracle_speedup, 0.001));
        log_sum_jf += std::log(std::max(m.jit_vs_fast_speedup, 0.001));
    }
    double geomean_fo = std::exp(log_sum_fo / static_cast<double>(metrics.size()));
    double geomean_jf = std::exp(log_sum_jf / static_cast<double>(metrics.size()));

    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left  << " " << std::setw(31) << "Geometric Mean"
              << std::right << std::setw(60) << " "
              << std::right << std::setw(13) << std::fixed << std::setprecision(2) << geomean_fo << "x"
              << std::right << std::setw(13) << std::fixed << std::setprecision(2) << geomean_jf << "x"
              << "\n";
    std::cout << "========================================================================================================================\n\n";
}

} // namespace

namespace brass::bench {

void run_interpreter_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet) {
    (void)ratchet;
    std::vector<InterpBenchmarkMetric> metrics;
    metrics.reserve(5);

    auto compute_metrics = [](InterpBenchmarkMetric& m, size_t iters, double oracle_ms, double fast_ms, double jit_ms, double native_ms) {
        m.iterations = iters;
        m.oracle_ms = oracle_ms;
        m.fast_ms = fast_ms;
        m.jit_ms = jit_ms;
        m.native_ms = native_ms;
        const double d_iters = static_cast<double>(iters);
        m.oracle_ns_op = (oracle_ms * 1e6) / d_iters;
        m.fast_ns_op = (fast_ms * 1e6) / d_iters;
        m.jit_ns_op = (jit_ms * 1e6) / d_iters;
        m.native_ns_op = (native_ms * 1e6) / d_iters;
        m.fast_vs_oracle_speedup = (fast_ms > 0.0) ? (oracle_ms / fast_ms) : 1.0;
        m.jit_vs_fast_speedup = (jit_ms > 0.0) ? (fast_ms / jit_ms) : 1.0;
    };

    print_interpreter_benchmark_header();

    // ========================================================================
    // Benchmark 1: Iterative Fibonacci (n = 40)
    // ========================================================================
    {
        const size_t iters = is_debug_build() ? 1000 : 25000;
        const int64_t n = 40;
        auto mod = build_fib_module();

        // 1. Oracle
        Interpreter oracle;
        oracle.run(*mod, "fib_iter", {RuntimeValue::from_i64(n)});
        Stopwatch sw;
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = oracle.run(*mod, "fib_iter", {RuntimeValue::from_i64(n)});
            DoNotOptimize(r);
        }
        double oracle_ms = sw.stop_ms();

        // 2. FastInterpreter
        FastInterpreter fast;
        fast.run(*mod, "fib_iter", {RuntimeValue::from_i64(n)});
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = fast.run(*mod, "fib_iter", {RuntimeValue::from_i64(n)});
            DoNotOptimize(r);
        }
        double fast_ms = sw.stop_ms();

        // 3. Baseline JIT
        BaselineJitCompiler jit_compiler;
        auto compiled_funcs = jit_compiler.compile_module(*mod);
        const BaselineCompiledFunction* compiled = nullptr;
        for (const auto& cf : compiled_funcs) {
            if (cf.name() == "fib_iter") { compiled = &cf; break; }
        }
        if (compiled) compiled->invoke({RuntimeValue::from_i64(n)});
        sw.start();
        if (compiled) {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke({RuntimeValue::from_i64(n)});
                DoNotOptimize(r);
            }
        }
        double jit_ms = sw.stop_ms();

        // 4. Native C++
        native_fib_iter(static_cast<uint64_t>(n));
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            uint64_t r = native_fib_iter(static_cast<uint64_t>(n));
            DoNotOptimize(r);
        }
        double native_ms = sw.stop_ms();

        InterpBenchmarkMetric m;
        m.key = "fib_iter_40";
        m.name = "Iterative Fibonacci (n=40)";
        compute_metrics(m, iters, oracle_ms, fast_ms, jit_ms, native_ms);

        print_interpreter_benchmark_row(m);
        metrics.push_back(m);
    }

    // ========================================================================
    // Benchmark 2: Collatz Steps (n = 1000)
    // ========================================================================
    {
        const size_t iters = is_debug_build() ? 200 : 2500;
        const int64_t max_n = 1000;
        auto mod = build_collatz_module();

        // 1. Oracle
        Interpreter oracle;
        oracle.run(*mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
        Stopwatch sw;
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = oracle.run(*mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
            DoNotOptimize(r);
        }
        double oracle_ms = sw.stop_ms();

        // 2. FastInterpreter
        FastInterpreter fast;
        fast.run(*mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = fast.run(*mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
            DoNotOptimize(r);
        }
        double fast_ms = sw.stop_ms();

        // 3. Baseline JIT
        BaselineJitCompiler jit_compiler;
        auto compiled_funcs = jit_compiler.compile_module(*mod);
        const BaselineCompiledFunction* compiled = nullptr;
        for (const auto& cf : compiled_funcs) {
            if (cf.name() == "collatz_sum") { compiled = &cf; break; }
        }
        if (compiled) compiled->invoke({RuntimeValue::from_i64(max_n)});
        sw.start();
        if (compiled) {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke({RuntimeValue::from_i64(max_n)});
                DoNotOptimize(r);
            }
        }
        double jit_ms = sw.stop_ms();

        // 4. Native C++
        native_collatz_sum(max_n);
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t r = native_collatz_sum(max_n);
            DoNotOptimize(r);
        }
        double native_ms = sw.stop_ms();

        InterpBenchmarkMetric m;
        m.key = "collatz_1000";
        m.name = "Collatz Steps (n=1000)";
        compute_metrics(m, iters, oracle_ms, fast_ms, jit_ms, native_ms);

        print_interpreter_benchmark_row(m);
        metrics.push_back(m);
    }

    // ========================================================================
    // Benchmark 3: Prime Sieve (limit = 100000)
    // ========================================================================
    {
        const size_t iters = is_debug_build() ? 5 : 40;
        const int64_t limit = 100000;
        std::vector<int64_t> buf(static_cast<size_t>(limit), 0);
        auto mod = build_sieve_module();

        // 1. Oracle
        Interpreter oracle;
        oracle.run(*mod, "prime_sieve", {RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
        Stopwatch sw;
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = oracle.run(*mod, "prime_sieve", {RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
            DoNotOptimize(r);
        }
        double oracle_ms = sw.stop_ms();

        // 2. FastInterpreter
        FastInterpreter fast;
        fast.run(*mod, "prime_sieve", {RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = fast.run(*mod, "prime_sieve", {RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
            DoNotOptimize(r);
        }
        double fast_ms = sw.stop_ms();

        // 3. Baseline JIT
        BaselineJitCompiler jit_compiler;
        auto compiled_funcs = jit_compiler.compile_module(*mod);
        const BaselineCompiledFunction* compiled = nullptr;
        for (const auto& cf : compiled_funcs) {
            if (cf.name() == "prime_sieve") { compiled = &cf; break; }
        }
        if (compiled) compiled->invoke({RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
        sw.start();
        if (compiled) {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke({RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)});
                DoNotOptimize(r);
            }
        }
        double jit_ms = sw.stop_ms();

        // 4. Native C++
        native_prime_sieve(buf.data(), limit);
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t r = native_prime_sieve(buf.data(), limit);
            DoNotOptimize(r);
        }
        double native_ms = sw.stop_ms();

        InterpBenchmarkMetric m;
        m.key = "prime_sieve_100k";
        m.name = "Prime Sieve (limit=100k)";
        compute_metrics(m, iters, oracle_ms, fast_ms, jit_ms, native_ms);

        print_interpreter_benchmark_row(m);
        metrics.push_back(m);
    }

    // ========================================================================
    // Benchmark 4: Recursive Fibonacci (n = 22)
    // ========================================================================
    {
        const size_t iters = is_debug_build() ? 2 : 12;
        const int64_t n = 22;
        auto mod = build_fib_rec_module();

        // 1. Oracle
        Interpreter oracle;
        oracle.run(*mod, "fib_rec", {RuntimeValue::from_i64(n)});
        Stopwatch sw;
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = oracle.run(*mod, "fib_rec", {RuntimeValue::from_i64(n)});
            DoNotOptimize(r);
        }
        double oracle_ms = sw.stop_ms();

        // 2. FastInterpreter
        FastInterpreter fast;
        fast.run(*mod, "fib_rec", {RuntimeValue::from_i64(n)});
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = fast.run(*mod, "fib_rec", {RuntimeValue::from_i64(n)});
            DoNotOptimize(r);
        }
        double fast_ms = sw.stop_ms();

        // 3. Baseline JIT
        BaselineJitCompiler jit_compiler;
        auto compiled_funcs = jit_compiler.compile_module(*mod);
        const BaselineCompiledFunction* compiled = nullptr;
        for (const auto& cf : compiled_funcs) {
            if (cf.name() == "fib_rec") { compiled = &cf; break; }
        }
        if (compiled) compiled->invoke({RuntimeValue::from_i64(n)});
        sw.start();
        if (compiled) {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke({RuntimeValue::from_i64(n)});
                DoNotOptimize(r);
            }
        }
        double jit_ms = sw.stop_ms();

        // 4. Native C++
        native_fib_rec(n);
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t r = native_fib_rec(n);
            DoNotOptimize(r);
        }
        double native_ms = sw.stop_ms();

        InterpBenchmarkMetric m;
        m.key = "fib_rec_22";
        m.name = "Recursive Fibonacci (n=22)";
        compute_metrics(m, iters, oracle_ms, fast_ms, jit_ms, native_ms);

        print_interpreter_benchmark_row(m);
        metrics.push_back(m);
    }

    // ========================================================================
    // Benchmark 5: Matrix Multiplication (32x32)
    // ========================================================================
    {
        const size_t iters = is_debug_build() ? 5 : 35;
        const int64_t N = 32;
        const size_t total_elements = static_cast<size_t>(N * N);
        std::vector<int64_t> A(total_elements, 1);
        std::vector<int64_t> B(total_elements, 2);
        std::vector<int64_t> C(total_elements, 0);

        for (size_t idx = 0; idx < total_elements; ++idx) {
            A[idx] = static_cast<int64_t>((idx % 17) + 1);
            B[idx] = static_cast<int64_t>((idx % 13) + 1);
        }

        auto mod = build_matmul_i64_naive_module();

        // 1. Oracle
        Interpreter oracle;
        oracle.run(*mod, "matmul_i64_naive", {
            RuntimeValue::from_ptr(A.data()),
            RuntimeValue::from_ptr(B.data()),
            RuntimeValue::from_ptr(C.data()),
            RuntimeValue::from_i64(N)
        });
        Stopwatch sw;
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = oracle.run(*mod, "matmul_i64_naive", {
                RuntimeValue::from_ptr(A.data()),
                RuntimeValue::from_ptr(B.data()),
                RuntimeValue::from_ptr(C.data()),
                RuntimeValue::from_i64(N)
            });
            DoNotOptimize(r);
        }
        double oracle_ms = sw.stop_ms();

        // 2. FastInterpreter
        FastInterpreter fast;
        fast.run(*mod, "matmul_i64_naive", {
            RuntimeValue::from_ptr(A.data()),
            RuntimeValue::from_ptr(B.data()),
            RuntimeValue::from_ptr(C.data()),
            RuntimeValue::from_i64(N)
        });
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            RuntimeValue r = fast.run(*mod, "matmul_i64_naive", {
                RuntimeValue::from_ptr(A.data()),
                RuntimeValue::from_ptr(B.data()),
                RuntimeValue::from_ptr(C.data()),
                RuntimeValue::from_i64(N)
            });
            DoNotOptimize(r);
        }
        double fast_ms = sw.stop_ms();

        // 3. Baseline JIT
        BaselineJitCompiler jit_compiler;
        auto compiled_funcs = jit_compiler.compile_module(*mod);
        const BaselineCompiledFunction* compiled = nullptr;
        for (const auto& cf : compiled_funcs) {
            if (cf.name() == "matmul_i64_naive") { compiled = &cf; break; }
        }
        if (compiled) {
            compiled->invoke({
                RuntimeValue::from_ptr(A.data()),
                RuntimeValue::from_ptr(B.data()),
                RuntimeValue::from_ptr(C.data()),
                RuntimeValue::from_i64(N)
            });
        }
        sw.start();
        if (compiled) {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke({
                    RuntimeValue::from_ptr(A.data()),
                    RuntimeValue::from_ptr(B.data()),
                    RuntimeValue::from_ptr(C.data()),
                    RuntimeValue::from_i64(N)
                });
                DoNotOptimize(r);
            }
        }
        double jit_ms = sw.stop_ms();

        // 4. Native C++
        native_matmul_i64(A.data(), B.data(), C.data(), N);
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            native_matmul_i64(A.data(), B.data(), C.data(), N);
            ClobberMemory();
        }
        double native_ms = sw.stop_ms();

        InterpBenchmarkMetric m;
        m.key = "matmul_32x32";
        m.name = "Matrix Multiply (32x32)";
        compute_metrics(m, iters, oracle_ms, fast_ms, jit_ms, native_ms);

        print_interpreter_benchmark_row(m);
        metrics.push_back(m);
    }

    print_interpreter_benchmark_summary(metrics);

    // Record results for benchmark suite
    for (const auto& m : metrics) {
        BenchmarkResult res;
        res.key = "fast_interp_" + m.key;
        res.name = "FastInterpreter: " + m.name;
        res.iterations = m.iterations;
        res.repetitions = 1;
        res.native_ms = m.oracle_ms; // Oracle is baseline comparator
        res.brass_ms = m.fast_ms;
        res.ratio = m.fast_vs_oracle_speedup;
        res.target_ratio = 1.25;
        res.passes_bar = (m.fast_vs_oracle_speedup >= 1.05);
        res.notes = std::to_string(m.fast_vs_oracle_speedup) + "x vs Oracle; " +
                    std::to_string(m.jit_vs_fast_speedup) + "x JIT/Fast";
        results.push_back(res);
    }
}

} // namespace brass::bench
