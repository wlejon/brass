#include "bench_numeric.hpp"
#include "bench_numeric_modules.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <vector>
#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

// ============================================================================
// 1. Native Baselines
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

// Vectorized -O3 baseline
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

// Scalar -O3 companion baseline (no auto-vectorization)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void native_matmul_i64_scalar(const int64_t* A, const int64_t* B, int64_t* C, int64_t N) {
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

// Vectorized -O3 baseline
void native_matmul_f64(const double* A, const double* B, double* C, int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Scalar -O3 companion baseline (no auto-vectorization)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void native_matmul_f64_scalar(const double* A, const double* B, double* C, int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

struct BenchListNode {
    int64_t value;
    BenchListNode* next;
};

int64_t native_list_traversal(const BenchListNode* head) {
    int64_t sum = 0;
    while (head) {
        sum += head->value;
        head = head->next;
    }
    return sum;
}

} // namespace

namespace brass::bench {

void run_numeric_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet) {
    Stopwatch sw;

    // 1. Iterative Fibonacci
    {
        size_t iters = 500000;
        uint64_t n = 45;

        auto (*volatile native_fn)(uint64_t) = &native_fib_iter;

        sw.start();
        uint64_t native_sink = 0;
        for (size_t i = 0; i < iters; ++i) {
            uint64_t input = n;
            DoNotOptimize(input);
            native_sink = native_fn(input);
            DoNotOptimize(native_sink);
        }
        double native_ms = sw.stop_ms();

        auto mod = build_fib_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto fib_fn = jit.get_function_ptr<uint64_t(*)(uint64_t)>("fib_iter");
        if (!fib_fn) {
            std::cerr << "FATAL: fib_iter function pointer is null!\n";
            std::abort();
        }

        sw.start();
        uint64_t jit_sink = 0;
        for (size_t i = 0; i < iters; ++i) {
            uint64_t input = n;
            DoNotOptimize(input);
            jit_sink = fib_fn(input);
            DoNotOptimize(jit_sink);
        }
        double brass_ms = sw.stop_ms();

        if (native_sink != jit_sink) {
            std::cerr << "FATAL: Fibonacci result mismatch: native=" << native_sink << ", JIT=" << jit_sink << "\n";
            std::abort();
        }

        double ratio = brass_ms / native_ms;
        double target = ratchet.get_ratio("fib", 1.15);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        results.push_back({"fib", "Iterative Fibonacci (N=45)", iters, native_ms, 0.0, brass_ms, ratio, 0.0, target, ratio <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 2. Prime Sieve
    {
        size_t iters = 500;
        int64_t limit = 100000;
        std::vector<int64_t> native_buf(limit, 0);
        std::vector<int64_t> jit_buf(limit, 0);

        sw.start();
        int64_t native_count = 0;
        for (size_t i = 0; i < iters; ++i) {
            int64_t lim = limit;
            DoNotOptimize(lim);
            native_count = native_prime_sieve(native_buf.data(), lim);
            DoNotOptimize(native_count);
            DoNotOptimize(native_buf.data());
        }
        double native_ms = sw.stop_ms();

        auto mod = build_sieve_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto sieve_fn = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t)>("prime_sieve");
        if (!sieve_fn) {
            std::cerr << "FATAL: prime_sieve function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_count = 0;
        for (size_t i = 0; i < iters; ++i) {
            int64_t lim = limit;
            DoNotOptimize(lim);
            jit_count = sieve_fn(jit_buf.data(), lim);
            DoNotOptimize(jit_count);
            DoNotOptimize(jit_buf.data());
        }
        double brass_ms = sw.stop_ms();

        if (native_count != jit_count || native_buf != jit_buf) {
            std::cerr << "FATAL: Prime Sieve result mismatch: native=" << native_count << ", JIT=" << jit_count << "\n";
            std::abort();
        }

        double ratio = brass_ms / native_ms;
        double target = ratchet.get_ratio("sieve", 1.25);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        results.push_back({"sieve", "Prime Sieve (N=100k)", iters, native_ms, 0.0, brass_ms, ratio, 0.0, target, ratio <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 3. Collatz Conjecture Sum
    {
        size_t iters = 50;
        int64_t max_n = 100000;

        sw.start();
        int64_t native_steps = 0;
        for (size_t i = 0; i < iters; ++i) {
            int64_t input = max_n;
            DoNotOptimize(input);
            native_steps = native_collatz_sum(input);
            DoNotOptimize(native_steps);
        }
        double native_ms = sw.stop_ms();

        auto mod = build_collatz_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto collatz_fn = jit.get_function_ptr<int64_t(*)(int64_t)>("collatz_sum");
        if (!collatz_fn) {
            std::cerr << "FATAL: collatz_sum function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_steps = 0;
        for (size_t i = 0; i < iters; ++i) {
            int64_t input = max_n;
            DoNotOptimize(input);
            jit_steps = collatz_fn(input);
            DoNotOptimize(jit_steps);
        }
        double brass_ms = sw.stop_ms();

        if (native_steps != jit_steps) {
            std::cerr << "FATAL: Collatz Sum result mismatch: native=" << native_steps << ", JIT=" << jit_steps << "\n";
            std::abort();
        }

        double ratio = brass_ms / native_ms;
        double target = ratchet.get_ratio("collatz", 1.35);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        results.push_back({"collatz", "Collatz Sum (1..100k)", iters, native_ms, 0.0, brass_ms, ratio, 0.0, target, ratio <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 4. Matrix Multiplication 32x32 Integer
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_scalar(N * N, 0), C_jit(N * N, 0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64_scalar(A.data(), B.data(), C_scalar.data(), size_n);
            DoNotOptimize(C_scalar.data());
        }
        double scalar_ms = sw.stop_ms();

        auto mod = build_matmul_i64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64");
        if (!matmul_fn) {
            std::cerr << "FATAL: matmul_i64 function pointer is null!\n";
            std::abort();
        }

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            matmul_fn(A.data(), B.data(), C_jit.data(), size_n);
            DoNotOptimize(C_jit.data());
        }
        double brass_ms = sw.stop_ms();

        if (C_native != C_jit) {
            std::cerr << "FATAL: MatMul 32x32 (i64) result mismatch: C_native[0]=" << C_native[0] << ", C_jit[0]=" << C_jit[0] << "\n";
            std::abort();
        }

        double ratio_scalar = brass_ms / scalar_ms;
        double ratio_vec = brass_ms / native_ms;
        double target = ratchet.get_ratio("matmul_i64_32", 1.65);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x scalar (vec: " << std::fixed << std::setprecision(2) << ratio_vec << "x)";
        results.push_back({"matmul_i64_32", "MatMul 32x32 (i64)", iters, native_ms, scalar_ms, brass_ms, ratio_scalar, ratio_vec, target, ratio_scalar <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 5. Matrix Multiplication 64x64 Integer
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_scalar(N * N, 0), C_jit(N * N, 0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64_scalar(A.data(), B.data(), C_scalar.data(), size_n);
            DoNotOptimize(C_scalar.data());
        }
        double scalar_ms = sw.stop_ms();

        auto mod = build_matmul_i64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64");
        if (!matmul_fn) {
            std::cerr << "FATAL: matmul_i64 function pointer is null!\n";
            std::abort();
        }

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            matmul_fn(A.data(), B.data(), C_jit.data(), size_n);
            DoNotOptimize(C_jit.data());
        }
        double brass_ms = sw.stop_ms();

        if (C_native != C_jit) {
            std::cerr << "FATAL: MatMul 64x64 (i64) result mismatch between native and JIT!\n";
            std::abort();
        }

        double ratio_scalar = brass_ms / scalar_ms;
        double ratio_vec = brass_ms / native_ms;
        double target = ratchet.get_ratio("matmul_i64_64", 1.75);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x scalar (vec: " << std::fixed << std::setprecision(2) << ratio_vec << "x)";
        results.push_back({"matmul_i64_64", "MatMul 64x64 (i64)", iters, native_ms, scalar_ms, brass_ms, ratio_scalar, ratio_vec, target, ratio_scalar <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 6. Matrix Multiplication 32x32 Float
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_scalar(N * N, 0.0), C_jit(N * N, 0.0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64_scalar(A.data(), B.data(), C_scalar.data(), size_n);
            DoNotOptimize(C_scalar.data());
        }
        double scalar_ms = sw.stop_ms();

        auto mod = build_matmul_f64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64");
        if (!matmul_fn) {
            std::cerr << "FATAL: matmul_f64 function pointer is null!\n";
            std::abort();
        }

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            matmul_fn(A.data(), B.data(), C_jit.data(), size_n);
            DoNotOptimize(C_jit.data());
        }
        double brass_ms = sw.stop_ms();

        for (size_t i = 0; i < C_native.size(); ++i) {
            if (std::abs(C_native[i] - C_jit[i]) > 1e-5) {
                std::cerr << "FATAL: MatMul 32x32 (f64) result mismatch at index " << i << ": native=" << C_native[i] << ", JIT=" << C_jit[i] << "\n";
                std::abort();
            }
        }

        double ratio_scalar = brass_ms / scalar_ms;
        double ratio_vec = brass_ms / native_ms;
        double target = ratchet.get_ratio("matmul_f64_32", 2.15);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x scalar (vec: " << std::fixed << std::setprecision(2) << ratio_vec << "x)";
        results.push_back({"matmul_f64_32", "MatMul 32x32 (f64)", iters, native_ms, scalar_ms, brass_ms, ratio_scalar, ratio_vec, target, ratio_scalar <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 7. Matrix Multiplication 64x64 Float
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_scalar(N * N, 0.0), C_jit(N * N, 0.0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64_scalar(A.data(), B.data(), C_scalar.data(), size_n);
            DoNotOptimize(C_scalar.data());
        }
        double scalar_ms = sw.stop_ms();

        auto mod = build_matmul_f64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64");
        if (!matmul_fn) {
            std::cerr << "FATAL: matmul_f64 function pointer is null!\n";
            std::abort();
        }

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            matmul_fn(A.data(), B.data(), C_jit.data(), size_n);
            DoNotOptimize(C_jit.data());
        }
        double brass_ms = sw.stop_ms();

        for (size_t i = 0; i < C_native.size(); ++i) {
            if (std::abs(C_native[i] - C_jit[i]) > 1e-5) {
                std::cerr << "FATAL: MatMul 64x64 (f64) result mismatch at index " << i << ": native=" << C_native[i] << ", JIT=" << C_jit[i] << "\n";
                std::abort();
            }
        }

        double ratio_scalar = brass_ms / scalar_ms;
        double ratio_vec = brass_ms / native_ms;
        double target = ratchet.get_ratio("matmul_f64_64", 1.65);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x scalar (vec: " << std::fixed << std::setprecision(2) << ratio_vec << "x)";
        results.push_back({"matmul_f64_64", "MatMul 64x64 (f64)", iters, native_ms, scalar_ms, brass_ms, ratio_scalar, ratio_vec, target, ratio_scalar <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }

    // 8. Pointer-Chasing Linked List Traversal
    {
        size_t iters = 500;
        size_t node_count = 50000;
        std::vector<BenchListNode> nodes(node_count);
        for (size_t i = 0; i < node_count; ++i) {
            nodes[i].value = static_cast<int64_t>(i + 1);
            nodes[i].next = (i + 1 < node_count) ? &nodes[i + 1] : nullptr;
        }

        sw.start();
        int64_t native_sum = 0;
        for (size_t i = 0; i < iters; ++i) {
            const BenchListNode* head_ptr = nodes.data();
            DoNotOptimize(head_ptr);
            native_sum = native_list_traversal(head_ptr);
            DoNotOptimize(native_sum);
        }
        double native_ms = sw.stop_ms();

        auto mod = build_list_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto list_fn = jit.get_function_ptr<int64_t(*)(const BenchListNode*)>("list_traversal");
        if (!list_fn) {
            std::cerr << "FATAL: list_traversal function pointer is null!\n";
            std::abort();
        }

        sw.start();
        int64_t jit_sum = 0;
        for (size_t i = 0; i < iters; ++i) {
            const BenchListNode* head_ptr = nodes.data();
            DoNotOptimize(head_ptr);
            jit_sum = list_fn(head_ptr);
            DoNotOptimize(jit_sum);
        }
        double brass_ms = sw.stop_ms();

        if (native_sum != jit_sum) {
            std::cerr << "FATAL: Linked List Traversal result mismatch: native=" << native_sum << ", JIT=" << jit_sum << "\n";
            std::abort();
        }

        double ratio = brass_ms / native_ms;
        double target = ratchet.get_ratio("linked_list", 0.95);
        std::ostringstream notes;
        notes << "<= " << std::fixed << std::setprecision(2) << target << "x baseline";
        results.push_back({"linked_list", "Linked List Traversal (50k)", iters, native_ms, 0.0, brass_ms, ratio, 0.0, target, ratio <= target, notes.str()});
        BenchmarkReporter::print_row(results.back());
    }
}

} // namespace brass::bench
