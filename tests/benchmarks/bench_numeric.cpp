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
    // 1. Iterative Fibonacci
    {
        size_t iters = 500000;
        uint64_t n = 45;

        auto (*volatile native_fn)(uint64_t) = &native_fib_iter;
        auto run_native = [native_fn, n, iters]() {
            uint64_t sink = 0;
            for (size_t i = 0; i < iters; ++i) {
                uint64_t input = n;
                DoNotOptimize(input);
                sink = native_fn(input);
                DoNotOptimize(sink);
            }
            return sink;
        };

        auto mod = build_fib_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto fib_fn = jit->get_function_ptr<uint64_t(*)(uint64_t)>("fib_iter");
            if (!fib_fn) {
                std::cerr << "FATAL: fib_iter function pointer is null!\n";
                std::abort();
            }
            return [jit, fib_fn, n, iters]() {
                uint64_t sink = 0;
                for (size_t i = 0; i < iters; ++i) {
                    uint64_t input = n;
                    DoNotOptimize(input);
                    sink = fib_fn(input);
                    DoNotOptimize(sink);
                }
                return sink;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        uint64_t native_sink = run_native();
        auto test_jit = make_jit_runner(0);
        uint64_t jit_sink = test_jit();

        if (native_sink != jit_sink) {
            std::cerr << "FATAL: Fibonacci result mismatch: native=" << native_sink << ", JIT=" << jit_sink << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("fib", "Iterative Fibonacci (N=45)", iters, paired, ratchet.get_ratio("fib", 1.25)));
        BenchmarkReporter::print_row(results.back());
    }

    // 2. Prime Sieve
    {
        size_t iters = 500;
        int64_t limit = 100000;
        std::vector<int64_t> native_buf(limit, 0);

        auto run_native = [buf = native_buf.data(), limit, iters]() {
            int64_t count = 0;
            for (size_t i = 0; i < iters; ++i) {
                int64_t lim = limit;
                DoNotOptimize(lim);
                count = native_prime_sieve(buf, lim);
                DoNotOptimize(count);
                DoNotOptimize(buf);
            }
            return count;
        };

        auto mod = build_sieve_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto sieve_fn = jit->get_function_ptr<int64_t(*)(int64_t*, int64_t)>("prime_sieve");
            if (!sieve_fn) {
                std::cerr << "FATAL: prime_sieve function pointer is null!\n";
                std::abort();
            }
            auto jit_buf = std::make_shared<std::vector<int64_t>>(limit, 0);
            return [jit, sieve_fn, jit_buf, limit, iters]() {
                int64_t count = 0;
                for (size_t i = 0; i < iters; ++i) {
                    int64_t lim = limit;
                    DoNotOptimize(lim);
                    count = sieve_fn(jit_buf->data(), lim);
                    DoNotOptimize(count);
                    DoNotOptimize(jit_buf->data());
                }
                return count;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        int64_t native_count = run_native();
        auto test_jit = make_jit_runner(0);
        int64_t jit_count = test_jit();

        if (native_count != jit_count) {
            std::cerr << "FATAL: Prime Sieve result mismatch: native=" << native_count << ", JIT=" << jit_count << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("sieve", "Prime Sieve (N=100k)", iters, paired, ratchet.get_ratio("sieve", 1.35)));
        BenchmarkReporter::print_row(results.back());
    }

    // 3. Collatz Conjecture Sum
    {
        size_t iters = 50;
        int64_t max_n = 100000;

        auto run_native = [max_n, iters]() {
            int64_t steps = 0;
            for (size_t i = 0; i < iters; ++i) {
                int64_t input = max_n;
                DoNotOptimize(input);
                steps = native_collatz_sum(input);
                DoNotOptimize(steps);
            }
            return steps;
        };

        auto mod = build_collatz_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto collatz_fn = jit->get_function_ptr<int64_t(*)(int64_t)>("collatz_sum");
            if (!collatz_fn) {
                std::cerr << "FATAL: collatz_sum function pointer is null!\n";
                std::abort();
            }
            return [jit, collatz_fn, max_n, iters]() {
                int64_t steps = 0;
                for (size_t i = 0; i < iters; ++i) {
                    int64_t input = max_n;
                    DoNotOptimize(input);
                    steps = collatz_fn(input);
                    DoNotOptimize(steps);
                }
                return steps;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        int64_t native_steps = run_native();
        auto test_jit = make_jit_runner(0);
        int64_t jit_steps = test_jit();

        if (native_steps != jit_steps) {
            std::cerr << "FATAL: Collatz Sum result mismatch: native=" << native_steps << ", JIT=" << jit_steps << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("collatz", "Collatz Sum (1..100k)", iters, paired, ratchet.get_ratio("collatz", 1.05)));
        BenchmarkReporter::print_row(results.back());
    }

    // 4. Matrix Multiplication 32x32 Integer (Naive and Preopt)
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_scalar(N * N, 0);

        auto run_vec = [a = A.data(), b = B.data(), c = C_native.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_i64(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_scalar = [a = A.data(), b = B.data(), c = C_scalar.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_i64_scalar(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        // 4a. Naive kernel
        {
            auto mod = build_matmul_i64_naive_module();
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_naive");
                if (!matmul_fn) {
                    std::cerr << "FATAL: matmul_i64_naive function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<int64_t>>(N * N, 0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<int64_t> C_jit_v(N * N, 0);
            auto test_jit = make_jit_runner(0);
            test_jit();
            native_matmul_i64(A.data(), B.data(), C_native.data(), N);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_naive");
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            if (C_native != C_jit_v) {
                std::cerr << "FATAL: MatMul 32x32 (i64, naive) result mismatch!\n";
                std::abort();
            }

            results.push_back(make_triplet_result("matmul_i64_32_naive", "MatMul 32x32 (i64, naive)", iters, triplet, ratchet.get_ratio("matmul_i64_32_naive", 1.50)));
            BenchmarkReporter::print_row(results.back());
        }

        // 4b. Preopt kernel
        {
            auto mod = build_matmul_i64_preopt_module();
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_preopt");
                if (!matmul_fn) {
                    std::cerr << "FATAL: matmul_i64_preopt function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<int64_t>>(N * N, 0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<int64_t> C_jit_v(N * N, 0);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_preopt");
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            if (C_native != C_jit_v) {
                std::cerr << "FATAL: MatMul 32x32 (i64, preopt) result mismatch!\n";
                std::abort();
            }

            results.push_back(make_triplet_result("matmul_i64_32_preopt", "MatMul 32x32 (i64, preopt)", iters, triplet, ratchet.get_ratio("matmul_i64_32_preopt", 1.50)));
            BenchmarkReporter::print_row(results.back());
        }
    }

    // 5. Matrix Multiplication 64x64 Integer (Naive and Preopt)
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_scalar(N * N, 0);

        auto run_vec = [a = A.data(), b = B.data(), c = C_native.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_i64(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_scalar = [a = A.data(), b = B.data(), c = C_scalar.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_i64_scalar(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        // 5a. Naive kernel
        {
            auto mod = build_matmul_i64_naive_module();
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_naive");
                if (!matmul_fn) {
                    std::cerr << "FATAL: matmul_i64_naive function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<int64_t>>(N * N, 0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<int64_t> C_jit_v(N * N, 0);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_naive");
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            if (C_native != C_jit_v) {
                std::cerr << "FATAL: MatMul 64x64 (i64, naive) result mismatch!\n";
                std::abort();
            }

            results.push_back(make_triplet_result("matmul_i64_64_naive", "MatMul 64x64 (i64, naive)", iters, triplet, ratchet.get_ratio("matmul_i64_64_naive", 1.45)));
            BenchmarkReporter::print_row(results.back());
        }

        // 5b. Preopt kernel
        {
            auto mod = build_matmul_i64_preopt_module();
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_preopt");
                if (!matmul_fn) {
                    std::cerr << "FATAL: matmul_i64_preopt function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<int64_t>>(N * N, 0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<int64_t> C_jit_v(N * N, 0);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64_preopt");
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            if (C_native != C_jit_v) {
                std::cerr << "FATAL: MatMul 64x64 (i64, preopt) result mismatch!\n";
                std::abort();
            }

            results.push_back(make_triplet_result("matmul_i64_64_preopt", "MatMul 64x64 (i64, preopt)", iters, triplet, ratchet.get_ratio("matmul_i64_64_preopt", 1.45)));
            BenchmarkReporter::print_row(results.back());
        }
    }

    // 6. Matrix Multiplication 32x32 Float (Strict and Reassoc)
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_scalar(N * N, 0.0);

        auto run_vec = [a = A.data(), b = B.data(), c = C_native.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_f64(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_scalar = [a = A.data(), b = B.data(), c = C_scalar.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_f64_scalar(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_f64_32 = [&](const char* key, const char* name, std::unique_ptr<Module> mod, const char* fn_sym, double def_target) {
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>(fn_sym);
                if (!matmul_fn) {
                    std::cerr << "FATAL: " << fn_sym << " function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<double>>(N * N, 0.0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<double> C_jit_v(N * N, 0.0);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>(fn_sym);
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            native_matmul_f64(A.data(), B.data(), C_native.data(), N);
            for (size_t i = 0; i < C_native.size(); ++i) {
                if (std::abs(C_native[i] - C_jit_v[i]) > 1e-4) {
                    std::cerr << "FATAL: " << name << " result mismatch at index " << i << "\n";
                    std::abort();
                }
            }

            results.push_back(make_triplet_result(key, name, iters, triplet, ratchet.get_ratio(key, def_target)));
            BenchmarkReporter::print_row(results.back());
        };

        // 6a. Strict Naive
        run_f64_32("matmul_f64_32_strict_naive", "MatMul 32x32 (f64, strict, naive)", build_matmul_f64_strict_naive_module(), "matmul_f64_strict_naive", 2.25);
        // 6b. Strict Preopt
        run_f64_32("matmul_f64_32_strict_preopt", "MatMul 32x32 (f64, strict, preopt)", build_matmul_f64_strict_preopt_module(), "matmul_f64_strict_preopt", 2.20);
        // 6c. Reassoc Naive (flagged opt-in)
        run_f64_32("matmul_f64_32_reassoc_naive", "MatMul 32x32 (f64, reassoc, naive)", build_matmul_f64_reassoc_naive_module(), "matmul_f64_reassoc_naive", 1.00);
        // 6d. Reassoc Preopt (flagged opt-in)
        run_f64_32("matmul_f64_32_reassoc_preopt", "MatMul 32x32 (f64, reassoc, preopt)", build_matmul_f64_reassoc_preopt_module(), "matmul_f64_reassoc_preopt", 1.00);
    }

    // 7. Matrix Multiplication 64x64 Float (Strict and Reassoc)
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_scalar(N * N, 0.0);

        auto run_vec = [a = A.data(), b = B.data(), c = C_native.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_f64(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_scalar = [a = A.data(), b = B.data(), c = C_scalar.data(), N, iters]() {
            for (size_t i = 0; i < iters; ++i) {
                int64_t size_n = N;
                DoNotOptimize(size_n);
                native_matmul_f64_scalar(a, b, c, size_n);
                DoNotOptimize(c);
            }
        };

        auto run_f64_64 = [&](const char* key, const char* name, std::unique_ptr<Module> mod, const char* fn_sym, double def_target) {
            auto make_jit_runner = [&](size_t padding) {
                auto jit = std::make_shared<JitExecutionEngine>();
                jit->compile_and_load(*mod, padding);
                auto matmul_fn = jit->get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>(fn_sym);
                if (!matmul_fn) {
                    std::cerr << "FATAL: " << fn_sym << " function pointer is null!\n";
                    std::abort();
                }
                auto c_jit = std::make_shared<std::vector<double>>(N * N, 0.0);
                return [jit, matmul_fn, a = A.data(), b = B.data(), c_jit, N, iters]() {
                    for (size_t i = 0; i < iters; ++i) {
                        int64_t size_n = N;
                        DoNotOptimize(size_n);
                        matmul_fn(a, b, c_jit->data(), size_n);
                        DoNotOptimize(c_jit->data());
                    }
                };
            };

            auto triplet = measure_triplet_multi_placement(DEFAULT_BENCH_REPETITIONS, run_vec, run_scalar, make_jit_runner);

            std::vector<double> C_jit_v(N * N, 0.0);
            JitExecutionEngine jit_v;
            jit_v.compile_and_load(*mod);
            auto fn_v = jit_v.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>(fn_sym);
            fn_v(A.data(), B.data(), C_jit_v.data(), N);
            native_matmul_f64(A.data(), B.data(), C_native.data(), N);
            for (size_t i = 0; i < C_native.size(); ++i) {
                if (std::abs(C_native[i] - C_jit_v[i]) > 1e-4) {
                    std::cerr << "FATAL: " << name << " result mismatch at index " << i << "\n";
                    std::abort();
                }
            }

            results.push_back(make_triplet_result(key, name, iters, triplet, ratchet.get_ratio(key, def_target)));
            BenchmarkReporter::print_row(results.back());
        };

        // 7a. Strict Naive
        run_f64_64("matmul_f64_64_strict_naive", "MatMul 64x64 (f64, strict, naive)", build_matmul_f64_strict_naive_module(), "matmul_f64_strict_naive", 1.70);
        // 7b. Strict Preopt
        run_f64_64("matmul_f64_64_strict_preopt", "MatMul 64x64 (f64, strict, preopt)", build_matmul_f64_strict_preopt_module(), "matmul_f64_strict_preopt", 1.70);
        // 7c. Reassoc Naive (flagged opt-in)
        run_f64_64("matmul_f64_64_reassoc_naive", "MatMul 64x64 (f64, reassoc, naive)", build_matmul_f64_reassoc_naive_module(), "matmul_f64_reassoc_naive", 0.85);
        // 7d. Reassoc Preopt (flagged opt-in)
        run_f64_64("matmul_f64_64_reassoc_preopt", "MatMul 64x64 (f64, reassoc, preopt)", build_matmul_f64_reassoc_preopt_module(), "matmul_f64_reassoc_preopt", 0.85);
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

        auto run_native = [head_ptr = nodes.data(), iters]() {
            int64_t sum = 0;
            for (size_t i = 0; i < iters; ++i) {
                const BenchListNode* p = head_ptr;
                DoNotOptimize(p);
                sum = native_list_traversal(p);
                DoNotOptimize(sum);
            }
            return sum;
        };

        auto mod = build_list_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto list_fn = jit->get_function_ptr<int64_t(*)(const BenchListNode*)>("list_traversal");
            if (!list_fn) {
                std::cerr << "FATAL: list_traversal function pointer is null!\n";
                std::abort();
            }
            return [jit, list_fn, head_ptr = nodes.data(), iters]() {
                int64_t sum = 0;
                for (size_t i = 0; i < iters; ++i) {
                    const BenchListNode* p = head_ptr;
                    DoNotOptimize(p);
                    sum = list_fn(p);
                    DoNotOptimize(sum);
                }
                return sum;
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_native, make_jit_runner);

        int64_t native_sum = run_native();
        auto test_jit = make_jit_runner(0);
        int64_t jit_sum = test_jit();

        if (native_sum != jit_sum) {
            std::cerr << "FATAL: Linked List Traversal result mismatch: native=" << native_sum << ", JIT=" << jit_sum << "\n";
            std::abort();
        }

        results.push_back(make_paired_result("linked_list", "Linked List Traversal (50k)", iters, paired, ratchet.get_ratio("linked_list", 0.90)));
        BenchmarkReporter::print_row(results.back());
    }
}

} // namespace brass::bench
