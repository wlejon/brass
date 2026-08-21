#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include "../benchmarks/bench_numeric_modules.hpp"
#include <random>
#include <vector>
#include <cmath>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - 2D Matrix Indexing and Nested Loops (i64)") {
    std::mt19937_64 rng(7777);

    for (uint64_t seed = 0; seed < 55; ++seed) {
        std::string mod_name = "fuzz_mat_i64_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_mat_i64_fn";

        generate_fuzz_matrix_i64(mod, fn_name, seed + 200);

        int64_t N = 4 + static_cast<int64_t>(seed % 13);
        size_t total_elements = static_cast<size_t>(N * N);

        std::vector<int64_t> A(total_elements);
        std::vector<int64_t> B(total_elements);
        std::vector<int64_t> C_interp(total_elements, 0);
        std::vector<int64_t> C_jit(total_elements, 0);

        for (size_t i = 0; i < total_elements; ++i) {
            A[i] = static_cast<int64_t>((rng() % 100) - 50);
            B[i] = static_cast<int64_t>((rng() % 100) - 50);
        }

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(A.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(B.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(C_interp.data())),
            RuntimeValue::from_i64(N)
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t*, int64_t*, int64_t)>(fn_name);
        REQUIRE(fn_ptr != nullptr);
        int64_t jit_res = fn_ptr(A.data(), B.data(), C_jit.data(), N);

        CHECK_EQ(interp_res.as_i64(), jit_res);
        CHECK(C_interp == C_jit);
    }
}

TEST_CASE("Differential Fuzzer - 2D Matrix Indexing and Nested Loops (f64)") {
    std::mt19937_64 rng(8888);

    for (uint64_t seed = 0; seed < 55; ++seed) {
        std::string mod_name = "fuzz_mat_f64_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_mat_f64_fn";

        generate_fuzz_matrix_f64(mod, fn_name, seed + 300);

        int64_t N = 4 + static_cast<int64_t>(seed % 13);
        size_t total_elements = static_cast<size_t>(N * N);

        std::vector<double> A(total_elements);
        std::vector<double> B(total_elements);
        std::vector<double> C_interp(total_elements, 0.0);
        std::vector<double> C_jit(total_elements, 0.0);

        for (size_t i = 0; i < total_elements; ++i) {
            A[i] = static_cast<double>(static_cast<int64_t>(rng() % 200) - 100) / 10.0;
            B[i] = static_cast<double>(static_cast<int64_t>(rng() % 200) - 100) / 10.0;
        }

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(A.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(B.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(C_interp.data())),
            RuntimeValue::from_i64(N)
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        auto fn_ptr = jit.get_function_ptr<double(*)(double*, double*, double*, int64_t)>(fn_name);
        REQUIRE(fn_ptr != nullptr);
        double jit_res = fn_ptr(A.data(), B.data(), C_jit.data(), N);

        double d_interp = interp_res.as_f64();
        CHECK(std::abs(d_interp - jit_res) < 1e-4);

        for (size_t i = 0; i < total_elements; ++i) {
            CHECK(std::abs(C_interp[i] - C_jit[i]) < 1e-4);
        }
    }
}

TEST_CASE("Differential Fuzzer - Dual Kernel MatMul Equivalence across Odd Dimensions") {
    for (int64_t N : {1, 2, 3, 5, 7, 9, 11, 13, 17, 23, 31, 33}) {
        size_t total_elements = static_cast<size_t>(N * N);
        std::vector<double> A(total_elements);
        std::vector<double> B(total_elements);
        std::vector<double> C_naive(total_elements, 0.0);
        std::vector<double> C_preopt(total_elements, 0.0);

        for (size_t i = 0; i < total_elements; ++i) {
            A[i] = static_cast<double>((i * 7 + 3) % 29) * 0.25;
            B[i] = static_cast<double>((i * 11 + 5) % 31) * 0.25;
        }

        auto mod_naive = bench::build_matmul_f64_naive_module();
        codegen::JitExecutionEngine jit_naive(Target::host());
        REQUIRE(jit_naive.compile_and_load(*mod_naive));
        auto fn_naive = jit_naive.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_naive");
        REQUIRE(fn_naive != nullptr);
        fn_naive(A.data(), B.data(), C_naive.data(), N);

        auto mod_preopt = bench::build_matmul_f64_preopt_module();
        codegen::JitExecutionEngine jit_preopt(Target::host());
        REQUIRE(jit_preopt.compile_and_load(*mod_preopt));
        auto fn_preopt = jit_preopt.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_preopt");
        REQUIRE(fn_preopt != nullptr);
        fn_preopt(A.data(), B.data(), C_preopt.data(), N);

        for (size_t i = 0; i < total_elements; ++i) {
            CHECK(std::abs(C_naive[i] - C_preopt[i]) < 1e-6);
        }
    }
}

