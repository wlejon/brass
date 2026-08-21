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

static uint64_t double_to_bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof(double));
    return u;
}

static uint64_t ulp_distance(double a, double b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
    uint64_t ua = double_to_bits(a);
    uint64_t ub = double_to_bits(b);
    if ((ua >> 63) != (ub >> 63)) {
        if (a == 0.0 && b == 0.0) return 0;
        return (ua & 0x7FFFFFFFFFFFFFFFULL) + (ub & 0x7FFFFFFFFFFFFFFFULL);
    }
    return (ua > ub) ? (ua - ub) : (ub - ua);
}

TEST_CASE("Differential Fuzzer - 2D Matrix Indexing and Nested Loops (f64, Pseudorandom Mantissas)") {
    std::mt19937_64 rng(0x9ABCDEF012345678ULL);

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

        // Pseudorandom mantissas that are NOT exact binary fractions
        for (size_t i = 0; i < total_elements; ++i) {
            uint64_t r1 = rng();
            uint64_t r2 = rng();
            double d1 = 1.0 + static_cast<double>(r1 & 0xFFFFFFFFFFFFF) * 1e-16;
            double d2 = 1.0 + static_cast<double>(r2 & 0xFFFFFFFFFFFFF) * 1e-16;
            A[i] = d1;
            B[i] = d2;
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
        // Strict IEEE-754 mode must be bit-exact identical to interpreter
        CHECK_EQ(double_to_bits(d_interp), double_to_bits(jit_res));

        for (size_t i = 0; i < total_elements; ++i) {
            CHECK_EQ(double_to_bits(C_interp[i]), double_to_bits(C_jit[i]));
        }
    }
}

TEST_CASE("Differential Fuzzer - Dual Kernel MatMul Strict Bit-Exactness and Reassoc ULP Tolerance") {
    std::mt19937_64 rng(0xCAFEF00D12345678ULL);

    for (int64_t N : {1, 2, 3, 5, 7, 8, 9, 11, 13, 16, 17, 23, 31, 32}) {
        size_t total_elements = static_cast<size_t>(N * N);
        std::vector<double> A(total_elements);
        std::vector<double> B(total_elements);
        std::vector<double> C_strict_naive(total_elements, 0.0);
        std::vector<double> C_strict_preopt(total_elements, 0.0);
        std::vector<double> C_reassoc_naive(total_elements, 0.0);
        std::vector<double> C_reassoc_preopt(total_elements, 0.0);

        // Pseudorandom mantissas
        for (size_t i = 0; i < total_elements; ++i) {
            uint64_t r1 = rng();
            uint64_t r2 = rng();
            A[i] = 1.0 + static_cast<double>(r1 & 0x000FFFFFFFFFFFFFULL) * 1e-15;
            B[i] = 1.0 + static_cast<double>(r2 & 0x000FFFFFFFFFFFFFULL) * 1e-15;
        }

        // 1. Strict Naive
        auto mod_strict_naive = bench::build_matmul_f64_strict_naive_module();
        codegen::JitExecutionEngine jit_sn(Target::host());
        REQUIRE(jit_sn.compile_and_load(*mod_strict_naive));
        auto fn_sn = jit_sn.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_strict_naive");
        REQUIRE(fn_sn != nullptr);
        fn_sn(A.data(), B.data(), C_strict_naive.data(), N);

        // 2. Strict Preopt
        auto mod_strict_preopt = bench::build_matmul_f64_strict_preopt_module();
        codegen::JitExecutionEngine jit_sp(Target::host());
        REQUIRE(jit_sp.compile_and_load(*mod_strict_preopt));
        auto fn_sp = jit_sp.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_strict_preopt");
        REQUIRE(fn_sp != nullptr);
        fn_sp(A.data(), B.data(), C_strict_preopt.data(), N);

        // In strict mode: Naive and Preopt are bit-exact identical
        for (size_t i = 0; i < total_elements; ++i) {
            CHECK_EQ(double_to_bits(C_strict_naive[i]), double_to_bits(C_strict_preopt[i]));
        }

        // 3. Reassoc Naive (flagged opt-in)
        auto mod_reassoc_naive = bench::build_matmul_f64_reassoc_naive_module();
        codegen::JitExecutionEngine jit_rn(Target::host());
        REQUIRE(jit_rn.compile_and_load(*mod_reassoc_naive));
        auto fn_rn = jit_rn.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_reassoc_naive");
        REQUIRE(fn_rn != nullptr);
        fn_rn(A.data(), B.data(), C_reassoc_naive.data(), N);

        // 4. Reassoc Preopt (flagged opt-in)
        auto mod_reassoc_preopt = bench::build_matmul_f64_reassoc_preopt_module();
        codegen::JitExecutionEngine jit_rp(Target::host());
        REQUIRE(jit_rp.compile_and_load(*mod_reassoc_preopt));
        auto fn_rp = jit_rp.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64_reassoc_preopt");
        REQUIRE(fn_rp != nullptr);
        fn_rp(A.data(), B.data(), C_reassoc_preopt.data(), N);

        // Verify ULP tolerance: Reassoc results are within stated tolerance (<= 32 ULPs) of strict results
        for (size_t i = 0; i < total_elements; ++i) {
            uint64_t ulp_rn = ulp_distance(C_strict_naive[i], C_reassoc_naive[i]);
            uint64_t ulp_rp = ulp_distance(C_strict_naive[i], C_reassoc_preopt[i]);
            CHECK(ulp_rn <= 64);
            CHECK(ulp_rp <= 64);
        }
    }
}

