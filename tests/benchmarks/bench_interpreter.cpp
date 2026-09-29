#include "bench_interpreter.hpp"
#include "bench_numeric_modules.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
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

// Fast-interpreter benchmarks. The ratcheted number (interp_<name>, lower is
// better) is the fast interpreter's time per op divided by the baseline
// JIT's on the same MIR, taken per repetition, then the median. The baseline
// JIT's code lives in JIT memory at a placement brass controls, so the ratio
// holds within a few percent across rebuilds of this binary; against native
// C++ it moved 10-13% on a relink that changed nothing measured, because the
// native loop's alignment in the executable moved. (Re-baseline the interp_
// keys when the baseline JIT's codegen changes.) Native and the reference
// interpreter (oracle) are printed for context and gate nothing; the oracle
// is timed on a fraction of the iterations because it is two orders of
// magnitude slower.

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

// ============================================================================
// 1. Native C++ Baselines (inputs arrive opaque so nothing constant-folds)
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

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
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
// 3. Measurement
// ============================================================================

struct InterpMetric {
    std::string key;
    std::string name;
    size_t iterations = 0;
    double oracle_ns_op = 0.0;
    TimingStats fast_ms;
    TimingStats jit_ms;
    TimingStats native_ms;
    TimingStats fast_vs_native; // context
    TimingStats fast_vs_jit;    // ratcheted
};

// Interleaves the three tiers inside each repetition so drift (clocks,
// another process waking up) lands on all of them, and takes each ratio per
// repetition before the median.
template <typename FFast, typename FJit, typename FNative>
void measure_tiers(InterpMetric& m, size_t reps, FFast&& fast, FJit&& jit, FNative&& native) {
    if (is_debug_build()) reps = 1;
    fast();
    jit();
    native();
    std::vector<double> f, j, n, fn, fj;
    Stopwatch sw;
    for (size_t r = 0; r < reps; ++r) {
        sw.start(); fast();   const double f_ms = sw.stop_ms();
        sw.start(); jit();    const double j_ms = sw.stop_ms();
        sw.start(); native(); const double n_ms = sw.stop_ms();
        f.push_back(f_ms);
        j.push_back(j_ms);
        n.push_back(n_ms);
        fn.push_back(n_ms > 0.0 ? f_ms / n_ms : 0.0);
        fj.push_back(j_ms > 0.0 ? f_ms / j_ms : 0.0);
    }
    m.fast_ms = TimingStats(std::move(f));
    m.jit_ms = TimingStats(std::move(j));
    m.native_ms = TimingStats(std::move(n));
    m.fast_vs_native = TimingStats(std::move(fn));
    m.fast_vs_jit = TimingStats(std::move(fj));
}

template <typename FOracle>
double time_oracle_ns_op(size_t oracle_iters, FOracle&& oracle_once) {
    oracle_once();
    Stopwatch sw;
    sw.start();
    for (size_t i = 0; i < oracle_iters; ++i) oracle_once();
    return sw.stop_ms() * 1e6 / static_cast<double>(oracle_iters);
}

const BaselineCompiledFunction* find_compiled(const std::vector<BaselineCompiledFunction>& fns, const char* name) {
    for (const auto& cf : fns) {
        if (cf.name() == name) return &cf;
    }
    std::cerr << "FATAL: baseline JIT did not produce " << name << "\n";
    std::abort();
}

constexpr size_t kInterpRepetitions = 9;

// Runs one benchmark: `args` is the call's argument list, `native_once` one
// native call. Every tier runs `iters` calls per repetition.
template <typename FNativeOnce>
InterpMetric run_one(const char* key, const char* name, Module& mod, const char* fn_name,
                     const std::vector<RuntimeValue>& args, size_t iters, size_t oracle_iters,
                     FNativeOnce&& native_once) {
    InterpMetric m;
    m.key = key;
    m.name = name;
    m.iterations = iters;

    runtime::FunctionDispatchTable oracle_program;
    Interpreter oracle;
    oracle.set_dispatch_table(&oracle_program);
    m.oracle_ns_op = time_oracle_ns_op(oracle_iters, [&] {
        RuntimeValue r = oracle.run(mod, fn_name, args);
        DoNotOptimize(r);
    });

    // A program of its own: the baseline JIT below registers its code by name
    // in the default program, and a FastInterpreter on that program would
    // call it instead of interpreting. An owned program's pipeline is not
    // running, so nothing tiers up (no OSR) either: this times the
    // interpreter alone.
    runtime::FunctionDispatchTable fast_program;
    FastInterpreter fast;
    fast.set_dispatch_table(&fast_program);
    BaselineJitCompiler jit_compiler;
    auto compiled_funcs = jit_compiler.compile_module(mod);
    const BaselineCompiledFunction* compiled = find_compiled(compiled_funcs, fn_name);

    measure_tiers(m, kInterpRepetitions,
        [&] {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = fast.run(mod, fn_name, args);
                DoNotOptimize(r);
            }
        },
        [&] {
            for (size_t i = 0; i < iters; ++i) {
                RuntimeValue r = compiled->invoke(args);
                DoNotOptimize(r);
            }
        },
        [&] {
            for (size_t i = 0; i < iters; ++i) native_once();
        });
    return m;
}

void print_header() {
    std::cout << "\n";
    std::cout << "==================================================================================================================================\n";
    std::cout << " Fast interpreter (ns/op, median of " << kInterpRepetitions << "): ratcheted as Fast/JIT; Oracle and native for context\n";
    std::cout << "==================================================================================================================================\n";
    std::cout << std::left  << std::setw(32) << " Benchmark"
              << std::right << std::setw(14) << "Oracle"
              << std::right << std::setw(14) << "Fast"
              << std::right << std::setw(14) << "JIT"
              << std::right << std::setw(14) << "Native"
              << std::right << std::setw(24) << "Fast/JIT [min-max]"
              << std::right << std::setw(14) << "Fast/Native"
              << std::right << std::setw(14) << "Oracle/Fast"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
}

void print_row(const InterpMetric& m) {
    const double it = static_cast<double>(m.iterations);
    auto ns = [&](const TimingStats& s) { return s.median * 1e6 / it; };
    const double fast_ns = ns(m.fast_ms);
    std::ostringstream fj;
    fj << std::fixed << std::setprecision(2) << m.fast_vs_jit.median << "x ["
       << m.fast_vs_jit.min << "-" << m.fast_vs_jit.max << "]";
    std::cout << std::left  << " " << std::setw(31) << m.name << std::fixed << std::setprecision(1)
              << std::right << std::setw(14) << m.oracle_ns_op
              << std::right << std::setw(14) << fast_ns
              << std::right << std::setw(14) << ns(m.jit_ms)
              << std::right << std::setw(14) << ns(m.native_ms)
              << std::right << std::setw(24) << fj.str()
              << std::right << std::setw(13) << m.fast_vs_native.median << "x"
              << std::right << std::setw(13) << (fast_ns > 0.0 ? m.oracle_ns_op / fast_ns : 0.0) << "x"
              << "\n";
}

} // namespace

namespace brass::bench {

void run_interpreter_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager&) {
    const bool dbg = is_debug_build();
    std::vector<InterpMetric> metrics;
    print_header();

    {
        const size_t iters = dbg ? 1000 : 25000;
        auto mod = build_fib_module();
        metrics.push_back(run_one("fib_iter_40", "Iterative Fibonacci (n=40)", *mod, "fib_iter",
            {RuntimeValue::from_i64(40)}, iters, iters / 10, [] {
                uint64_t n = 40;
                DoNotOptimize(n);
                uint64_t r = native_fib_iter(n);
                DoNotOptimize(r);
            }));
    }
    {
        const size_t iters = dbg ? 20 : 100;
        auto mod = build_collatz_module();
        metrics.push_back(run_one("collatz_1000", "Collatz Steps (n=1000)", *mod, "collatz_sum",
            {RuntimeValue::from_i64(1000)}, iters, dbg ? 2 : 10, [] {
                int64_t n = 1000;
                DoNotOptimize(n);
                int64_t r = native_collatz_sum(n);
                DoNotOptimize(r);
            }));
    }
    {
        const size_t iters = dbg ? 4 : 20;
        const int64_t limit = 100000;
        std::vector<int64_t> buf(static_cast<size_t>(limit), 0);
        std::vector<int64_t> native_buf(static_cast<size_t>(limit), 0);
        auto mod = build_sieve_module();
        metrics.push_back(run_one("prime_sieve_100k", "Prime Sieve (limit=100k)", *mod, "prime_sieve",
            {RuntimeValue::from_ptr(buf.data()), RuntimeValue::from_i64(limit)}, iters, dbg ? 1 : 4, [&] {
                int64_t lim = limit;
                DoNotOptimize(lim);
                int64_t r = native_prime_sieve(native_buf.data(), lim);
                DoNotOptimize(r);
            }));
    }
    {
        const size_t iters = dbg ? 4 : 40;
        auto mod = build_fib_rec_module();
        metrics.push_back(run_one("fib_rec_22", "Recursive Fibonacci (n=22)", *mod, "fib_rec",
            {RuntimeValue::from_i64(22)}, iters, dbg ? 1 : 4, [] {
                int64_t n = 22;
                DoNotOptimize(n);
                int64_t r = native_fib_rec(n);
                DoNotOptimize(r);
            }));
    }
    {
        const size_t iters = dbg ? 5 : 60;
        const int64_t N = 32;
        const size_t total = static_cast<size_t>(N * N);
        std::vector<int64_t> A(total), B(total), C(total, 0), NC(total, 0);
        for (size_t idx = 0; idx < total; ++idx) {
            A[idx] = static_cast<int64_t>((idx % 17) + 1);
            B[idx] = static_cast<int64_t>((idx % 13) + 1);
        }
        auto mod = build_matmul_i64_naive_module();
        metrics.push_back(run_one("matmul_32x32", "Matrix Multiply (32x32)", *mod, "matmul_i64_naive",
            {RuntimeValue::from_ptr(A.data()), RuntimeValue::from_ptr(B.data()),
             RuntimeValue::from_ptr(C.data()), RuntimeValue::from_i64(N)},
            iters, dbg ? 1 : 6, [&] {
                int64_t n = N;
                DoNotOptimize(n);
                native_matmul_i64(A.data(), B.data(), NC.data(), n);
                ClobberMemory();
            }));
    }

    for (const auto& m : metrics) print_row(m);
    std::cout << "==================================================================================================================================\n\n";

    for (const auto& m : metrics) {
        BenchmarkResult res;
        res.key = "interp_" + m.key;
        res.name = "FastInterpreter: " + m.name;
        res.iterations = m.iterations;
        res.repetitions = m.fast_vs_jit.samples.size();
        res.native_ms = m.jit_ms.median;
        res.native_min_ms = m.jit_ms.min;
        res.native_max_ms = m.jit_ms.max;
        res.brass_ms = m.fast_ms.median;
        res.brass_min_ms = m.fast_ms.min;
        res.brass_max_ms = m.fast_ms.max;
        res.ratio = m.fast_vs_jit.median;
        res.ratio_min = m.fast_vs_jit.min;
        res.ratio_max = m.fast_vs_jit.max;
        std::ostringstream notes;
        notes << std::fixed << std::setprecision(1) << "fast/native " << m.fast_vs_native.median << "x";
        res.notes = notes.str();
        results.push_back(res);
    }
}

} // namespace brass::bench
