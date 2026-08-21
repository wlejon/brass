#include "bench_numeric.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <vector>
#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;
using namespace brass::x64;

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

// ============================================================================
// 2. Brass MIR Generators
// ============================================================================

std::unique_ptr<Module> build_fib_module() {
    auto mod = std::make_unique<Module>("bench_fib");
    Builder b(*mod);

    Function* fn = mod->create_function("fib_iter", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* base_case = b.create_block("base_case");
    BasicBlock* loop_init = b.create_block("loop_init");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* c2 = b.build_iconst_i64(2);
    Value* is_small = b.build_slt(n, c2);
    b.build_br_if(is_small, base_case, {}, loop_init, {});

    fn->append_block(base_case);
    b.position_at_end(base_case);
    b.build_ret(n);

    fn->append_block(loop_init);
    b.position_at_end(loop_init);
    Value* i_init = b.build_iconst_i64(2);
    Value* a_init = b.build_iconst_i64(0);
    Value* b_init = b.build_iconst_i64(1);
    b.build_br(loop_body, {i_init, a_init, b_init});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* i = b.add_block_param(loop_body, Type::i64());
    Value* a = b.add_block_param(loop_body, Type::i64());
    Value* cur_b = b.add_block_param(loop_body, Type::i64());
    Value* c = b.build_add(a, cur_b);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    Value* in_range = b.build_sle(next_i, n);
    b.build_br_if(in_range, loop_body, {next_i, cur_b, c}, exit_bb, {c});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_sieve_module() {
    auto mod = std::make_unique<Module>("bench_sieve");
    Builder b(*mod);

    Function* fn = mod->create_function("prime_sieve", Type::i64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* buf = b.add_block_param(entry, Type::ptr());
    Value* limit = b.add_block_param(entry, Type::i64());

    BasicBlock* fill_body = b.create_block("fill_body");
    BasicBlock* fill_done = b.create_block("fill_done");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(fill_body, {zero});

    fn->append_block(fill_body);
    b.position_at_end(fill_body);
    Value* fill_i = b.add_block_param(fill_body, Type::i64());
    b.build_store_indexed(Type::i64(), buf, fill_i, 8, 0, one);
    Value* next_fill_i = b.build_add(fill_i, one);
    Value* fill_cond = b.build_slt(next_fill_i, limit);
    b.build_br_if(fill_cond, fill_body, {next_fill_i}, fill_done, {});

    fn->append_block(fill_done);
    b.position_at_end(fill_done);
    b.build_store_indexed(Type::i64(), buf, zero, 8, 0, zero);
    b.build_store_indexed(Type::i64(), buf, one, 8, 0, zero);

    BasicBlock* outer_hdr = b.create_block("outer_hdr");
    BasicBlock* outer_body = b.create_block("outer_body");
    BasicBlock* inner_init = b.create_block("inner_init");
    BasicBlock* inner_body = b.create_block("inner_body");
    BasicBlock* next_p_bb = b.create_block("next_p");
    BasicBlock* count_init = b.create_block("count_init");

    Value* two = b.build_iconst_i64(2);
    b.build_br(outer_hdr, {two});

    fn->append_block(outer_hdr);
    b.position_at_end(outer_hdr);
    Value* p = b.add_block_param(outer_hdr, Type::i64());
    Value* p_sq = b.build_mul(p, p);
    Value* outer_cond = b.build_slt(p_sq, limit);
    b.build_br_if(outer_cond, outer_body, {}, count_init, {});

    fn->append_block(outer_body);
    b.position_at_end(outer_body);
    Value* p_val = b.build_load_indexed(Type::i64(), buf, p, 8, 0);
    Value* is_p = b.build_ne(p_val, zero);
    b.build_br_if(is_p, inner_init, {}, next_p_bb, {});

    fn->append_block(inner_init);
    b.position_at_end(inner_init);
    b.build_br(inner_body, {p_sq});

    fn->append_block(inner_body);
    b.position_at_end(inner_body);
    Value* k = b.add_block_param(inner_body, Type::i64());
    b.build_store_indexed(Type::i64(), buf, k, 8, 0, zero);
    Value* next_k = b.build_add(k, p);
    Value* in_k = b.build_slt(next_k, limit);
    b.build_br_if(in_k, inner_body, {next_k}, next_p_bb, {});

    fn->append_block(next_p_bb);
    b.position_at_end(next_p_bb);
    Value* next_p = b.build_add(p, one);
    b.build_br(outer_hdr, {next_p});

    BasicBlock* count_body = b.create_block("count_body");
    BasicBlock* exit_bb = b.create_block("exit");

    fn->append_block(count_init);
    b.position_at_end(count_init);
    b.build_br(count_body, {two, zero});

    fn->append_block(count_body);
    b.position_at_end(count_body);
    Value* ci = b.add_block_param(count_body, Type::i64());
    Value* acc = b.add_block_param(count_body, Type::i64());
    Value* is_p_val = b.build_load_indexed(Type::i64(), buf, ci, 8, 0);
    Value* next_acc = b.build_add(acc, is_p_val);
    Value* next_ci = b.build_add(ci, one);
    Value* c_cond = b.build_slt(next_ci, limit);
    b.build_br_if(c_cond, count_body, {next_ci, next_acc}, exit_bb, {next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_count = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_count);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_collatz_module() {
    auto mod = std::make_unique<Module>("bench_collatz");
    Builder b(*mod);

    Function* fn = mod->create_function("collatz_sum", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* max_n = b.add_block_param(entry, Type::i64());

    BasicBlock* outer_hdr = b.create_block("outer_hdr");
    BasicBlock* outer_body = b.create_block("outer_body");
    BasicBlock* inner_hdr = b.create_block("inner_hdr");
    BasicBlock* inner_check = b.create_block("inner_check");
    BasicBlock* inner_even = b.create_block("inner_even");
    BasicBlock* inner_odd = b.create_block("inner_odd");
    BasicBlock* outer_next = b.create_block("outer_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* one = b.build_iconst_i64(1);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(outer_hdr, {one, zero});

    fn->append_block(outer_hdr);
    b.position_at_end(outer_hdr);
    Value* i = b.add_block_param(outer_hdr, Type::i64());
    Value* total_steps = b.add_block_param(outer_hdr, Type::i64());
    Value* in_range = b.build_sle(i, max_n);
    b.build_br_if(in_range, outer_body, {}, exit_bb, {total_steps});

    fn->append_block(outer_body);
    b.position_at_end(outer_body);
    b.build_br(inner_hdr, {i, zero});

    fn->append_block(inner_hdr);
    b.position_at_end(inner_hdr);
    Value* cur_n = b.add_block_param(inner_hdr, Type::i64());
    Value* cur_steps = b.add_block_param(inner_hdr, Type::i64());
    Value* is_done = b.build_sle(cur_n, one);
    b.build_br_if(is_done, outer_next, {}, inner_check, {});

    fn->append_block(inner_check);
    b.position_at_end(inner_check);
    Value* rem = b.build_and(cur_n, one);
    Value* is_even = b.build_eq(rem, zero);
    b.build_br_if(is_even, inner_even, {}, inner_odd, {});

    fn->append_block(inner_even);
    b.position_at_end(inner_even);
    Value* half = b.build_ashr(cur_n, one);
    Value* next_steps_even = b.build_add(cur_steps, one);
    b.build_br(inner_hdr, {half, next_steps_even});

    fn->append_block(inner_odd);
    b.position_at_end(inner_odd);
    Value* three = b.build_iconst_i64(3);
    Value* odd_next = b.build_add(b.build_mul(cur_n, three), one);
    Value* next_steps_odd = b.build_add(cur_steps, one);
    b.build_br(inner_hdr, {odd_next, next_steps_odd});

    fn->append_block(outer_next);
    b.position_at_end(outer_next);
    Value* next_tot = b.build_add(total_steps, cur_steps);
    Value* next_i = b.build_add(i, one);
    b.build_br(outer_hdr, {next_i, next_tot});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_steps = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_steps);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_matmul_i64_module() {
    auto mod = std::make_unique<Module>("bench_matmul_i64");
    Builder b(*mod);

    Function* fn = mod->create_function("matmul_i64", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* A = b.add_block_param(entry, Type::ptr());
    Value* B = b.add_block_param(entry, Type::ptr());
    Value* C = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_i_hdr = b.create_block("loop_i_hdr");
    BasicBlock* loop_i_body = b.create_block("loop_i_body");
    BasicBlock* loop_j_hdr = b.create_block("loop_j_hdr");
    BasicBlock* loop_j_body = b.create_block("loop_j_body");
    BasicBlock* loop_k_hdr = b.create_block("loop_k_hdr");
    BasicBlock* loop_k_body = b.create_block("loop_k_body");
    BasicBlock* loop_j_next = b.create_block("loop_j_next");
    BasicBlock* loop_i_next = b.create_block("loop_i_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_i_hdr, {zero});

    fn->append_block(loop_i_hdr);
    b.position_at_end(loop_i_hdr);
    Value* i = b.add_block_param(loop_i_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, loop_i_body, {}, exit_bb, {});

    fn->append_block(loop_i_body);
    b.position_at_end(loop_i_body);
    Value* row_a = b.build_mul(i, N);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    b.build_br(loop_k_hdr, {zero, zero, zero});

    fn->append_block(loop_k_hdr);
    b.position_at_end(loop_k_hdr);
    Value* k = b.add_block_param(loop_k_hdr, Type::i64());
    Value* kN = b.add_block_param(loop_k_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_k_hdr, Type::i64());
    Value* cond_k = b.build_slt(k, N);
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {});

    fn->append_block(loop_k_body);
    b.position_at_end(loop_k_body);
    Value* idx_a = b.build_add(row_a, k);
    Value* val_a = b.build_load_indexed(Type::i64(), A, idx_a, 8, 0);

    Value* idx_b = b.build_add(kN, j);
    Value* val_b = b.build_load_indexed(Type::i64(), B, idx_b, 8, 0);

    Value* term = b.build_mul(val_a, val_b);
    Value* next_sum = b.build_add(sum, term);
    Value* next_kN = b.build_add(kN, N);
    Value* next_k = b.build_add(k, one);
    b.build_br(loop_k_hdr, {next_k, next_kN, next_sum});

    fn->append_block(loop_j_next);
    b.position_at_end(loop_j_next);
    Value* idx_c = b.build_add(row_a, j);
    b.build_store_indexed(Type::i64(), C, idx_c, 8, 0, sum);
    Value* next_j = b.build_add(j, one);
    b.build_br(loop_j_hdr, {next_j});

    fn->append_block(loop_i_next);
    b.position_at_end(loop_i_next);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_i_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(nullptr);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_matmul_f64_module() {
    auto mod = std::make_unique<Module>("bench_matmul_f64");
    Builder b(*mod);

    Function* fn = mod->create_function("matmul_f64", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* A = b.add_block_param(entry, Type::ptr());
    Value* B = b.add_block_param(entry, Type::ptr());
    Value* C = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_i_hdr = b.create_block("loop_i_hdr");
    BasicBlock* loop_i_body = b.create_block("loop_i_body");
    BasicBlock* loop_j_hdr = b.create_block("loop_j_hdr");
    BasicBlock* loop_j_body = b.create_block("loop_j_body");
    BasicBlock* loop_k_hdr = b.create_block("loop_k_hdr");
    BasicBlock* loop_k_body = b.create_block("loop_k_body");
    BasicBlock* loop_j_next = b.create_block("loop_j_next");
    BasicBlock* loop_i_next = b.create_block("loop_i_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* zero_f = b.build_fconst_f64(0.0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_i_hdr, {zero});

    fn->append_block(loop_i_hdr);
    b.position_at_end(loop_i_hdr);
    Value* i = b.add_block_param(loop_i_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, loop_i_body, {}, exit_bb, {});

    fn->append_block(loop_i_body);
    b.position_at_end(loop_i_body);
    Value* row_a = b.build_mul(i, N);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    b.build_br(loop_k_hdr, {zero, zero, zero_f});

    fn->append_block(loop_k_hdr);
    b.position_at_end(loop_k_hdr);
    Value* k = b.add_block_param(loop_k_hdr, Type::i64());
    Value* kN = b.add_block_param(loop_k_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_k_hdr, Type::f64());
    Value* cond_k = b.build_slt(k, N);
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {});

    fn->append_block(loop_k_body);
    b.position_at_end(loop_k_body);
    Value* idx_a = b.build_add(row_a, k);
    Value* val_a = b.build_load_indexed(Type::f64(), A, idx_a, 8, 0);

    Value* idx_b = b.build_add(kN, j);
    Value* val_b = b.build_load_indexed(Type::f64(), B, idx_b, 8, 0);

    Value* term = b.build_mul(val_a, val_b);
    Value* next_sum = b.build_add(sum, term);
    Value* next_kN = b.build_add(kN, N);
    Value* next_k = b.build_add(k, one);
    b.build_br(loop_k_hdr, {next_k, next_kN, next_sum});

    fn->append_block(loop_j_next);
    b.position_at_end(loop_j_next);
    Value* idx_c = b.build_add(row_a, j);
    b.build_store_indexed(Type::f64(), C, idx_c, 8, 0, sum);
    Value* next_j = b.build_add(j, one);
    b.build_br(loop_j_hdr, {next_j});

    fn->append_block(loop_i_next);
    b.position_at_end(loop_i_next);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_i_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(nullptr);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_list_module() {
    auto mod = std::make_unique<Module>("bench_list");
    Builder b(*mod);

    Function* fn = mod->create_function("list_traversal", Type::i64(), {Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* head = b.add_block_param(entry, Type::ptr());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {head, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* cur = b.add_block_param(loop_hdr, Type::ptr());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* has_node = b.build_ne(cur, zero);
    b.build_br_if(has_node, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* val = b.build_load(Type::i64(), cur, 0);
    Value* next_ptr = b.build_load(Type::ptr(), cur, 8);
    Value* next_acc = b.build_add(acc, val);
    b.build_br(loop_hdr, {next_ptr, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

namespace brass::bench {

void run_numeric_benchmarks(std::vector<BenchmarkResult>& results) {
    Stopwatch sw;

    // 1. Iterative Fibonacci
    {
        size_t iters = 100000;
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
        results.push_back({"Iterative Fibonacci (N=45)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
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
        results.push_back({"Prime Sieve (N=100k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
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
        results.push_back({"Collatz Sum (1..100k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // 4. Matrix Multiplication 32x32 Integer
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_jit(N * N, 0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

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
            std::cerr << "FATAL: MatMul 32x32 (i64) result mismatch between native and JIT!\n";
            std::abort();
        }

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 32x32 (i64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // 5. Matrix Multiplication 64x64 Integer
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_jit(N * N, 0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_i64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

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

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 64x64 (i64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // 6. Matrix Multiplication 32x32 Float
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_jit(N * N, 0.0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

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

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 32x32 (f64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // 7. Matrix Multiplication 64x64 Float
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_jit(N * N, 0.0);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            int64_t size_n = N;
            DoNotOptimize(size_n);
            native_matmul_f64(A.data(), B.data(), C_native.data(), size_n);
            DoNotOptimize(C_native.data());
        }
        double native_ms = sw.stop_ms();

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

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 64x64 (f64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
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
        results.push_back({"Linked List Traversal (50k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }
}

} // namespace brass::bench
