#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <vector>
#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

// Subroutine for GC benchmark
extern "C" int64_t leaf_gc_subroutine(int64_t x) {
    return (x * 1664525 + 1013904223) & 0x7FFFFFFF;
}

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

// Shadow-stack simulation of live GC references
int64_t shadow_stack_gc_runner(
    void* r0, void* r1, void* r2, void* r3,
    int64_t iters, ThreadShadowStack& ss
) {
    int64_t acc = 0;
    for (int64_t i = 0; i < iters; ++i) {
        ShadowStackFrame frame;
        frame.roots[0] = r0;
        frame.roots[1] = r1;
        frame.roots[2] = r2;
        frame.roots[3] = r3;
        ss.push(&frame, 4);

        int64_t val0 = *reinterpret_cast<int64_t*>(frame.roots[0]);
        int64_t val1 = *reinterpret_cast<int64_t*>(frame.roots[1]);
        int64_t val2 = *reinterpret_cast<int64_t*>(frame.roots[2]);
        int64_t val3 = *reinterpret_cast<int64_t*>(frame.roots[3]);

        acc += leaf_gc_subroutine(val0 + val1 + val2 + val3 + acc);

        ss.pop();
    }
    return acc;
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
    BasicBlock* loop_hdr = b.create_block("loop_hdr");
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
    b.build_br(loop_hdr, {i_init, a_init, b_init});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* a = b.add_block_param(loop_hdr, Type::i64());
    Value* cur_b = b.add_block_param(loop_hdr, Type::i64());
    Value* in_range = b.build_sle(i, n);
    b.build_br_if(in_range, loop_body, {}, exit_bb, {cur_b});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* sum = b.build_add(a, cur_b);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, cur_b, sum});

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

    // 1. Fill buffer with 1s
    BasicBlock* fill_hdr = b.create_block("fill_hdr");
    BasicBlock* fill_body = b.create_block("fill_body");
    BasicBlock* fill_done = b.create_block("fill_done");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(fill_hdr, {zero});

    fn->append_block(fill_hdr);
    b.position_at_end(fill_hdr);
    Value* fill_i = b.add_block_param(fill_hdr, Type::i64());
    Value* fill_cond = b.build_slt(fill_i, limit);
    b.build_br_if(fill_cond, fill_body, {}, fill_done, {});

    fn->append_block(fill_body);
    b.position_at_end(fill_body);
    b.build_store_indexed(Type::i64(), buf, fill_i, 8, 0, one);
    Value* next_fill_i = b.build_add(fill_i, one);
    b.build_br(fill_hdr, {next_fill_i});

    fn->append_block(fill_done);
    b.position_at_end(fill_done);
    b.build_store_indexed(Type::i64(), buf, zero, 8, 0, zero);
    b.build_store_indexed(Type::i64(), buf, one, 8, 0, zero);

    // 2. Outer sieve loop
    BasicBlock* outer_hdr = b.create_block("outer_hdr");
    BasicBlock* outer_body = b.create_block("outer_body");
    BasicBlock* inner_init = b.create_block("inner_init");
    BasicBlock* inner_hdr = b.create_block("inner_hdr");
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
    b.build_br(inner_hdr, {p_sq});

    fn->append_block(inner_hdr);
    b.position_at_end(inner_hdr);
    Value* k = b.add_block_param(inner_hdr, Type::i64());
    Value* in_k = b.build_slt(k, limit);
    b.build_br_if(in_k, inner_body, {}, next_p_bb, {});

    fn->append_block(inner_body);
    b.position_at_end(inner_body);
    b.build_store_indexed(Type::i64(), buf, k, 8, 0, zero);
    Value* next_k = b.build_add(k, p);
    b.build_br(inner_hdr, {next_k});

    fn->append_block(next_p_bb);
    b.position_at_end(next_p_bb);
    Value* next_p = b.build_add(p, one);
    b.build_br(outer_hdr, {next_p});

    // 3. Count primes
    BasicBlock* count_hdr = b.create_block("count_hdr");
    BasicBlock* count_body = b.create_block("count_body");
    BasicBlock* exit_bb = b.create_block("exit");

    fn->append_block(count_init);
    b.position_at_end(count_init);
    b.build_br(count_hdr, {two, zero});

    fn->append_block(count_hdr);
    b.position_at_end(count_hdr);
    Value* ci = b.add_block_param(count_hdr, Type::i64());
    Value* acc = b.add_block_param(count_hdr, Type::i64());
    Value* c_cond = b.build_slt(ci, limit);
    b.build_br_if(c_cond, count_body, {}, exit_bb, {acc});

    fn->append_block(count_body);
    b.position_at_end(count_body);
    Value* is_p_val = b.build_load_indexed(Type::i64(), buf, ci, 8, 0);
    Value* next_acc = b.build_add(acc, is_p_val);
    Value* next_ci = b.build_add(ci, one);
    b.build_br(count_hdr, {next_ci, next_acc});

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
    BasicBlock* inner_even = b.create_block("inner_even");
    BasicBlock* inner_odd = b.create_block("inner_odd");
    BasicBlock* inner_step = b.create_block("inner_step");
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
    b.build_br_if(is_done, outer_next, {}, inner_even, {});

    fn->append_block(inner_even);
    b.position_at_end(inner_even);
    Value* rem = b.build_and(cur_n, one);
    Value* is_even = b.build_eq(rem, zero);
    Value* half = b.build_ashr(cur_n, one);
    b.build_br_if(is_even, inner_step, {half, cur_steps},
                          inner_odd, {});

    fn->append_block(inner_odd);
    b.position_at_end(inner_odd);
    Value* three = b.build_iconst_i64(3);
    Value* odd_next = b.build_add(b.build_mul(cur_n, three), one);
    b.build_br(inner_step, {odd_next, cur_steps});

    fn->append_block(inner_step);
    b.position_at_end(inner_step);
    Value* next_n = b.add_block_param(inner_step, Type::i64());
    Value* step_accum = b.add_block_param(inner_step, Type::i64());
    Value* next_steps = b.build_add(step_accum, one);
    b.build_br(inner_hdr, {next_n, next_steps});

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
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    b.build_br(loop_k_hdr, {zero, zero});

    fn->append_block(loop_k_hdr);
    b.position_at_end(loop_k_hdr);
    Value* k = b.add_block_param(loop_k_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_k_hdr, Type::i64());
    Value* cond_k = b.build_slt(k, N);
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {});

    fn->append_block(loop_k_body);
    b.position_at_end(loop_k_body);
    Value* idx_a = b.build_add(b.build_mul(i, N), k);
    Value* val_a = b.build_load_indexed(Type::i64(), A, idx_a, 8, 0);

    Value* idx_b = b.build_add(b.build_mul(k, N), j);
    Value* val_b = b.build_load_indexed(Type::i64(), B, idx_b, 8, 0);

    Value* term = b.build_mul(val_a, val_b);
    Value* next_sum = b.build_add(sum, term);
    Value* next_k = b.build_add(k, one);
    b.build_br(loop_k_hdr, {next_k, next_sum});

    fn->append_block(loop_j_next);
    b.position_at_end(loop_j_next);
    Value* idx_c = b.build_add(b.build_mul(i, N), j);
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
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    b.build_br(loop_k_hdr, {zero, zero_f});

    fn->append_block(loop_k_hdr);
    b.position_at_end(loop_k_hdr);
    Value* k = b.add_block_param(loop_k_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_k_hdr, Type::f64());
    Value* cond_k = b.build_slt(k, N);
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {});

    fn->append_block(loop_k_body);
    b.position_at_end(loop_k_body);
    Value* idx_a = b.build_add(b.build_mul(i, N), k);
    Value* val_a = b.build_load_indexed(Type::f64(), A, idx_a, 8, 0);

    Value* idx_b = b.build_add(b.build_mul(k, N), j);
    Value* val_b = b.build_load_indexed(Type::f64(), B, idx_b, 8, 0);

    Value* term = b.build_mul(val_a, val_b);
    Value* next_sum = b.build_add(sum, term);
    Value* next_k = b.build_add(k, one);
    b.build_br(loop_k_hdr, {next_k, next_sum});

    fn->append_block(loop_j_next);
    b.position_at_end(loop_j_next);
    Value* idx_c = b.build_add(b.build_mul(i, N), j);
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

std::unique_ptr<Module> build_gc_benchmark_module() {
    auto mod = std::make_unique<Module>("bench_gc_model");
    mod->add_external_symbol("leaf_gc_subroutine");
    Builder b(*mod);

    Function* fn = mod->create_function("brass_gc_caller", Type::i64(), {
        Type::gcref(), Type::gcref(), Type::gcref(), Type::gcref(), Type::i64()
    });
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* r0 = b.add_block_param(entry, Type::gcref());
    Value* r1 = b.add_block_param(entry, Type::gcref());
    Value* r2 = b.add_block_param(entry, Type::gcref());
    Value* r3 = b.add_block_param(entry, Type::gcref());
    Value* iters = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, iters);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* v0 = b.build_load(Type::i64(), r0, 0);
    Value* v1 = b.build_load(Type::i64(), r1, 0);
    Value* v2 = b.build_load(Type::i64(), r2, 0);
    Value* v3 = b.build_load(Type::i64(), r3, 0);

    Value* sum4 = b.build_add(b.build_add(v0, v1), b.build_add(v2, v3));
    Value* arg = b.build_add(sum4, acc);

    // Call subroutine with live gcrefs
    Value* sub_res = b.build_call("leaf_gc_subroutine", Type::i64(), {arg});
    Value* next_acc = b.build_add(acc, sub_res);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    BenchmarkReporter::print_header("Brass JIT vs Native C++ (-O2) & GC Performance");

    Stopwatch sw;
    std::vector<BenchmarkResult> results;

    // ------------------------------------------------------------------------
    // Benchmark 1: Iterative Fibonacci (fib(45) x 100,000)
    // ------------------------------------------------------------------------
    {
        size_t iters = 100000;
        uint64_t n = 45;

        // Native C++
        sw.start();
        volatile uint64_t native_sink = 0;
        for (size_t i = 0; i < iters; ++i) {
            native_sink = native_fib_iter(n);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_fib_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto fib_fn = jit.get_function_ptr<uint64_t(*)(uint64_t)>("fib_iter");
        assert(fib_fn != nullptr);

        sw.start();
        volatile uint64_t jit_sink = 0;
        for (size_t i = 0; i < iters; ++i) {
            jit_sink = fib_fn(n);
        }
        double brass_ms = sw.stop_ms();

        assert(native_sink == jit_sink);
        double ratio = brass_ms / native_ms;
        results.push_back({"Iterative Fibonacci (N=45)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 2: Prime Sieve (Eratosthenes N=100,000 x 500)
    // ------------------------------------------------------------------------
    {
        size_t iters = 500;
        int64_t limit = 100000;
        std::vector<int64_t> native_buf(limit, 0);
        std::vector<int64_t> jit_buf(limit, 0);

        // Native C++
        sw.start();
        int64_t native_count = 0;
        for (size_t i = 0; i < iters; ++i) {
            native_count = native_prime_sieve(native_buf.data(), limit);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_sieve_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto sieve_fn = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t)>("prime_sieve");
        assert(sieve_fn != nullptr);

        sw.start();
        int64_t jit_count = 0;
        for (size_t i = 0; i < iters; ++i) {
            jit_count = sieve_fn(jit_buf.data(), limit);
        }
        double brass_ms = sw.stop_ms();

        assert(native_count == jit_count);
        double ratio = brass_ms / native_ms;
        results.push_back({"Prime Sieve (N=100k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 3: Collatz Conjecture Sum (1..100,000 x 50)
    // ------------------------------------------------------------------------
    {
        size_t iters = 50;
        int64_t max_n = 100000;

        // Native C++
        sw.start();
        int64_t native_steps = 0;
        for (size_t i = 0; i < iters; ++i) {
            native_steps = native_collatz_sum(max_n);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_collatz_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto collatz_fn = jit.get_function_ptr<int64_t(*)(int64_t)>("collatz_sum");
        assert(collatz_fn != nullptr);

        sw.start();
        int64_t jit_steps = 0;
        for (size_t i = 0; i < iters; ++i) {
            jit_steps = collatz_fn(max_n);
        }
        double brass_ms = sw.stop_ms();

        assert(native_steps == jit_steps);
        double ratio = brass_ms / native_ms;
        results.push_back({"Collatz Sum (1..100k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 4: Matrix Multiplication 32x32 Integer (i64 x 2,000)
    // ------------------------------------------------------------------------
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_jit(N * N, 0);

        // Native C++
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            native_matmul_i64(A.data(), B.data(), C_native.data(), N);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_matmul_i64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64");
        assert(matmul_fn != nullptr);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            matmul_fn(A.data(), B.data(), C_jit.data(), N);
        }
        double brass_ms = sw.stop_ms();

        assert(C_native == C_jit);
        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 32x32 (i64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 5: Matrix Multiplication 64x64 Integer (i64 x 250)
    // ------------------------------------------------------------------------
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<int64_t> A(N * N, 2), B(N * N, 3), C_native(N * N, 0), C_jit(N * N, 0);

        // Native C++
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            native_matmul_i64(A.data(), B.data(), C_native.data(), N);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_matmul_i64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const int64_t*, const int64_t*, int64_t*, int64_t)>("matmul_i64");
        assert(matmul_fn != nullptr);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            matmul_fn(A.data(), B.data(), C_jit.data(), N);
        }
        double brass_ms = sw.stop_ms();

        assert(C_native == C_jit);
        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 64x64 (i64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 6: Matrix Multiplication 32x32 Float (f64 x 2,000)
    // ------------------------------------------------------------------------
    {
        size_t iters = 2000;
        int64_t N = 32;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_jit(N * N, 0.0);

        // Native C++
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            native_matmul_f64(A.data(), B.data(), C_native.data(), N);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_matmul_f64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64");
        assert(matmul_fn != nullptr);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            matmul_fn(A.data(), B.data(), C_jit.data(), N);
        }
        double brass_ms = sw.stop_ms();

        double diff = 0.0;
        for (size_t i = 0; i < C_native.size(); ++i) diff += std::abs(C_native[i] - C_jit[i]);
        assert(diff < 1e-6);

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 32x32 (f64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 7: Matrix Multiplication 64x64 Float (f64 x 250)
    // ------------------------------------------------------------------------
    {
        size_t iters = 250;
        int64_t N = 64;
        std::vector<double> A(N * N, 1.5), B(N * N, 2.5), C_native(N * N, 0.0), C_jit(N * N, 0.0);

        // Native C++
        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            native_matmul_f64(A.data(), B.data(), C_native.data(), N);
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_matmul_f64_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto matmul_fn = jit.get_function_ptr<void(*)(const double*, const double*, double*, int64_t)>("matmul_f64");
        assert(matmul_fn != nullptr);

        sw.start();
        for (size_t i = 0; i < iters; ++i) {
            matmul_fn(A.data(), B.data(), C_jit.data(), N);
        }
        double brass_ms = sw.stop_ms();

        double diff = 0.0;
        for (size_t i = 0; i < C_native.size(); ++i) diff += std::abs(C_native[i] - C_jit[i]);
        assert(diff < 1e-6);

        double ratio = brass_ms / native_ms;
        results.push_back({"MatMul 64x64 (f64)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 8: Pointer-Chasing Linked List Traversal (50,000 nodes x 500)
    // ------------------------------------------------------------------------
    {
        size_t iters = 500;
        size_t node_count = 50000;
        std::vector<BenchListNode> nodes(node_count);
        for (size_t i = 0; i < node_count; ++i) {
            nodes[i].value = static_cast<int64_t>(i + 1);
            nodes[i].next = (i + 1 < node_count) ? &nodes[i + 1] : nullptr;
        }

        // Native C++
        sw.start();
        int64_t native_sum = 0;
        for (size_t i = 0; i < iters; ++i) {
            native_sum = native_list_traversal(nodes.data());
        }
        double native_ms = sw.stop_ms();

        // Brass JIT
        auto mod = build_list_module();
        JitExecutionEngine jit;
        jit.compile_and_load(*mod);
        auto list_fn = jit.get_function_ptr<int64_t(*)(const BenchListNode*)>("list_traversal");
        assert(list_fn != nullptr);

        sw.start();
        int64_t jit_sum = 0;
        for (size_t i = 0; i < iters; ++i) {
            jit_sum = list_fn(nodes.data());
        }
        double brass_ms = sw.stop_ms();

        assert(native_sum == jit_sum);
        double ratio = brass_ms / native_ms;
        results.push_back({"Linked List Traversal (50k)", iters, native_ms, brass_ms, ratio, ratio <= 1.30, "<= 1.3x baseline"});
        BenchmarkReporter::print_row(results.back());
    }

    // ------------------------------------------------------------------------
    // Benchmark 9: GC Model: Brass Stack Maps vs Shadow Stack (200,000 calls)
    // ------------------------------------------------------------------------
    {
        size_t iters = 200000;
        int64_t heap_obj0 = 10;
        int64_t heap_obj1 = 20;
        int64_t heap_obj2 = 30;
        int64_t heap_obj3 = 40;

        // (a) Shadow-Stack Model
        ThreadShadowStack ss;
        sw.start();
        int64_t shadow_res = shadow_stack_gc_runner(
            &heap_obj0, &heap_obj1, &heap_obj2, &heap_obj3,
            static_cast<int64_t>(iters), ss
        );
        double shadow_stack_ms = sw.stop_ms();

        // (b) Brass Model (Live gcref registers & stack maps, zero shadow-stack pushing)
        auto mod = build_gc_benchmark_module();
        JitExecutionEngine jit;
        jit.register_external_symbol("leaf_gc_subroutine", reinterpret_cast<void*>(&leaf_gc_subroutine));
        jit.compile_and_load(*mod);
        auto gc_fn = jit.get_function_ptr<int64_t(*)(void*, void*, void*, void*, int64_t)>("brass_gc_caller");
        assert(gc_fn != nullptr);

        sw.start();
        int64_t brass_res = gc_fn(
            &heap_obj0, &heap_obj1, &heap_obj2, &heap_obj3,
            static_cast<int64_t>(iters)
        );
        double brass_stack_map_ms = sw.stop_ms();

        assert(shadow_res == brass_res);
        double speedup = shadow_stack_ms / brass_stack_map_ms;
        bool passes_gc_bar = (speedup >= 1.50);

        BenchmarkReporter::print_gc_comparison(shadow_stack_ms, brass_stack_map_ms, speedup, passes_gc_bar);
        assert(passes_gc_bar);
    }

    return 0;
}
