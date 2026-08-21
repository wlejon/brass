#include "bench_numeric_modules.hpp"

namespace brass::bench {

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
    BasicBlock* inner_body = b.create_block("inner_body");
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
    b.build_br_if(is_done, outer_next, {}, inner_body, {});

    fn->append_block(inner_body);
    b.position_at_end(inner_body);
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
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {sum});

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
    Value* final_sum = b.add_block_param(loop_j_next, Type::i64());
    Value* idx_c = b.build_add(row_a, j);
    b.build_store_indexed(Type::i64(), C, idx_c, 8, 0, final_sum);
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
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {sum});

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
    Value* final_sum = b.add_block_param(loop_j_next, Type::f64());
    Value* idx_c = b.build_add(row_a, j);
    b.build_store_indexed(Type::f64(), C, idx_c, 8, 0, final_sum);
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

} // namespace brass::bench
