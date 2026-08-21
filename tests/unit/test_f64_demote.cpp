#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <cmath>
#include <vector>

using namespace brass;
using namespace brass::il;
using namespace brass::test;

TEST_CASE("F64 Demote - Collatz Inner Loop Demotion & JIT Execution") {
    Module mod("test_collatz_demote");
    Builder b(mod);

    Function* fn = mod.create_function("collatz", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* arg0 = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");
    BasicBlock* b4 = b.create_block("b4");
    BasicBlock* b5 = b.create_block("b5");
    BasicBlock* b6 = b.create_block("b6");

    Value* zero_f = b.build_fconst_f64(0.0);
    b.build_br(b1, {arg0, zero_f});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* n = b.add_block_param(b1, Type::f64());
    Value* steps = b.add_block_param(b1, Type::f64());
    Value* one_f = b.build_fconst_f64(1.0);
    Value* cond = b.build_sgt(n, one_f);
    b.build_br_if(cond, b2, {}, b3, {n, steps});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* two_f = b.build_fconst_f64(2.0);
    Value* mod_val = b.build_call("bronze_f64_mod", Type::f64(), {n, two_f});
    Value* is_even = b.build_eq(mod_val, zero_f);
    b.build_br_if(is_even, b4, {}, b5, {});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_n = b.add_block_param(b3, Type::f64());
    Value* ret_steps = b.add_block_param(b3, Type::f64());
    (void)ret_n;
    b.build_ret(ret_steps);

    fn->append_block(b4);
    b.position_at_end(b4);
    Value* two_f4 = b.build_fconst_f64(2.0);
    Value* half = b.build_sdiv(n, two_f4);
    b.build_br(b6, {half});

    fn->append_block(b5);
    b.position_at_end(b5);
    Value* three_f = b.build_fconst_f64(3.0);
    Value* mul3 = b.build_mul(n, three_f);
    Value* one_f5 = b.build_fconst_f64(1.0);
    Value* odd_next = b.build_add(mul3, one_f5);
    b.build_br(b6, {odd_next});

    fn->append_block(b6);
    b.position_at_end(b6);
    Value* next_n = b.add_block_param(b6, Type::f64());
    Value* one_f6 = b.build_fconst_f64(1.0);
    Value* next_steps = b.build_add(steps, one_f6);
    b.build_br(b1, {next_n, next_steps});

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    bool demoted = f64_demote_pass(*fn);
    CHECK(demoted);
    REQUIRE(verify_function(*fn, &diag));

    // Verify b1 loop parameters were demoted to i64
    CHECK_EQ(b1->param(0)->type(), Type::i64());
    CHECK_EQ(b1->param(1)->type(), Type::i64());

    // Verify b6 param was demoted to i64
    CHECK_EQ(b6->param(0)->type(), Type::i64());

    // Optimize and run with JIT
    LoopOptOptions opt;
    optimize_function_loops(*fn, opt);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    il::register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto collatz_fn = jit.get_function_ptr<double(*)(double)>("collatz");
    REQUIRE(collatz_fn != nullptr);
    CHECK_EQ(collatz_fn(27.0), 111.0);
    CHECK_EQ(collatz_fn(1.0), 0.0);
    CHECK_EQ(collatz_fn(6.0), 8.0);
    CHECK_EQ(collatz_fn(12.0), 9.0);
}

TEST_CASE("F64 Demote - Fib Iteration Demotion & Loop Unrolling") {
    Module mod("test_fib_demote");
    Builder b(mod);

    Function* fn = mod.create_function("fibIter", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");

    Value* zero = b.build_fconst_f64(0.0);
    Value* one = b.build_fconst_f64(1.0);
    b.build_br(b1, {zero, one, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* a = b.add_block_param(b1, Type::f64());
    Value* cur_b = b.add_block_param(b1, Type::f64());
    Value* i = b.add_block_param(b1, Type::f64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, b2, {}, b3, {a, cur_b, i});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* next_b = b.build_add(a, cur_b);
    Value* one_b2 = b.build_fconst_f64(1.0);
    Value* next_i = b.build_add(i, one_b2);
    b.build_br(b1, {cur_b, next_b, next_i});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_a = b.add_block_param(b3, Type::f64());
    Value* ret_b = b.add_block_param(b3, Type::f64());
    Value* ret_i = b.add_block_param(b3, Type::f64());
    (void)ret_b;
    (void)ret_i;
    b.build_ret(ret_a);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    LoopOptOptions opt;
    opt.enable_unroll = true;
    opt.unroll_factor = 4;
    bool changed = optimize_function_loops(*fn, opt);
    CHECK(changed);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto fib_fn = jit.get_function_ptr<double(*)(double)>("fibIter");
    REQUIRE(fib_fn != nullptr);
    CHECK_EQ(fib_fn(0.0), 0.0);
    CHECK_EQ(fib_fn(1.0), 1.0);
    CHECK_EQ(fib_fn(2.0), 1.0);
    CHECK_EQ(fib_fn(10.0), 55.0);
    CHECK_EQ(fib_fn(20.0), 6765.0);
    CHECK_EQ(fib_fn(30.0), 832040.0);
}

TEST_CASE("F64 Demote - Prime Sieve and Count Demotion & JIT Execution") {
    Module mod("test_prime_demote");
    Builder b(mod);

    Function* fn = mod.create_function("isPrime", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b_small = b.create_block("b_small");
    BasicBlock* b_init = b.create_block("b_init");
    BasicBlock* b_loop = b.create_block("b_loop");
    BasicBlock* b_body = b.create_block("b_body");
    BasicBlock* b_div = b.create_block("b_div");
    BasicBlock* b_next = b.create_block("b_next");
    BasicBlock* b_exit = b.create_block("b_exit");

    Value* two_f = b.build_fconst_f64(2.0);
    Value* is_small = b.build_slt(n, two_f);
    b.build_br_if(is_small, b_small, {}, b_init, {});

    fn->append_block(b_small);
    b.position_at_end(b_small);
    b.build_ret(b.build_fconst_f64(0.0));

    fn->append_block(b_init);
    b.position_at_end(b_init);
    b.build_br(b_loop, {two_f});

    fn->append_block(b_loop);
    b.position_at_end(b_loop);
    Value* p = b.add_block_param(b_loop, Type::f64());
    Value* p_sq = b.build_mul(p, p);
    Value* in_range = b.build_sle(p_sq, n);
    b.build_br_if(in_range, b_body, {}, b_exit, {});

    fn->append_block(b_body);
    b.position_at_end(b_body);
    Value* rem = b.build_call("bronze_f64_mod", Type::f64(), {n, p});
    Value* is_div = b.build_eq(rem, b.build_fconst_f64(0.0));
    b.build_br_if(is_div, b_div, {}, b_next, {});

    fn->append_block(b_div);
    b.position_at_end(b_div);
    b.build_ret(b.build_fconst_f64(0.0));

    fn->append_block(b_next);
    b.position_at_end(b_next);
    Value* next_p = b.build_add(p, b.build_fconst_f64(1.0));
    b.build_br(b_loop, {next_p});

    fn->append_block(b_exit);
    b.position_at_end(b_exit);
    b.build_ret(b.build_fconst_f64(1.0));

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    bool demoted = f64_demote_pass(*fn);
    CHECK(demoted);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    il::register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto is_prime_fn = jit.get_function_ptr<double(*)(double)>("isPrime");
    REQUIRE(is_prime_fn != nullptr);
    CHECK_EQ(is_prime_fn(1.0), 0.0);
    CHECK_EQ(is_prime_fn(2.0), 1.0);
    CHECK_EQ(is_prime_fn(3.0), 1.0);
    CHECK_EQ(is_prime_fn(4.0), 0.0);
    CHECK_EQ(is_prime_fn(29.0), 1.0);
    CHECK_EQ(is_prime_fn(100.0), 0.0);
    CHECK_EQ(is_prime_fn(101.0), 1.0);
}

TEST_CASE("F64 Demote - Matrix 2D Recurrence Nested Loop Demotion & Loop Unrolling") {
    Module mod("test_matrix_recurrence_demote");
    Builder b(mod);

    Function* fn = mod.create_function("recurrence2D", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");
    BasicBlock* b4 = b.create_block("b4");
    BasicBlock* b5 = b.create_block("b5");
    BasicBlock* b6 = b.create_block("b6");

    Value* zero = b.build_fconst_f64(0.0);
    Value* c1 = b.build_fconst_f64(1.0);
    Value* c3 = b.build_fconst_f64(3.0);
    Value* c7 = b.build_fconst_f64(7.0);
    b.build_br(b1, {zero, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* acc_i = b.add_block_param(b1, Type::f64());
    Value* i = b.add_block_param(b1, Type::f64());
    Value* cond_i = b.build_slt(i, n);
    b.build_br_if(cond_i, b2, {}, b3, {acc_i, i});

    fn->append_block(b2);
    b.position_at_end(b2);
    b.build_br(b4, {acc_i, zero});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_acc = b.add_block_param(b3, Type::f64());
    Value* ret_i = b.add_block_param(b3, Type::f64());
    (void)ret_i;
    b.build_ret(ret_acc);

    fn->append_block(b4);
    b.position_at_end(b4);
    Value* acc_j = b.add_block_param(b4, Type::f64());
    Value* j = b.add_block_param(b4, Type::f64());
    Value* cond_j = b.build_slt(j, n);
    b.build_br_if(cond_j, b5, {}, b6, {acc_j, j});

    fn->append_block(b5);
    b.position_at_end(b5);
    Value* t3 = b.build_mul(i, c3);
    Value* t7 = b.build_mul(j, c7);
    Value* sum_t = b.build_add(t3, t7);
    Value* term = b.build_add(sum_t, c1);
    Value* next_acc_j = b.build_add(acc_j, term);
    Value* next_j = b.build_add(j, c1);
    b.build_br(b4, {next_acc_j, next_j});

    fn->append_block(b6);
    b.position_at_end(b6);
    Value* final_acc_j = b.add_block_param(b6, Type::f64());
    Value* final_j = b.add_block_param(b6, Type::f64());
    (void)final_j;
    Value* next_i = b.build_add(i, c1);
    b.build_br(b1, {final_acc_j, next_i});

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    LoopOptOptions opt;
    opt.enable_unroll = true;
    opt.unroll_factor = 4;
    bool changed = optimize_function_loops(*fn, opt);
    CHECK(changed);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto rec_fn = jit.get_function_ptr<double(*)(double)>("recurrence2D");
    REQUIRE(rec_fn != nullptr);
    CHECK_EQ(rec_fn(5.0), 525.0);
    CHECK_EQ(rec_fn(10.0), 4600.0);
}

TEST_CASE("F64 Demote - Non-exact Floating Point Division Preserved") {
    Module mod("test_float_div_preserved");
    Builder b(mod);

    Function* fn = mod.create_function("floatDiv", Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* arg0 = b.add_block_param(b0, Type::f64());
    Value* arg1 = b.add_block_param(b0, Type::f64());

    Value* div_res = b.build_sdiv(arg0, arg1);
    b.build_ret(div_res);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // f64_demote must not demote unguarded non-exact division
    f64_demote_pass(*fn);
    REQUIRE(verify_function(*fn, &diag));

    CHECK_EQ(div_res->type(), Type::f64());

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto div_fn = jit.get_function_ptr<double(*)(double, double)>("floatDiv");
    REQUIRE(div_fn != nullptr);
    CHECK(std::abs(div_fn(7.0, 3.0) - (7.0 / 3.0)) < 1e-9);
}
