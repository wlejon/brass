#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/generational_gc.hpp>
#include "../benchmarks/bench_numeric_modules.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>

using namespace brass;
using namespace brass::runtime;

namespace {

void assert_oracle_vs_fast(
    const Module& mod,
    std::string_view fn_name,
    const std::vector<RuntimeValue>& args
) {
    Interpreter oracle;
    RuntimeValue o_res = oracle.run(mod, fn_name, args);

    FastInterpreter fast;
    RuntimeValue f_res = fast.run(mod, fn_name, args);

    if (o_res.is_f64()) {
        double d1 = o_res.as_f64();
        double d2 = f_res.as_f64();
        if (std::isnan(d1)) {
            CHECK(std::isnan(d2));
        } else {
            CHECK(std::abs(d1 - d2) < 1e-6);
        }
    } else if (o_res.is_f32()) {
        float f1 = o_res.as_f32();
        float f2 = f_res.as_f32();
        if (std::isnan(f1)) {
            CHECK(std::isnan(f2));
        } else {
            CHECK(std::abs(f1 - f2) < 1e-6f);
        }
    } else {
        CHECK_EQ(o_res.raw_bits(), f_res.raw_bits());
    }
}

} // namespace

// ============================================================================
// 1. Numeric Modules Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Numeric Modules") {
    // 1a. Iterative Fibonacci
    {
        Module mod("diff_fib_iter");
        Builder b(mod);
        Function* fn = mod.create_function("fib_iter", Type::i64(), {Type::i64()});
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
        REQUIRE(verify_function(*fn));

        for (int64_t test_n : {0, 1, 2, 5, 10, 20, 30}) {
            assert_oracle_vs_fast(mod, "fib_iter", {RuntimeValue::from_i64(test_n)});
        }
    }

    // 1b. Recursive Fibonacci
    {
        Module mod("diff_fib_rec");
        Function* fn = mod.create_function("fib_rec", Type::i64(), {Type::i64()});
        Builder b(mod);
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
        Value* n1 = b.build_sub(n, one);
        Value* n2 = b.build_sub(n, two);
        Value* fib1 = b.build_call("fib_rec", Type::i64(), {n1});
        Value* fib2 = b.build_call("fib_rec", Type::i64(), {n2});
        b.build_ret(b.build_add(fib1, fib2));

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int64_t test_n : {0, 1, 3, 6, 8, 11}) {
            assert_oracle_vs_fast(mod, "fib_rec", {RuntimeValue::from_i64(test_n)});
        }
    }

    // 1c. Collatz Sequence Steps
    {
        Module mod("diff_collatz");
        Builder b(mod);
        Function* fn = mod.create_function("collatz_steps", Type::i64(), {Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* start_n = b.add_block_param(entry, Type::i64());
        BasicBlock* loop_hdr = b.append_block("loop_hdr");
        BasicBlock* loop_body = b.append_block("loop_body");
        BasicBlock* even_bb = b.append_block("even_bb");
        BasicBlock* odd_bb = b.append_block("odd_bb");
        BasicBlock* exit_bb = b.append_block("exit_bb");

        b.position_at_end(entry);
        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop_hdr, {start_n, zero});

        b.position_at_end(loop_hdr);
        Value* cur_n = b.add_block_param(loop_hdr, Type::i64());
        Value* steps = b.add_block_param(loop_hdr, Type::i64());
        Value* one = b.build_iconst_i64(1);
        Value* not_one = b.build_sgt(cur_n, one);
        b.build_br_if(not_one, loop_body, exit_bb);

        b.position_at_end(loop_body);
        Value* bit_mask = b.build_iconst_i64(1);
        Value* is_odd = b.build_and(cur_n, bit_mask);
        Value* odd_cond = b.build_eq(is_odd, one);
        b.build_br_if(odd_cond, odd_bb, even_bb);

        b.position_at_end(even_bb);
        Value* half_n = b.build_ashr(cur_n, one);
        Value* next_steps_e = b.build_add(steps, one);
        b.build_br(loop_hdr, {half_n, next_steps_e});

        b.position_at_end(odd_bb);
        Value* three = b.build_iconst_i64(3);
        Value* mul3 = b.build_mul(cur_n, three);
        Value* next_n_odd = b.build_add(mul3, one);
        Value* next_steps_o = b.build_add(steps, one);
        b.build_br(loop_hdr, {next_n_odd, next_steps_o});

        b.position_at_end(exit_bb);
        b.build_ret(steps);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int64_t n_val : {1, 2, 3, 6, 7, 12, 19, 27}) {
            assert_oracle_vs_fast(mod, "collatz_steps", {RuntimeValue::from_i64(n_val)});
        }
    }

    // 1d. Matrix Multiplication (Integer 4x4)
    {
        auto mod = bench::build_matmul_i64_naive_module();
        int64_t matA[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        int64_t matB[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        int64_t matC_oracle[16] = {0};
        int64_t matC_fast[16] = {0};

        Interpreter oracle;
        oracle.run(*mod, "matmul_i64_naive", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matA)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matB)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matC_oracle)),
            RuntimeValue::from_i64(4)
        });

        FastInterpreter fast;
        fast.run(*mod, "matmul_i64_naive", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matA)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matB)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(matC_fast)),
            RuntimeValue::from_i64(4)
        });

        for (int i = 0; i < 16; ++i) {
            CHECK_EQ(matC_oracle[i], matC_fast[i]);
        }
    }

    // 1e. Prime Sieve
    {
        auto mod = bench::build_sieve_module();
        int64_t limit = 100;
        std::vector<int64_t> buf_oracle(limit, 0);
        std::vector<int64_t> buf_fast(limit, 0);

        Interpreter oracle;
        RuntimeValue o_res = oracle.run(*mod, "prime_sieve", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buf_oracle.data())),
            RuntimeValue::from_i64(limit)
        });

        FastInterpreter fast;
        RuntimeValue f_res = fast.run(*mod, "prime_sieve", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buf_fast.data())),
            RuntimeValue::from_i64(limit)
        });

        CHECK_EQ(o_res.as_i64(), f_res.as_i64());
        for (int i = 0; i < limit; ++i) {
            CHECK_EQ(buf_oracle[i], buf_fast[i]);
        }
    }
}

// ============================================================================
// 2. Loops and Complex Branch Reductions
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Loops and Reductions") {
    // 2a. Double nested loop with accumulator
    {
        Module mod("diff_nested_loops");
        Builder b(mod);
        Function* fn = mod.create_function("nested_sum", Type::i64(), {Type::i64(), Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* n_i = b.add_block_param(entry, Type::i64());
        Value* n_j = b.add_block_param(entry, Type::i64());

        BasicBlock* i_hdr = b.append_block("i_hdr");
        BasicBlock* j_hdr = b.append_block("j_hdr");
        BasicBlock* j_body = b.append_block("j_body");
        BasicBlock* exit_bb = b.append_block("exit_bb");

        b.position_at_end(entry);
        Value* zero = b.build_iconst_i64(0);
        Value* one = b.build_iconst_i64(1);
        b.build_br(i_hdr, {zero, zero});

        // i loop: param i, acc
        b.position_at_end(i_hdr);
        Value* i = b.add_block_param(i_hdr, Type::i64());
        Value* acc_i = b.add_block_param(i_hdr, Type::i64());
        Value* i_cond = b.build_slt(i, n_i);
        b.build_br_if(i_cond, j_hdr, {zero, acc_i}, exit_bb, {acc_i});

        // j loop: param j, acc
        b.position_at_end(j_hdr);
        Value* j = b.add_block_param(j_hdr, Type::i64());
        Value* acc_j = b.add_block_param(j_hdr, Type::i64());
        Value* j_cond = b.build_slt(j, n_j);
        Value* next_i = b.build_add(i, one);
        b.build_br_if(j_cond, j_body, {}, i_hdr, {next_i, acc_j});

        // j body: compute acc + (i * n_j + j)
        b.position_at_end(j_body);
        Value* i_term = b.build_mul(i, n_j);
        Value* term = b.build_add(i_term, j);
        Value* new_acc = b.build_add(acc_j, term);
        Value* next_j = b.build_add(j, one);
        b.build_br(j_hdr, {next_j, new_acc});

        b.position_at_end(exit_bb);
        Value* final_res = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(final_res);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        bool ok = verify_function(*fn, &diag);
        if (!ok) {
            std::cerr << "VERIFY ERROR:\n" << diag.format_all() << "\n";
        }
        REQUIRE(ok);

        for (int64_t ni : {1, 3, 5, 8}) {
            for (int64_t nj : {1, 4, 7}) {
                assert_oracle_vs_fast(mod, "nested_sum", {RuntimeValue::from_i64(ni), RuntimeValue::from_i64(nj)});
            }
        }
    }

    // 2b. Float reduction loop
    {
        Module mod("diff_fp_loop");
        Builder b(mod);
        Function* fn = mod.create_function("fp_reciprocal_sum", Type::f64(), {Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_hdr = b.append_block("loop_hdr");
        BasicBlock* loop_body = b.append_block("loop_body");
        BasicBlock* exit_bb = b.append_block("exit_bb");

        b.position_at_end(entry);
        Value* one_i = b.build_iconst_i64(1);
        Value* zero_f = b.build_fconst_f64(0.0);
        b.build_br(loop_hdr, {one_i, zero_f});

        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::f64());
        Value* cond = b.build_sle(i, n);
        b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

        b.position_at_end(loop_body);
        Value* i_f = b.build_sitofp_f64_i64(i);
        Value* half_f = b.build_fconst_f64(0.5);
        Value* term = b.build_mul(i_f, half_f);
        Value* new_acc = b.build_fadd(acc, term);
        Value* next_i = b.build_add(i, one_i);
        b.build_br(loop_hdr, {next_i, new_acc});

        b.position_at_end(exit_bb);
        Value* final_acc = b.add_block_param(exit_bb, Type::f64());
        b.build_ret(final_acc);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int64_t lim : {1, 5, 10, 50, 100}) {
            assert_oracle_vs_fast(mod, "fp_reciprocal_sum", {RuntimeValue::from_i64(lim)});
        }
    }
}

// ============================================================================
// 3. Closures and Environments Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Closures and Environments") {
    // Test closure pattern: heap/stack-allocated environments with captured mutable state
    // dispatched via indirect function pointers.
    Module mod("diff_closures");
    Builder b(mod);

    // 1. Counter increment function: takes (env_ptr, extra_step) -> returns updated count
    // env[0] = counter_value (i64)
    // env[1] = base_step (i64)
    Function* fn_inc = mod.create_function("counter_inc", Type::i64(), {Type::ptr(), Type::i64()});
    {
        b.set_function(fn_inc);
        BasicBlock* bb = b.append_block("entry");
        Value* env = b.add_block_param(bb, Type::ptr());
        Value* extra_step = b.add_block_param(bb, Type::i64());

        Value* cur_val = b.build_load(Type::i64(), env, 0);
        Value* base_step = b.build_load(Type::i64(), env, 8);
        Value* step_sum = b.build_add(base_step, extra_step);
        Value* new_val = b.build_add(cur_val, step_sum);
        b.build_store(Type::i64(), env, 0, new_val);
        b.build_ret(new_val);
    }

    // 2. Counter scale function: takes (env_ptr, factor) -> returns new scaled count
    Function* fn_scale = mod.create_function("counter_scale", Type::i64(), {Type::ptr(), Type::i64()});
    {
        b.set_function(fn_scale);
        BasicBlock* bb = b.append_block("entry");
        Value* env = b.add_block_param(bb, Type::ptr());
        Value* factor = b.add_block_param(bb, Type::i64());

        Value* cur_val = b.build_load(Type::i64(), env, 0);
        Value* new_val = b.build_mul(cur_val, factor);
        b.build_store(Type::i64(), env, 0, new_val);
        b.build_ret(new_val);
    }

    // 3. Orchestrator function:
    // Creates two independent lexical environments (env1 and env2),
    // interacts with both through indirect calls using function addresses,
    // verifies mutual independence and cumulative updates.
    Function* fn_run = mod.create_function("run_closure_pipeline", Type::i64(),
                                           {Type::i64(), Type::i64(), Type::i64(), Type::i64()});
    {
        b.set_function(fn_run);
        BasicBlock* bb = b.append_block("entry");
        Value* start1 = b.add_block_param(bb, Type::i64());
        Value* step1 = b.add_block_param(bb, Type::i64());
        Value* start2 = b.add_block_param(bb, Type::i64());
        Value* step2 = b.add_block_param(bb, Type::i64());

        Value* inc_fn = b.build_func_addr("counter_inc");
        Value* scale_fn = b.build_func_addr("counter_scale");

        // env1: [val, step]
        Value* env1 = b.build_alloca(16, 8);
        b.build_store(Type::i64(), env1, 0, start1);
        b.build_store(Type::i64(), env1, 8, step1);

        // env2: [val, step]
        Value* env2 = b.build_alloca(16, 8);
        b.build_store(Type::i64(), env2, 0, start2);
        b.build_store(Type::i64(), env2, 8, step2);

        Value* one = b.build_iconst_i64(1);
        Value* two = b.build_iconst_i64(2);
        Value* three = b.build_iconst_i64(3);

        // Call inc on env1 twice:
        Value* r1a = b.build_call_indirect(inc_fn, Type::i64(), {env1, one});
        Value* r1b = b.build_call_indirect(inc_fn, Type::i64(), {env1, two});

        // Call inc on env2 twice:
        Value* r2a = b.build_call_indirect(inc_fn, Type::i64(), {env2, one});
        Value* r2b = b.build_call_indirect(inc_fn, Type::i64(), {env2, three});

        // Scale env1 by 3:
        Value* r1c = b.build_call_indirect(scale_fn, Type::i64(), {env1, three});

        // Scale env2 by 2:
        Value* r2c = b.build_call_indirect(scale_fn, Type::i64(), {env2, two});

        // Combine: r1a + r1b + r1c + r2a + r2b + r2c
        Value* sum1 = b.build_add(r1a, r1b);
        Value* sum2 = b.build_add(sum1, r1c);
        Value* sum3 = b.build_add(sum2, r2a);
        Value* sum4 = b.build_add(sum3, r2b);
        Value* total = b.build_add(sum4, r2c);
        b.build_ret(total);
    }

    REQUIRE(verify_function(*fn_inc));
    REQUIRE(verify_function(*fn_scale));
    REQUIRE(verify_function(*fn_run));

    // Differential assertions across diverse input states
    std::vector<std::pair<int64_t, int64_t>> test_cases = {
        {10, 2}, {100, 5}, {0, 1}, {-20, 3}, {42, 7}
    };

    for (auto [s1, st1] : test_cases) {
        for (auto [s2, st2] : test_cases) {
            assert_oracle_vs_fast(mod, "run_closure_pipeline", {
                RuntimeValue::from_i64(s1),
                RuntimeValue::from_i64(st1),
                RuntimeValue::from_i64(s2),
                RuntimeValue::from_i64(st2)
            });
        }
    }
}

// ============================================================================
// 4. Shapes and Property Access Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Shapes and Inline Caches") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    Shape* s_x = registry.transition_to(root, "x");
    Shape* s_xy = registry.transition_to(s_x, "y");
    Shape* s_xyz = registry.transition_to(s_xy, "z");

    DynamicObject* obj = DynamicObject::create(nullptr, root);
    obj->set_property("x", HostValue::from_i32(10), registry);
    obj->set_property("y", HostValue::from_i32(20), registry);
    obj->set_property("z", HostValue::from_i32(30), registry);

    CHECK_EQ(obj->shape, s_xyz);
    CHECK_EQ(obj->get_property("x").as_i32(), 10);
    CHECK_EQ(obj->get_property("y").as_i32(), 20);
    CHECK_EQ(obj->get_property("z").as_i32(), 30);

    // Module accessing properties through host object functions
    Module mod("diff_shapes");
    Builder b(mod);

    Function* fn = mod.create_function("read_slots_sum", Type::i64(), {Type::ptr()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* obj_ptr = b.add_block_param(entry, Type::ptr());

    // Call external brass_obj_get_prop
    Value* vx = b.build_call("brass_get_prop_x", Type::i64(), {obj_ptr});
    Value* vy = b.build_call("brass_get_prop_y", Type::i64(), {obj_ptr});
    Value* vz = b.build_call("brass_get_prop_z", Type::i64(), {obj_ptr});
    Value* s1 = b.build_add(vx, vy);
    Value* sum = b.build_add(s1, vz);
    b.build_ret(sum);
    fn->rebuild_cfg_predecessors();

    auto setup_props = [](auto& interp) {
        brass::HostFn fn_x = [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
            auto* o = reinterpret_cast<DynamicObject*>(args[0].as_ptr());
            return RuntimeValue::from_i64(o->get_property("x").as_i32());
        };
        brass::HostFn fn_y = [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
            auto* o = reinterpret_cast<DynamicObject*>(args[0].as_ptr());
            return RuntimeValue::from_i64(o->get_property("y").as_i32());
        };
        brass::HostFn fn_z = [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
            auto* o = reinterpret_cast<DynamicObject*>(args[0].as_ptr());
            return RuntimeValue::from_i64(o->get_property("z").as_i32());
        };
        interp.register_external_function("brass_get_prop_x", fn_x);
        interp.register_external_function("brass_get_prop_y", fn_y);
        interp.register_external_function("brass_get_prop_z", fn_z);
    };

    Interpreter oracle;
    setup_props(oracle);
    RuntimeValue o_res = oracle.run(mod, "read_slots_sum", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(obj))});

    FastInterpreter fast;
    setup_props(fast);
    RuntimeValue f_res = fast.run(mod, "read_slots_sum", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(obj))});

    CHECK_EQ(o_res.as_i64(), 60);
    CHECK_EQ(f_res.as_i64(), 60);
    CHECK_EQ(o_res.as_i64(), f_res.as_i64());

    DynamicObject::destroy_non_gc(obj);
}

// ============================================================================
// 5. Exception Handling & Unwinding Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Exceptions and Unwinding") {
    Module mod("diff_exceptions");
    Builder b(mod);

    // callee: func @compute(%val: i64) -> i64
    // if %val < 0 throw %val
    // else return %val * 3 + 1
    Function* callee = mod.create_function("compute", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    BasicBlock* c_throw = b.append_block("c_throw");
    BasicBlock* c_ok = b.append_block("c_ok");

    b.position_at_end(c_entry);
    Value* val = b.add_block_param(c_entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    Value* is_neg = b.build_slt(val, zero);
    b.build_br_if(is_neg, c_throw, c_ok);

    b.position_at_end(c_throw);
    b.build_throw(val);

    b.position_at_end(c_ok);
    Value* three = b.build_iconst_i64(3);
    Value* one = b.build_iconst_i64(1);
    Value* p = b.build_mul(val, three);
    Value* s = b.build_add(p, one);
    b.build_ret(s);
    callee->rebuild_cfg_predecessors();

    // caller: func @run_compute(%x: i64) -> i64
    // try { return compute(%x); } catch (%e) { return %e * -10; }
    Function* caller = mod.create_function("run_compute", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("compute", Type::i64(), {x}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* minus_ten = b.build_iconst_i64(-10);
    Value* catch_res = b.build_mul(exc, minus_ten);
    b.build_ret(catch_res);
    caller->rebuild_cfg_predecessors();

    for (int64_t input : {0, 1, 5, 10, 42, -1, -5, -12, -99}) {
        assert_oracle_vs_fast(mod, "run_compute", {RuntimeValue::from_i64(input)});
    }
}

// ============================================================================
// 6. Coroutines Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Coroutines") {
    Module mod("diff_coro_mod");
    Builder b(mod);

    // Coroutine accumulating sum across suspends:
    // yields 1 (sum=1)
    // receives arg + 10 (sum = 1 + 10 = 11), yields 11
    // receives arg + 20 (sum = 11 + 20 = 31), yields 31
    // returns sum
    Function* fn = mod.create_function("diff_coro_fn", Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* sum0 = b.build_iconst_i64(1);
    Value* r1 = b.build_coro_suspend(sum0, 1, Type::i64());

    Value* sum1 = b.build_add(sum0, r1);
    Value* r2 = b.build_coro_suspend(sum1, 2, Type::i64());

    Value* sum2 = b.build_add(sum1, r2);
    b.build_ret(sum2);

    CoroTransformPass pass;
    CHECK(pass.run_on_module(mod));
    REQUIRE(verify_module(mod));

    Function* coro_fn = mod.get_function("diff_coro_fn");
    REQUIRE(coro_fn != nullptr);

    // Run in Oracle Interpreter
    Interpreter oracle;
    uintptr_t o_frame = brass_coro_create(nullptr, 16, 0);
    REQUIRE(o_frame != 0);
    auto* o_fptr = reinterpret_cast<runtime::BrassCoroFrame*>(o_frame);

    RuntimeValue o_y1 = oracle.run(*coro_fn, {RuntimeValue::from_ptr(o_frame)});
    o_fptr->resume_arg = 10;
    RuntimeValue o_y2 = oracle.run(*coro_fn, {RuntimeValue::from_ptr(o_frame)});
    o_fptr->resume_arg = 20;
    RuntimeValue o_y3 = oracle.run(*coro_fn, {RuntimeValue::from_ptr(o_frame)});
    uint32_t o_done = o_fptr->is_done;
    brass_coro_destroy(o_frame);

    // Run in FastInterpreter
    FastInterpreter fast;
    uintptr_t f_frame = brass_coro_create(nullptr, 16, 0);
    REQUIRE(f_frame != 0);
    auto* f_fptr = reinterpret_cast<runtime::BrassCoroFrame*>(f_frame);

    RuntimeValue f_y1 = fast.run(*coro_fn, {RuntimeValue::from_ptr(f_frame)});
    f_fptr->resume_arg = 10;
    RuntimeValue f_y2 = fast.run(*coro_fn, {RuntimeValue::from_ptr(f_frame)});
    f_fptr->resume_arg = 20;
    RuntimeValue f_y3 = fast.run(*coro_fn, {RuntimeValue::from_ptr(f_frame)});
    uint32_t f_done = f_fptr->is_done;
    brass_coro_destroy(f_frame);

    CHECK_EQ(o_y1.as_i64(), 1);
    CHECK_EQ(f_y1.as_i64(), o_y1.as_i64());

    CHECK_EQ(o_y2.as_i64(), 11);
    CHECK_EQ(f_y2.as_i64(), o_y2.as_i64());

    CHECK_EQ(o_y3.as_i64(), 31);
    CHECK_EQ(f_y3.as_i64(), o_y3.as_i64());

    CHECK_EQ(o_done, 1u);
    CHECK_EQ(f_done, o_done);
}

// ============================================================================
// 7. Garbage Collection Differential Verification
// ============================================================================

TEST_CASE("Differential - FastInterpreter vs Oracle: Garbage Collection Roots") {
    Module mod("diff_gc_roots");
    Builder b(mod);

    Function* fn = mod.create_function("gc_root_runner", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* val = b.add_block_param(entry, Type::i64());
    Value* res = b.build_call("scavenge_and_check", Type::i64(), {val});
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();

    // 1. Oracle Interpreter
    Interpreter oracle(128 * 1024);
    uintptr_t o_obj = oracle.gc().allocate(24, 0, 7);
    oracle.gc().write_field(o_obj, 0, 0x123456789ABCDEF0ULL);
    uintptr_t o_updated = 0;

    oracle.register_external_function("scavenge_and_check", [&](Interpreter& in, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        std::vector<uintptr_t*> roots;
        uintptr_t root = o_obj;
        roots.push_back(&root);
        in.gc().collect(roots);
        o_updated = root;
        int64_t field = static_cast<int64_t>(in.gc().read_field(o_updated, 0));
        return RuntimeValue::from_i64(field + args[0].as_i64());
    });
    RuntimeValue o_res = oracle.run(mod, "gc_root_runner", {RuntimeValue::from_i64(5)});

    // 2. FastInterpreter
    FastInterpreter fast(128 * 1024);
    uintptr_t f_obj = fast.gc().allocate(24, 0, 7);
    fast.gc().write_field(f_obj, 0, 0x123456789ABCDEF0ULL);
    uintptr_t f_updated = 0;

    fast.register_external_function("scavenge_and_check", [&](FastInterpreter& in, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        std::vector<uintptr_t*> roots;
        uintptr_t root = f_obj;
        roots.push_back(&root);
        in.gc().collect(roots);
        f_updated = root;
        int64_t field = static_cast<int64_t>(in.gc().read_field(f_updated, 0));
        return RuntimeValue::from_i64(field + args[0].as_i64());
    });
    RuntimeValue f_res = fast.run(mod, "gc_root_runner", {RuntimeValue::from_i64(5)});

    CHECK_EQ(o_res.as_i64(), f_res.as_i64());
    CHECK(oracle.gc().is_valid_object(o_updated));
    CHECK(fast.gc().is_valid_object(f_updated));
}

// ============================================================================
// 8. Multi-Tier Pipeline Integration with FastInterpreter
// ============================================================================

TEST_CASE("Differential - FastInterpreter in MultiTierPipeline") {
    Module mod("diff_pipeline_mod");
    Function* fn = mod.create_function("pipeline_tiered_fn", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* three = b.build_iconst_i64(3);
    Value* tripled = b.build_mul(x, three);
    b.build_ret(tripled);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    TieringConfig config;
    config.invocation_tier1_threshold = 3;
    config.invocation_tier2_threshold = 10;
    config.tier0_interpreter = Tier0Interpreter::Fast;
    config.enable_background_compile = false;

    auto& pipeline = MultiTierPipeline::instance();
    pipeline.initialize(config);
    pipeline.reset_stats();

    CHECK(pipeline.use_fast_interpreter());
    CHECK_EQ(pipeline.tier0_interpreter(), Tier0Interpreter::Fast);

    TieringRegistry::instance().set_active_module(&mod);
    auto* handle = FunctionDispatchTable::instance().get_or_create("pipeline_tiered_fn", fn);
    handle->set_tier(TierLevel::Tier0_Interpreter);

    // Call 1: Tier 0 FastInterpreter execution
    RuntimeValue r1 = pipeline.execute(mod, "pipeline_tiered_fn", {RuntimeValue::from_i64(10)});
    CHECK_EQ(r1.as_i64(), 30);
    CHECK_EQ(handle->tier(), TierLevel::Tier0_Interpreter);

    // Call 2: Tier 0 FastInterpreter execution
    RuntimeValue r2 = pipeline.execute(mod, "pipeline_tiered_fn", {RuntimeValue::from_i64(20)});
    CHECK_EQ(r2.as_i64(), 60);
    CHECK_EQ(handle->tier(), TierLevel::Tier0_Interpreter);

    // Call 3: triggers Tier 1 baseline compilation
    RuntimeValue r3 = pipeline.execute(mod, "pipeline_tiered_fn", {RuntimeValue::from_i64(30)});
    CHECK_EQ(r3.as_i64(), 90);
    CHECK_EQ(handle->tier(), TierLevel::Tier1_Baseline);
    CHECK(handle->has_native_entry());

    // Call 4: Execute via handle at Tier 1 Baseline JIT
    RuntimeValue r4 = pipeline.execute(mod, "pipeline_tiered_fn", {RuntimeValue::from_i64(40)});
    CHECK_EQ(r4.as_i64(), 120);

    pipeline.shutdown();
}
