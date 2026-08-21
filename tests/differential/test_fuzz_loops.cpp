#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Complex Nested Loops with Block Arguments") {
    std::mt19937_64 rng(4242);

    for (uint64_t seed = 0; seed < 50; ++seed) {
        std::string mod_name = "fuzz_loop_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_loop_fn";

        generate_fuzz_loops(mod, fn_name, seed + 100);

        for (int t = 0; t < 3; ++t) {
            int64_t limit = static_cast<int64_t>(rng() % 50);
            int64_t init_val = static_cast<int64_t>(rng() % 1000);

            assert_diff(mod, fn_name, {
                RuntimeValue::from_i64(limit),
                RuntimeValue::from_i64(init_val)
            });
        }
    }
}

TEST_CASE("Differential Fuzzer - Collatz Arithmetic Fusion Loop") {
    for (int64_t max_n : {1, 5, 10, 27, 50, 100, 500, 1000}) {
        Module mod("collatz_diff_mod");
        Builder b(mod);

        Function* fn = mod.create_function("collatz_sum", Type::i64(), {Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* mn = b.add_block_param(entry, Type::i64());

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
        Value* in_range = b.build_sle(i, mn);
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

        assert_diff(mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
    }
}

TEST_CASE("Differential Fuzzer - Float Reduction Loop Unrolling with Prime/Odd Trip Counts") {
    // Tests unroll factor remainder loops for f64 reductions
    for (int64_t trip_count : {0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 31, 33, 63, 65, 100}) {
        Module mod("f64_reduction_diff_mod");
        Builder b(mod);

        Function* fn = mod.create_function("f64_sum_squares", Type::f64(), {Type::ptr(), Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* arr_ptr = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero_i = b.build_iconst_i64(0);
        Value* zero_f = b.build_fconst_f64(0.0);
        Value* one_i = b.build_iconst_i64(1);

        b.build_br(loop_hdr, {zero_i, zero_f});

        fn->append_block(loop_hdr);
        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::f64());
        Value* cond = b.build_slt(i, n);
        b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* val = b.build_load_indexed(Type::f64(), arr_ptr, i, 8, 0);
        Value* sq = b.build_mul(val, val);
        Value* next_acc = b.build_add(acc, sq);
        Value* next_i = b.build_add(i, one_i);
        b.build_br(loop_hdr, {next_i, next_acc});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_res = b.add_block_param(exit_bb, Type::f64());
        b.build_ret(final_res);

        fn->rebuild_cfg_predecessors();

        size_t count = static_cast<size_t>(trip_count);
        std::vector<double> data(count + 4);
        for (size_t k = 0; k < count + 4; ++k) {
            data[k] = static_cast<double>(k + 1) * 0.5;
        }

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, "f64_sum_squares", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(data.data())),
            RuntimeValue::from_i64(trip_count)
        });

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));
        auto fn_ptr = jit.get_function_ptr<double(*)(const double*, int64_t)>("f64_sum_squares");
        REQUIRE(fn_ptr != nullptr);

        double jit_res = fn_ptr(data.data(), trip_count);
        CHECK(std::abs(interp_res.as_f64() - jit_res) < 1e-6);
    }
}

TEST_CASE("Differential Fuzzer - Loop Carried Non-Linear Dependence") {
    // Tests loops with dependencies that forbid unroll-and-jam (acc = acc * 3 + i)
    for (int64_t n_val : {0, 1, 2, 3, 5, 8, 12, 17, 25}) {
        Module mod("nonlinear_dep_mod");
        Builder b(mod);

        Function* fn = mod.create_function("nonlinear_acc", Type::i64(), {Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero = b.build_iconst_i64(0);
        Value* one = b.build_iconst_i64(1);
        Value* three = b.build_iconst_i64(3);
        b.build_br(loop_hdr, {zero, one});

        fn->append_block(loop_hdr);
        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::i64());
        Value* cond = b.build_slt(i, n);
        b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* acc3 = b.build_mul(acc, three);
        Value* next_acc = b.build_add(acc3, i);
        Value* next_i = b.build_add(i, one);
        b.build_br(loop_hdr, {next_i, next_acc});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_res = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(final_res);

        fn->rebuild_cfg_predecessors();

        assert_diff(mod, "nonlinear_acc", {RuntimeValue::from_i64(n_val)});
    }
}

TEST_CASE("Differential Fuzzer - Loops with In-Body Stores") {
    // Tests loops with stores to ensure memory order is preserved
    for (int64_t n_val : {1, 2, 4, 7, 15, 23, 32, 47}) {
        Module mod("store_loop_mod");
        Builder b(mod);

        Function* fn = mod.create_function("fill_fib_array", Type::void_type(), {Type::ptr(), Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* arr = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* two = b.build_iconst_i64(2);
        Value* one = b.build_iconst_i64(1);
        b.build_br(loop_hdr, {two});

        fn->append_block(loop_hdr);
        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::i64());
        Value* cond = b.build_slt(i, n);
        b.build_br_if(cond, loop_body, {}, exit_bb, {});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* i_minus_1 = b.build_sub(i, one);
        Value* i_minus_2 = b.build_sub(i, two);
        Value* f1 = b.build_load_indexed(Type::i64(), arr, i_minus_1, 8, 0);
        Value* f2 = b.build_load_indexed(Type::i64(), arr, i_minus_2, 8, 0);
        Value* f_next = b.build_add(f1, f2);
        b.build_store_indexed(Type::i64(), arr, i, 8, 0, f_next);
        Value* next_i = b.build_add(i, one);
        b.build_br(loop_hdr, {next_i});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        b.build_ret(nullptr);

        fn->rebuild_cfg_predecessors();

        size_t count = static_cast<size_t>(n_val + 4);
        std::vector<int64_t> arr_interp(count, 0);
        std::vector<int64_t> arr_jit(count, 0);
        arr_interp[0] = 0; arr_interp[1] = 1;
        arr_jit[0] = 0; arr_jit[1] = 1;

        Interpreter interp;
        interp.run(mod, "fill_fib_array", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(arr_interp.data())),
            RuntimeValue::from_i64(n_val)
        });

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));
        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t)>("fill_fib_array");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(arr_jit.data(), n_val);

        CHECK(arr_interp == arr_jit);
    }
}


