#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>

using namespace brass;

namespace {

void assert_diff(
    const Module& mod,
    std::string_view fn_name,
    const std::vector<RuntimeValue>& args
) {
    Interpreter interp;
    RuntimeValue interp_res = interp.run(mod, fn_name, args);

    codegen::JitExecutionEngine jit(Target::host());
    bool ok = jit.compile_and_load(mod);
    REQUIRE(ok);

    RuntimeValue jit_res = jit.invoke(fn_name, args);

    if (interp_res.is_f64()) {
        double d1 = interp_res.as_f64();
        double d2 = jit_res.as_f64();
        if (std::isnan(d1)) {
            CHECK(std::isnan(d2));
        } else {
            CHECK(std::abs(d1 - d2) < 1e-9);
        }
    } else {
        if (interp_res.raw_bits() != jit_res.raw_bits()) {
            std::cout << "MISMATCH in " << fn_name << "\n";
            std::cout << "Interp: " << interp_res.raw_bits() << " JIT: " << jit_res.raw_bits() << "\n";
            print_module(mod, std::cout);
        }
        CHECK_EQ(interp_res.raw_bits(), jit_res.raw_bits());
    }
}

} // namespace

TEST_CASE("Differential Exec - Recursive Fibonacci") {
    Module mod("diff_fib");
    Function* fn = mod.create_function("fib", Type::i64(), {Type::i64()});

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
    Value* fib1 = b.build_call("fib", Type::i64(), {n1});
    Value* fib2 = b.build_call("fib", Type::i64(), {n2});
    Value* sum = b.build_add(fib1, fib2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    for (int64_t v : {0, 1, 2, 3, 5, 7, 10, 12}) {
        assert_diff(mod, "fib", {RuntimeValue::from_i64(v)});
    }
}

TEST_CASE("Differential Exec - Factorial and Sum Loops") {
    // 1. Factorial Loop
    {
        Module mod("diff_fact");
        Function* fn = mod.create_function("fact", Type::i64(), {Type::i64()});

        Builder b(mod);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop_header = b.create_block("loop_header");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        fn->append_block(loop_header);
        fn->append_block(loop_body);
        fn->append_block(exit_bb);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* one = b.build_iconst_i64(1);
        b.build_br(loop_header, {n, one});

        b.position_at_end(loop_header);
        Value* cur_n = b.add_block_param(loop_header, Type::i64());
        Value* acc = b.add_block_param(loop_header, Type::i64());
        Value* cond = b.build_sgt(cur_n, one);
        b.build_br_if(cond, loop_body, exit_bb);

        b.position_at_end(loop_body);
        Value* new_acc = b.build_mul(acc, cur_n);
        Value* new_n = b.build_sub(cur_n, one);
        b.build_br(loop_header, {new_n, new_acc});

        b.position_at_end(exit_bb);
        b.build_ret(acc);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int64_t v : {0, 1, 2, 4, 6, 8, 10}) {
            assert_diff(mod, "fact", {RuntimeValue::from_i64(v)});
        }
    }

    // 2. Sum 1 to N Loop
    {
        Module mod("diff_sum");
        Function* fn = mod.create_function("sum_to_n", Type::i64(), {Type::i64()});

        Builder b(mod);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop_header = b.create_block("loop_header");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        fn->append_block(loop_header);
        fn->append_block(loop_body);
        fn->append_block(exit_bb);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* i_init = b.build_iconst_i64(1);
        Value* acc_init = b.build_iconst_i64(0);
        b.build_br(loop_header, {i_init, acc_init});

        b.position_at_end(loop_header);
        Value* i_val = b.add_block_param(loop_header, Type::i64());
        Value* acc_val = b.add_block_param(loop_header, Type::i64());
        Value* cond = b.build_sle(i_val, n);
        b.build_br_if(cond, loop_body, exit_bb);

        b.position_at_end(loop_body);
        Value* new_acc = b.build_add(acc_val, i_val);
        Value* one = b.build_iconst_i64(1);
        Value* new_i = b.build_add(i_val, one);
        b.build_br(loop_header, {new_i, new_acc});

        b.position_at_end(exit_bb);
        b.build_ret(acc_val);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int64_t v : {0, 1, 5, 10, 50, 100}) {
            assert_diff(mod, "sum_to_n", {RuntimeValue::from_i64(v)});
        }
    }
}

TEST_CASE("Differential Exec - Collatz Sequence") {
    Module mod("diff_collatz");
    Function* fn = mod.create_function("collatz_steps", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.create_block("loop_header");
    BasicBlock* test_even = b.create_block("test_even");
    BasicBlock* do_even = b.create_block("do_even");
    BasicBlock* do_odd = b.create_block("do_odd");
    BasicBlock* exit_bb = b.create_block("exit");

    fn->append_block(loop_header);
    fn->append_block(test_even);
    fn->append_block(do_even);
    fn->append_block(do_odd);
    fn->append_block(exit_bb);

    Value* start_n = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_header, {start_n, zero});

    // Header: if n > 1 go to test_even else exit
    b.position_at_end(loop_header);
    Value* cur_n = b.add_block_param(loop_header, Type::i64());
    Value* steps = b.add_block_param(loop_header, Type::i64());
    Value* one = b.build_iconst_i64(1);
    Value* cond = b.build_sgt(cur_n, one);
    b.build_br_if(cond, test_even, exit_bb);

    // Test even: n % 2 == 0
    b.position_at_end(test_even);
    Value* two = b.build_iconst_i64(2);
    Value* rem = b.build_smod(cur_n, two);
    Value* is_even = b.build_eq(rem, zero);
    b.build_br_if(is_even, do_even, do_odd);

    // Even: n = n / 2, steps += 1
    b.position_at_end(do_even);
    Value* even_n = b.build_sdiv(cur_n, two);
    Value* next_steps1 = b.build_add(steps, one);
    b.build_br(loop_header, {even_n, next_steps1});

    // Odd: n = 3 * n + 1, steps += 1
    b.position_at_end(do_odd);
    Value* three = b.build_iconst_i64(3);
    Value* odd_n = b.build_add(b.build_mul(cur_n, three), one);
    Value* next_steps2 = b.build_add(steps, one);
    b.build_br(loop_header, {odd_n, next_steps2});

    // Exit
    b.position_at_end(exit_bb);
    b.build_ret(steps);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    for (int64_t n_val : {1, 2, 3, 6, 7, 9, 15, 27}) {
        assert_diff(mod, "collatz_steps", {RuntimeValue::from_i64(n_val)});
    }
}

TEST_CASE("Differential Exec - Bitwise Operations") {
    Module mod("diff_bitwise");
    Function* fn = mod.create_function("bitwise_tests", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());

    Value* v_and = b.build_and(x, y);
    Value* v_or  = b.build_or(x, y);
    Value* v_xor = b.build_xor(v_and, v_or);
    Value* v_not = b.build_not(v_xor);

    Value* shift_amt = b.build_iconst_i64(3);
    Value* v_shl = b.build_shl(v_not, shift_amt);
    Value* v_lshr = b.build_lshr(v_shl, shift_amt);
    Value* v_ashr = b.build_ashr(v_lshr, shift_amt);

    Value* v_pop = b.build_popcnt(v_ashr);
    Value* v_clz = b.build_clz(x);
    Value* v_ctz = b.build_ctz(y);

    Value* sum1 = b.build_add(v_pop, v_clz);
    Value* sum2 = b.build_add(sum1, v_ctz);
    b.build_ret(sum2);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    std::vector<std::pair<int64_t, int64_t>> test_pairs = {
        {0, 0},
        {1, 2},
        {-1, 1},
        {0x5555555555555555LL, 0x3333333333333333LL},
        {0x123456789ABCDEF0LL, 0x0FEDCBA987654321LL},
        {0x8000000000000000LL, 0x7FFFFFFFFFFFFFFFLL},
        {1024, 2048}
    };

    for (const auto& [a, c] : test_pairs) {
        assert_diff(mod, "bitwise_tests", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(c)});
    }
}

TEST_CASE("Differential Exec - Signed and Unsigned Division and Modulo") {
    Module mod("diff_div_mod");
    Function* fn_sdiv = mod.create_function("test_sdiv", Type::i64(), {Type::i64(), Type::i64()});
    Function* fn_udiv = mod.create_function("test_udiv", Type::i64(), {Type::i64(), Type::i64()});
    Function* fn_smod = mod.create_function("test_smod", Type::i64(), {Type::i64(), Type::i64()});
    Function* fn_umod = mod.create_function("test_umod", Type::i64(), {Type::i64(), Type::i64()});

    {
        Builder b(mod);
        b.set_function(fn_sdiv);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_sdiv(x, y));
        fn_sdiv->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_sdiv));
    }
    {
        Builder b(mod);
        b.set_function(fn_udiv);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_udiv(x, y));
        fn_udiv->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_udiv));
    }
    {
        Builder b(mod);
        b.set_function(fn_smod);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_smod(x, y));
        fn_smod->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_smod));
    }
    {
        Builder b(mod);
        b.set_function(fn_umod);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_umod(x, y));
        fn_umod->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_umod));
    }

    std::vector<std::pair<int64_t, int64_t>> div_pairs = {
        {100, 3},
        {-100, 3},
        {100, -3},
        {-100, -3},
        {0x7FFFFFFFFFFFFFFFLL, 2},
        {0x8000000000000000LL, 2},
        {0xFFFFFFFF00000000LL, 0x10000000LL},
        {42, 42},
        {0, 10}
    };

    for (const auto& [a, c] : div_pairs) {
        assert_diff(mod, "test_sdiv", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(c)});
        assert_diff(mod, "test_udiv", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(c)});
        assert_diff(mod, "test_smod", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(c)});
        assert_diff(mod, "test_umod", {RuntimeValue::from_i64(a), RuntimeValue::from_i64(c)});
    }
}

TEST_CASE("Differential Exec - Float Arithmetic and Conversions") {
    Module mod("diff_float");
    Function* fn = mod.create_function("float_ops", Type::f64(), {Type::f64(), Type::f64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::f64());
    Value* y = b.add_block_param(entry, Type::f64());
    Value* i = b.add_block_param(entry, Type::i64());

    Value* f_add = b.build_add(x, y);
    Value* f_sub = b.build_sub(x, y);
    Value* f_mul = b.build_mul(f_add, f_sub);
    Value* f_div = b.build_sdiv(f_mul, y);
    Value* f_neg = b.build_neg(f_div);

    Value* f_from_i = b.build_sitofp_f64_i64(i);
    Value* combined = b.build_add(f_neg, f_from_i);

    Value* as_int = b.build_fptosi_i64(combined);
    Value* back_f = b.build_sitofp_f64_i64(as_int);
    b.build_ret(back_f);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    std::vector<std::tuple<double, double, int64_t>> test_tuples = {
        {10.5, 2.5, 4},
        {-5.0, 3.0, 10},
        {100.25, 4.0, -5},
        {1.0, 1.0, 0}
    };

    for (const auto& [fx, fy, ival] : test_tuples) {
        assert_diff(mod, "float_ops", {RuntimeValue::from_f64(fx), RuntimeValue::from_f64(fy), RuntimeValue::from_i64(ival)});
    }
}

TEST_CASE("Differential Exec - Memory Loads and Stores with Indexed Addressing") {
    Module mod("diff_memory");
    Function* fn = mod.create_function("array_sum", Type::i64(), {Type::ptr(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.create_block("loop_header");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    fn->append_block(loop_header);
    fn->append_block(loop_body);
    fn->append_block(exit_bb);

    Value* buf_ptr = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_header, {zero, zero});

    b.position_at_end(loop_header);
    Value* idx = b.add_block_param(loop_header, Type::i64());
    Value* acc = b.add_block_param(loop_header, Type::i64());
    Value* cond = b.build_slt(idx, count);
    b.build_br_if(cond, loop_body, exit_bb);

    b.position_at_end(loop_body);
    // Load indexed: buf_ptr[idx * 8 + 0]
    Value* val = b.build_load_indexed(Type::i64(), buf_ptr, idx, 8, 0);
    Value* new_acc = b.build_add(acc, val);
    Value* one = b.build_iconst_i64(1);
    Value* next_idx = b.build_add(idx, one);
    b.build_br(loop_header, {next_idx, new_acc});

    b.position_at_end(exit_bb);
    b.build_ret(acc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    int64_t arr[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(arr);

    assert_diff(mod, "array_sum", {RuntimeValue::from_ptr(ptr_val), RuntimeValue::from_i64(8)});
    assert_diff(mod, "array_sum", {RuntimeValue::from_ptr(ptr_val), RuntimeValue::from_i64(4)});
    assert_diff(mod, "array_sum", {RuntimeValue::from_ptr(ptr_val), RuntimeValue::from_i64(0)});
}

TEST_CASE("Differential Exec - Multi-Argument Calls (Host Calling Convention)") {
    Module mod("diff_multi_arg");
    std::vector<Type> params(8, Type::i64());
    Function* fn = mod.create_function("calc_8", Type::i64(), params);

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    std::vector<Value*> args;
    for (size_t i = 0; i < 8; ++i) {
        args.push_back(b.add_block_param(entry, Type::i64()));
    }

    // (a0 + a1 - a2) * a3 + (a4 - a5) * a6 + a7
    Value* p1 = b.build_sub(b.build_add(args[0], args[1]), args[2]);
    Value* t1 = b.build_mul(p1, args[3]);
    Value* p2 = b.build_sub(args[4], args[5]);
    Value* t2 = b.build_mul(p2, args[6]);
    Value* res = b.build_add(b.build_add(t1, t2), args[7]);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    std::vector<RuntimeValue> run_args = {
        RuntimeValue::from_i64(10),
        RuntimeValue::from_i64(20),
        RuntimeValue::from_i64(5),
        RuntimeValue::from_i64(3),
        RuntimeValue::from_i64(100),
        RuntimeValue::from_i64(30),
        RuntimeValue::from_i64(2),
        RuntimeValue::from_i64(7)
    };

    assert_diff(mod, "calc_8", run_args);
}

TEST_CASE("Differential Exec - Mutual Recursion Across Functions") {
    Module mod("diff_mutual");
    Function* f_even = mod.create_function("is_even", Type::i64(), {Type::i64()});
    Function* f_odd  = mod.create_function("is_odd",  Type::i64(), {Type::i64()});

    // is_even(n): if n == 0 return 1; return is_odd(n - 1)
    {
        Builder b(mod);
        b.set_function(f_even);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* base = b.create_block("base");
        BasicBlock* rec = b.create_block("rec");

        f_even->append_block(base);
        f_even->append_block(rec);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* is_zero = b.build_eq(n, zero);
        b.build_br_if(is_zero, base, rec);

        b.position_at_end(base);
        Value* one_base = b.build_iconst_i64(1);
        b.build_ret(one_base);

        b.position_at_end(rec);
        Value* one_rec = b.build_iconst_i64(1);
        Value* n_minus_1 = b.build_sub(n, one_rec);
        Value* res = b.build_call("is_odd", Type::i64(), {n_minus_1});
        b.build_ret(res);

        f_even->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_even));
    }

    // is_odd(n): if n == 0 return 0; return is_even(n - 1)
    {
        Builder b(mod);
        b.set_function(f_odd);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* base = b.create_block("base");
        BasicBlock* rec = b.create_block("rec");

        f_odd->append_block(base);
        f_odd->append_block(rec);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* is_zero = b.build_eq(n, zero);
        b.build_br_if(is_zero, base, rec);

        b.position_at_end(base);
        b.build_ret(zero);

        b.position_at_end(rec);
        Value* one_rec = b.build_iconst_i64(1);
        Value* n_minus_1 = b.build_sub(n, one_rec);
        Value* res = b.build_call("is_even", Type::i64(), {n_minus_1});
        b.build_ret(res);

        f_odd->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f_odd));
    }

    for (int64_t v : {0, 1, 2, 3, 4, 5, 10, 15}) {
        assert_diff(mod, "is_even", {RuntimeValue::from_i64(v)});
        assert_diff(mod, "is_odd",  {RuntimeValue::from_i64(v)});
    }
}

TEST_CASE("Differential Exec - Generative MIR Fuzzer") {
    std::mt19937_64 rng(1337);

    for (int seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_mod_" + std::to_string(seed);
        Module mod(mod_name);
        Function* fn = mod.create_function("fuzz_fn", Type::i64(), {Type::i64(), Type::i64()});

        Builder b(mod);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());

        std::vector<Value*> pool = {x, y};
        int num_insts = 10 + (seed % 15);

        for (int i = 0; i < num_insts; ++i) {
            uint32_t op_choice = static_cast<uint32_t>(rng() % 10);
            Value* opA = pool[rng() % pool.size()];
            Value* opB = pool[rng() % pool.size()];

            Value* res = nullptr;
            switch (op_choice) {
                case 0:
                    res = b.build_add(opA, opB);
                    break;
                case 1:
                    res = b.build_sub(opA, opB);
                    break;
                case 2:
                    res = b.build_and(opA, opB);
                    break;
                case 3:
                    res = b.build_or(opA, opB);
                    break;
                case 4:
                    res = b.build_xor(opA, opB);
                    break;
                case 5:
                    res = b.build_not(opA);
                    break;
                case 6: {
                    Value* shift_c = b.build_iconst_i64(static_cast<int64_t>(rng() % 16));
                    res = b.build_shl(opA, shift_c);
                    break;
                }
                case 7: {
                    Value* shift_c = b.build_iconst_i64(static_cast<int64_t>(rng() % 16));
                    res = b.build_lshr(opA, shift_c);
                    break;
                }
                case 8:
                    res = b.build_popcnt(opA);
                    break;
                case 9:
                    res = b.build_clz(opA);
                    break;
            }
            if (res) pool.push_back(res);
        }

        b.build_ret(pool.back());

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        for (int t = 0; t < 3; ++t) {
            int64_t vx = static_cast<int64_t>(rng() % 10000);
            int64_t vy = static_cast<int64_t>(rng() % 10000);
            assert_diff(mod, "fuzz_fn", {RuntimeValue::from_i64(vx), RuntimeValue::from_i64(vy)});
        }
    }
}
