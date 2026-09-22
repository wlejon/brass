#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("Fast Interpreter - Arithmetic i32 and i64") {
    Module mod("arith_mod");
    Builder b(mod);

    // i32 arith: (a + b - 5) * 3 / 2
    Function* fn_i32 = mod.create_function("calc_i32", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn_i32);
    BasicBlock* bb1 = b.append_block("entry");
    Value* a = b.add_block_param(bb1, Type::i32());
    Value* b_param = b.add_block_param(bb1, Type::i32());
    Value* sum = b.build_add(a, b_param);
    Value* c5 = b.build_iconst_i32(5);
    Value* diff = b.build_sub(sum, c5);
    Value* c3 = b.build_iconst_i32(3);
    Value* prod = b.build_mul(diff, c3);
    Value* c2 = b.build_iconst_i32(2);
    Value* quot = b.build_sdiv(prod, c2);
    b.build_ret(quot);

    // i64 arith: (x * y) % m - (-z)
    Function* fn_i64 = mod.create_function("calc_i64", Type::i64(), {Type::i64(), Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn_i64);
    BasicBlock* bb2 = b.append_block("entry");
    Value* x = b.add_block_param(bb2, Type::i64());
    Value* y = b.add_block_param(bb2, Type::i64());
    Value* m = b.add_block_param(bb2, Type::i64());
    Value* z = b.add_block_param(bb2, Type::i64());
    Value* p64 = b.build_mul(x, y);
    Value* mod64 = b.build_smod(p64, m);
    Value* negz = b.build_neg(z);
    Value* res64 = b.build_sub(mod64, negz);
    b.build_ret(res64);

    FastInterpreter interp;
    interp.set_module(&mod);

    // (10 + 20 - 5) * 3 / 2 = 25 * 3 / 2 = 75 / 2 = 37
    RuntimeValue r_i32 = interp.run(*fn_i32, {RuntimeValue::from_i32(10), RuntimeValue::from_i32(20)});
    CHECK_EQ(r_i32.as_i32(), 37);

    // (10000000000LL * 3LL) % 7LL - (-15LL) = 30000000000 % 7 + 15 = 5 + 15 = 20
    RuntimeValue r_i64 = interp.run(*fn_i64, {
        RuntimeValue::from_i64(10000000000LL),
        RuntimeValue::from_i64(3LL),
        RuntimeValue::from_i64(7LL),
        RuntimeValue::from_i64(15LL)
    });
    CHECK_EQ(r_i64.as_i64(), 20LL);
}

TEST_CASE("Fast Interpreter - Float f32 and f64 Arithmetic") {
    Module mod("float_mod");
    Builder b(mod);

    Function* fn_f64 = mod.create_function("calc_f64", Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(fn_f64);
    BasicBlock* bb = b.append_block("entry");
    Value* fa = b.add_block_param(bb, Type::f64());
    Value* fb = b.add_block_param(bb, Type::f64());
    Value* fsum = b.build_add(fa, fb);
    Value* fprod = b.build_mul(fsum, fa);
    Value* fquot = b.build_sdiv(fprod, fb);
    Value* fneg = b.build_neg(fquot);
    b.build_ret(fneg);

    FastInterpreter interp;
    interp.set_module(&mod);

    // fa = 3.0, fb = 2.0 -> (3 + 2) * 3 / 2 = 15 / 2 = 7.5 -> neg = -7.5
    RuntimeValue r = interp.run(*fn_f64, {RuntimeValue::from_f64(3.0), RuntimeValue::from_f64(2.0)});
    CHECK(std::fabs(r.as_f64() - (-7.5)) < 1e-6);
}

TEST_CASE("Fast Interpreter - Float Math Builtins") {
    Module mod("math_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_builtins", Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* x = b.add_block_param(bb, Type::f64());
    Value* y = b.add_block_param(bb, Type::f64());
    Value* z = b.add_block_param(bb, Type::f64());

    Value* sq = b.build_sqrt(x);          // sqrt(16.0) = 4.0
    Value* fm = b.build_fma(sq, y, z);     // 4.0 * 2.5 + 3.2 = 13.2
    Value* fl = b.build_floor(fm);         // floor(13.2) = 13.0
    Value* ce = b.build_ceil(fm);          // ceil(13.2) = 14.0
    Value* diff = b.build_sub(fl, ce);     // 13.0 - 14.0 = -1.0
    Value* ab = b.build_fabs(diff);        // fabs(-1.0) = 1.0
    b.build_ret(ab);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue res = interp.run(*fn, {
        RuntimeValue::from_f64(16.0),
        RuntimeValue::from_f64(2.5),
        RuntimeValue::from_f64(3.2)
    });
    CHECK(std::fabs(res.as_f64() - 1.0) < 1e-6);
}

TEST_CASE("Fast Interpreter - Bitwise and Bit-Manipulation") {
    Module mod("bitwise_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_bitwise", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* val = b.add_block_param(bb, Type::i32());

    // val = 0x00F00000
    // clz(val) = 8
    // ctz(val) = 20
    // popcnt(val) = 4
    Value* v_clz = b.build_clz(val);
    Value* v_ctz = b.build_ctz(val);
    Value* v_pop = b.build_popcnt(val);

    Value* s1 = b.build_add(v_clz, v_ctz); // 28
    Value* s2 = b.build_add(s1, v_pop);    // 32

    Value* c4 = b.build_iconst_i32(4);
    Value* sh = b.build_shl(s2, c4);       // 32 << 4 = 512
    Value* c1 = b.build_iconst_i32(1);
    Value* rsh = b.build_lshr(sh, c1);     // 512 >> 1 = 256
    b.build_ret(rsh);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i32(0x00F00000)});
    CHECK_EQ(res.as_i32(), 256);
}

TEST_CASE("Fast Interpreter - Comparisons and Select") {
    Module mod("cmp_mod");
    Builder b(mod);

    Function* fn = mod.create_function("max_val", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* a = b.add_block_param(bb, Type::i32());
    Value* c = b.add_block_param(bb, Type::i32());

    Value* is_gt = b.build_sgt(a, c);
    Value* sel = b.build_select(is_gt, a, c);
    b.build_ret(sel);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue r1 = interp.run(*fn, {RuntimeValue::from_i32(42), RuntimeValue::from_i32(99)});
    CHECK_EQ(r1.as_i32(), 99);

    RuntimeValue r2 = interp.run(*fn, {RuntimeValue::from_i32(100), RuntimeValue::from_i32(20)});
    CHECK_EQ(r2.as_i32(), 100);
}

TEST_CASE("Fast Interpreter - Conversions and Bitcasts") {
    Module mod("conv_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_conv", Type::i64(), {Type::f64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* f = b.add_block_param(bb, Type::f64());

    // Convert f64 -> i64, then bitcast i64 -> f64, then bitcast back -> i64
    Value* as_int = b.build_fptosi_i64(f);
    Value* as_float_bits = b.build_bitcast_i64_f64(as_int);
    Value* back_to_int = b.build_bitcast_f64_i64(as_float_bits);
    b.build_ret(back_to_int);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_f64(12345.67)});
    CHECK_EQ(res.as_i64(), 12345LL);
}

TEST_CASE("Fast Interpreter - Iterative Loop with Block Parameters (Sum 1 to N)") {
    Module mod("loop_mod");
    Builder b(mod);

    // int sum_n(int n) {
    //   int i = 1, acc = 0;
    //   while (i <= n) { acc += i; i++; }
    //   return acc;
    // }
    Function* fn = mod.create_function("sum_n", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* exit_bb = b.append_block("exit");

    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i32());
    Value* init_i = b.build_iconst_i32(1);
    Value* init_acc = b.build_iconst_i32(0);
    b.build_br(loop_header, {init_i, init_acc});

    b.position_at_end(loop_header);
    Value* cur_i = b.add_block_param(loop_header, Type::i32());
    Value* cur_acc = b.add_block_param(loop_header, Type::i32());
    Value* cond = b.build_sle(cur_i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    b.position_at_end(loop_body);
    Value* next_acc = b.build_add(cur_acc, cur_i);
    Value* c1 = b.build_iconst_i32(1);
    Value* next_i = b.build_add(cur_i, c1);
    b.build_br(loop_header, {next_i, next_acc});

    b.position_at_end(exit_bb);
    b.build_ret(cur_acc);

    FastInterpreter interp;
    interp.set_module(&mod);

    // sum(1..10) = 55
    RuntimeValue r10 = interp.run(*fn, {RuntimeValue::from_i32(10)});
    CHECK_EQ(r10.as_i32(), 55);

    // sum(1..100) = 5050
    RuntimeValue r100 = interp.run(*fn, {RuntimeValue::from_i32(100)});
    CHECK_EQ(r100.as_i32(), 5050);

    // Compare against verification oracle Interpreter!
    Interpreter oracle;
    oracle.set_module(&mod);
    RuntimeValue oracle_res = oracle.run(*fn, {RuntimeValue::from_i32(100)});
    CHECK_EQ(r100.as_i32(), oracle_res.as_i32());
}

TEST_CASE("Fast Interpreter - Collatz Sequence") {
    Module mod("collatz_mod");
    Builder b(mod);

    // Counts steps for Collatz sequence to reach 1:
    // while (n > 1) {
    //   if (n % 2 == 0) n /= 2; else n = 3 * n + 1;
    //   steps++;
    // }
    Function* fn = mod.create_function("collatz", Type::i32(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* even_bb = b.append_block("even_bb");
    BasicBlock* odd_bb = b.append_block("odd_bb");
    BasicBlock* exit_bb = b.append_block("exit");

    b.position_at_end(entry);
    Value* init_n = b.add_block_param(entry, Type::i64());
    Value* zero_steps = b.build_iconst_i32(0);
    b.build_br(loop_header, {init_n, zero_steps});

    b.position_at_end(loop_header);
    Value* n = b.add_block_param(loop_header, Type::i64());
    Value* steps = b.add_block_param(loop_header, Type::i32());
    Value* c1_64 = b.build_iconst_i64(1);
    Value* is_gt_1 = b.build_sgt(n, c1_64);
    b.build_br_if(is_gt_1, loop_body, {}, exit_bb, {});

    b.position_at_end(loop_body);
    Value* c2_64 = b.build_iconst_i64(2);
    Value* rem = b.build_smod(n, c2_64);
    Value* c0_64 = b.build_iconst_i64(0);
    Value* is_even = b.build_eq(rem, c0_64);
    b.build_br_if(is_even, even_bb, {}, odd_bb, {});

    b.position_at_end(even_bb);
    Value* n_even = b.build_sdiv(n, c2_64);
    Value* c1_32 = b.build_iconst_i32(1);
    Value* steps_even = b.build_add(steps, c1_32);
    b.build_br(loop_header, {n_even, steps_even});

    b.position_at_end(odd_bb);
    Value* c3_64 = b.build_iconst_i64(3);
    Value* prod = b.build_mul(n, c3_64);
    Value* n_odd = b.build_add(prod, c1_64);
    Value* c1_32_odd = b.build_iconst_i32(1);
    Value* steps_odd = b.build_add(steps, c1_32_odd);
    b.build_br(loop_header, {n_odd, steps_odd});

    b.position_at_end(exit_bb);
    b.build_ret(steps);

    FastInterpreter interp;
    interp.set_module(&mod);

    // Collatz(27) takes 111 steps to reach 1
    RuntimeValue r27 = interp.run(*fn, {RuntimeValue::from_i64(27LL)});
    CHECK_EQ(r27.as_i32(), 111);

    // Compare against verification oracle
    Interpreter oracle;
    oracle.set_module(&mod);
    CHECK_EQ(r27.as_i32(), oracle.run(*fn, {RuntimeValue::from_i64(27LL)}).as_i32());
}

TEST_CASE("Fast Interpreter - Recursive Fibonacci") {
    Module mod("fib_mod");
    Builder b(mod);

    // int fib(int n) {
    //   if (n <= 1) return n;
    //   return fib(n - 1) + fib(n - 2);
    // }
    Function* fn = mod.create_function("fib", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_bb = b.append_block("base_bb");
    BasicBlock* rec_bb = b.append_block("rec_bb");

    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i32());
    Value* c1 = b.build_iconst_i32(1);
    Value* is_base = b.build_sle(n, c1);
    b.build_br_if(is_base, base_bb, {}, rec_bb, {});

    b.position_at_end(base_bb);
    b.build_ret(n);

    b.position_at_end(rec_bb);
    Value* n_minus_1 = b.build_sub(n, c1);
    Value* fib1 = b.build_call("fib", Type::i32(), {n_minus_1});
    Value* c2 = b.build_iconst_i32(2);
    Value* n_minus_2 = b.build_sub(n, c2);
    Value* fib2 = b.build_call("fib", Type::i32(), {n_minus_2});
    Value* res = b.build_add(fib1, fib2);
    b.build_ret(res);

    FastInterpreter interp;
    interp.set_module(&mod);

    // fib(10) = 55
    RuntimeValue r10 = interp.run(*fn, {RuntimeValue::from_i32(10)});
    CHECK_EQ(r10.as_i32(), 55);

    // fib(12) = 144
    RuntimeValue r12 = interp.run(*fn, {RuntimeValue::from_i32(12)});
    CHECK_EQ(r12.as_i32(), 144);

    // Compare against verification oracle
    Interpreter oracle;
    oracle.set_module(&mod);
    CHECK_EQ(r12.as_i32(), oracle.run(*fn, {RuntimeValue::from_i32(12)}).as_i32());
}

TEST_CASE("Fast Interpreter - Memory Operations") {
    Module mod("mem_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_mem", Type::i32(), {});
    b.set_function(fn);
    b.append_block("entry");

    // Allocate 32 bytes on frame
    Value* buf = b.build_alloca(32, 8);

    // Store 42 at offset 0
    Value* c42 = b.build_iconst_i32(42);
    b.build_store(Type::i32(), buf, 0, c42);

    // Store 99 at offset 4
    Value* c99 = b.build_iconst_i32(99);
    b.build_store(Type::i32(), buf, 4, c99);

    // Load back and add
    Value* v0 = b.build_load(Type::i32(), buf, 0);
    Value* v4 = b.build_load(Type::i32(), buf, 4);
    Value* sum = b.build_add(v0, v4);
    b.build_ret(sum);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue r = interp.run(*fn);
    CHECK_EQ(r.as_i32(), 141);
}

TEST_CASE("Fast Interpreter - Host External Function Calls") {
    Module mod("host_mod");
    Builder b(mod);

    Function* fn = mod.create_function("call_host", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* x = b.add_block_param(bb, Type::i32());
    Value* call_res = b.build_call("my_host_callback", Type::i32(), {x});
    b.build_ret(call_res);

    FastInterpreter interp;
    interp.set_module(&mod);

    int callback_count = 0;
    interp.register_external_function("my_host_callback", [&](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        callback_count++;
        int32_t val = args[0].as_i32();
        return RuntimeValue::from_i32(val * 10);
    });

    RuntimeValue r = interp.run(*fn, {RuntimeValue::from_i32(7)});
    CHECK_EQ(callback_count, 1);
    CHECK_EQ(r.as_i32(), 70);
}

TEST_CASE("Fast Interpreter - Function Pointers and Indirect Calls") {
    Module mod("indir_mod");
    Builder b(mod);

    Function* target = mod.create_function("target_add", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(target);
    BasicBlock* tb = b.append_block("entry");
    Value* ta = b.add_block_param(tb, Type::i32());
    Value* tb_val = b.add_block_param(tb, Type::i32());
    b.build_ret(b.build_add(ta, tb_val));

    Function* caller = mod.create_function("call_indir", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(caller);
    BasicBlock* cb = b.append_block("entry");
    Value* ca = b.add_block_param(cb, Type::i32());
    Value* cb_val = b.add_block_param(cb, Type::i32());
    Value* target_addr = b.build_func_addr("target_add");
    Value* indir_res = b.build_call_indirect(target_addr, Type::i32(), {ca, cb_val});
    b.build_ret(indir_res);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue r = interp.run(*caller, {RuntimeValue::from_i32(15), RuntimeValue::from_i32(25)});
    CHECK_EQ(r.as_i32(), 40);
}

TEST_CASE("Fast Interpreter - Dynamic Patching") {
    Module mod("patch_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_patch", Type::i32(), {});
    b.set_function(fn);
    b.append_block("entry");
    Value* cv = b.build_patchable_const_i32("my_patch_var", 100);
    b.build_ret(cv);

    FastInterpreter interp;
    interp.set_module(&mod);

    interp.patch_const("my_patch_var", 999);
    RuntimeValue r = interp.run(*fn);
    CHECK_EQ(r.as_i32(), 999);
}

TEST_CASE("Fast Interpreter - Execution Limits") {
    Module mod("limit_mod");
    Builder b(mod);

    // Infinite loop
    Function* fn = mod.create_function("infinite_loop", Type::void_type(), {});
    b.set_function(fn);
    BasicBlock* loop = b.append_block("loop");
    b.build_br(loop);

    FastInterpreter interp;
    interp.set_module(&mod);
    interp.set_max_instructions(500);

    bool threw = false;
    try {
        interp.run(*fn);
    } catch (const InterpreterException&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Fast Interpreter - Vector SIMD Operations") {
    Module mod("simd_mod");
    Builder b(mod);

    Function* fn = mod.create_function("simd_test", Type::i32(), {});
    b.set_function(fn);
    b.append_block("entry");

    Value* vz = b.build_vzero(Type::i32x4());
    Value* c10 = b.build_iconst_i32(10);
    Value* vb = b.build_vbroadcast(Type::i32x4(), c10);
    Value* vsum = b.build_vadd(vz, vb);
    Value* lane0 = b.build_vextract_lane(vsum, 0);
    b.build_ret(lane0);

    FastInterpreter interp;
    interp.set_module(&mod);

    RuntimeValue r = interp.run(*fn);
    CHECK_EQ(r.as_i32(), 10);
}
