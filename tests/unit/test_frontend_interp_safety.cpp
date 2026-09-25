#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <cstdint>
#include <vector>

using namespace brass;
using namespace brass::test;

// 1. alloca_ execution in Interpreter: allocate memory, store values, read back.
TEST_CASE("Frontend & Interp Safety - alloca execution in Interpreter") {
    Module mod("interp_alloca_mod");
    Function* fn = mod.create_function("test_interp_alloca", Type::i64(), {});
    Builder b(*fn);

    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);

    Value* buf1 = b.build_alloca(32, 16);
    Value* buf2 = b.build_alloca(32, 16);

    // Store into buf1 and buf2
    b.build_store(Type::i64(), buf1, 0, b.build_iconst_i64(42));
    b.build_store(Type::i64(), buf1, 8, b.build_iconst_i64(99));
    b.build_store(Type::i64(), buf2, 0, b.build_iconst_i64(1000));
    b.build_store(Type::i64(), buf2, 8, b.build_iconst_i64(2000));

    // Read back and sum
    Value* v1 = b.build_load(Type::i64(), buf1, 0);
    Value* v2 = b.build_load(Type::i64(), buf1, 8);
    Value* v3 = b.build_load(Type::i64(), buf2, 0);
    Value* v4 = b.build_load(Type::i64(), buf2, 8);

    Value* sum1 = b.build_add(v1, v2);
    Value* sum2 = b.build_add(v3, v4);
    Value* total = b.build_add(sum1, sum2);
    b.build_ret(total);

    Interpreter interp;
    RuntimeValue res = interp.run(*fn, {});
    CHECK_EQ(res.as_i64(), 42 + 99 + 1000 + 2000);
}

// 2. alloca_ compilation and execution in Baseline JIT
TEST_CASE("Frontend & Interp Safety - alloca execution in Baseline JIT") {
    Module mod("jit_alloca_mod");
    Function* fn = mod.create_function("test_jit_alloca", Type::i64(), {});
    Builder b(*fn);

    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);

    Value* buf1 = b.build_alloca(32, 16);
    Value* buf2 = b.build_alloca(32, 16);

    b.build_store(Type::i64(), buf1, 0, b.build_iconst_i64(100));
    b.build_store(Type::i64(), buf1, 8, b.build_iconst_i64(250));
    b.build_store(Type::i64(), buf2, 0, b.build_iconst_i64(500));
    b.build_store(Type::i64(), buf2, 8, b.build_iconst_i64(150));

    Value* v1 = b.build_load(Type::i64(), buf1, 0);
    Value* v2 = b.build_load(Type::i64(), buf1, 8);
    Value* v3 = b.build_load(Type::i64(), buf2, 0);
    Value* v4 = b.build_load(Type::i64(), buf2, 8);

    Value* s1 = b.build_add(v1, v2);
    Value* s2 = b.build_add(v3, v4);
    Value* tot = b.build_add(s1, s2);
    b.build_ret(tot);

    codegen::BaselineJitCompiler jit;
    auto compiled = jit.compile(*fn);
    CHECK(compiled.is_valid());

    RuntimeValue res = compiled.invoke({});
    CHECK_EQ(res.as_i64(), 100 + 250 + 500 + 150);
}

// Dynamic call (>16 args) with alloca buffer in Baseline JIT
static int64_t helper_sum_18_args(int64_t callee, int64_t this_val, int32_t argc, const int64_t* argv) {
    (void)callee;
    (void)this_val;
    int64_t sum = 0;
    for (int32_t i = 0; i < argc; ++i) {
        sum += argv[i];
    }
    return sum;
}

TEST_CASE("Frontend & Interp Safety - dynamic call >16 args with alloca buffer") {
    Module mod("dyn_call_18_mod");
    Function* fn = mod.create_function("test_dyn_call_18", Type::i64(), {});
    Builder b(*fn);

    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);

    const size_t argc = 18;
    Value* argv = b.build_alloca(static_cast<uint32_t>(argc * sizeof(uint64_t)), 8);
    for (size_t i = 0; i < argc; ++i) {
        b.build_store(Type::i64(), argv, static_cast<int32_t>(i * sizeof(uint64_t)), b.build_iconst_i64(static_cast<int64_t>(i + 1)));
    }
    Value* callee_dummy = b.build_iconst_i64(0);
    Value* this_dummy = b.build_iconst_i64(0);
    Value* argc_val = b.build_iconst_i32(static_cast<int32_t>(argc));
    Value* call_res = b.build_call("test_helper_sum_18", Type::i64(), {callee_dummy, this_dummy, argc_val, argv});
    b.build_ret(call_res);

    codegen::BaselineJitCompiler jit;
    jit.register_external_symbol("test_helper_sum_18", reinterpret_cast<void*>(&helper_sum_18_args));
    auto compiled = jit.compile(*fn);
    CHECK(compiled.is_valid());

    RuntimeValue res = compiled.invoke({});
    // sum of 1..18 = 18 * 19 / 2 = 171
    CHECK_EQ(res.as_i64(), 171);
}

// 4. Interpreter instruction limit: execute exactly N instructions with max_instructions = N without double counting
TEST_CASE("Frontend & Interp Safety - Interpreter instruction limit exact counting") {
    Module mod("inst_limit_mod");
    Function* fn = mod.create_function("test_limit", Type::i32(), {});
    Builder b(*fn);

    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);

    // Exactly 5 instructions:
    // 1: iconst 10
    // 2: iconst 20
    // 3: add
    // 4: iconst 5
    // 5: ret
    Value* v1 = b.build_iconst_i32(10);
    Value* v2 = b.build_iconst_i32(20);
    Value* sum = b.build_add(v1, v2);
    (void)b.build_iconst_i32(5);
    b.build_ret(sum);

    Interpreter interp;
    interp.set_max_instructions(5);
    // Executing exactly 5 instructions must succeed without throwing
    RuntimeValue res = interp.run(*fn, {});
    CHECK_EQ(res.as_i32(), 30);

    // With max_instructions = 4, executing 5 instructions must exceed limit and throw
    Interpreter interp2;
    interp2.set_max_instructions(4);
    bool threw = false;
    try {
        interp2.run(*fn, {});
    } catch (const InterpreterException&) {
        threw = true;
    }
    CHECK(threw);
}

// 5. f32 comparisons: -0.0f == +0.0f is true, -1.0f < +1.0f is true
TEST_CASE("Frontend & Interp Safety - f32 comparisons IEEE 754") {
    RuntimeValue neg_zero = RuntimeValue::from_f32(-0.0f);
    RuntimeValue pos_zero = RuntimeValue::from_f32(0.0f);
    RuntimeValue neg_one = RuntimeValue::from_f32(-1.0f);
    RuntimeValue pos_one = RuntimeValue::from_f32(1.0f);

    CHECK_EQ(val_eq(neg_zero, pos_zero).as_i32(), 1);
    CHECK_EQ(val_ne(neg_zero, pos_zero).as_i32(), 0);
    CHECK_EQ(val_sle(neg_zero, pos_zero).as_i32(), 1);
    CHECK_EQ(val_sge(neg_zero, pos_zero).as_i32(), 1);

    CHECK_EQ(val_slt(neg_one, pos_one).as_i32(), 1);
    CHECK_EQ(val_slt(pos_one, neg_one).as_i32(), 0);
    CHECK_EQ(val_sle(neg_one, pos_one).as_i32(), 1);
    CHECK_EQ(val_sgt(pos_one, neg_one).as_i32(), 1);
    CHECK_EQ(val_sgt(neg_one, pos_one).as_i32(), 0);
    CHECK_EQ(val_sge(pos_one, neg_one).as_i32(), 1);

    // Also verify in Interpreter through MIR instructions
    Module mod("f32_cmp_mod");
    Function* fn = mod.create_function("f32_cmp", Type::i32(), {});
    Builder b(*fn);

    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);

    Value* f_neg0 = b.build_fconst_f32(-0.0f);
    Value* f_pos0 = b.build_fconst_f32(0.0f);
    Value* cmp_zero = b.build_eq(f_neg0, f_pos0);

    Value* f_neg1 = b.build_fconst_f32(-1.0f);
    Value* f_pos1 = b.build_fconst_f32(1.0f);
    Value* cmp_neg = b.build_slt(f_neg1, f_pos1);

    Value* both = b.build_and(cmp_zero, cmp_neg);
    b.build_ret(both);

    Interpreter interp;
    RuntimeValue r = interp.run(*fn, {});
    CHECK_EQ(r.as_i32(), 1);
}
