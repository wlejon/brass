#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <vector>
#include <cmath>

using namespace brass;
using namespace brass::test;

TEST_CASE("Sched Differential - Arithmetic and Latency Hiding") {
    Module mod("sched_diff_arith");
    Function* fn = mod.create_function("arith_kernel", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* z = b.add_block_param(entry, Type::i64());

    // Path 1 (long latency mul & div)
    Value* m1 = b.build_mul(x, y);
    Value* d1 = b.build_sdiv(m1, z);

    // Path 2 (independent ALU ops)
    Value* a1 = b.build_add(x, z);
    Value* s1 = b.build_sub(y, x);
    Value* o1 = b.build_or(a1, s1);
    Value* x1 = b.build_xor(o1, y);

    // Path 3 (more independent operations)
    Value* sh1 = b.build_shl(z, b.build_iconst_i64(2));
    Value* an1 = b.build_and(sh1, x);

    // Recombine
    Value* r1 = b.build_add(d1, x1);
    Value* r2 = b.build_add(r1, an1);
    b.build_ret(r2);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "arith_kernel", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(20), RuntimeValue::from_i64(5)});
    assert_diff(mod, "arith_kernel", {RuntimeValue::from_i64(123), RuntimeValue::from_i64(456), RuntimeValue::from_i64(7)});
    assert_diff(mod, "arith_kernel", {RuntimeValue::from_i64(-50), RuntimeValue::from_i64(30), RuntimeValue::from_i64(3)});
}

TEST_CASE("Sched Differential - SIMD Vector Arithmetic") {
    Module mod("sched_diff_simd");
    Function* fn = mod.create_function("simd_poly", Type::f32x4(), {Type::f32x4(), Type::f32x4()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* v0 = b.add_block_param(entry, Type::f32x4());
    Value* v1 = b.add_block_param(entry, Type::f32x4());

    Value* v0_sq = b.build_vmul(v0, v0);
    Value* v0_v1 = b.build_vmul(v0, v1);
    Value* sum1 = b.build_vadd(v0_sq, v0_v1);
    Value* res = b.build_vadd(sum1, v1);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    RuntimeValue a = RuntimeValue::from_f32x4(1.0f, 2.0f, 3.0f, 4.0f);
    RuntimeValue c = RuntimeValue::from_f32x4(0.5f, 1.5f, -1.0f, 2.0f);
    assert_diff(mod, "simd_poly", {a, c});
}

TEST_CASE("Sched Differential - High Register Pressure Kernel") {
    Module mod("sched_diff_pressure");
    Function* fn = mod.create_function("pressure_kernel", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* p = b.add_block_param(entry, Type::i64());

    // Generate 18 simultaneous live values to exceed physical register count (14 GPRs)
    std::vector<Value*> live_vals;
    live_vals.reserve(18);
    for (int i = 0; i < 18; ++i) {
        live_vals.push_back(b.build_add(p, b.build_iconst_i64(i + 1)));
    }

    // Interleave operations using all 18 values
    Value* acc = b.build_iconst_i64(0);
    for (int i = 0; i < 18; ++i) {
        if (i % 2 == 0) {
            acc = b.build_add(acc, live_vals[static_cast<size_t>(i)]);
        } else {
            acc = b.build_xor(acc, live_vals[static_cast<size_t>(i)]);
        }
    }
    b.build_ret(acc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "pressure_kernel", {RuntimeValue::from_i64(42)});
    assert_diff(mod, "pressure_kernel", {RuntimeValue::from_i64(100)});
    assert_diff(mod, "pressure_kernel", {RuntimeValue::from_i64(-7)});
}

TEST_CASE("Sched Differential - Floating Point Math") {
    Module mod("sched_diff_fp");
    Function* fn = mod.create_function("fp_kernel", Type::f64(), {Type::f64(), Type::f64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::f64());
    Value* y = b.add_block_param(entry, Type::f64());

    Value* add_val = b.build_add(x, y);
    Value* mul_val = b.build_mul(x, y);
    Value* sub_val = b.build_sub(mul_val, add_val);
    Value* div_val = b.build_sdiv(sub_val, x);
    b.build_ret(div_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "fp_kernel", {RuntimeValue::from_f64(3.5), RuntimeValue::from_f64(2.0)});
    assert_diff(mod, "fp_kernel", {RuntimeValue::from_f64(10.0), RuntimeValue::from_f64(4.5)});
}

TEST_CASE("Sched Differential - Loop Execution with Pipelining") {
    Module mod("sched_diff_loop");
    Builder b(mod);

    Function* fn = mod.create_function("sum_loop", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_bb = b.create_block("loop");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_bb, {zero, zero});

    fn->append_block(loop_bb);
    b.position_at_end(loop_bb);
    Value* i = b.add_block_param(loop_bb, Type::i64());
    Value* sum = b.add_block_param(loop_bb, Type::i64());

    Value* term = b.build_mul(i, b.build_iconst_i64(3));
    Value* next_sum = b.build_add(sum, term);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);

    Value* cond = b.build_slt(next_i, n);
    b.build_br_if(cond, loop_bb, {next_i, next_sum}, exit_bb, {next_sum});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "sum_loop", {RuntimeValue::from_i64(10)});
    assert_diff(mod, "sum_loop", {RuntimeValue::from_i64(50)});
}
