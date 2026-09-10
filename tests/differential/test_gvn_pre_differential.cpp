#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential GVN-PRE - Nested Diamonds and Multi-Branch Calculations") {
    Module mod("diff_nested_pre");
    Builder b(mod);

    // func @nested_pre(%c1: i32, %c2: i32, %a: i64, %b: i64, %c: i64) -> i64
    Function* fn = mod.create_function("nested_pre", Type::i64(),
        {Type::i32(), Type::i32(), Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b_left = b.append_block("b_left");
    BasicBlock* b_right = b.append_block("b_right");
    BasicBlock* b_mid = b.append_block("b_mid");
    BasicBlock* b_mid_left = b.append_block("b_mid_left");
    BasicBlock* b_mid_right = b.append_block("b_mid_right");
    BasicBlock* b_exit = b.append_block("b_exit");

    b.position_at_end(b0);
    Value* c1 = b.add_block_param(b0, Type::i32());
    Value* c2 = b.add_block_param(b0, Type::i32());
    Value* a = b.add_block_param(b0, Type::i64());
    Value* b_val = b.add_block_param(b0, Type::i64());
    Value* c = b.add_block_param(b0, Type::i64());
    b.build_br_if(c1, b_left, b_right);

    // Diamond 1 - Left: computes (a * 7 + b)
    b.position_at_end(b_left);
    Value* mul_l = b.build_mul(a, b.build_iconst_i64(7));
    Value* add_l = b.build_add(mul_l, b_val);
    b.build_br(b_mid, {add_l});

    // Diamond 1 - Right: does other work, no (a * 7 + b)
    b.position_at_end(b_right);
    Value* dummy_r = b.build_add(a, c);
    b.build_br(b_mid, {dummy_r});

    // Diamond 1 - Merge (b_mid): computes (a * 7 + b) partially redundant!
    b.position_at_end(b_mid);
    Value* mid_in = b.add_block_param(b_mid, Type::i64());
    Value* mul_mid = b.build_mul(a, b.build_iconst_i64(7));
    Value* add_mid = b.build_add(mul_mid, b_val);
    Value* acc_mid = b.build_add(mid_in, add_mid);
    b.build_br_if(c2, b_mid_left, b_mid_right);

    // Diamond 2 - Left: computes (b * 3 - c)
    b.position_at_end(b_mid_left);
    Value* mul2_l = b.build_mul(b_val, b.build_iconst_i64(3));
    Value* sub2_l = b.build_sub(mul2_l, c);
    b.build_br(b_exit, {acc_mid, sub2_l});

    // Diamond 2 - Right
    b.position_at_end(b_mid_right);
    Value* dummy2_r = b.build_iconst_i64(10);
    b.build_br(b_exit, {acc_mid, dummy2_r});

    // Diamond 2 - Merge (b_exit): computes (b * 3 - c) partially redundant!
    b.position_at_end(b_exit);
    Value* exit_acc = b.add_block_param(b_exit, Type::i64());
    Value* exit_delta = b.add_block_param(b_exit, Type::i64());
    Value* mul2_exit = b.build_mul(b_val, b.build_iconst_i64(3));
    Value* sub2_exit = b.build_sub(mul2_exit, c);
    Value* sum1 = b.build_add(exit_acc, exit_delta);
    Value* total = b.build_add(sum1, sub2_exit);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Optimize with GVN-PRE
    GvnPreStats stats;
    GvnPreOptions opts;
    opts.stats = &stats;
    bool changed = gvn_pre_module(mod, opts);
    CHECK(changed);
    CHECK(stats.expressions_hoisted > 0);
    CHECK(stats.expressions_eliminated > 0);
    REQUIRE(verify_function(*fn));

    // Verify differential execution (JIT vs Interpreter) across multiple branch paths
    std::vector<std::vector<RuntimeValue>> test_inputs = {
        {RuntimeValue::from_i32(1), RuntimeValue::from_i32(1), RuntimeValue::from_i64(5), RuntimeValue::from_i64(3), RuntimeValue::from_i64(2)},
        {RuntimeValue::from_i32(1), RuntimeValue::from_i32(0), RuntimeValue::from_i64(5), RuntimeValue::from_i64(3), RuntimeValue::from_i64(2)},
        {RuntimeValue::from_i32(0), RuntimeValue::from_i32(1), RuntimeValue::from_i64(5), RuntimeValue::from_i64(3), RuntimeValue::from_i64(2)},
        {RuntimeValue::from_i32(0), RuntimeValue::from_i32(0), RuntimeValue::from_i64(5), RuntimeValue::from_i64(3), RuntimeValue::from_i64(2)},
        {RuntimeValue::from_i32(1), RuntimeValue::from_i32(1), RuntimeValue::from_i64(-4), RuntimeValue::from_i64(12), RuntimeValue::from_i64(7)},
        {RuntimeValue::from_i32(0), RuntimeValue::from_i32(0), RuntimeValue::from_i64(100), RuntimeValue::from_i64(-20), RuntimeValue::from_i64(50)},
    };

    for (const auto& args : test_inputs) {
        assert_diff(mod, "nested_pre", args);
    }
}

TEST_CASE("Differential GVN-PRE - Memory Load Elimination across Complex Branching") {
    Module mod("diff_load_pre");
    Builder b(mod);

    // func @mem_load_pre(%buf: ptr, %c1: i32, %c2: i32, %v: i64) -> i64
    Function* fn = mod.create_function("mem_load_pre", Type::i64(),
        {Type::ptr(), Type::i32(), Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b_a = b.append_block("b_a");
    BasicBlock* b_b = b.append_block("b_b");
    BasicBlock* b_join = b.append_block("b_join");

    b.position_at_end(b0);
    Value* buf = b.add_block_param(b0, Type::ptr());
    Value* c1 = b.add_block_param(b0, Type::i32());
    Value* c2 = b.add_block_param(b0, Type::i32());
    (void)c2;
    Value* v = b.add_block_param(b0, Type::i64());
    b.build_br_if(c1, b_a, b_b);

    // Branch A: loads buf[0]
    b.position_at_end(b_a);
    Value* la = b.build_load(Type::i64(), buf, 0);
    Value* va = b.build_add(la, v);
    b.build_br(b_join, {va});

    // Branch B: stores v to buf[8], does not touch buf[0]
    b.position_at_end(b_b);
    b.build_store(Type::i64(), buf, 8, v);
    Value* vb = b.build_mul(v, b.build_iconst_i64(2));
    b.build_br(b_join, {vb});

    // Join: loads buf[0] (partially redundant!)
    b.position_at_end(b_join);
    Value* in_val = b.add_block_param(b_join, Type::i64());
    Value* l_join = b.build_load(Type::i64(), buf, 0);
    Value* res = b.build_add(in_val, l_join);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    GvnPreStats stats;
    GvnPreOptions opts;
    opts.stats = &stats;
    bool changed = gvn_pre_module(mod, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    int64_t memory_buffer[4] = {42, 0, 0, 0};
    uintptr_t buf_ptr = reinterpret_cast<uintptr_t>(memory_buffer);

    assert_diff(mod, "mem_load_pre", {
        RuntimeValue::from_ptr(buf_ptr), RuntimeValue::from_i32(1), RuntimeValue::from_i32(0), RuntimeValue::from_i64(5)
    });

    assert_diff(mod, "mem_load_pre", {
        RuntimeValue::from_ptr(buf_ptr), RuntimeValue::from_i32(0), RuntimeValue::from_i32(0), RuntimeValue::from_i64(15)
    });
}
