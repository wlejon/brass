#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential SCCP - Complex Nested Branching and Constant Folding") {
    Module mod("diff_sccp_branching");
    Builder b(mod);

    // func @nested_decision(%x: i64, %mode: i32) -> i64
    Function* fn = mod.create_function("nested_decision", Type::i64(),
        {Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_pos = b.append_block("b_pos");
    BasicBlock* b_neg = b.append_block("b_neg");
    BasicBlock* b_pos_even = b.append_block("b_pos_even");
    BasicBlock* b_pos_odd = b.append_block("b_pos_odd");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* mode = b.add_block_param(entry, Type::i32());

    // Constant arithmetic mixed with inputs
    Value* c10 = b.build_iconst_i64(10);
    Value* c2 = b.build_iconst_i64(2);
    Value* base = b.build_add(c10, c2); // 12 (constant)

    Value* is_pos = b.build_sgt(x, b.build_iconst_i64(0));
    b.build_br_if(is_pos, b_pos, b_neg);

    // Positive path
    b.position_at_end(b_pos);
    Value* mod2 = b.build_smod(x, c2);
    Value* is_even = b.build_eq(mod2, b.build_iconst_i64(0));
    b.build_br_if(is_even, b_pos_even, b_pos_odd);

    b.position_at_end(b_pos_even);
    Value* res_even = b.build_mul(x, base); // x * 12
    b.build_br(merge, {res_even});

    b.position_at_end(b_pos_odd);
    Value* res_odd = b.build_add(x, base); // x + 12
    b.build_br(merge, {res_odd});

    // Negative path
    b.position_at_end(b_neg);
    Value* res_neg = b.build_sub(b.build_iconst_i64(0), x); // -x
    Value* res_neg_scaled = b.build_add(res_neg, base);
    b.build_br(merge, {res_neg_scaled});

    b.position_at_end(merge);
    Value* final_val = b.add_block_param(merge, Type::i64());
    Value* mode_ext = b.build_sext_i64(mode);
    Value* result = b.build_add(final_val, mode_ext);
    b.build_ret(result);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Run SCCP & CFG simplify
    sccp_module(mod);
    cfg_simplify_module(mod);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "nested_decision", {RuntimeValue::from_i64(10), RuntimeValue::from_i32(1)});
    assert_diff(mod, "nested_decision", {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    assert_diff(mod, "nested_decision", {RuntimeValue::from_i64(-4), RuntimeValue::from_i32(5)});
    assert_diff(mod, "nested_decision", {RuntimeValue::from_i64(0), RuntimeValue::from_i32(-2)});
    assert_diff(mod, "nested_decision", {RuntimeValue::from_i64(100), RuntimeValue::from_i32(42)});
}

TEST_CASE("Differential SCCP - Loop with Constant Induction Step & Accumulation") {
    Module mod("diff_sccp_loop");
    Builder b(mod);

    // func @loop_sum(%n: i64) -> i64
    Function* fn = mod.create_function("loop_sum", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(header, {zero, zero});

    b.position_at_end(header);
    Value* i = b.add_block_param(header, Type::i64());
    Value* acc = b.add_block_param(header, Type::i64());

    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, body, exit);

    b.position_at_end(body);
    Value* one = b.build_iconst_i64(1);
    Value* i_next = b.build_add(i, one);
    Value* scale = b.build_iconst_i64(3);
    Value* term = b.build_mul(i, scale);
    Value* acc_next = b.build_add(acc, term);
    b.build_br(header, {i_next, acc_next});

    b.position_at_end(exit);
    b.build_ret(acc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    sccp_module(mod);
    cfg_simplify_module(mod);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "loop_sum", {RuntimeValue::from_i64(0)});
    assert_diff(mod, "loop_sum", {RuntimeValue::from_i64(1)});
    assert_diff(mod, "loop_sum", {RuntimeValue::from_i64(5)});
    assert_diff(mod, "loop_sum", {RuntimeValue::from_i64(10)});
    assert_diff(mod, "loop_sum", {RuntimeValue::from_i64(50)});
}

TEST_CASE("Differential SCCP - Speculative Guard Elimination with State Capture") {
    Module mod("diff_sccp_guards");
    Builder b(mod);

    // func @spec_opt(%a: i64, %b: i64, %dyn_flag: i32) -> i64
    // Contains:
    //   1) A provably true guard: guard %c1, "stub", [%a, %b] -> eliminated by SCCP!
    //   2) Fast path math: (a * 2) + b + 5
    //   3) Select with dynamic condition
    Function* fn = mod.create_function("spec_opt", Type::i64(),
        {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* a = b.add_block_param(entry, Type::i64());
    Value* b_val = b.add_block_param(entry, Type::i64());
    Value* dyn_flag = b.add_block_param(entry, Type::i32());

    // Provably true condition
    Value* c10 = b.build_iconst_i32(10);
    Value* c5 = b.build_iconst_i32(5);
    Value* cond_true = b.build_sgt(c10, c5); // 10 > 5 -> 1

    // Guard with true condition (provably never fails)
    b.build_guard(cond_true, "fallback_stub", {a, b_val});

    // Computation
    Value* a_scaled = b.build_mul(a, b.build_iconst_i64(2));
    Value* sum = b.build_add(a_scaled, b_val);

    // Select with dynamic flag
    Value* extra = b.build_select(dyn_flag, b.build_iconst_i64(100), b.build_iconst_i64(50));
    Value* res = b.build_add(sum, extra);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    sccp_module(mod);
    cfg_simplify_module(mod);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "spec_opt", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(20), RuntimeValue::from_i32(1)});
    assert_diff(mod, "spec_opt", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(20), RuntimeValue::from_i32(0)});
    assert_diff(mod, "spec_opt", {RuntimeValue::from_i64(-5), RuntimeValue::from_i64(100), RuntimeValue::from_i32(1)});
    assert_diff(mod, "spec_opt", {RuntimeValue::from_i64(0), RuntimeValue::from_i64(0), RuntimeValue::from_i32(0)});
}

TEST_CASE("Differential SCCP - Multi-way Switch Constant Selection and Propagation") {
    Module mod("diff_sccp_switch");
    Builder b(mod);

    // func @switch_kernel(%code: i32, %input: i64) -> i64
    Function* fn = mod.create_function("switch_kernel", Type::i64(),
        {Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* c0 = b.append_block("c0");
    BasicBlock* c1 = b.append_block("c1");
    BasicBlock* c2 = b.append_block("c2");
    BasicBlock* def_bb = b.append_block("def_bb");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* code = b.add_block_param(entry, Type::i32());
    Value* input = b.add_block_param(entry, Type::i64());

    b.build_switch(code, def_bb, {SwitchCase(0, c0), SwitchCase(1, c1), SwitchCase(2, c2)});

    b.position_at_end(c0);
    b.build_br(merge, {b.build_add(input, b.build_iconst_i64(10))});

    b.position_at_end(c1);
    b.build_br(merge, {b.build_mul(input, b.build_iconst_i64(3))});

    b.position_at_end(c2);
    b.build_br(merge, {b.build_sub(input, b.build_iconst_i64(5))});

    b.position_at_end(def_bb);
    b.build_br(merge, {b.build_iconst_i64(0)});

    b.position_at_end(merge);
    Value* out = b.add_block_param(merge, Type::i64());
    b.build_ret(out);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    sccp_module(mod);
    cfg_simplify_module(mod);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "switch_kernel", {RuntimeValue::from_i32(0), RuntimeValue::from_i64(15)});
    assert_diff(mod, "switch_kernel", {RuntimeValue::from_i32(1), RuntimeValue::from_i64(15)});
    assert_diff(mod, "switch_kernel", {RuntimeValue::from_i32(2), RuntimeValue::from_i64(15)});
    assert_diff(mod, "switch_kernel", {RuntimeValue::from_i32(99), RuntimeValue::from_i64(15)});
    assert_diff(mod, "switch_kernel", {RuntimeValue::from_i32(-1), RuntimeValue::from_i64(15)});
}
