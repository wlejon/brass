#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/range_analysis.hpp>
#include <iostream>

using namespace brass;

TEST_CASE("RangeAnalysis - ValueRange Interval Arithmetic Primitives") {
    ValueRange r0 = ValueRange::constant(10);
    CHECK(r0.is_constant());
    CHECK(r0.is_non_negative());
    CHECK_EQ(r0.min_val, 10);
    CHECK_EQ(r0.max_val, 10);

    ValueRange r1 = ValueRange::range(0, 100);
    CHECK(!r1.is_constant());
    CHECK(r1.is_non_negative());
    CHECK(r1.contains(0));
    CHECK(r1.contains(50));
    CHECK(r1.contains(100));
    CHECK(!r1.contains(-1));
    CHECK(!r1.contains(101));

    // Subrange
    CHECK(r0.is_subrange_of(r1));
    CHECK(!r1.is_subrange_of(r0));

    // Intersection
    ValueRange r2 = ValueRange::range(50, 150);
    ValueRange inter = r1;
    inter.intersect_with(r2);
    CHECK_EQ(inter.min_val, 50);
    CHECK_EQ(inter.max_val, 100);

    // Union
    ValueRange un = r1;
    un.union_with(r2);
    CHECK_EQ(un.min_val, 0);
    CHECK_EQ(un.max_val, 150);

    // Empty
    ValueRange empty = ValueRange::empty();
    CHECK(empty.is_empty());
    ValueRange inter_empty = r1;
    inter_empty.intersect_with(empty);
    CHECK(inter_empty.is_empty());
}

TEST_CASE("RangeAnalysis - Arithmetic and Bitwise Interval Operations") {
    ValueRange a = ValueRange::range(10, 20);
    ValueRange b = ValueRange::range(2, 5);

    // Add
    ValueRange add_r = ValueRange::add(a, b);
    CHECK_EQ(add_r.min_val, 12);
    CHECK_EQ(add_r.max_val, 25);

    // Sub
    ValueRange sub_r = ValueRange::sub(a, b);
    CHECK_EQ(sub_r.min_val, 5);
    CHECK_EQ(sub_r.max_val, 18);

    // Mul
    ValueRange mul_r = ValueRange::mul(a, b);
    CHECK_EQ(mul_r.min_val, 20);
    CHECK_EQ(mul_r.max_val, 100);

    // Saturated Add
    ValueRange huge = ValueRange::range(INT64_MAX - 10, INT64_MAX);
    ValueRange sat_r = ValueRange::add(huge, ValueRange::constant(20));
    CHECK_EQ(sat_r.max_val, INT64_MAX);

    // Bitwise AND with mask
    ValueRange nonneg = ValueRange::range(0, 1000);
    ValueRange mask = ValueRange::constant(7);
    ValueRange and_r = ValueRange::and_(nonneg, mask);
    CHECK(and_r.is_non_negative());
    CHECK_EQ(and_r.min_val, 0);
    CHECK_EQ(and_r.max_val, 7);

    // Bitwise OR
    ValueRange or_r = ValueRange::or_(ValueRange::range(0, 3), ValueRange::range(0, 4));
    CHECK_EQ(or_r.min_val, 0);
    CHECK(or_r.max_val >= 7);

    // Shifts
    ValueRange shl_r = ValueRange::shl(ValueRange::range(1, 4), ValueRange::constant(3));
    CHECK_EQ(shl_r.min_val, 8);
    CHECK_EQ(shl_r.max_val, 32);

    ValueRange ashr_r = ValueRange::ashr(ValueRange::range(16, 64), ValueRange::constant(2));
    CHECK_EQ(ashr_r.min_val, 4);
    CHECK_EQ(ashr_r.max_val, 16);

    // Trunc & Extensions
    ValueRange ext = ValueRange::zext(ValueRange::range(0, 255));
    CHECK_EQ(ext.min_val, 0);
    CHECK_EQ(ext.max_val, 255);

    ValueRange sext_r = ValueRange::sext(ValueRange::range(-10, 10));
    CHECK_EQ(sext_r.min_val, -10);
    CHECK_EQ(sext_r.max_val, 10);
}

TEST_CASE("RangeAnalysis - Evaluation of SSA Arithmetic in Function") {
    Module mod("test_ssa_ranges");
    Builder b(mod);

    // func @eval_test(%x: i64) -> i64
    // b0:
    //   %m = and %x, 15
    //   %p = add %m, 5
    //   %q = mul %p, 2
    //   ret %q
    Function* fn = mod.create_function("eval_test", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    b.position_at_end(b0);
    Value* x = b.add_block_param(b0, Type::i64());
    Value* m = b.build_and(x, b.build_iconst_i64(15));
    Value* p = b.build_add(m, b.build_iconst_i64(5));
    Value* q = b.build_mul(p, b.build_iconst_i64(2));
    b.build_ret(q);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    RangeAnalysis ra(*fn);
    ValueRange m_range = ra.get_range(m);
    CHECK_EQ(m_range.min_val, 0);
    CHECK_EQ(m_range.max_val, 15);

    ValueRange p_range = ra.get_range(p);
    CHECK_EQ(p_range.min_val, 5);
    CHECK_EQ(p_range.max_val, 20);

    ValueRange q_range = ra.get_range(q);
    CHECK_EQ(q_range.min_val, 10);
    CHECK_EQ(q_range.max_val, 40);
}

TEST_CASE("RangeAnalysis - Path-Sensitive Branch Range Narrowing") {
    Module mod("test_path_narrowing");
    Builder b(mod);

    // func @path_test(%x: i64) -> i64
    // b0:
    //   %c = slt %x, 10
    //   br_if %c, bb_then, bb_else
    // bb_then:
    //   ret %x
    // bb_else:
    //   ret %x
    Function* fn = mod.create_function("path_test", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* bb_then = b.append_block("bb_then");
    BasicBlock* bb_else = b.append_block("bb_else");

    b.position_at_end(b0);
    Value* x = b.add_block_param(b0, Type::i64());
    Value* c = b.build_slt(x, b.build_iconst_i64(10));
    b.build_br_if(c, bb_then, bb_else);

    b.position_at_end(bb_then);
    b.build_ret(x);

    b.position_at_end(bb_else);
    b.build_ret(x);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    DominatorTree dom(*fn);
    LoopAnalysis loops(*fn, dom);
    RangeAnalysis ra(*fn, dom, loops);

    ValueRange r_then = ra.get_range_at(x, bb_then);
    CHECK(r_then.max_val <= 9);

    ValueRange r_else = ra.get_range_at(x, bb_else);
    CHECK(r_else.min_val >= 10);
}

TEST_CASE("RangeAnalysis - Loop Induction Variable Range Inference") {
    Module mod("test_loop_iv_range");
    Builder b(mod);

    // func @loop_iv() -> i64
    // ph:
    //   br hdr(0, 0)
    // hdr(%i: i64, %sum: i64):
    //   %cond = slt %i, 16
    //   br_if %cond, body, exit
    // body:
    //   %sum2 = add %sum, %i
    //   %i_next = add %i, 1
    //   br hdr(%i_next, %sum2)
    // exit:
    //   ret %sum
    Function* fn = mod.create_function("loop_iv", Type::i64(), {});
    b.set_function(fn);

    BasicBlock* ph = b.append_block("ph");
    BasicBlock* hdr = b.append_block("hdr");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    b.position_at_end(ph);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(hdr, {zero, zero});

    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    Value* sum = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(i, b.build_iconst_i64(16));
    b.build_br_if(cond, body, exit);

    b.position_at_end(body);
    Value* sum2 = b.build_add(sum, i);
    Value* i_next = b.build_add(i, b.build_iconst_i64(1));
    b.build_br(hdr, {i_next, sum2});

    b.position_at_end(exit);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    DominatorTree dom(*fn);
    LoopAnalysis loops(*fn, dom);
    RangeAnalysis ra(*fn, dom, loops);

    ValueRange i_body_range = ra.get_range_at(i, body);
    CHECK_EQ(i_body_range.min_val, 0);
    CHECK_EQ(i_body_range.max_val, 15);
    CHECK(i_body_range.is_non_negative());
}
