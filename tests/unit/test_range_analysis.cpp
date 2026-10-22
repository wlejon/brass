#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/bounds_check_elim.hpp>
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

TEST_CASE("RangeAnalysis - ValueRange::sext Constant Folding and Overflow Wrapping") {
    // 1. Constant out of signed 32-bit range wraps via static_cast<int32_t>
    ValueRange c_wrap = ValueRange::sext(ValueRange::constant(0x80000000LL));
    CHECK(c_wrap.is_constant());
    CHECK_EQ(c_wrap.min_val, static_cast<int64_t>(INT32_MIN));

    ValueRange c_pos = ValueRange::sext(ValueRange::constant(0x10000002ALL));
    CHECK(c_pos.is_constant());
    CHECK_EQ(c_pos.min_val, 42);

    // 2. Range entirely within [INT32_MIN, INT32_MAX] is preserved
    ValueRange r_in = ValueRange::sext(ValueRange::range(-100, 200));
    CHECK_EQ(r_in.min_val, -100);
    CHECK_EQ(r_in.max_val, 200);

    // 3. Range exceeding INT32_MAX can wrap to negative 32-bit values and must not clamp to [0, INT32_MAX]
    ValueRange r_overflow = ValueRange::sext(ValueRange::range(0, 0x80000000LL));
    CHECK_EQ(r_overflow.min_val, static_cast<int64_t>(INT32_MIN));
    CHECK_EQ(r_overflow.max_val, static_cast<int64_t>(INT32_MAX));
}

TEST_CASE("BCE - ImpliedCheckFolder Soundness: 32/64-bit DomExpr, trunc_i32/overflow bounds, and negative slt") {
    // 1. 32-bit wrapping shl.i32 must not canonicalize to 64-bit shl.i64 (or vice versa)
    {
        Module mod("test_bce_dom_expr_width");
        Builder b(mod);
        Function* fn = mod.create_function("dom_width", Type::i32(), {Type::i32(), Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* bb_pos = b.append_block("bb_pos");
        BasicBlock* bb_check = b.append_block("bb_check");
        BasicBlock* bb_exit = b.append_block("bb_exit");

        b.position_at_end(entry);
        Value* a32 = b.add_block_param(entry, Type::i32());
        Value* len64 = b.add_block_param(entry, Type::i64());
        Value* non_neg = b.build_sge(a32, b.build_iconst_i32(0));
        b.build_br_if(non_neg, bb_pos, {}, bb_exit, {b.build_iconst_i32(0)});

        b.position_at_end(bb_pos);
        // 32-bit shift (can wrap for e.g. a32 = 0x40000001 -> shl32 = 4)
        Value* shl32 = b.build_shl(a32, b.build_iconst_i32(2));
        Value* shl32_zext = b.build_zext_i64(shl32);
        Value* guard_32 = b.build_ult(shl32_zext, len64);
        b.build_br_if(guard_32, bb_check, {}, bb_exit, {b.build_iconst_i32(0)});

        b.position_at_end(bb_check);
        // 64-bit shift of zext(a32): for a32 = 0x40000001, shl64 = 0x100000004 != 4
        Value* a64 = b.build_zext_i64(a32);
        Value* shl64 = b.build_shl(a64, b.build_iconst_i64(2));
        Value* cmp64 = b.build_ult(shl64, len64);
        b.build_br(bb_exit, {cmp64});

        b.position_at_end(bb_exit);
        Value* res = b.add_block_param(bb_exit, Type::i32());
        b.build_ret(res);

        fn->rebuild_cfg_predecessors();
        RangeAnalysisStats stats;
        RangeAnalysisOptions opts;
        opts.stats = &stats;
        run_bounds_check_elimination(*fn, mod, opts);
        CHECK_EQ(stats.checks_implied, size_t{0});
    }

    // 2. is_length_bound must not strip trunc_i32 on arbitrary i64 or accept overflowing cn << shift,
    //    and check_implied must not treat slt with negative RHS as implying ult.
    {
        Module mod("test_bce_neg_slt_and_overflow_bound");
        Builder b(mod);
        Function* fn = mod.create_function("neg_slt", Type::i32(), {Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* bb_body = b.append_block("bb_body");
        BasicBlock* bb_exit = b.append_block("bb_exit");

        b.position_at_end(entry);
        Value* idx = b.add_block_param(entry, Type::i64());
        Value* n64 = b.add_block_param(entry, Type::i64());
        Value* idx_nn = b.build_sge(idx, b.build_iconst_i64(0));
        b.build_br_if(idx_nn, bb_body, {}, bb_exit, {b.build_iconst_i32(0)});

        b.position_at_end(bb_body);
        // Guard with negative RHS (-8): slt(idx, -8) does NOT imply ult(idx << 3, -8 >> 3)
        Value* neg_bound = b.build_iconst_i64(-8);
        Value* bad_slt = b.build_slt(idx, neg_bound);
        b.build_guard(bad_slt, "@deopt");
        // Also unbounded shift cbl = n64 << 3 (can overflow 64 bits)
        Value* cbl = b.build_shl(n64, b.build_iconst_i64(3));
        Value* lt_n = b.build_ult(idx, n64);
        b.build_guard(lt_n, "@deopt");
        Value* sh_idx = b.build_shl(idx, b.build_iconst_i64(3));
        Value* check = b.build_ult(sh_idx, cbl);
        b.build_br(bb_exit, {check});

        b.position_at_end(bb_exit);
        Value* res = b.add_block_param(bb_exit, Type::i32());
        b.build_ret(res);

        fn->rebuild_cfg_predecessors();
        RangeAnalysisStats stats;
        RangeAnalysisOptions opts;
        opts.stats = &stats;
        run_bounds_check_elimination(*fn, mod, opts);
        CHECK_EQ(stats.checks_implied, size_t{0});
    }
}


