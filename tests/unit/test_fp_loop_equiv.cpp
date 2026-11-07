#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/verify/bv_solver.hpp>
#include <brass/mir/builder.hpp>
#include <cmath>
#include <cstring>
#include <limits>

using namespace brass;
using namespace brass::verify;

namespace {

uint64_t f64_bits(double d) {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof(double));
    return u;
}

double bits_f64(uint64_t u) {
    double d = 0.0;
    std::memcpy(&d, &u, sizeof(double));
    return d;
}

} // namespace

TEST_CASE("FP Bit-Blasting - Exact IEEE-754 FAbs, FNeg, FEq, FLt, FLe, FMin, FMax, FpExt") {
    // 1. +0.0 == -0.0 is true in FEq, and NaN == NaN is false in FEq
    {
        BVContext ctx;
        BVNodeId pos_zero = ctx.make_const(f64_bits(0.0), 64);
        BVNodeId neg_zero = ctx.make_const(f64_bits(-0.0), 64);
        BVNodeId qnan = ctx.make_const(f64_bits(std::numeric_limits<double>::quiet_NaN()), 64);

        BVNodeId eq_zeros = ctx.make_binary(BVOp::FEq, pos_zero, neg_zero, 1);
        CHECK(ctx.nodes()[eq_zeros].op == BVOp::Const);
        CHECK_EQ(ctx.nodes()[eq_zeros].imm, 1ULL);

        BVNodeId eq_nans = ctx.make_binary(BVOp::FEq, qnan, qnan, 1);
        CHECK(ctx.nodes()[eq_nans].op == BVOp::Const);
        CHECK_EQ(ctx.nodes()[eq_nans].imm, 0ULL);

        // Symbolic x: FEq(x, x) != 1 is SAT (counterexample is NaN!)
        BVNodeId x = ctx.make_var("x", 64, true);
        BVNodeId x_eq_x = ctx.make_binary(BVOp::FEq, x, x, 1);
        BVNodeId not_refl = ctx.make_unary(BVOp::Not, x_eq_x, 1);
        SolveOutcome s_nan = solve_nonzero(ctx, not_refl);
        CHECK_EQ(s_nan.status, SolveOutcome::Sat);
        REQUIRE_EQ(s_nan.model.size(), 1u);
        CHECK(std::isnan(bits_f64(s_nan.model[0])));
    }

    // 2. Asymmetry & trichotomy of IEEE-754 FLt / FEq / FGt across all 64-bit floats
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 64, true);
        BVNodeId y = ctx.make_var("y", 64, true);

        // Cannot have both (x < y) and (y < x)
        BVNodeId lt_xy = ctx.make_binary(BVOp::FLt, x, y, 1);
        BVNodeId lt_yx = ctx.make_binary(BVOp::FLt, y, x, 1);
        BVNodeId both_lt = ctx.make_binary(BVOp::And, lt_xy, lt_yx, 1);
        CHECK_EQ(solve_nonzero(ctx, both_lt).status, SolveOutcome::Unsat);

        // When neither x nor y is NaN (FEq(x, x) && FEq(y, y)), (x <= y) || (y < x) always holds
        BVNodeId non_nan = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::FEq, x, x, 1),
            ctx.make_binary(BVOp::FEq, y, y, 1), 1);
        BVNodeId le_xy = ctx.make_binary(BVOp::FLe, x, y, 1);
        BVNodeId covered = ctx.make_binary(BVOp::Or, le_xy, lt_yx, 1);
        BVNodeId trichotomy_viol = ctx.make_binary(
            BVOp::And, non_nan, ctx.make_unary(BVOp::Not, covered, 1), 1);
        CHECK_EQ(solve_nonzero(ctx, trichotomy_viol).status, SolveOutcome::Unsat);
    }

    // 3. FMin / FMax commutativity and ordering on non-NaN floats
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 64, true);
        BVNodeId y = ctx.make_var("y", 64, true);
        BVNodeId non_nan = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::FEq, x, x, 1),
            ctx.make_binary(BVOp::FEq, y, y, 1), 1);
        BVNodeId min_xy = ctx.make_binary(BVOp::FMin, x, y, 64);
        BVNodeId min_yx = ctx.make_binary(BVOp::FMin, y, x, 64);
        BVNodeId max_xy = ctx.make_binary(BVOp::FMax, x, y, 64);

        // Bitwise commutativity of FMin(x, y) == FMin(y, x) when non-NaN
        BVNodeId min_comm_viol = ctx.make_binary(
            BVOp::And, non_nan, ctx.make_binary(BVOp::Ne, min_xy, min_yx, 1), 1);
        CHECK_EQ(solve_nonzero(ctx, min_comm_viol).status, SolveOutcome::Unsat);

        // FLe(FMin(x, y), FMax(x, y)) holds for all non-NaN x, y
        BVNodeId min_le_max = ctx.make_binary(BVOp::FLe, min_xy, max_xy, 1);
        BVNodeId ord_viol = ctx.make_binary(
            BVOp::And, non_nan, ctx.make_unary(BVOp::Not, min_le_max, 1), 1);
        CHECK_EQ(solve_nonzero(ctx, ord_viol).status, SolveOutcome::Unsat);
    }

    // 4. FpExt (f32 -> f64) preserves strict ordering across all 32-bit floats (including subnormals)
    {
        BVContext ctx;
        BVNodeId x32 = ctx.make_var("x32", 32, true);
        BVNodeId y32 = ctx.make_var("y32", 32, true);
        BVNodeId x64 = ctx.make_unary(BVOp::FpExt, x32, 64);
        BVNodeId y64 = ctx.make_unary(BVOp::FpExt, y32, 64);

        BVNodeId lt32 = ctx.make_binary(BVOp::FLt, x32, y32, 1);
        BVNodeId lt64 = ctx.make_binary(BVOp::FLt, x64, y64, 1);
        BVNodeId mismatch = ctx.make_binary(BVOp::Xor, lt32, lt64, 1);
        CHECK_EQ(solve_nonzero(ctx, mismatch).status, SolveOutcome::Unsat);
    }
}

TEST_CASE("FP Interval Flags & Soundness of MIR Floating-Point Rewrites") {
    // 1. Check BVInterval FP flags on SiToFp, FAbs, FNeg, Const
    {
        BVContext ctx;
        BVNodeId i32_v = ctx.make_var("i", 32, false);
        BVNodeId f64_from_i32 = ctx.make_unary(BVOp::SiToFp, i32_v, 64);
        const BVInterval& iv_si = ctx.interval(f64_from_i32);
        CHECK(iv_si.fp_non_nan);
        CHECK(iv_si.fp_non_inf);
        CHECK(iv_si.fp_non_neg_zero);
        CHECK(iv_si.fp_exact_int);

        // Floor / Ceil / Round of exact integer f64 simplifies to identity
        CHECK_EQ(ctx.make_unary(BVOp::FFloor, f64_from_i32, 64), f64_from_i32);
        CHECK_EQ(ctx.make_unary(BVOp::FCeil, f64_from_i32, 64), f64_from_i32);
        CHECK_EQ(ctx.make_unary(BVOp::FRound, f64_from_i32, 64), f64_from_i32);
        CHECK_EQ(ctx.make_unary(BVOp::FpToSi, f64_from_i32, 32), i32_v);

        BVNodeId sym_f = ctx.make_var("f", 64, true);
        BVNodeId abs_f = ctx.make_unary(BVOp::FAbs, sym_f, 64);
        CHECK(ctx.interval(abs_f).fp_non_neg_zero);
        CHECK(!ctx.interval(abs_f).fp_non_nan);
    }

    // 2. Prove x * 2.0 == x + x and fabs(-x) == fabs(x) in MIR
    {
        Module m1("m_fp1");
        Builder b1(m1);
        Function* f1 = m1.create_function("scale_abs", Type::f64(), {Type::f64()});
        b1.set_function(f1);
        BasicBlock* e1 = b1.append_block("entry");
        Value* x1 = b1.add_block_param(e1, Type::f64());
        b1.position_at_end(e1);
        Value* neg_x1 = b1.build_neg(x1);
        Value* abs1 = b1.build_fabs_f64(neg_x1);
        Value* mul2 = b1.build_mul(abs1, b1.build_fconst_f64(2.0));
        b1.build_ret(mul2);
        f1->rebuild_cfg_predecessors();

        Module m2("m_fp2");
        Builder b2(m2);
        Function* f2 = m2.create_function("scale_abs", Type::f64(), {Type::f64()});
        b2.set_function(f2);
        BasicBlock* e2 = b2.append_block("entry");
        Value* x2 = b2.add_block_param(e2, Type::f64());
        b2.position_at_end(e2);
        Value* abs2 = b2.build_fabs_f64(x2);
        Value* add_self = b2.build_add(abs2, abs2);
        b2.build_ret(add_self);
        f2->rebuild_cfg_predecessors();

        EquivResult res = verify_mir_equivalence(*f1, *f2);
        CHECK_EQ(res.status, EquivStatus::ProvenEquivalent);
    }

    // 3. Refute unsound unguarded (x + 0.0 == x) on -0.0 or sNaN
    {
        Module m1("m_unsound1");
        Builder b1(m1);
        Function* f1 = m1.create_function("add_zero", Type::f64(), {Type::f64()});
        b1.set_function(f1);
        BasicBlock* e1 = b1.append_block("entry");
        Value* x1 = b1.add_block_param(e1, Type::f64());
        b1.position_at_end(e1);
        b1.build_ret(b1.build_add(x1, b1.build_fconst_f64(0.0)));
        f1->rebuild_cfg_predecessors();

        Module m2("m_unsound2");
        Builder b2(m2);
        Function* f2 = m2.create_function("add_zero", Type::f64(), {Type::f64()});
        b2.set_function(f2);
        BasicBlock* e2 = b2.append_block("entry");
        Value* x2 = b2.add_block_param(e2, Type::f64());
        b2.position_at_end(e2);
        b2.build_ret(x2);
        f2->rebuild_cfg_predecessors();

        EquivResult res = verify_mir_equivalence(*f1, *f2);
        CHECK_EQ(res.status, EquivStatus::Counterexample);
        REQUIRE_EQ(res.counterexample.size(), 1u);
    }

    // 4. Prove guarded: if (x > 0.0) return x + 0.0; else return 0.0; == if (x > 0.0) return x; else return 0.0;
    {
        Module m1("m_guarded1");
        Builder b1(m1);
        Function* f1 = m1.create_function("guarded_add_zero", Type::f64(), {Type::f64()});
        b1.set_function(f1);
        BasicBlock* e1 = b1.append_block("entry");
        BasicBlock* t1 = b1.append_block("then");
        BasicBlock* f_blk1 = b1.append_block("else");
        Value* x1 = b1.add_block_param(e1, Type::f64());
        b1.position_at_end(e1);
        Value* gt1 = b1.build_sgt(x1, b1.build_fconst_f64(0.0));
        b1.build_br_if(gt1, t1, {}, f_blk1, {});
        b1.position_at_end(t1);
        b1.build_ret(b1.build_add(x1, b1.build_fconst_f64(0.0)));
        b1.position_at_end(f_blk1);
        b1.build_ret(b1.build_fconst_f64(0.0));
        f1->rebuild_cfg_predecessors();

        Module m2("m_guarded2");
        Builder b2(m2);
        Function* f2 = m2.create_function("guarded_add_zero", Type::f64(), {Type::f64()});
        b2.set_function(f2);
        BasicBlock* e2 = b2.append_block("entry");
        BasicBlock* t2 = b2.append_block("then");
        BasicBlock* f_blk2 = b2.append_block("else");
        Value* x2 = b2.add_block_param(e2, Type::f64());
        b2.position_at_end(e2);
        Value* gt2 = b2.build_sgt(x2, b2.build_fconst_f64(0.0));
        b2.build_br_if(gt2, t2, {}, f_blk2, {});
        b2.position_at_end(t2);
        b2.build_ret(x2);
        b2.position_at_end(f_blk2);
        b2.build_ret(b2.build_fconst_f64(0.0));
        f2->rebuild_cfg_predecessors();

        EquivResult res = verify_mir_equivalence(*f1, *f2);
        CHECK_EQ(res.status, EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("Relational Loop Equivalence - Unbounded Trip Count Proofs and Refutations") {
    // 1. Unbounded loop: acc += (i * 4) + (i * k) vs acc += (i << 2) + (k * i)
    {
        Module m1("m_loop1");
        Builder b1(m1);
        Function* f1 = m1.create_function("sum_loop", Type::i64(), {Type::i64(), Type::i64()});
        b1.set_function(f1);
        BasicBlock* entry1 = b1.append_block("entry");
        BasicBlock* hdr1 = b1.append_block("header");
        BasicBlock* body1 = b1.append_block("body");
        BasicBlock* exit1 = b1.append_block("exit");
        Value* n1 = b1.add_block_param(entry1, Type::i64());
        Value* k1 = b1.add_block_param(entry1, Type::i64());
        Value* i1 = b1.add_block_param(hdr1, Type::i64());
        Value* acc1 = b1.add_block_param(hdr1, Type::i64());

        b1.position_at_end(entry1);
        Value* z1 = b1.build_iconst_i64(0);
        b1.build_br(hdr1, {z1, z1});

        b1.position_at_end(hdr1);
        b1.build_br_if(b1.build_slt(i1, n1), body1, {}, exit1, {});

        b1.position_at_end(body1);
        Value* term1_a = b1.build_mul(i1, b1.build_iconst_i64(4));
        Value* term1_b = b1.build_mul(i1, k1);
        Value* delta1 = b1.build_add(term1_a, term1_b);
        Value* next_acc1 = b1.build_add(acc1, delta1);
        Value* next_i1 = b1.build_add(i1, b1.build_iconst_i64(1));
        b1.build_br(hdr1, {next_i1, next_acc1});

        b1.position_at_end(exit1);
        b1.build_ret(acc1);
        f1->rebuild_cfg_predecessors();

        Module m2("m_loop2");
        Builder b2(m2);
        Function* f2 = m2.create_function("sum_loop", Type::i64(), {Type::i64(), Type::i64()});
        b2.set_function(f2);
        BasicBlock* entry2 = b2.append_block("entry");
        BasicBlock* hdr2 = b2.append_block("header");
        BasicBlock* body2 = b2.append_block("body");
        BasicBlock* exit2 = b2.append_block("exit");
        Value* n2 = b2.add_block_param(entry2, Type::i64());
        Value* k2 = b2.add_block_param(entry2, Type::i64());
        Value* i2 = b2.add_block_param(hdr2, Type::i64());
        Value* acc2 = b2.add_block_param(hdr2, Type::i64());

        b2.position_at_end(entry2);
        Value* z2 = b2.build_iconst_i64(0);
        // LICM-style preheader factor (k + 4)
        Value* k_plus_4 = b2.build_add(k2, b2.build_iconst_i64(4));
        b2.build_br(hdr2, {z2, z2});

        b2.position_at_end(hdr2);
        b2.build_br_if(b2.build_slt(i2, n2), body2, {}, exit2, {});

        b2.position_at_end(body2);
        Value* delta2 = b2.build_mul(i2, k_plus_4);
        Value* next_acc2 = b2.build_add(acc2, delta2);
        Value* next_i2 = b2.build_add(i2, b2.build_iconst_i64(1));
        b2.build_br(hdr2, {next_i2, next_acc2});

        b2.position_at_end(exit2);
        b2.build_ret(acc2);
        f2->rebuild_cfg_predecessors();

        EquivResult loop_res = verify_mir_loop_equivalence(*f1, *f2);
        CHECK_EQ(loop_res.status, EquivStatus::ProvenEquivalent);

        // Also works via verify_mir_equivalence even with loop_unroll_bound = 0
        EquivOptions zero_unroll;
        zero_unroll.loop_unroll_bound = 0;
        EquivResult eq_res = verify_mir_equivalence(*f1, *f2, zero_unroll);
        CHECK_EQ(eq_res.status, EquivStatus::ProvenEquivalent);
    }

    // 2. Refute diverging loop where accumulator or step is wrong
    {
        Module m1("m_div1");
        Builder b1(m1);
        Function* f1 = m1.create_function("bad_loop", Type::i64(), {Type::i64()});
        b1.set_function(f1);
        BasicBlock* entry1 = b1.append_block("entry");
        BasicBlock* hdr1 = b1.append_block("header");
        BasicBlock* body1 = b1.append_block("body");
        BasicBlock* exit1 = b1.append_block("exit");
        Value* n1 = b1.add_block_param(entry1, Type::i64());
        Value* i1 = b1.add_block_param(hdr1, Type::i64());
        Value* acc1 = b1.add_block_param(hdr1, Type::i64());

        b1.position_at_end(entry1);
        Value* z1 = b1.build_iconst_i64(0);
        b1.build_br(hdr1, {z1, z1});
        b1.position_at_end(hdr1);
        b1.build_br_if(b1.build_slt(i1, n1), body1, {}, exit1, {});
        b1.position_at_end(body1);
        Value* next_acc1 = b1.build_add(acc1, b1.build_shl(i1, b1.build_iconst_i64(2)));
        Value* next_i1 = b1.build_add(i1, b1.build_iconst_i64(1));
        b1.build_br(hdr1, {next_i1, next_acc1});
        b1.position_at_end(exit1);
        b1.build_ret(acc1);
        f1->rebuild_cfg_predecessors();

        Module m2("m_div2");
        Builder b2(m2);
        Function* f2 = m2.create_function("bad_loop", Type::i64(), {Type::i64()});
        b2.set_function(f2);
        BasicBlock* entry2 = b2.append_block("entry");
        BasicBlock* hdr2 = b2.append_block("header");
        BasicBlock* body2 = b2.append_block("body");
        BasicBlock* exit2 = b2.append_block("exit");
        Value* n2 = b2.add_block_param(entry2, Type::i64());
        Value* i2 = b2.add_block_param(hdr2, Type::i64());
        Value* acc2 = b2.add_block_param(hdr2, Type::i64());

        b2.position_at_end(entry2);
        Value* z2 = b2.build_iconst_i64(0);
        b2.build_br(hdr2, {z2, z2});
        b2.position_at_end(hdr2);
        b2.build_br_if(b2.build_slt(i2, n2), body2, {}, exit2, {});
        b2.position_at_end(body2);
        // Bug: shifts by 3 (i * 8) instead of 2 (i * 4)
        Value* next_acc2 = b2.build_add(acc2, b2.build_shl(i2, b2.build_iconst_i64(3)));
        Value* next_i2 = b2.build_add(i2, b2.build_iconst_i64(1));
        b2.build_br(hdr2, {next_i2, next_acc2});
        b2.position_at_end(exit2);
        b2.build_ret(acc2);
        f2->rebuild_cfg_predecessors();

        EquivResult bad_res = verify_mir_loop_equivalence(*f1, *f2);
        CHECK_EQ(bad_res.status, EquivStatus::Counterexample);
        REQUIRE_EQ(bad_res.counterexample.size(), 1u);
        CHECK(bad_res.counterexample[0].as_i64() >= 2);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
