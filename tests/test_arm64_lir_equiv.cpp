#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/arm64_lir_equiv.hpp>
#include <brass/verify/bv_solver.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/ml_fusion.hpp>

using namespace brass;
using namespace brass::verify;
using namespace brass::codegen;
using namespace brass::aarch64;

TEST_CASE("ARM64 LIR Equiv - All 31 GPRs (X0..X30) & W-Register 32-Bit Zero-Extension Rule") {
    // 1. Verify 32-bit W-register write zero-extends across high-numbered AArch64 GPRs (X19..X30)
    {
        LirBlock b1(0, "add32_zero_id");
        auto i1 = std::make_unique<LirInst>(LirOpcode::Add32);
        i1->add_def(LirOperand::preg_aarch64_gpr(GPR::X23, 4));
        i1->add_use(LirOperand::preg_aarch64_gpr(GPR::X23, 4));
        i1->add_use(LirOperand::imm32(0));
        b1.append_inst(std::move(i1));

        LirBlock b2(0, "mov32_self");
        auto j1 = std::make_unique<LirInst>(LirOpcode::Mov32);
        j1->add_def(LirOperand::preg_aarch64_gpr(GPR::X23, 4));
        j1->add_use(LirOperand::preg_aarch64_gpr(GPR::X23, 4));
        b2.append_inst(std::move(j1));

        // On AArch64, neither add32 nor mov32 touches NZCV, and both zero bits [63:32] of X23!
        LirEquivResult r_ok = verify_arm64_lir_block_equivalence(b1, b2);
        CHECK_EQ(r_ok.status, EquivStatus::ProvenEquivalent);

        // Seeded bug #1: Dropping `mov32 w23, w23` / `add32 w23, w23, #0` into a no-op
        // leaves dirty upper 32 bits in X23!
        LirBlock buggy_noop(0, "buggy_noop");
        LirEquivResult r_bug = verify_arm64_lir_block_equivalence(b1, buggy_noop);
        CHECK_EQ(r_bug.status, EquivStatus::Counterexample);
        CHECK(r_bug.diagnostic.find("x23") != std::string::npos);
    }

    // 2. Seeded bug #2: Propagating a 64-bit source across a 32-bit W-register zero-extending move
    // (`mov32 w0, w1; add x2, x0, x20` vs `mov32 w0, w1; add x2, x1, x20`)
    {
        LirBlock before_blk(0, "w_zero_ext_before");
        auto m32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        m32->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        m32->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
        before_blk.append_inst(std::move(m32));

        auto add64 = std::make_unique<LirInst>(LirOpcode::Add);
        add64->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        add64->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        add64->add_use(LirOperand::preg_aarch64_gpr(GPR::X20, 8));
        before_blk.append_inst(std::move(add64));

        LirBlock buggy_after(0, "w_zero_ext_buggy");
        auto m32_b = std::make_unique<LirInst>(LirOpcode::Mov32);
        m32_b->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        m32_b->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
        buggy_after.append_inst(std::move(m32_b));

        auto add64_b = std::make_unique<LirInst>(LirOpcode::Add);
        add64_b->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        add64_b->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8)); // Bug: upper 32 bits of X1 not zeroed!
        add64_b->add_use(LirOperand::preg_aarch64_gpr(GPR::X20, 8));
        buggy_after.append_inst(std::move(add64_b));

        LirEquivResult r_bug = verify_arm64_lir_block_equivalence(before_blk, buggy_after);
        CHECK_EQ(r_bug.status, EquivStatus::Counterexample);
        CHECK(r_bug.diagnostic.find("x2") != std::string::npos);
    }

    // 3. XZR register semantics (reads 0, ignores writes)
    {
        LirBlock b1(0, "xzr_add");
        auto i1 = std::make_unique<LirInst>(LirOpcode::Add);
        i1->add_def(LirOperand::preg_aarch64_gpr(GPR::X30, 8));
        i1->add_use(LirOperand::preg_aarch64_gpr(GPR::X28, 8));
        i1->add_use(LirOperand::preg_aarch64_gpr(GPR::XZR, 8));
        b1.append_inst(std::move(i1));

        LirBlock b2(0, "xzr_mov");
        auto j1 = std::make_unique<LirInst>(LirOpcode::Mov);
        j1->add_def(LirOperand::preg_aarch64_gpr(GPR::X30, 8));
        j1->add_use(LirOperand::preg_aarch64_gpr(GPR::X28, 8));
        b2.append_inst(std::move(j1));

        CHECK_EQ(verify_arm64_lir_block_equivalence(b1, b2).status, EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("ARM64 LIR Equiv - NZCV Flags, Not-Borrow Carry Polarity, and Live Flag Clobber Detection") {
    // 1. `subs xzr, x0, x1` is bit-exact equivalent to `cmp x0, x1` across all GPRs and NZCV
    {
        LirBlock b1(0, "subs_xzr");
        auto s = std::make_unique<LirInst>(LirOpcode::Subs);
        s->add_def(LirOperand::preg_aarch64_gpr(GPR::XZR, 8));
        s->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        s->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        b1.append_inst(std::move(s));

        LirBlock b2(0, "cmp_x0_x1");
        auto c = std::make_unique<LirInst>(LirOpcode::Cmp);
        c->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        c->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        b2.append_inst(std::move(c));

        CHECK_EQ(verify_arm64_lir_block_equivalence(b1, b2).status, EquivStatus::ProvenEquivalent);
    }

    // 2. Non-S ALU instructions (Add, Sub, Eor/Xor32) between Cmp and Cset/Csel preserve live NZCV on ARM64,
    //    whereas rewriting Add -> Adds (or inserting Test/Cmp) clobbers live NZCV!
    {
        LirBlock before_blk(0, "cmp_add_cset");
        auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        before_blk.append_inst(std::move(cmp));

        auto mov0 = std::make_unique<LirInst>(LirOpcode::Mov);
        mov0->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        mov0->add_use(LirOperand::imm(0, 8));
        before_blk.append_inst(std::move(mov0));

        auto cset = std::make_unique<LirInst>(LirOpcode::Setcc);
        cset->condition = x64::Condition::L; // ARM64 LT (N != V)
        cset->add_def(LirOperand::preg_aarch64_gpr(GPR::X3, 1));
        before_blk.append_inst(std::move(cset));

        // Valid ARM64 rewrite: `mov x2, 0` -> `eor w2, w2, w2` (`Xor32`) does NOT touch NZCV on ARM64!
        LirBlock valid_eor(0, "cmp_eor_cset");
        auto cmp_v = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp_v->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        cmp_v->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        valid_eor.append_inst(std::move(cmp_v));

        auto eor0 = std::make_unique<LirInst>(LirOpcode::Xor32);
        eor0->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        eor0->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        eor0->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        valid_eor.append_inst(std::move(eor0));

        auto cset_v = std::make_unique<LirInst>(LirOpcode::Setcc);
        cset_v->condition = x64::Condition::L;
        cset_v->add_def(LirOperand::preg_aarch64_gpr(GPR::X3, 1));
        valid_eor.append_inst(std::move(cset_v));

        CHECK_EQ(verify_arm64_lir_block_equivalence(before_blk, valid_eor).status,
                 EquivStatus::ProvenEquivalent);

        // Seeded bug: `subs w2, w2, w2` (`Subs32`) between `cmp` and `cset` clobbers live NZCV!
        LirBlock buggy_subs(0, "cmp_subs_cset_bug");
        auto cmp_b = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp_b->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        cmp_b->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        buggy_subs.append_inst(std::move(cmp_b));

        auto subs0 = std::make_unique<LirInst>(LirOpcode::Subs32);
        subs0->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        subs0->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        subs0->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
        buggy_subs.append_inst(std::move(subs0));

        auto cset_b = std::make_unique<LirInst>(LirOpcode::Setcc);
        cset_b->condition = x64::Condition::L;
        cset_b->add_def(LirOperand::preg_aarch64_gpr(GPR::X3, 1));
        buggy_subs.append_inst(std::move(cset_b));

        LirEquivResult r_clobber = verify_arm64_lir_block_equivalence(before_blk, buggy_subs);
        CHECK_EQ(r_clobber.status, EquivStatus::Counterexample);
    }

    // 3. ARM64 Carry Flag Polarity on `tst x0, x0` vs `cmp x0, #0`:
    //    - Equivalent for signed/equality conditions (EQ, NE, LT, GE, GT, LE) when flags are dead afterwards.
    //    - NOT equivalent for unsigned conditions (HI / A, HS / AE, LO / B, LS / BE) or when C flag is live,
    //      because `cmp x0, #0` sets C=1 (no borrow) while `tst x0, x0` sets C=0!
    {
        LirBlock cmp_zero_lt(0, "cmp0_lt");
        auto c1 = std::make_unique<LirInst>(LirOpcode::Cmp);
        c1->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        c1->add_use(LirOperand::imm(0, 8));
        cmp_zero_lt.append_inst(std::move(c1));
        auto s1 = std::make_unique<LirInst>(LirOpcode::Setcc);
        s1->condition = x64::Condition::L; // LT (N != V)
        s1->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 1));
        cmp_zero_lt.append_inst(std::move(s1));

        LirBlock tst_zero_lt(0, "tst0_lt");
        auto t1 = std::make_unique<LirInst>(LirOpcode::Test);
        t1->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        t1->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        tst_zero_lt.append_inst(std::move(t1));
        auto s2 = std::make_unique<LirInst>(LirOpcode::Setcc);
        s2->condition = x64::Condition::L;
        s2->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 1));
        tst_zero_lt.append_inst(std::move(s2));

        // When flags are not live-out, `cmp x0, #0; cset x1, lt` == `tst x0, x0; cset x1, lt`
        CHECK_EQ(verify_arm64_lir_block_equivalence(
                     cmp_zero_lt, tst_zero_lt, kArm64AllGprsMask, 0, false).status,
                 EquivStatus::ProvenEquivalent);

        // When C flag is live-out (check_flags = true), C differs (C=1 vs C=0)!
        LirEquivResult r_c_live = verify_arm64_lir_block_equivalence(
            cmp_zero_lt, tst_zero_lt, kArm64AllGprsMask, 0, true);
        CHECK_EQ(r_c_live.status, EquivStatus::Counterexample);
        CHECK(r_c_live.diagnostic.find("C mismatch") != std::string::npos);

        // And if used with unsigned HI (`Condition::A`), `cmp x0, #0` (C=1 && !Z) != `tst x0, x0` (C=0 && !Z == 0)!
        cmp_zero_lt.instructions.back()->condition = x64::Condition::A;
        tst_zero_lt.instructions.back()->condition = x64::Condition::A;
        LirEquivResult r_hi_bug = verify_arm64_lir_block_equivalence(
            cmp_zero_lt, tst_zero_lt, kArm64AllGprsMask, 0, false);
        CHECK_EQ(r_hi_bug.status, EquivStatus::Counterexample);
        CHECK(r_hi_bug.diagnostic.find("x1") != std::string::npos);
    }
}

TEST_CASE("ARM64 LIR Equiv - CSEL, CSET Full-Register Write, and CCMP Chain Verification") {
    // 1. `cmp x0, x1; csel x2, x3, x4, lt` == `cmp x1, x0; csel x2, x3, x4, gt`
    {
        LirBlock b1(0, "csel_lt");
        auto m1 = std::make_unique<LirInst>(LirOpcode::Mov);
        m1->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        m1->add_use(LirOperand::preg_aarch64_gpr(GPR::X4, 8));
        b1.append_inst(std::move(m1));
        auto c1 = std::make_unique<LirInst>(LirOpcode::Cmp);
        c1->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        c1->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        b1.append_inst(std::move(c1));
        auto sel1 = std::make_unique<LirInst>(LirOpcode::Cmovcc);
        sel1->condition = x64::Condition::L;
        sel1->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        sel1->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        sel1->add_use(LirOperand::preg_aarch64_gpr(GPR::X3, 8));
        b1.append_inst(std::move(sel1));

        LirBlock b2(0, "csel_gt_swapped");
        auto m2 = std::make_unique<LirInst>(LirOpcode::Mov);
        m2->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        m2->add_use(LirOperand::preg_aarch64_gpr(GPR::X4, 8));
        b2.append_inst(std::move(m2));
        auto c2 = std::make_unique<LirInst>(LirOpcode::Cmp);
        c2->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        c2->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        b2.append_inst(std::move(c2));
        auto sel2 = std::make_unique<LirInst>(LirOpcode::Cmovcc);
        sel2->condition = x64::Condition::G;
        sel2->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        sel2->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
        sel2->add_use(LirOperand::preg_aarch64_gpr(GPR::X3, 8));
        b2.append_inst(std::move(sel2));

        CHECK_EQ(verify_arm64_lir_block_equivalence(b1, b2, reg_mask(GPR::X2), 0, false).status,
                 EquivStatus::ProvenEquivalent);
    }

    // 2. CCMP chain verification: `(x0 >s 0) && (x1 <s 10)`
    //    `cmp x0, #0; ccmp x1, #10, #0b0000, gt; b.lt`
    {
        Arm64CcmpStep valid_step;
        valid_step.is_64 = true;
        valid_step.is_cmn = false;
        valid_step.lhs = LirOperand::preg_aarch64_gpr(GPR::X1, 8);
        valid_step.rhs = LirOperand::imm(10, 8);
        valid_step.nzcv_imm = 0b0000; // N=0, Z=0, C=0, V=0 -> LT (N != V) is false!
        valid_step.cond = aarch64::Condition::GT;

        LirEquivResult r_ok = verify_arm64_ccmp_chain(
            true, false,
            LirOperand::preg_aarch64_gpr(GPR::X0, 8),
            LirOperand::imm(0, 8),
            valid_step,
            aarch64::Condition::LT,
            true,
            aarch64::Condition::GT,
            aarch64::Condition::LT);
        CHECK_EQ(r_ok.status, EquivStatus::ProvenEquivalent);

        // Seeded CCMP bug: wrong fallback #nzcv immediate (0b1000 sets N=1, V=0 so LT becomes true on false gate!)
        Arm64CcmpStep buggy_step = valid_step;
        buggy_step.nzcv_imm = 0b1000;
        LirEquivResult r_bug = verify_arm64_ccmp_chain(
            true, false,
            LirOperand::preg_aarch64_gpr(GPR::X0, 8),
            LirOperand::imm(0, 8),
            buggy_step,
            aarch64::Condition::LT,
            true,
            aarch64::Condition::GT,
            aarch64::Condition::LT);
        CHECK_EQ(r_bug.status, EquivStatus::Counterexample);
    }
}

TEST_CASE("ARM64 LIR Equiv - SIMD/FP V0..V31 & FCMP Ordered/Unordered NaN Semantics") {
    // 1. AArch64 fused float compare swaps operands for `a > b` (`fcmp b, a; b.lo`)
    //    because `fcmp a, b; b.hi` (`C==1 && Z==0`) wrongly succeeds when `a` or `b` is NaN (`NZCV = 0011`)!
    {
        // Swapped `fcmp v1, v0; cset x0, lo` (Condition::B -> CC/LO: C==0, which is false on NaN!)
        LirBlock safe_gt(0, "fcmp_swapped_lo");
        auto fc1 = std::make_unique<LirInst>(LirOpcode::Ucomisd);
        fc1->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        fc1->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        safe_gt.append_inst(std::move(fc1));
        auto s1 = std::make_unique<LirInst>(LirOpcode::Setcc);
        s1->condition = x64::Condition::B; // Maps to CC/LO
        s1->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 1));
        safe_gt.append_inst(std::move(s1));

        // Unswapped `fcmp v0, v1; cset x0, hi; cset x1, vc; and x0, x0, x1` (explicitly masking out V=1 NaN)
        LirBlock explicit_gt(0, "fcmp_hi_and_vc");
        auto fc2 = std::make_unique<LirInst>(LirOpcode::Ucomisd);
        fc2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        fc2->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        explicit_gt.append_inst(std::move(fc2));
        auto s2 = std::make_unique<LirInst>(LirOpcode::Setcc);
        s2->condition = x64::Condition::A; // Maps to HI (C==1 && Z==0)
        s2->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 1));
        explicit_gt.append_inst(std::move(s2));
        auto s_vc = std::make_unique<LirInst>(LirOpcode::Setcc);
        s_vc->condition = x64::Condition::NP; // Maps to VC (V==0, ordered)
        s_vc->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 1));
        explicit_gt.append_inst(std::move(s_vc));
        auto and_i = std::make_unique<LirInst>(LirOpcode::And);
        and_i->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        and_i->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        and_i->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        explicit_gt.append_inst(std::move(and_i));

        CHECK_EQ(verify_arm64_lir_block_equivalence(
                     safe_gt, explicit_gt, reg_mask(GPR::X0), 0, false).status,
                 EquivStatus::ProvenEquivalent);

        // Seeded bug: `fcmp v0, v1; cset x0, hi` WITHOUT checking VC (returns 1 on NaN!)
        LirBlock buggy_gt(0, "fcmp_unswapped_hi_bug");
        auto fc3 = std::make_unique<LirInst>(LirOpcode::Ucomisd);
        fc3->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        fc3->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        buggy_gt.append_inst(std::move(fc3));
        auto s3 = std::make_unique<LirInst>(LirOpcode::Setcc);
        s3->condition = x64::Condition::A; // HI: true when unordered (C=1, Z=0, V=1)!
        s3->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 1));
        buggy_gt.append_inst(std::move(s3));

        LirEquivResult r_nan_bug = verify_arm64_lir_block_equivalence(
            safe_gt, buggy_gt, reg_mask(GPR::X0), 0, false);
        CHECK_EQ(r_nan_bug.status, EquivStatus::Counterexample);
    }
}

TEST_CASE("ARM64 LIR Equiv - PeepholeOptimizer & AArch64ISel End-to-End Verification") {
    // 1. Verify PeepholeOptimizer across multi-block AArch64 LirFunction (moves, spill forwarding,
    //    narrowed 32-bit spill reload, lea->add, mov 0 -> eor, branch inversion)
    LirFunction fn;
    fn.name = "arm64_peephole_fn";
    fn.calling_conv = CallingConvention::apple_aapcs64();
    LirBlock* b0 = fn.create_block("entry");
    LirBlock* b1 = fn.create_block("then_blk");
    LirBlock* b2 = fn.create_block("else_blk");
    b0->id = 0;
    b1->id = 1;
    b2->id = 2;
    link_blocks(*b0, *b1);
    link_blocks(*b0, *b2);

    // Redundant self move: mov x0, x0
    auto self_m = std::make_unique<LirInst>(LirOpcode::Mov);
    self_m->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    self_m->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    b0->append_inst(std::move(self_m));

    // 32-bit identity op: add32 w1, w1, 0 -> mov32 w1, w1 (must preserve W1 zero-extension!)
    auto add32_0 = std::make_unique<LirInst>(LirOpcode::Add32);
    add32_0->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
    add32_0->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
    add32_0->add_use(LirOperand::imm32(0));
    b0->append_inst(std::move(add32_0));

    // Spill store + narrowed 32-bit reload: mov [slot 0], x0; mov32 w2, [slot 0, 4]
    auto st_slot = std::make_unique<LirInst>(LirOpcode::Mov);
    st_slot->add_def(LirOperand::slot(0, 8));
    st_slot->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    b0->append_inst(std::move(st_slot));

    auto ld_slot = std::make_unique<LirInst>(LirOpcode::Mov32);
    ld_slot->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 4));
    ld_slot->add_use(LirOperand::slot(0, 4));
    b0->append_inst(std::move(ld_slot));

    // lea x3, [x3 + 16] -> add x3, x3, 16
    auto lea = std::make_unique<LirInst>(LirOpcode::Lea);
    lea->add_def(LirOperand::preg_aarch64_gpr(GPR::X3, 8));
    lea->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X3), 16, 8));
    b0->append_inst(std::move(lea));

    // cmp x0, x1; jcc E, b1 (fallthrough); jmp b2 -> jcc NE, b2
    auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
    cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    b0->append_inst(std::move(cmp));

    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = x64::Condition::E;
    jcc->add_use(LirOperand::label(1));
    b0->append_inst(std::move(jcc));

    auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp->add_use(LirOperand::label(2));
    b0->append_inst(std::move(jmp));

    auto ret1 = std::make_unique<LirInst>(LirOpcode::Ret);
    b1->append_inst(std::move(ret1));
    auto ret2 = std::make_unique<LirInst>(LirOpcode::Ret);
    b2->append_inst(std::move(ret2));

    LirEquivResult peep_res = verify_arm64_peephole(fn);
    CHECK_EQ(peep_res.status, EquivStatus::ProvenEquivalent);

    // 2. Also verify AArch64ISel + LinearScan + PeepholeOptimizer on a real MIR function
    Module mod("arm64_isel_mod");
    Builder b(mod);
    Function* mir_fn = mod.create_function("calc", Type::i64(), {Type::i32(), Type::i64()});
    b.set_function(mir_fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, Type::i32());
    Value* p1 = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);
    Value* zext0 = b.build_zext_i64(p0);
    Value* sum = b.build_add(zext0, p1);
    Value* cond = b.build_slt(sum, b.build_iconst_i64(100));
    Value* sel = b.build_select(cond, sum, p1);
    b.build_ret(sel);
    mir_fn->rebuild_cfg_predecessors();

    CallingConvention cc = CallingConvention::apple_aapcs64();
    auto lir_fn = lower_to_aarch64_lir(*mir_fn, Target::aarch64_macos(), cc);
    REQUIRE(lir_fn != nullptr);
    LivenessAnalysis liveness(*lir_fn);
    liveness.run();
    run_linear_scan_regalloc(*lir_fn, liveness, cc);
    LirEquivResult isel_peep_res = verify_arm64_peephole(*lir_fn);
    CHECK_EQ(isel_peep_res.status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("ARM64 LIR Equiv - 128-Bit VRegs, 16-Byte Spills, Scalar/Vector FMA, and NEON Ops") {
    // 1. 128-bit virtual register upper 64 bits + 16-byte spill slot forwarding
    {
        VReg vr0{0, RegClass::XMM, 16};
        VReg vr1{1, RegClass::XMM, 16};
        VReg vr2{2, RegClass::XMM, 16};

        LirBlock b1(0, "vreg128_before");
        auto st = std::make_unique<LirInst>(LirOpcode::Movaps);
        st->add_def(LirOperand::slot(0, 16));
        st->add_use(LirOperand::vreg(vr0, 16));
        b1.append_inst(std::move(st));
        auto ld = std::make_unique<LirInst>(LirOpcode::Movaps);
        ld->add_def(LirOperand::vreg(vr1, 16));
        ld->add_use(LirOperand::slot(0, 16));
        b1.append_inst(std::move(ld));

        LirBlock b2(0, "vreg128_after");
        auto st2 = std::make_unique<LirInst>(LirOpcode::Movaps);
        st2->add_def(LirOperand::slot(0, 16));
        st2->add_use(LirOperand::vreg(vr0, 16));
        b2.append_inst(std::move(st2));
        auto mv = std::make_unique<LirInst>(LirOpcode::Movaps);
        mv->add_def(LirOperand::vreg(vr1, 16));
        mv->add_use(LirOperand::vreg(vr0, 16));
        b2.append_inst(std::move(mv));

        CHECK_EQ(verify_arm64_lir_block_equivalence(b1, b2).status, EquivStatus::ProvenEquivalent);

        // Seeded bug: replacing 128-bit Movaps with 64-bit Movsd zeroes vreg1.hi!
        LirBlock buggy_64(0, "vreg128_trunc_bug");
        auto st3 = std::make_unique<LirInst>(LirOpcode::Movaps);
        st3->add_def(LirOperand::slot(0, 16));
        st3->add_use(LirOperand::vreg(vr0, 16));
        buggy_64.append_inst(std::move(st3));
        auto mv64 = std::make_unique<LirInst>(LirOpcode::Movsd);
        mv64->add_def(LirOperand::vreg(vr1, 8));
        mv64->add_use(LirOperand::vreg(vr0, 8));
        buggy_64.append_inst(std::move(mv64));

        LirEquivResult r_hi = verify_arm64_lir_block_equivalence(b1, buggy_64);
        CHECK_EQ(r_hi.status, EquivStatus::Counterexample);
        CHECK(r_hi.diagnostic.find("vreg1.hi") != std::string::npos);

        // Self-move on 128-bit vreg vs Paddd check
        LirBlock b_self(0, "vreg128_self");
        auto self_m = std::make_unique<LirInst>(LirOpcode::Movaps);
        self_m->add_def(LirOperand::vreg(vr2, 16));
        self_m->add_use(LirOperand::vreg(vr2, 16));
        b_self.append_inst(std::move(self_m));
        LirBlock b_empty(0, "vreg128_empty");
        CHECK_EQ(verify_arm64_lir_block_equivalence(b_self, b_empty).status, EquivStatus::ProvenEquivalent);
    }

    // 2. Scalar and vector FMA equivalence + mul+add counterexample
    {
        // Scalar: Vfmadd213sd(a, b, c) == Vfmadd231sd(c, a, b)
        LirBlock f213(0, "fma_213_sd");
        auto i1 = std::make_unique<LirInst>(LirOpcode::Vfmadd213sd);
        i1->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        i1->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        i1->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        i1->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 8));
        f213.append_inst(std::move(i1));

        LirBlock f231(0, "fma_231_sd");
        auto i2 = std::make_unique<LirInst>(LirOpcode::Vfmadd231sd);
        i2->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        i2->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 8));
        i2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        i2->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        f231.append_inst(std::move(i2));

        CHECK_EQ(verify_arm64_lir_block_equivalence(f213, f231, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);

        // Unfused mul + add is NOT bit-exact equivalent to FMA!
        LirBlock unfused(0, "mul_add_sd");
        auto m = std::make_unique<LirInst>(LirOpcode::Mulsd);
        m->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        m->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        m->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
        unfused.append_inst(std::move(m));
        auto a = std::make_unique<LirInst>(LirOpcode::Addsd);
        a->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        a->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
        a->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 8));
        unfused.append_inst(std::move(a));

        CHECK_EQ(verify_arm64_lir_block_equivalence(f213, unfused, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::Counterexample);

        // Vector 4x32 FMA: Vfmadd213ps(a, b, c) == Vfmadd231ps(c, a, b), != Mulps+Addps
        LirBlock vf213(0, "vfma_213_ps");
        auto vi1 = std::make_unique<LirInst>(LirOpcode::Vfmadd213ps);
        vi1->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vi1->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vi1->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        vi1->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        vf213.append_inst(std::move(vi1));

        LirBlock vf231(0, "vfma_231_ps");
        auto vi2 = std::make_unique<LirInst>(LirOpcode::Vfmadd231ps);
        vi2->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vi2->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        vi2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vi2->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        vf231.append_inst(std::move(vi2));

        CHECK_EQ(verify_arm64_lir_block_equivalence(vf213, vf231, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);

        LirBlock vunfused(0, "vmul_add_ps");
        auto vm = std::make_unique<LirInst>(LirOpcode::Mulps);
        vm->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vm->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vm->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        vunfused.append_inst(std::move(vm));
        auto va = std::make_unique<LirInst>(LirOpcode::Addps);
        va->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        va->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        va->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        vunfused.append_inst(std::move(va));

        CHECK_EQ(verify_arm64_lir_block_equivalence(vf213, vunfused, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::Counterexample);
    }

    // 3. NEON broadcast / shuffle / extract / fneg / int lane equivalence
    {
        // Pshufd(v1, v1, 0x00) == Vbroadcastss(v1)
        LirBlock b1(0, "bcast_ss");
        auto bc = std::make_unique<LirInst>(LirOpcode::Vbroadcastss);
        bc->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        bc->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 4));
        b1.append_inst(std::move(bc));

        LirBlock b2(0, "pshufd_0");
        auto sh = std::make_unique<LirInst>(LirOpcode::Pshufd);
        sh->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        sh->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        sh->add_use(LirOperand::imm(0, 1));
        b2.append_inst(std::move(sh));

        CHECK_EQ(verify_arm64_lir_block_equivalence(b1, b2, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);

        // Pextrd(w0, v0, #2) == Pshufd(v1, v0, #2); Movd_gx(w0, v1)
        LirBlock ex1(0, "pextrd_2");
        auto pe = std::make_unique<LirInst>(LirOpcode::Pextrd);
        pe->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        pe->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        pe->add_use(LirOperand::imm(2, 1));
        ex1.append_inst(std::move(pe));

        LirBlock ex2(0, "shuf_movd_2");
        auto sh2 = std::make_unique<LirInst>(LirOpcode::Pshufd);
        sh2->add_def(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        sh2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        sh2->add_use(LirOperand::imm(2, 1));
        ex2.append_inst(std::move(sh2));
        auto md = std::make_unique<LirInst>(LirOpcode::Movd_gx);
        md->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        md->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 4));
        ex2.append_inst(std::move(md));

        CHECK_EQ(verify_arm64_lir_block_equivalence(ex1, ex2, reg_mask(GPR::X0), 0, false).status,
                 EquivStatus::ProvenEquivalent);
    }

    // 4. End-to-end real ML fusion & NEON kernels via verify_arm64_peephole (must NOT return Unsupported!)
    {
        Module mod("arm64_vec_real_peep");
        MlFusionCompiler fusion;
        Function* adaln = fusion.build_adaln_modulate(mod, true, "fused_adaln_modulate_gated");
        Function* swiglu = fusion.build_swiglu(mod, "fused_swiglu");
        KernelOptions kopts;
        kopts.opt_level = 2;
        kopts.enable_avx2 = false;
        kopts.enable_vectorize = true;
        KernelJit::prepare_module(mod, kopts);

        CallingConvention cc = CallingConvention::apple_aapcs64();
        for (Function* f : {adaln, swiglu}) {
            auto lir_fn = lower_to_aarch64_lir(*f, Target::aarch64_macos(), cc);
            REQUIRE(lir_fn != nullptr);
            LivenessAnalysis live(*lir_fn);
            live.run();
            run_linear_scan_regalloc(*lir_fn, live, cc);

            LirEquivResult peep = verify_arm64_peephole(*lir_fn);
            CHECK_EQ(peep.status, EquivStatus::ProvenEquivalent);
        }
    }

    // 5. NEON Priority 2 vector lane symbolic equivalence checks
    {
        // Pshufd(0) == id for Vcmpps(EQ) result? No: Psrld(v0, 4) == Vpsrlvd(v0, Vpbroadcastd(4))
        LirBlock s1(0, "psrld_4");
        auto psr = std::make_unique<LirInst>(LirOpcode::Psrld);
        psr->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        psr->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        psr->add_use(LirOperand::imm(4, 1));
        s1.append_inst(std::move(psr));

        LirBlock s2(0, "vpsrlvd_4");
        auto mr = std::make_unique<LirInst>(LirOpcode::Mov32);
        mr->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        mr->add_use(LirOperand::imm(4, 4));
        s2.append_inst(std::move(mr));
        auto bc = std::make_unique<LirInst>(LirOpcode::Vpbroadcastd);
        bc->add_def(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        bc->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
        s2.append_inst(std::move(bc));
        auto vps = std::make_unique<LirInst>(LirOpcode::Vpsrlvd);
        vps->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vps->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        vps->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        s2.append_inst(std::move(vps));

        CHECK_EQ(verify_arm64_lir_block_equivalence(s1, s2, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);

        // Psraq(v0, 3) == Vpsravq(v0, Vpbroadcastq(3))
        LirBlock sq1(0, "psraq_3");
        auto paq = std::make_unique<LirInst>(LirOpcode::Psraq);
        paq->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        paq->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        paq->add_use(LirOperand::imm(3, 1));
        sq1.append_inst(std::move(paq));

        LirBlock sq2(0, "vpsravq_3");
        auto m64 = std::make_unique<LirInst>(LirOpcode::Mov);
        m64->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        m64->add_use(LirOperand::imm(3, 8));
        sq2.append_inst(std::move(m64));
        auto bcq = std::make_unique<LirInst>(LirOpcode::Vpbroadcastq);
        bcq->add_def(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        bcq->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        sq2.append_inst(std::move(bcq));
        auto vpaq = std::make_unique<LirInst>(LirOpcode::Vpsravq);
        vpaq->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        vpaq->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        vpaq->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
        sq2.append_inst(std::move(vpaq));

        CHECK_EQ(verify_arm64_lir_block_equivalence(sq1, sq2, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);

        // Vroundps/Vroundpd/Vcvtdq2ps/Vcvttps2dq/Vcvttpd2dq/Vcmpps/Vblendvps/Vmovmskps self-equiv & non-equiv
        auto mk_self = [](LirBlock& b, LirOpcode op) {
            auto i = std::make_unique<LirInst>(op);
            i->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
            i->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
            if (op == LirOpcode::Vroundps || op == LirOpcode::Vroundpd) {
                i->add_use(LirOperand::imm(0x8, 1));
            } else if (op == LirOpcode::Vcvttpd2dq) {
                i->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
            }
            b.append_inst(std::move(i));
        };
        for (LirOpcode rop : {LirOpcode::Vroundps, LirOpcode::Vroundpd,
                              LirOpcode::Vcvtdq2ps, LirOpcode::Vcvttps2dq, LirOpcode::Vcvttpd2dq}) {
            LirBlock r1(0, "r1"), r2(0, "r2");
            mk_self(r1, rop);
            mk_self(r2, rop);
            CHECK_EQ(verify_arm64_lir_block_equivalence(r1, r2, 0, reg_mask(FPR::V0), false).status,
                     EquivStatus::ProvenEquivalent);
        }

        // Vcmpps(0x11 LT) + Vblendvps(b, a, m) self-equiv and swapped a/b Counterexample
        auto mk_blend = [](LirBlock& b, bool swap_ab) {
            auto cmp = std::make_unique<LirInst>(LirOpcode::Vcmpps);
            cmp->add_def(LirOperand::preg_aarch64_fpr(FPR::V3, 16));
            cmp->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
            cmp->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 16));
            cmp->add_use(LirOperand::imm(0x11, 1));
            b.append_inst(std::move(cmp));
            auto bl = std::make_unique<LirInst>(LirOpcode::Vblendvps);
            bl->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
            bl->add_use(LirOperand::preg_aarch64_fpr(swap_ab ? FPR::V1 : FPR::V2, 16));
            bl->add_use(LirOperand::preg_aarch64_fpr(swap_ab ? FPR::V2 : FPR::V1, 16));
            bl->add_use(LirOperand::preg_aarch64_fpr(FPR::V3, 16));
            b.append_inst(std::move(bl));
        };
        LirBlock bl1(0, "bl1"), bl2(0, "bl2"), bl3(0, "bl3");
        mk_blend(bl1, false);
        mk_blend(bl2, false);
        mk_blend(bl3, true);
        CHECK_EQ(verify_arm64_lir_block_equivalence(bl1, bl2, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::ProvenEquivalent);
        CHECK_EQ(verify_arm64_lir_block_equivalence(bl1, bl3, 0, reg_mask(FPR::V0), false).status,
                 EquivStatus::Counterexample);

        // Vmovmskps / Vmovmskpd self-equiv
        for (LirOpcode mop : {LirOpcode::Vmovmskps, LirOpcode::Vmovmskpd}) {
            LirBlock m1(0, "m1"), m2(0, "m2");
            auto i1 = std::make_unique<LirInst>(mop);
            i1->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
            i1->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
            m1.append_inst(std::move(i1));
            auto i2 = std::make_unique<LirInst>(mop);
            i2->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
            i2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
            m2.append_inst(std::move(i2));
            CHECK_EQ(verify_arm64_lir_block_equivalence(m1, m2, reg_mask(GPR::X0), 0, false).status,
                     EquivStatus::ProvenEquivalent);
        }
    }
}

TEST_CASE("ARM64 LIR Equiv - BVContext::make_extract Memoizes Shared Diamond DAGs") {
    BVContext ctx;
    BVNodeId cond = ctx.make_var("cond", 1);
    BVNodeId x = ctx.make_var("x", 64);
    BVNodeId one = ctx.make_const(1, 64);
    for (int i = 0; i < 48; ++i) {
        BVNodeId inc = ctx.make_binary(BVOp::Add, x, one, 64);
        x = ctx.make_ite(cond, x, inc);
    }
    BVNodeId e1 = ctx.make_extract(x, 31, 0);
    BVNodeId e2 = ctx.make_extract(x, 31, 0);
    CHECK_EQ(ctx.width(e1), 32u);
    CHECK_EQ(e1, e2);
    CHECK_EQ(solve_nonzero(ctx, ctx.make_binary(BVOp::Ne, e1, e2, 1)).status, SolveOutcome::Unsat);
}

TEST_CASE("ARM64 Vector ISEL - Emits 3-Address NEON LIR Without Redundant Movaps") {
    Module mod("a64_vec3");
    Function* fn = mod.create_function(
        "vec3_ops",
        Type::f32x4(),
        {Type::f32x4(), Type::f32x4(), Type::f32x4(), Type::i32x4(), Type::i32x4()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::f32x4());
    Value* c = b.add_block_param(entry, Type::f32x4());
    Value* d = b.add_block_param(entry, Type::f32x4());
    Value* ia = b.add_block_param(entry, Type::i32x4());
    Value* ib = b.add_block_param(entry, Type::i32x4());

    Value* sum = b.build_vadd(a, c);
    Value* prod = b.build_vmul(sum, d);
    Value* fma_v = b.build_vfma(prod, c, d);
    Value* shuf = b.build_vshuffle(fma_v, a, 0x4E);
    Value* masked = b.build_vand(ia, ib);
    Value* masked_f = b.build_vbitcast(Type::f32x4(), masked);
    Value* res = b.build_vadd(shuf, masked_f);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();

    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    REQUIRE(!lir->blocks.empty());

    bool saw_addps = false, saw_mulps = false, saw_fma = false, saw_shuf = false, saw_pand = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Movaps && inst->mir_origin) {
            Opcode origin = inst->mir_origin->opcode();
            CHECK(origin != Opcode::vadd);
            CHECK(origin != Opcode::vmul);
            CHECK(origin != Opcode::vfma);
            CHECK(origin != Opcode::vshuffle);
            CHECK(origin != Opcode::vand);
        }
        if (inst->opcode == LirOpcode::Addps) {
            saw_addps = true;
            CHECK(inst->uses.size() == 2);
            CHECK(inst->defs[0].vreg_val != inst->uses[0].vreg_val);
        }
        if (inst->opcode == LirOpcode::Mulps) {
            saw_mulps = true;
            CHECK(inst->uses.size() == 2);
            CHECK(inst->defs[0].vreg_val != inst->uses[0].vreg_val);
        }
        if (inst->opcode == LirOpcode::Vfmadd213ps) {
            saw_fma = true;
            CHECK(inst->uses.size() == 3);
            CHECK(inst->defs[0].vreg_val != inst->uses[0].vreg_val);
        }
        if (inst->opcode == LirOpcode::Shufps) {
            saw_shuf = true;
            CHECK(inst->uses.size() == 3);
            CHECK(inst->defs[0].vreg_val != inst->uses[0].vreg_val);
        }
        if (inst->opcode == LirOpcode::Pand) {
            saw_pand = true;
            CHECK(inst->uses.size() == 2);
            CHECK(inst->defs[0].vreg_val != inst->uses[0].vreg_val);
        }
    }
    CHECK(saw_addps);
    CHECK(saw_mulps);
    CHECK(saw_fma);
    CHECK(saw_shuf);
    CHECK(saw_pand);

    // Register allocation + AArch64 encoding must succeed cleanly on the 3-address LIR.
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, lir->calling_conv);
    regalloc.allocate();

    auto compiled = compile_lir_to_aarch64(*lir, Target::aarch64_macos());
    CHECK(!compiled.code_buffer.empty());
}

TEST_CASE("ARM64 ISEL - Constant Multiply (2^k+1) and Broadcast Lane Extraction") {
    // 1. mul by 3, 5, 9 (i32 & i64) lowers to Add32Shifted / AddShifted with zero Imul,
    //    and proves bit-exact equivalence against Imul.
    for (int64_t c : {int64_t(3), int64_t(5), int64_t(9)}) {
        for (Type ty : {Type::i32(), Type::i64()}) {
            Module mod("a64_mul_sh");
            Function* fn = mod.create_function("mul_c", ty, {ty});
            Builder b(mod);
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, ty);
            Value* kc = (ty == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(c)) : b.build_iconst_i64(c);
            b.build_ret(b.build_mul(x, kc));
            fn->rebuild_cfg_predecessors();

            AArch64ISel isel(Target::aarch64_macos(), CallingConvention::aapcs64());
            auto lir = isel.lower(*fn);
            REQUIRE(lir != nullptr);
            REQUIRE(!lir->blocks.empty());

            const LirOpcode want_op = (ty == Type::i32()) ? LirOpcode::Add32Shifted : LirOpcode::AddShifted;
            const uint8_t sz = static_cast<uint8_t>(ty.size_in_bytes());
            bool saw_add_sh = false, saw_imul = false;
            LirBlock opt_blk(0, "opt_mul");
            for (const auto& inst : lir->blocks[0]->instructions) {
                if (inst->opcode == LirOpcode::Imul || inst->opcode == LirOpcode::Imul32) saw_imul = true;
                if (inst->opcode == want_op) {
                    saw_add_sh = true;
                    auto copy = std::make_unique<LirInst>(want_op);
                    copy->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, sz));
                    copy->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, sz));
                    copy->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, sz));
                    copy->add_use(inst->uses[2]);
                    opt_blk.append_inst(std::move(copy));
                }
            }
            CHECK(saw_add_sh);
            CHECK(!saw_imul);

            LirBlock ref_blk(0, "ref_mul");
            auto mv = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mv->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, sz));
            mv->add_use(LirOperand::imm(c, sz));
            ref_blk.append_inst(std::move(mv));
            auto im = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Imul32 : LirOpcode::Imul);
            im->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, sz));
            im->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, sz));
            im->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, sz));
            ref_blk.append_inst(std::move(im));

            CHECK_EQ(verify_arm64_lir_block_equivalence(ref_blk, opt_blk, reg_mask(GPR::X0), 0, false).status,
                     EquivStatus::ProvenEquivalent);
        }
    }

    // 2. vextract_lane(f32x4, 1/2/3) and vextract_lane(f64x2, 1) emit broadcast masks
    //    (0x55/0xAA/0xFF and 0x03) and encode to a single DUP instruction (4 bytes, no V31).
    {
        Module mod("a64_ext");
        Function* fn = mod.create_function("ext", Type::f32(), {Type::f32x4(), Type::f64x2()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* vf = b.add_block_param(entry, Type::f32x4());
        Value* vd = b.add_block_param(entry, Type::f64x2());
        Value* f1 = b.build_vextract_lane(vf, 1);
        Value* f2 = b.build_vextract_lane(vf, 2);
        Value* f3 = b.build_vextract_lane(vf, 3);
        Value* d1 = b.build_vextract_lane(vd, 1);
        Value* sum = b.build_add(b.build_add(f1, f2), b.build_add(f3, b.build_fptrunc_f32_f64(d1)));
        b.build_ret(sum);
        fn->rebuild_cfg_predecessors();

        AArch64ISel isel(Target::aarch64_macos(), CallingConvention::aapcs64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        std::vector<int64_t> pshuf_masks, shufpd_masks;
        for (const auto& inst : lir->blocks[0]->instructions) {
            if (inst->opcode == LirOpcode::Pshufd) pshuf_masks.push_back(inst->uses.back().imm_int);
            if (inst->opcode == LirOpcode::Shufpd) shufpd_masks.push_back(inst->uses.back().imm_int);
        }
        REQUIRE_EQ(pshuf_masks.size(), size_t(3));
        CHECK_EQ(pshuf_masks[0], int64_t(0x55));
        CHECK_EQ(pshuf_masks[1], int64_t(0xAA));
        CHECK_EQ(pshuf_masks[2], int64_t(0xFF));
        REQUIRE_EQ(shufpd_masks.size(), size_t(1));
        CHECK_EQ(shufpd_masks[0], int64_t(3));

        LivenessAnalysis liveness(*lir);
        liveness.run();
        LinearScanAllocator regalloc(*lir, liveness, lir->calling_conv);
        regalloc.allocate();
        auto compiled = compile_lir_to_aarch64(*lir, Target::aarch64_macos());
        CHECK(!compiled.code_buffer.empty());
    }

    // 3. Insertps where dst == uses[1] (scalar source) must preserve uses[1] across src0 copy.
    {
        LirFunction fn;
        fn.name = "ins_alias_test";
        fn.calling_conv = CallingConvention::aapcs64();
        auto bb = std::make_unique<LirBlock>(0, "entry");
        auto ins2 = std::make_unique<LirInst>(LirOpcode::Insertps);
        ins2->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        ins2->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16));
        ins2->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16));
        ins2->add_use(LirOperand::imm(3 << 4, 1));
        bb->append_inst(std::move(ins2));
        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        bb->append_inst(std::move(ret));
        fn.blocks.push_back(std::move(bb));

        auto compiled = compile_lir_to_aarch64(fn, Target::aarch64_macos());
        // Expected 3 instructions before ret: orr v31, v0, v0; orr v0, v1, v1; ins v0.s[3], v31.s[0]; ret
        CHECK_EQ(compiled.code_buffer.size(), size_t(16));
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
