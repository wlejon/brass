#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/arm64_lir_equiv.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/peephole.hpp>

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

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
