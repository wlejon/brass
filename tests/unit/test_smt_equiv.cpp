#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/verify/bv_solver.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <cstdio>
#include <cstdlib>

using namespace brass;
using namespace brass::verify;

TEST_CASE("SMT Equiv - Algebraic and Bitwise Identities Proven Equivalent") {
    Module m1("m1");
    Builder b1(m1);
    Function* f1 = m1.create_function("f", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b1.set_function(f1);
    BasicBlock* e1 = b1.append_block("entry");
    Value* x1 = b1.add_block_param(e1, Type::i64());
    Value* y1 = b1.add_block_param(e1, Type::i64());
    Value* z1 = b1.add_block_param(e1, Type::i64());
    b1.position_at_end(e1);
    Value* a1 = b1.build_and(x1, y1);
    Value* a2 = b1.build_and(x1, z1);
    Value* or1 = b1.build_or(a1, a2);
    Value* mul4 = b1.build_mul(or1, b1.build_iconst_i64(4));
    Value* res1 = b1.build_add(mul4, b1.build_iconst_i64(0));
    b1.build_ret(res1);
    f1->rebuild_cfg_predecessors();

    Module m2("m2");
    Builder b2(m2);
    Function* f2 = m2.create_function("f", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b2.set_function(f2);
    BasicBlock* e2 = b2.append_block("entry");
    Value* x2 = b2.add_block_param(e2, Type::i64());
    Value* y2 = b2.add_block_param(e2, Type::i64());
    Value* z2 = b2.add_block_param(e2, Type::i64());
    b2.position_at_end(e2);
    Value* yz = b2.build_or(y2, z2);
    Value* and1 = b2.build_and(x2, yz);
    Value* res2 = b2.build_shl(and1, b2.build_iconst_i64(2));
    b2.build_ret(res2);
    f2->rebuild_cfg_predecessors();

    EquivResult res = verify_mir_equivalence(*f1, *f2);
    CHECK_EQ(res.status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("SMT Equiv - Real Optimization Passes (GVN, SCCP, BCE, CFG Simplify) Proven Equivalent") {
    Module before_mod("opt_test");
    Builder b(before_mod);
    Function* fn = before_mod.create_function("compute", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a0 = b.add_block_param(entry, Type::i32());
    Value* b0 = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    Value* c0 = b.build_iconst_i32(0);
    Value* c10 = b.build_iconst_i32(10);
    Value* ge0 = b.build_sge(a0, c0);
    b.build_guard(ge0, "deopt");
    Value* lt10 = b.build_slt(a0, c10);
    b.build_guard(lt10, "deopt");
    Value* redundant_ge = b.build_sge(a0, c0);
    b.build_guard(redundant_ge, "deopt");

    Value* sum1 = b.build_add(a0, b0);
    Value* sum2 = b.build_add(a0, b0);
    Value* diff = b.build_sub(sum1, sum2);
    Value* ans = b.build_add(sum1, diff);
    b.build_ret(ans);
    fn->rebuild_cfg_predecessors();

    auto after_mod = clone_module(before_mod);
    REQUIRE(after_mod != nullptr);
    Function* fn_after = after_mod->get_function("compute");
    REQUIRE(fn_after != nullptr);

    gvn_function(*fn_after);
    sccp_function(*fn_after);
    run_bounds_check_elimination(*fn_after, *after_mod);
    cfg_simplify_function(*fn_after);

    EquivResult res = verify_mir_equivalence(*fn, *fn_after);
    CHECK_EQ(res.status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("SMT Equiv - Catches Bogus Guard Elimination with Concrete Counterexample") {
    Module m1("m_guard_b");
    Builder b1(m1);
    Function* f1 = m1.create_function("check_idx", Type::i32(), {Type::i32()});
    b1.set_function(f1);
    BasicBlock* e1 = b1.append_block("entry");
    Value* i1 = b1.add_block_param(e1, Type::i32());
    b1.position_at_end(e1);
    b1.build_guard(b1.build_sge(i1, b1.build_iconst_i32(0)), "deopt");
    b1.build_guard(b1.build_slt(i1, b1.build_iconst_i32(10)), "deopt");
    b1.build_ret(i1);
    f1->rebuild_cfg_predecessors();

    // Bogus optimization: dropped guard (i < 10)
    Module m2("m_guard_a");
    Builder b2(m2);
    Function* f2 = m2.create_function("check_idx", Type::i32(), {Type::i32()});
    b2.set_function(f2);
    BasicBlock* e2 = b2.append_block("entry");
    Value* i2 = b2.add_block_param(e2, Type::i32());
    b2.position_at_end(e2);
    b2.build_guard(b2.build_sge(i2, b2.build_iconst_i32(0)), "deopt");
    b2.build_ret(i2);
    f2->rebuild_cfg_predecessors();

    EquivResult res = verify_mir_equivalence(*f1, *f2);
    CHECK_EQ(res.status, EquivStatus::Counterexample);
    REQUIRE_EQ(res.counterexample.size(), 1u);
    CHECK(res.counterexample[0].as_i32() >= 10);
}

TEST_CASE("SMT Equiv - Verifies Deopt state_map() and resume_id Equivalence on Triggered Guards (#75)") {
    // Baseline function: guards (x < 10) with live interpreter state [x, x + y]
    Module m_base("m_deopt_base");
    Builder b1(m_base);
    Function* f_base = m_base.create_function("deopt_fn", Type::i64(), {Type::i64(), Type::i64()});
    b1.set_function(f_base);
    BasicBlock* e1 = b1.append_block("entry");
    Value* x1 = b1.add_block_param(e1, Type::i64());
    Value* y1 = b1.add_block_param(e1, Type::i64());
    b1.position_at_end(e1);
    Value* sum1 = b1.build_add(x1, y1);
    Value* cond1 = b1.build_slt(x1, b1.build_iconst_i64(10));
    Instruction* g1 = b1.build_guard(cond1, "deopt", {x1, sum1});
    g1->set_resume_id(7);
    b1.build_ret(sum1);
    f_base->rebuild_cfg_predecessors();

    // 1. Valid state_map rewrite: sum computed as (y + x)
    Module m_valid("m_deopt_valid");
    Builder b2(m_valid);
    Function* f_valid = m_valid.create_function("deopt_fn", Type::i64(), {Type::i64(), Type::i64()});
    b2.set_function(f_valid);
    BasicBlock* e2 = b2.append_block("entry");
    Value* x2 = b2.add_block_param(e2, Type::i64());
    Value* y2 = b2.add_block_param(e2, Type::i64());
    b2.position_at_end(e2);
    Value* sum2 = b2.build_add(y2, x2);
    Value* cond2 = b2.build_slt(x2, b2.build_iconst_i64(10));
    Instruction* g2 = b2.build_guard(cond2, "deopt", {x2, sum2});
    g2->set_resume_id(7);
    b2.build_ret(sum2);
    f_valid->rebuild_cfg_predecessors();

    EquivResult r_ok = verify_mir_equivalence(*f_base, *f_valid);
    CHECK_EQ(r_ok.status, EquivStatus::ProvenEquivalent);

    // 2. Corrupted state_map: identical guard condition and return value, but state_map swaps [sum, x]
    Module m_swapped("m_deopt_swapped");
    Builder b3(m_swapped);
    Function* f_swapped = m_swapped.create_function("deopt_fn", Type::i64(), {Type::i64(), Type::i64()});
    b3.set_function(f_swapped);
    BasicBlock* e3 = b3.append_block("entry");
    Value* x3 = b3.add_block_param(e3, Type::i64());
    Value* y3 = b3.add_block_param(e3, Type::i64());
    b3.position_at_end(e3);
    Value* sum3 = b3.build_add(x3, y3);
    Value* cond3 = b3.build_slt(x3, b3.build_iconst_i64(10));
    Instruction* g3 = b3.build_guard(cond3, "deopt", {sum3, x3});
    g3->set_resume_id(7);
    b3.build_ret(sum3);
    f_swapped->rebuild_cfg_predecessors();

    EquivResult r_swapped = verify_mir_equivalence(*f_base, *f_swapped);
    CHECK_EQ(r_swapped.status, EquivStatus::Counterexample);
    REQUIRE_EQ(r_swapped.counterexample.size(), 2u);
    CHECK(r_swapped.counterexample[0].as_i64() >= 10);
    CHECK(r_swapped.diagnostic.find("deopt state_map") != std::string::npos);

    // 3. Corrupted resume_id: identical state_map and return value, but wrong resume_id
    auto m_bad_id = clone_module(m_base);
    Function* f_bad_id = m_bad_id->get_function("deopt_fn");
    for (Instruction* inst : *f_bad_id->entry_block()) {
        if (inst->opcode() == Opcode::guard) inst->set_resume_id(99);
    }
    EquivResult r_bad_id = verify_mir_equivalence(*f_base, *f_bad_id);
    CHECK_EQ(r_bad_id.status, EquivStatus::Counterexample);
}

TEST_CASE("SMT Equiv - Catches Missing Sign-Extension and Signed/Unsigned Mismatch") {
    Module m1("m_sext");
    Builder b1(m1);
    Function* f1 = m1.create_function("ext", Type::i64(), {Type::i32()});
    b1.set_function(f1);
    BasicBlock* e1 = b1.append_block("entry");
    Value* x1 = b1.add_block_param(e1, Type::i32());
    b1.position_at_end(e1);
    b1.build_ret(b1.build_sext_i64(x1));
    f1->rebuild_cfg_predecessors();

    Module m2("m_zext");
    Builder b2(m2);
    Function* f2 = m2.create_function("ext", Type::i64(), {Type::i32()});
    b2.set_function(f2);
    BasicBlock* e2 = b2.append_block("entry");
    Value* x2 = b2.add_block_param(e2, Type::i32());
    b2.position_at_end(e2);
    b2.build_ret(b2.build_zext_i64(x2));
    f2->rebuild_cfg_predecessors();

    EquivResult res = verify_mir_equivalence(*f1, *f2);
    CHECK_EQ(res.status, EquivStatus::Counterexample);
    REQUIRE_EQ(res.counterexample.size(), 1u);
    CHECK(res.counterexample[0].as_i32() < 0);
}

TEST_CASE("SMT Equiv - Catches Signed Overflow Bug in (x * 2) / 2 != x") {
    Module m1("m_ovf1");
    Builder b1(m1);
    Function* f1 = m1.create_function("f", Type::i32(), {Type::i32()});
    b1.set_function(f1);
    BasicBlock* e1 = b1.append_block("entry");
    Value* x1 = b1.add_block_param(e1, Type::i32());
    b1.position_at_end(e1);
    Value* two = b1.build_iconst_i32(2);
    Value* p = b1.build_mul(x1, two);
    Value* q = b1.build_sdiv(p, two);
    b1.build_ret(q);
    f1->rebuild_cfg_predecessors();

    Module m2("m_ovf2");
    Builder b2(m2);
    Function* f2 = m2.create_function("f", Type::i32(), {Type::i32()});
    b2.set_function(f2);
    BasicBlock* e2 = b2.append_block("entry");
    Value* x2 = b2.add_block_param(e2, Type::i32());
    b2.position_at_end(e2);
    b2.build_ret(x2);
    f2->rebuild_cfg_predecessors();

    EquivResult res = verify_mir_equivalence(*f1, *f2);
    CHECK_EQ(res.status, EquivStatus::Counterexample);
    REQUIRE_EQ(res.counterexample.size(), 1u);
    int32_t cx = res.counterexample[0].as_i32();
    int32_t wrapped = static_cast<int32_t>(static_cast<uint32_t>(cx) * 2u) / 2;
    CHECK(wrapped != cx);
}

TEST_CASE("SMT Equiv - Memory Store Forwarding vs Bogus Dead Store Elimination") {
    Module m1("m_mem1");
    Builder b1(m1);
    Function* f1 = m1.create_function("mem_op", Type::i64(), {Type::ptr(), Type::i64()});
    b1.set_function(f1);
    BasicBlock* e1 = b1.append_block("entry");
    Value* p1 = b1.add_block_param(e1, Type::ptr());
    Value* v1 = b1.add_block_param(e1, Type::i64());
    b1.position_at_end(e1);
    b1.build_store(Type::i64(), p1, 0, b1.build_iconst_i64(100));
    b1.build_store(Type::i64(), p1, 0, v1);
    Value* ld = b1.build_load(Type::i64(), p1, 0);
    b1.build_ret(ld);
    f1->rebuild_cfg_predecessors();

    Module m2("m_mem2");
    Builder b2(m2);
    Function* f2 = m2.create_function("mem_op", Type::i64(), {Type::ptr(), Type::i64()});
    b2.set_function(f2);
    BasicBlock* e2 = b2.append_block("entry");
    Value* p2 = b2.add_block_param(e2, Type::ptr());
    Value* v2 = b2.add_block_param(e2, Type::i64());
    b2.position_at_end(e2);
    b2.build_store(Type::i64(), p2, 0, v2);
    b2.build_ret(v2);
    f2->rebuild_cfg_predecessors();

    Module m3("m_mem3");
    Builder b3(m3);
    Function* f3 = m3.create_function("mem_op", Type::i64(), {Type::ptr(), Type::i64()});
    b3.set_function(f3);
    BasicBlock* e3 = b3.append_block("entry");
    b3.add_block_param(e3, Type::ptr());
    Value* v3 = b3.add_block_param(e3, Type::i64());
    b3.position_at_end(e3);
    b3.build_ret(v3);
    f3->rebuild_cfg_predecessors();

    EquivResult r_valid = verify_mir_equivalence(*f1, *f2);
    CHECK_EQ(r_valid.status, EquivStatus::ProvenEquivalent);

    EquivResult r_bogus = verify_mir_equivalence(*f1, *f3);
    CHECK_EQ(r_bogus.status, EquivStatus::Counterexample);

    std::string smt2 = export_mir_equivalence_smt2(*f1, *f2);
    CHECK(smt2.find("(set-logic QF_BV)") != std::string::npos);
    CHECK(smt2.find("(check-sat)") != std::string::npos);
}

TEST_CASE("SMT Equiv - x64 LIR Block Equivalence Catches #59-Class Upper-32-Bit and RFLAGS Bugs") {
    using namespace brass::codegen;

    // 1. Valid peephole: mov rax, rbx; mov rcx, rax -> mov rax, rbx; mov rcx, rbx
    {
        LirBlock b1(0, "b1");
        auto i1 = std::make_unique<LirInst>(LirOpcode::Mov);
        i1->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        i1->add_use(LirOperand::preg_gpr(x64::GPR::RBX, 8));
        b1.append_inst(std::move(i1));
        auto i2 = std::make_unique<LirInst>(LirOpcode::Mov);
        i2->add_def(LirOperand::preg_gpr(x64::GPR::RCX, 8));
        i2->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        b1.append_inst(std::move(i2));

        LirBlock b2(0, "b2");
        auto j1 = std::make_unique<LirInst>(LirOpcode::Mov);
        j1->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        j1->add_use(LirOperand::preg_gpr(x64::GPR::RBX, 8));
        b2.append_inst(std::move(j1));
        auto j2 = std::make_unique<LirInst>(LirOpcode::Mov);
        j2->add_def(LirOperand::preg_gpr(x64::GPR::RCX, 8));
        j2->add_use(LirOperand::preg_gpr(x64::GPR::RBX, 8));
        b2.append_inst(std::move(j2));

        LirEquivResult r = verify_x64_lir_block_equivalence(b1, b2);
        CHECK_EQ(r.status, EquivStatus::ProvenEquivalent);
    }

    // 2. #59-class bug: add32 eax, 0 dropped completely instead of mov32 eax, eax
    // (fails to zero-extend upper 32 bits of RAX!)
    {
        LirBlock before_blk(0, "before_add32_0");
        auto add32 = std::make_unique<LirInst>(LirOpcode::Add32);
        add32->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add32->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add32->add_use(LirOperand::imm32(0));
        before_blk.append_inst(std::move(add32));

        LirBlock valid_after(0, "valid_mov32");
        auto mov32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        mov32->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        mov32->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        valid_after.append_inst(std::move(mov32));

        LirBlock buggy_empty(0, "buggy_empty");

        // When flags are not checked, add32 eax, 0 == mov32 eax, eax
        LirEquivResult r_ok = verify_x64_lir_block_equivalence(
            before_blk, valid_after, x64::reg_mask(x64::GPR::RAX), 0, false);
        CHECK_EQ(r_ok.status, EquivStatus::ProvenEquivalent);

        // Dropping add32 eax, 0 leaves upper 32 bits of RAX dirty!
        LirEquivResult r_bug = verify_x64_lir_block_equivalence(
            before_blk, buggy_empty, x64::reg_mask(x64::GPR::RAX), 0, false);
        CHECK_EQ(r_bug.status, EquivStatus::Counterexample);
    }

    // 3. #59-class bug: mov rax, 0 rewritten to xor32 eax, eax while flags are live (before setcc)
    {
        LirBlock before_blk(0, "before_mov0_setcc");
        auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp->add_use(LirOperand::preg_gpr(x64::GPR::RDI, 8));
        cmp->add_use(LirOperand::preg_gpr(x64::GPR::RSI, 8));
        before_blk.append_inst(std::move(cmp));

        auto mov0 = std::make_unique<LirInst>(LirOpcode::Mov);
        mov0->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        mov0->add_use(LirOperand::imm(0, 8));
        before_blk.append_inst(std::move(mov0));

        auto setcc1 = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc1->condition = x64::Condition::L;
        setcc1->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 1));
        before_blk.append_inst(std::move(setcc1));

        LirBlock buggy_after(0, "buggy_xor_setcc");
        auto cmp2 = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp2->add_use(LirOperand::preg_gpr(x64::GPR::RDI, 8));
        cmp2->add_use(LirOperand::preg_gpr(x64::GPR::RSI, 8));
        buggy_after.append_inst(std::move(cmp2));

        // Buggy rewrite: xor32 eax, eax clobbers ZF/SF/OF/CF from cmp!
        auto xor0 = std::make_unique<LirInst>(LirOpcode::Xor32);
        xor0->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        xor0->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        xor0->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        buggy_after.append_inst(std::move(xor0));

        auto setcc2 = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc2->condition = x64::Condition::L;
        setcc2->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 1));
        buggy_after.append_inst(std::move(setcc2));

        LirEquivResult r_flag_bug = verify_x64_lir_block_equivalence(
            before_blk, buggy_after, x64::reg_mask(x64::GPR::RAX), 0, false);
        CHECK_EQ(r_flag_bug.status, EquivStatus::Counterexample);
    }
}

TEST_CASE("SMT Equiv - DiffFuzzer smt_verify_passes Integration") {
    Module mod("fuzz_smt");
    Builder b(mod);
    Function* fn = mod.create_function("test_fn", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);
    Value* s1 = b.build_add(x, y);
    Value* s2 = b.build_add(x, y);
    Value* r = b.build_sub(s1, s2);
    Value* ans = b.build_add(x, r);
    b.build_ret(ans);
    fn->rebuild_cfg_predecessors();

    fuzz::DiffFuzzerOptions opts;
    opts.smt_verify_passes = true;
    opts.save_reproducers = false;
    fuzz::DiffFuzzer fuzzer(opts);

    fuzz::DiffResult res = fuzzer.run_test(mod, "test_fn",
        {RuntimeValue::from_i64(7), RuntimeValue::from_i64(13)}, 1);
    CHECK(res.passed);
}

TEST_CASE("BVSolver #153 - Soundness Regressions: Unsupported Op Unknown, Bit-Sliced MSB, and Interval Bounds") {
    // 1. Unsupported floating-point op whose uninterpreted SAT model fails eval_concrete
    //    must return SolveOutcome::Unknown (never SolveOutcome::Unsat).
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 32, true);
        BVNodeId y = ctx.make_var("y", 32, true);
        // Constrain x and y to a hard algebraic hash so Stage 1 concrete search does not hit it,
        // while Stage 2 bit-blaster treats FAdd as uninterpreted and produces a SAT model that fails eval_concrete.
        BVNodeId magic_x = ctx.make_const(0x3F800000u, 32); // 1.0f
        BVNodeId magic_y = ctx.make_const(0x40000000u, 32); // 2.0f
        BVNodeId x_eq = ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Xor, x, ctx.make_const(0x13579BDFu, 32), 32),
                                        ctx.make_const(0x3F800000u ^ 0x13579BDFu, 32), 1);
        BVNodeId y_eq = ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Xor, y, ctx.make_const(0x2468ACE0u, 32), 32),
                                        ctx.make_const(0x40000000u ^ 0x2468ACE0u, 32), 1);
        BVNodeId fsum = ctx.make_binary(BVOp::FAdd, x, y, 32);
        // Impossible floating-point result for 1.0f + 2.0f == 3.0f (0x40400000), but ask for 0x40800000 (4.0f)
        // Wait: even better, ask for 1.0f + 2.0f == 3.0f (0x40400000), which IS satisfiable at x=1.0f, y=2.0f,
        // while the uninterpreted FAdd node gets assigned 0 by SAT unless constrained to 0x40400000:
        // Wait: if we constrain fsum != 0x40400000 && fsum == 0x40400000 it's UNSAT.
        // If we constrain x_eq && y_eq && (FAdd(x, y) == 0x40400000):
        // The SAT solver assigns x = 1.0f, y = 2.0f, and uninterpreted fsum = 0x40400000 -> eval_concrete is TRUE!
        // What if we constrain x_eq && y_eq && (FEq(x, y) == 0) && (FAdd(x, y) == 0)?
        // For x = 1.0f, y = 2.0f, FAdd(x, y) is 3.0f (0x40400000 != 0), whereas uninterpreted FAdd is assigned 0!
        // Wait: to make the query genuinely satisfiable on some non-boundary float while the SAT solver picks x=0, y=0:
        // Let's require FAdd(x, y) == 0x40400000 (3.0f) with no constraint on x, y!
        // Then Stage 1 checks boundary floats (where 1.0f + 2.0f = 3.0f might be found if pair search runs).
        // To bypass Stage 1 pair search, use 5 variables so nvars > 4:
        BVNodeId z1 = ctx.make_var("z1", 32, false);
        BVNodeId z2 = ctx.make_var("z2", 32, false);
        BVNodeId z3 = ctx.make_var("z3", 32, false);
        BVNodeId dom = ctx.make_binary(BVOp::And,
            ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Add, z1, z2, 32), ctx.make_const(0x12345678u, 32), 1),
            ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Xor, z2, z3, 32), ctx.make_const(0x9ABCDEF0u, 32), 1), 1);
        BVNodeId f_cond = ctx.make_binary(BVOp::Eq, fsum, ctx.make_const(0x40400000u, 32), 1);
        BVNodeId bad = ctx.make_binary(BVOp::And, dom, f_cond, 1);
        SolveOutcome out = solve_nonzero(ctx, bad, 10000);
        // Satisfiable (e.g. x = 1.0f, y = 2.0f, z2 = 0, z1 = 0x12345678, z3 = 0x9ABCDEF0),
        // so solver must NEVER claim Unsat!
        CHECK(out.status != SolveOutcome::Unsat);
        (void)magic_x;
        (void)magic_y;
        (void)x_eq;
        (void)y_eq;
    }

    // 2. Pin bit-sliced `(a != b) && dom` path where `a` and `b` differ ONLY in the MSB under `dom`.
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 16, false);
        BVNodeId y = ctx.make_var("y", 16, false);
        // Under dom: (x ^ 0x5A3C) == 0x8000 && y == 0x5A3C -> x = 0xDA3C, y = 0x5A3C (differ ONLY in bit 15, the MSB!)
        BVNodeId d1 = ctx.make_binary(BVOp::Eq,
            ctx.make_binary(BVOp::Xor, x, ctx.make_const(0x5A3Cu, 16), 16),
            ctx.make_const(0x8000u, 16), 1);
        BVNodeId d2 = ctx.make_binary(BVOp::Eq,
            ctx.make_binary(BVOp::Add, y, ctx.make_const(0x1111u, 16), 16),
            ctx.make_const(0x5A3Cu + 0x1111u, 16), 1);
        BVNodeId dom = ctx.make_binary(BVOp::And, d1, d2, 1);
        BVNodeId neq = ctx.make_binary(BVOp::Ne, x, y, 1);
        BVNodeId bad = ctx.make_binary(BVOp::And, neq, dom, 1);

        SolveOutcome res = solve_nonzero(ctx, bad, 10000);
        CHECK_EQ(res.status, SolveOutcome::Sat);
        REQUIRE_EQ(res.model.size(), 2u);
        CHECK_EQ(ctx.eval_concrete(bad, res.model), 1u);
        CHECK_EQ(res.model[0], 0xDA3Cu);
        CHECK_EQ(res.model[1], 0x5A3Cu);
    }

    // 3. Pin `Or`-rooted query that must skip the top-level `And` bit-sliced shortcut.
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 16, false);
        BVNodeId y = ctx.make_var("y", 16, false);
        BVNodeId neq = ctx.make_binary(BVOp::Ne, x, y, 1);
        BVNodeId eq_magic = ctx.make_binary(BVOp::Eq, x, ctx.make_const(0xBEEFu, 16), 1);
        BVNodeId or_root = ctx.make_binary(BVOp::Or, neq, eq_magic, 1);

        SolveOutcome res = solve_nonzero(ctx, or_root, 10000);
        CHECK_EQ(res.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(or_root, res.model), 1u);
    }

    // 4. Interval soundness regressions for BVOp::And (umax) and BVOp::Or (umin)
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 8, false);
        // a in [0, 4] (via zext of 1-bit << 2, so known_zeros allows bit 2, umax = 4)
        BVNodeId b0 = ctx.make_extract(x, 0, 0);
        BVNodeId a = ctx.make_binary(BVOp::Shl, ctx.make_zext(b0, 7), ctx.make_const(2, 8), 8);
        // Or with 3-bit value in [0, 3]: a2 has umax = 4 + 3 = 7, wait:
        // Let's construct `u` with umax = 4 and known_zeros = ~7 (bits 0..2 can be 1):
        // u = ite(c, 4, 3) -> umin = 3, umax = 4, known_zeros = ~7 (0xF8)
        BVNodeId c = ctx.make_extract(x, 1, 1);
        BVNodeId u = ctx.make_ite(c, ctx.make_const(4, 8), ctx.make_const(3, 8));
        // v = 3 -> umin = 3, umax = 3, known_zeros = ~3 (0xFC)
        // Wait: if v = ite(c2, 6, 3) -> umin = 3, umax = 6, known_zeros = ~7 (0xF8)
        // And w = zext(extract(x, 3, 2)) -> umin = 0, umax = 3, known_zeros = ~3 (0xFC)
        // Then u & w: u.umax = 4, w.umax = 3, known_zeros = ~3.
        // What if u has umax = 4 (known_zeros = ~7) and v has umax = 6 (known_zeros = ~3)?
        // E.g. v = ite(c2, 2, 3) | ... wait: if u = ite(c, 4, 3) (umax = 4) and v = 3 (const):
        // When c = 0, u = 3, so (u & 3) == 3!
        BVNodeId u_and_3 = ctx.make_binary(BVOp::And, u, ctx.make_const(3, 8), 8);
        BVNodeId eq_3 = ctx.make_binary(BVOp::Eq, u_and_3, ctx.make_const(3, 8), 1);
        SolveOutcome r_and = solve_nonzero(ctx, eq_3, 1000);
        CHECK_EQ(r_and.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(eq_3, r_and.model), 1u);

        // BVOp::Or umin regression:
        // p = ite(c, 5, 6) -> umin = 5, umax = 6, known_ones = 4 (0b100)
        // q = 2 -> umin = 2, umax = 2, known_ones = 2 (0b010)
        // When c = 0, p = 6, so (p | 2) = 6 | 2 = 6!
        // Previously, Or computed umin = max(5, 2) | (4 | 2) = 5 | 6 = 7 > 6, falsely folding (p | 2) == 6 to 0!
        BVNodeId p = ctx.make_ite(c, ctx.make_const(5, 8), ctx.make_const(6, 8));
        BVNodeId p_or_2 = ctx.make_binary(BVOp::Or, p, ctx.make_const(2, 8), 8);
        BVNodeId eq_6 = ctx.make_binary(BVOp::Eq, p_or_2, ctx.make_const(6, 8), 1);
        SolveOutcome r_or = solve_nonzero(ctx, eq_6, 1000);
        CHECK_EQ(r_or.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(eq_6, r_or.model), 1u);
        (void)a;
    }
}

TEST_CASE("BVSolver #153 - Per-Op Differential Test Across Widths 1, 7, 8, 32, 64") {
    const uint32_t widths[] = {1, 7, 8, 32, 64};
    const BVOp bin_ops[] = {
        BVOp::And, BVOp::Or, BVOp::Xor, BVOp::Add, BVOp::Sub, BVOp::Mul,
        BVOp::UDiv, BVOp::SDiv, BVOp::URem, BVOp::SRem,
        BVOp::Shl, BVOp::LShr, BVOp::AShr,
        BVOp::Eq, BVOp::Ne, BVOp::Ult, BVOp::Ule, BVOp::Ugt, BVOp::Uge,
        BVOp::Slt, BVOp::Sle, BVOp::Sgt, BVOp::Sge,
        BVOp::SAddOvf, BVOp::SSubOvf, BVOp::SMulOvf,
        BVOp::UAddOvf, BVOp::USubOvf, BVOp::UMulOvf
    };
    const BVOp un_ops[] = {
        BVOp::Not, BVOp::Neg, BVOp::Clz, BVOp::Ctz, BVOp::Popcnt
    };

    uint64_t rng = 0xCAFEBABE12345678ULL;
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };

    for (uint32_t w : widths) {
        const uint64_t m = width_mask(w);
        const uint64_t smin_u = (w >= 64) ? (1ULL << 63) : (1ULL << (w - 1));
        const uint64_t smax_u = smin_u - 1ULL;

        // Edge-case constants: 0, 1, -1, INT_MIN, INT_MAX, w-1, w, w+1, randoms
        std::vector<uint64_t> edge_vals = {
            0ULL,
            1ULL & m,
            m, // -1 (all ones)
            smin_u & m, // INT_MIN
            smax_u & m, // INT_MAX
            static_cast<uint64_t>(w - 1) & m,
            static_cast<uint64_t>(w) & m,
            static_cast<uint64_t>(w + 1) & m,
            next_rng() & m,
            next_rng() & m
        };

        // 1. Test all unary ops at width w
        for (BVOp uop : un_ops) {
            for (uint64_t cx : edge_vals) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId z = ctx.make_var("z", w, false);
                BVNodeId op_node = ctx.make_unary(uop, x, w);
                uint64_t expected_z = ctx.eval_concrete(op_node, {cx, 0});

                BVNodeId cond = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Eq, z, op_node, 1), 1);
                SolveOutcome sat_res = solve_nonzero(ctx, cond, 10000);
                CHECK_EQ(sat_res.status, SolveOutcome::Sat);
                REQUIRE_EQ(sat_res.model.size(), 2u);
                CHECK_EQ(sat_res.model[1], expected_z);

                // Also verify that (x == cx) && (op(x) != expected_z) is UNSAT
                BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Ne, op_node, ctx.make_const(expected_z, w), 1), 1);
                SolveOutcome unsat_res = solve_nonzero(ctx, bad_neq, 10000);
                CHECK_EQ(unsat_res.status, SolveOutcome::Unsat);
            }
        }

        // 2. Test Extract, ZExt, SExt, Concat, Ite at width w
        for (uint64_t cx : edge_vals) {
            uint64_t cy = edge_vals[(cx + 3) % edge_vals.size()];
            // ZExt & SExt to min(64, w + 5)
            if (w <= 59) {
                for (bool is_sext : {false, true}) {
                    BVContext ctx;
                    BVNodeId x = ctx.make_var("x", w, false);
                    BVNodeId z = ctx.make_var("z", w + 5, false);
                    BVNodeId ext = is_sext ? ctx.make_sext(x, 5) : ctx.make_zext(x, 5);
                    uint64_t expected = ctx.eval_concrete(ext, {cx, 0});
                    BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                        ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                        ctx.make_binary(BVOp::Ne, ext, ctx.make_const(expected, w + 5), 1), 1);
                    CHECK_EQ(solve_nonzero(ctx, bad_neq, 10000).status, SolveOutcome::Unsat);
                    (void)z;
                }
            }
            // Extract boundaries (low bit, high bit, mid slice)
            {
                uint32_t hi = (w > 1) ? (w - 1) : 0;
                uint32_t lo = (w > 2) ? 1 : 0;
                uint32_t ew = hi - lo + 1;
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId ex = ctx.make_extract(x, hi, lo);
                uint64_t expected = ctx.eval_concrete(ex, {cx});
                BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Ne, ex, ctx.make_const(expected, ew), 1), 1);
                CHECK_EQ(solve_nonzero(ctx, bad_neq, 10000).status, SolveOutcome::Unsat);
            }
            // Concat & Ite
            if (w <= 32) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId y = ctx.make_var("y", w, false);
                BVNodeId c = ctx.make_var("c", 1, false);
                BVNodeId cat = ctx.make_concat(x, y);
                BVNodeId ite = ctx.make_ite(c, x, y);
                uint64_t exp_cat = ctx.eval_concrete(cat, {cx, cy, 1});
                uint64_t exp_ite = ctx.eval_concrete(ite, {cx, cy, 1});
                BVNodeId fix = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::And,
                        ctx.make_binary(BVOp::Eq, y, ctx.make_const(cy, w), 1),
                        ctx.make_binary(BVOp::Eq, c, ctx.make_const(1, 1), 1), 1), 1);
                CHECK_EQ(solve_nonzero(ctx, ctx.make_binary(BVOp::And, fix,
                    ctx.make_binary(BVOp::Ne, cat, ctx.make_const(exp_cat, 2 * w), 1), 1), 10000).status,
                    SolveOutcome::Unsat);
                CHECK_EQ(solve_nonzero(ctx, ctx.make_binary(BVOp::And, fix,
                    ctx.make_binary(BVOp::Ne, ite, ctx.make_const(exp_ite, w), 1), 1), 10000).status,
                    SolveOutcome::Unsat);
            }
        }

        // 3. Test all binary ops at width w across critical pairs (including INT_MIN / -1, div-by-0, shift >= w)
        std::vector<std::pair<uint64_t, uint64_t>> pairs = {
            {smin_u & m, m},                       // INT_MIN, -1
            {smin_u & m, 0ULL},                    // INT_MIN, 0 (div/rem by 0)
            {m, 0ULL},                             // -1, 0 (div/rem by 0)
            {smax_u & m, 1ULL & m},                // INT_MAX, 1
            {m, static_cast<uint64_t>(w) & m},     // shift == width
            {m, static_cast<uint64_t>(w + 1) & m}, // shift > width
            {smin_u & m, static_cast<uint64_t>(w - 1) & m},
            {edge_vals[8], edge_vals[9]},
            {edge_vals[9], edge_vals[8]}
        };

        for (BVOp bop : bin_ops) {
            bool is_pred = (bop == BVOp::Eq || bop == BVOp::Ne ||
                            bop == BVOp::Ult || bop == BVOp::Ule || bop == BVOp::Ugt || bop == BVOp::Uge ||
                            bop == BVOp::Slt || bop == BVOp::Sle || bop == BVOp::Sgt || bop == BVOp::Sge ||
                            bop == BVOp::SAddOvf || bop == BVOp::SSubOvf || bop == BVOp::SMulOvf ||
                            bop == BVOp::UAddOvf || bop == BVOp::USubOvf || bop == BVOp::UMulOvf);
            uint32_t zw = is_pred ? 1u : w;

            for (const auto& [cx, cy] : pairs) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId y = ctx.make_var("y", w, false);
                BVNodeId z = ctx.make_var("z", zw, false);
                BVNodeId op_node = ctx.make_binary(bop, x, y, zw);
                uint64_t expected_z = ctx.eval_concrete(op_node, {cx, cy, 0});

                BVNodeId fix_xy = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Eq, y, ctx.make_const(cy, w), 1), 1);

                BVNodeId sat_q = ctx.make_binary(BVOp::And, fix_xy,
                    ctx.make_binary(BVOp::Eq, z, op_node, 1), 1);
                SolveOutcome sat_res = solve_nonzero(ctx, sat_q, 10000);
                CHECK_EQ(sat_res.status, SolveOutcome::Sat);
                REQUIRE_EQ(sat_res.model.size(), 3u);
                CHECK_EQ(sat_res.model[2], expected_z);

                BVNodeId unsat_q = ctx.make_binary(BVOp::And, fix_xy,
                    ctx.make_binary(BVOp::Ne, op_node, ctx.make_const(expected_z, zw), 1), 1);
                SolveOutcome unsat_res = solve_nonzero(ctx, unsat_q, 10000);
                CHECK_EQ(unsat_res.status, SolveOutcome::Unsat);
            }
        }
    }
}

TEST_CASE("BVSolver #153 - Exhaustive 4-Bit Truth-Table Cross-Check on Random DAGs") {
    uint64_t rng = 0x7F4A7C159E3779B9ULL;
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };

    const BVOp ops[] = {
        BVOp::And, BVOp::Or, BVOp::Xor, BVOp::Add, BVOp::Sub, BVOp::Mul,
        BVOp::UDiv, BVOp::SDiv, BVOp::URem, BVOp::SRem,
        BVOp::Shl, BVOp::LShr, BVOp::AShr
    };
    const BVOp cmp_ops[] = {
        BVOp::Eq, BVOp::Ne, BVOp::Ult, BVOp::Ule, BVOp::Ugt, BVOp::Uge,
        BVOp::Slt, BVOp::Sle, BVOp::Sgt, BVOp::Sge,
        BVOp::SAddOvf, BVOp::SSubOvf, BVOp::SMulOvf,
        BVOp::UAddOvf, BVOp::USubOvf, BVOp::UMulOvf
    };

    uint32_t sat_count = 0;
    uint32_t unsat_count = 0;

    // 200 random DAGs over three 4-bit variables (2^12 = 4096 concrete inputs each)
    for (uint32_t trial = 0; trial < 200; ++trial) {
        BVContext ctx;
        BVNodeId v0 = ctx.make_var("x", 4, false);
        BVNodeId v1 = ctx.make_var("y", 4, false);
        BVNodeId v2 = ctx.make_var("z", 4, false);
        std::vector<BVNodeId> pool = {
            v0, v1, v2,
            ctx.make_const(next_rng() & 0xFULL, 4),
            ctx.make_const(next_rng() & 0xFULL, 4)
        };

        for (int step = 0; step < 6; ++step) {
            uint64_t r = next_rng();
            BVNodeId a = pool[static_cast<size_t>(r % pool.size())];
            BVNodeId b = pool[static_cast<size_t>((r >> 8) % pool.size())];
            uint32_t kind = static_cast<uint32_t>((r >> 16) % 10);
            if (kind == 0) {
                pool.push_back(ctx.make_unary(BVOp::Not, a, 4));
            } else if (kind == 1) {
                pool.push_back(ctx.make_unary(BVOp::Neg, a, 4));
            } else if (kind == 2) {
                BVOp uop = ((r >> 24) % 3 == 0) ? BVOp::Clz : (((r >> 24) % 3 == 1) ? BVOp::Ctz : BVOp::Popcnt);
                pool.push_back(ctx.make_unary(uop, a, 4));
            } else if (kind == 3) {
                BVNodeId c = ctx.make_binary(cmp_ops[(r >> 24) % 16], a, b, 1);
                BVNodeId d = pool[static_cast<size_t>((r >> 32) % pool.size())];
                pool.push_back(ctx.make_ite(c, a, d));
            } else {
                BVOp bop = ops[(r >> 24) % (sizeof(ops) / sizeof(ops[0]))];
                pool.push_back(ctx.make_binary(bop, a, b, 4));
            }
        }

        BVNodeId lhs = pool[pool.size() - 1];
        BVNodeId rhs = pool[pool.size() - 2];
        BVOp root_cmp = cmp_ops[next_rng() % 16];
        BVNodeId c1 = ctx.make_binary(root_cmp, lhs, rhs, 1);
        BVNodeId c2 = ctx.make_binary(cmp_ops[next_rng() % 16], pool[pool.size() - 3], v0, 1);
        BVNodeId bad_cond = ((trial & 1u) == 0)
            ? ctx.make_binary(BVOp::And, c1, c2, 1)
            : ctx.make_binary(BVOp::Ne, lhs, rhs, 1);

        SolveOutcome res = solve_nonzero(ctx, bad_cond, 10000);
        CHECK(res.status != SolveOutcome::Unknown);

        bool brute_force_sat = false;
        std::vector<uint64_t> vals(3, 0);
        std::vector<uint64_t> node_vals;
        for (uint64_t x = 0; x < 16 && !brute_force_sat; ++x) {
            vals[0] = x;
            for (uint64_t y = 0; y < 16 && !brute_force_sat; ++y) {
                vals[1] = y;
                for (uint64_t z = 0; z < 16; ++z) {
                    vals[2] = z;
                    ctx.evaluate_all(vals, node_vals);
                    if (node_vals[bad_cond] != 0) {
                        brute_force_sat = true;
                        break;
                    }
                }
            }
        }

        if (brute_force_sat) {
            ++sat_count;
            CHECK_EQ(res.status, SolveOutcome::Sat);
            CHECK_EQ(ctx.eval_concrete(bad_cond, res.model), 1u);
        } else {
            ++unsat_count;
            CHECK_EQ(res.status, SolveOutcome::Unsat);
        }
    }
    CHECK(sat_count > 20u);
    CHECK(unsat_count > 20u);
}

TEST_CASE("BVSolver #153 - Checked-In Z3 / SMT-LIB2 Corpus Verification") {
    struct CorpusCase {
        const char* name;
        SolveOutcome::Status expected;
        BVNodeId (*build)(BVContext&);
    };

    const CorpusCase corpus[] = {
        {"de_morgan_64", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 64), y = c.make_var("y", 64);
            BVNodeId lhs = c.make_unary(BVOp::Not, c.make_binary(BVOp::And, x, y, 64), 64);
            BVNodeId rhs = c.make_binary(BVOp::Or, c.make_unary(BVOp::Not, x, 64), c.make_unary(BVOp::Not, y, 64), 64);
            return c.make_binary(BVOp::Ne, lhs, rhs, 1);
        }},
        {"add_xor_and_identity_32", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 32), y = c.make_var("y", 32);
            BVNodeId lhs = c.make_binary(BVOp::Add, x, y, 32);
            BVNodeId rhs = c.make_binary(BVOp::Add,
                c.make_binary(BVOp::Xor, x, y, 32),
                c.make_binary(BVOp::Shl, c.make_binary(BVOp::And, x, y, 32), c.make_const(1, 32), 32), 32);
            return c.make_binary(BVOp::Ne, lhs, rhs, 1);
        }},
        {"sdiv_int_min_neg1_8", SolveOutcome::Sat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8), y = c.make_var("y", 8);
            // x <s 0 && y <s 0 && sdiv(x, y) <s 0 (true ONLY when x == -128 && y == -1)
            BVNodeId zero = c.make_const(0, 8);
            BVNodeId q = c.make_binary(BVOp::SDiv, x, y, 8);
            return c.make_binary(BVOp::And,
                c.make_binary(BVOp::Slt, x, zero, 1),
                c.make_binary(BVOp::And,
                    c.make_binary(BVOp::Slt, y, zero, 1),
                    c.make_binary(BVOp::Slt, q, zero, 1), 1), 1);
        }},
        {"udiv_rem_euclid_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId q = c.make_binary(BVOp::UDiv, x, y, 5);
            BVNodeId r = c.make_binary(BVOp::URem, x, y, 5);
            BVNodeId recon = c.make_binary(BVOp::Add, c.make_binary(BVOp::Mul, q, y, 5), r, 5);
            BVNodeId y_nz = c.make_binary(BVOp::Ne, y, c.make_const(0, 5), 1);
            return c.make_binary(BVOp::And, y_nz, c.make_binary(BVOp::Ne, recon, x, 1), 1);
        }},
        {"sdiv_srem_euclid_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId q = c.make_binary(BVOp::SDiv, x, y, 5);
            BVNodeId r = c.make_binary(BVOp::SRem, x, y, 5);
            BVNodeId recon = c.make_binary(BVOp::Add, c.make_binary(BVOp::Mul, q, y, 5), r, 5);
            BVNodeId y_nz = c.make_binary(BVOp::Ne, y, c.make_const(0, 5), 1);
            BVNodeId not_ovf = c.make_unary(BVOp::Not,
                c.make_binary(BVOp::And,
                    c.make_binary(BVOp::Eq, x, c.make_const(0x10, 5), 1),
                    c.make_binary(BVOp::Eq, y, c.make_const(0x1F, 5), 1), 1), 1);
            return c.make_binary(BVOp::And, c.make_binary(BVOp::And, y_nz, not_ovf, 1),
                                 c.make_binary(BVOp::Ne, recon, x, 1), 1);
        }},
        {"div_by_zero_is_zero_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8), y = c.make_var("y", 8);
            BVNodeId y_zero = c.make_binary(BVOp::Eq, y, c.make_const(0, 8), 1);
            BVNodeId uq = c.make_binary(BVOp::UDiv, x, y, 8);
            BVNodeId sq = c.make_binary(BVOp::SDiv, x, y, 8);
            BVNodeId ur = c.make_binary(BVOp::URem, x, y, 8);
            BVNodeId sr = c.make_binary(BVOp::SRem, x, y, 8);
            BVNodeId any_nz = c.make_binary(BVOp::Ne,
                c.make_binary(BVOp::Or, c.make_binary(BVOp::Or, uq, sq, 8),
                              c.make_binary(BVOp::Or, ur, sr, 8), 8),
                c.make_const(0, 8), 1);
            return c.make_binary(BVOp::And, y_zero, any_nz, 1);
        }},
        {"non_power_of_two_width7_shift_add", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 7);
            BVNodeId sum = c.make_binary(BVOp::Add, x, x, 7);
            BVNodeId sh1 = c.make_binary(BVOp::Shl, x, c.make_const(1, 7), 7);
            return c.make_binary(BVOp::Ne, sum, sh1, 1);
        }},
        {"sext_zext_non_neg_equiv_16", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId non_neg = c.make_binary(BVOp::Sge, x, c.make_const(0, 8), 1);
            BVNodeId diff = c.make_binary(BVOp::Ne, c.make_sext(x, 8), c.make_zext(x, 8), 1);
            return c.make_binary(BVOp::And, non_neg, diff, 1);
        }},
        {"clz_ctz_single_bit_sum_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId nz = c.make_binary(BVOp::Ne, x, c.make_const(0, 8), 1);
            BVNodeId pow2 = c.make_binary(BVOp::Eq,
                c.make_binary(BVOp::And, x, c.make_binary(BVOp::Sub, x, c.make_const(1, 8), 8), 8),
                c.make_const(0, 8), 1);
            BVNodeId sum = c.make_binary(BVOp::Add,
                c.make_unary(BVOp::Clz, x, 8), c.make_unary(BVOp::Ctz, x, 8), 8);
            return c.make_binary(BVOp::And, c.make_binary(BVOp::And, nz, pow2, 1),
                                 c.make_binary(BVOp::Ne, sum, c.make_const(7, 8), 1), 1);
        }},
        {"overshift_mask_32", SolveOutcome::Sat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 32), sh = c.make_var("sh", 32);
            BVNodeId ge32 = c.make_binary(BVOp::Uge, sh, c.make_const(32, 32), 1);
            BVNodeId nz = c.make_binary(BVOp::Ne, c.make_binary(BVOp::Shl, x, sh, 32), c.make_const(0, 32), 1);
            return c.make_binary(BVOp::And, ge32, nz, 1);
        }},
        {"smul_ovf_vs_sext_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId prod10 = c.make_binary(BVOp::Mul, c.make_sext(x, 5), c.make_sext(y, 5), 10);
            BVNodeId sext_lo = c.make_sext(c.make_extract(prod10, 4, 0), 5);
            BVNodeId ref_ovf = c.make_binary(BVOp::Ne, prod10, sext_lo, 1);
            BVNodeId ovf = c.make_binary(BVOp::SMulOvf, x, y, 1);
            return c.make_binary(BVOp::Ne, ovf, ref_ovf, 1);
        }},
        {"umul_ovf_vs_zext_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId prod10 = c.make_binary(BVOp::Mul, c.make_zext(x, 5), c.make_zext(y, 5), 10);
            BVNodeId hi5 = c.make_extract(prod10, 9, 5);
            BVNodeId ref_ovf = c.make_binary(BVOp::Ne, hi5, c.make_const(0, 5), 1);
            BVNodeId ovf = c.make_binary(BVOp::UMulOvf, x, y, 1);
            return c.make_binary(BVOp::Ne, ovf, ref_ovf, 1);
        }},
        {"popcnt_clear_lsb_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId nz = c.make_binary(BVOp::Ne, x, c.make_const(0, 8), 1);
            BVNodeId x_clr = c.make_binary(BVOp::And, x, c.make_binary(BVOp::Sub, x, c.make_const(1, 8), 8), 8);
            BVNodeId p1 = c.make_unary(BVOp::Popcnt, x, 8);
            BVNodeId p2 = c.make_binary(BVOp::Add, c.make_unary(BVOp::Popcnt, x_clr, 8), c.make_const(1, 8), 8);
            return c.make_binary(BVOp::And, nz, c.make_binary(BVOp::Ne, p1, p2, 1), 1);
        }}
    };

    for (const auto& tc : corpus) {
        BVContext ctx;
        BVNodeId bad = tc.build(ctx);
        SolveOutcome res = solve_nonzero(ctx, bad, 10000);
        CHECK_EQ(res.status, tc.expected);
        if (res.status == SolveOutcome::Sat) {
            CHECK_EQ(ctx.eval_concrete(bad, res.model), 1u);
        }
        std::string smt2 = ctx.export_smt2(bad, tc.name);
        CHECK(smt2.find("(set-logic QF_BV)") != std::string::npos);
        CHECK(smt2.find("(check-sat)") != std::string::npos);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
