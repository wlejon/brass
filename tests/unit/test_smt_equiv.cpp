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

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
