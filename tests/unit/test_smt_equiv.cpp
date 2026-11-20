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

TEST_CASE("SMT Equiv - Soundness: Truncated Loops, INT_MIN/-1 Traps, TLS, Stateful Calls & XMM/PF") {
    // 1. Truncated loop must return Unknown instead of falsely claiming ProvenEquivalent
    {
        Module mod("trunc_loop");
        auto make_loop = [&](const char* name, int64_t final_add) -> Function* {
            Builder b(mod);
            Function* fn = mod.create_function(name, Type::i32(), {Type::i32()});
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* hdr = b.append_block("hdr");
            BasicBlock* body = b.append_block("body");
            BasicBlock* exit = b.append_block("exit");
            Value* n = b.add_block_param(entry, Type::i32());
            Value* i = b.add_block_param(hdr, Type::i32());
            b.position_at_end(entry);
            b.build_br(hdr, {b.build_iconst_i32(0)});
            b.position_at_end(hdr);
            b.build_br_if(b.build_slt(i, n), body, {}, exit, {});
            b.position_at_end(body);
            // Non-linear step i*2 + 1 prevents simple inductive affine proof -> bounded unroll truncates
            Value* next_i = b.build_add(b.build_mul(i, b.build_iconst_i32(2)), b.build_iconst_i32(1));
            b.build_br(hdr, {next_i});
            b.position_at_end(exit);
            Value* past_bound = b.build_sge(i, b.build_iconst_i32(100));
            Value* ret_val = b.build_select(
                past_bound,
                b.build_add(i, b.build_iconst_i32(static_cast<int32_t>(final_add))),
                i);
            b.build_ret(ret_val);
            fn->rebuild_cfg_predecessors();
            return fn;
        };
        Function* f1 = make_loop("f1", 0);
        Function* f2 = make_loop("f2", 1);
        EquivOptions opts;
        opts.loop_unroll_bound = 2;
        CHECK_EQ(verify_mir_equivalence(*f1, *f2, opts).status, EquivStatus::Unknown);
    }

    // 2. Signed division INT_MIN / -1 trap must prevent unsafe simplification
    {
        Module mod("sdiv_int_min");
        Builder b(mod);
        Function* before = mod.create_function("before", Type::i32(), {Type::i32()});
        b.set_function(before);
        BasicBlock* eb = b.append_block("entry");
        Value* xb = b.add_block_param(eb, Type::i32());
        b.position_at_end(eb);
        b.build_ret(b.build_sdiv(xb, b.build_iconst_i32(-1)));
        before->rebuild_cfg_predecessors();

        Function* after = mod.create_function("after", Type::i32(), {Type::i32()});
        b.set_function(after);
        BasicBlock* ea = b.append_block("entry");
        Value* xa = b.add_block_param(ea, Type::i32());
        b.position_at_end(ea);
        b.build_ret(b.build_neg(xa));
        after->rebuild_cfg_predecessors();

        CHECK_EQ(verify_mir_equivalence(*before, *after).status, EquivStatus::Counterexample);
    }

    // 3. Unknown / potentially stateful call yields Unsupported when assume_calls_pure == false
    {
        Module mod("call_unsup");
        Builder b(mod);
        auto make_f = [&](const char* name) -> Function* {
            Function* fn = mod.create_function(name, Type::i32(), {Type::i32()});
            b.set_function(fn);
            BasicBlock* e = b.append_block("entry");
            Value* x = b.add_block_param(e, Type::i32());
            b.position_at_end(e);
            b.build_ret(b.build_call("ext_side_effect", Type::i32(), {x}));
            fn->rebuild_cfg_predecessors();
            return fn;
        };
        Function* f1 = make_f("f1");
        Function* f2 = make_f("f2");
        CHECK_EQ(verify_mir_equivalence(*f1, *f2).status, EquivStatus::Unsupported);
        EquivOptions pure_opts;
        pure_opts.assume_calls_pure = true;
        CHECK_EQ(verify_mir_equivalence(*f1, *f2, pure_opts).status, EquivStatus::ProvenEquivalent);

        // Pure Brass math intrinsics (e.g. ptx_fmax, ptx_fabs) prove equivalent even when assume_calls_pure == false
        auto make_math = [&](const char* name) -> Function* {
            Function* fn = mod.create_function(name, Type::f32(), {Type::f32(), Type::f32()});
            b.set_function(fn);
            BasicBlock* e = b.append_block("entry");
            Value* x = b.add_block_param(e, Type::f32());
            Value* y = b.add_block_param(e, Type::f32());
            b.position_at_end(e);
            Value* ax = b.build_call("ptx_fabs", Type::f32(), {x});
            Value* m = b.build_call("ptx_fmax", Type::f32(), {ax, y});
            b.build_ret(m);
            fn->rebuild_cfg_predecessors();
            return fn;
        };
        Function* m1 = make_math("math1");
        Function* m2 = make_math("math2");
        CHECK_EQ(verify_mir_equivalence(*m1, *m2).status, EquivStatus::ProvenEquivalent);

        auto make_unk = [&](const char* name) -> Function* {
            Function* fn = mod.create_function(name, Type::f32(), {Type::f32()});
            b.set_function(fn);
            BasicBlock* e = b.append_block("entry");
            Value* x = b.add_block_param(e, Type::f32());
            b.position_at_end(e);
            b.build_ret(b.build_call("unknown_stateful_fn", Type::f32(), {x}));
            fn->rebuild_cfg_predecessors();
            return fn;
        };
        Function* u1 = make_unk("unk1");
        Function* u2 = make_unk("unk2");
        CHECK_EQ(verify_mir_equivalence(*u1, *u2).status, EquivStatus::Unsupported);
    }

    // 4. LIR equivalence: catches 128-bit XMM.hi mismatch, PF flag mismatch, and propagates Unsupported
    {
        using namespace brass::codegen;
        LirBlock b1(0, "xmm_before");
        auto m1 = std::make_unique<LirInst>(LirOpcode::Movaps);
        m1->add_def(LirOperand::preg_xmm(x64::XMM::XMM0));
        m1->add_use(LirOperand::preg_xmm(x64::XMM::XMM1));
        b1.append_inst(std::move(m1));

        LirBlock b2(0, "xmm_after_scalar_only");
        auto m2 = std::make_unique<LirInst>(LirOpcode::Movsd);
        m2->add_def(LirOperand::preg_xmm(x64::XMM::XMM0));
        m2->add_use(LirOperand::preg_xmm(x64::XMM::XMM1));
        b2.append_inst(std::move(m2));

        CHECK_EQ(verify_x64_lir_block_equivalence(b1, b2, 0u, 1u, false).status, EquivStatus::Counterexample);

        LirBlock b_ymm(0, "ymm");
        auto vm = std::make_unique<LirInst>(LirOpcode::Vmovaps);
        vm->add_def(LirOperand::preg_xmm(x64::XMM::XMM0));
        vm->add_use(LirOperand::preg_xmm(x64::XMM::XMM1));
        b_ymm.append_inst(std::move(vm));
        CHECK_EQ(verify_x64_lir_block_equivalence(b_ymm, b_ymm, 0u, 1u, false).status, EquivStatus::Unsupported);

        LirFunction fn_u;
        fn_u.name = "fn_unsup";
        LirBlock* ub = fn_u.create_block("entry");
        ub->id = 0;
        auto vm2 = std::make_unique<LirInst>(LirOpcode::Vmovaps);
        vm2->add_def(LirOperand::preg_xmm(x64::XMM::XMM0));
        vm2->add_use(LirOperand::preg_xmm(x64::XMM::XMM1));
        ub->append_inst(std::move(vm2));
        CHECK_EQ(verify_x64_lir_function_equivalence(fn_u, fn_u).status, EquivStatus::Unsupported);

        LirBlock pf_before(0, "pf_b");
        auto add_a = std::make_unique<LirInst>(LirOpcode::Add32);
        add_a->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add_a->add_use(LirOperand::preg_gpr(x64::GPR::RDI, 4));
        add_a->add_use(LirOperand::imm(3, 4));
        pf_before.append_inst(std::move(add_a));

        LirBlock pf_after(0, "pf_a");
        auto add_b = std::make_unique<LirInst>(LirOpcode::Add32);
        add_b->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add_b->add_use(LirOperand::preg_gpr(x64::GPR::RDI, 4));
        add_b->add_use(LirOperand::imm(3, 4));
        pf_after.append_inst(std::move(add_b));
        // Setcc P depends on PF: check RAX is preserved if not check_flags, whereas check_flags catches wrong PF vs cleared PF
        LirBlock pf_bad(0, "pf_bad");
        auto mov_a = std::make_unique<LirInst>(LirOpcode::Mov32);
        mov_a->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        mov_a->add_use(LirOperand::preg_gpr(x64::GPR::RDI, 4));
        pf_bad.append_inst(std::move(mov_a));
        auto add0 = std::make_unique<LirInst>(LirOpcode::Add32);
        add0->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add0->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        add0->add_use(LirOperand::imm(3, 4));
        pf_bad.append_inst(std::move(add0));
        auto jp = std::make_unique<LirInst>(LirOpcode::Setcc);
        jp->condition = x64::Condition::P;
        jp->add_def(LirOperand::preg_gpr(x64::GPR::RCX, 1));
        pf_bad.append_inst(std::move(jp));

        LirBlock pf_bad_wrong(0, "pf_bad_wrong");
        auto mov_c = std::make_unique<LirInst>(LirOpcode::Mov32);
        mov_c->add_def(LirOperand::preg_gpr(x64::GPR::RCX, 4));
        mov_c->add_use(LirOperand::imm(1, 4));
        pf_bad_wrong.append_inst(std::move(mov_c));
        CHECK_EQ(verify_x64_lir_block_equivalence(pf_bad, pf_bad_wrong, x64::reg_mask(x64::GPR::RCX), 0u, false).status,
                 EquivStatus::Counterexample);
        CHECK_EQ(verify_x64_lir_block_equivalence(pf_before, pf_after, x64::reg_mask(x64::GPR::RAX), 0u, true).status,
                 EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("SMT Equiv - Multi-Loop Cutset Relational Induction (Sequential & Nested Loops)") {
    // 1. Sequential 2-loop function: algebraic rewrite & umod-8->and-7 prove equivalent;
    //    mutating second loop body yields Counterexample.
    {
        Module mod("seq_2loop");
        Builder b(mod);
        auto build_seq = [&](const char* name, bool opt_style, bool buggy) -> Function* {
            Function* fn = mod.create_function(name, Type::i32(), {Type::i32()});
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* h1 = b.append_block("h1");
            BasicBlock* b1 = b.append_block("b1");
            BasicBlock* h2 = b.append_block("h2");
            BasicBlock* b2 = b.append_block("b2");
            BasicBlock* done = b.append_block("done");
            Value* n = b.add_block_param(entry, Type::i32());
            Value* i1 = b.add_block_param(h1, Type::i32());
            Value* a1 = b.add_block_param(h1, Type::i32());
            Value* i2 = b.add_block_param(h2, Type::i32());
            Value* a2 = b.add_block_param(h2, Type::i32());

            b.position_at_end(entry);
            b.build_br(h1, {b.build_iconst_i32(0), b.build_iconst_i32(0)});

            b.position_at_end(h1);
            b.build_br_if(b.build_slt(i1, n), b1, {}, h2, {b.build_iconst_i32(0), a1});

            b.position_at_end(b1);
            Value* step1 = opt_style ? i1 : b.build_add(i1, b.build_iconst_i32(0));
            b.build_br(h1, {b.build_add(i1, b.build_iconst_i32(1)), b.build_add(a1, step1)});

            b.position_at_end(h2);
            b.build_br_if(b.build_slt(i2, n), b2, {}, done, {});

            b.position_at_end(b2);
            Value* rem = opt_style
                ? b.build_and(i2, b.build_iconst_i32(7))
                : b.build_umod(i2, b.build_iconst_i32(8));
            if (buggy) rem = b.build_add(rem, b.build_iconst_i32(1));
            b.build_br(h2, {b.build_add(i2, b.build_iconst_i32(1)), b.build_xor(a2, rem)});

            b.position_at_end(done);
            b.build_ret(a2);
            fn->rebuild_cfg_predecessors();
            return fn;
        };

        Function* f_unopt = build_seq("seq_unopt", false, false);
        Function* f_opt   = build_seq("seq_opt",   true,  false);
        Function* f_buggy = build_seq("seq_buggy", true,  true);

        EquivOptions opts;
        opts.loop_unroll_bound = 1;
        CHECK_EQ(verify_mir_equivalence(*f_unopt, *f_opt, opts).status, EquivStatus::ProvenEquivalent);
        CHECK_EQ(verify_mir_equivalence(*f_unopt, *f_buggy, opts).status, EquivStatus::Counterexample);
    }

    // 2. Nested 2-loop function: outer + inner loop equivalence and bug detection.
    {
        Module mod("nested_2loop");
        Builder b(mod);
        auto build_nested = [&](const char* name, bool opt_style, bool buggy) -> Function* {
            Function* fn = mod.create_function(name, Type::i32(), {Type::i32(), Type::i32()});
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* outer_h = b.append_block("outer_h");
            BasicBlock* inner_h = b.append_block("inner_h");
            BasicBlock* inner_b = b.append_block("inner_b");
            BasicBlock* outer_latch = b.append_block("outer_latch");
            BasicBlock* done = b.append_block("done");

            Value* n = b.add_block_param(entry, Type::i32());
            Value* m = b.add_block_param(entry, Type::i32());
            Value* oi = b.add_block_param(outer_h, Type::i32());
            Value* oa = b.add_block_param(outer_h, Type::i32());
            Value* ij = b.add_block_param(inner_h, Type::i32());
            Value* ia = b.add_block_param(inner_h, Type::i32());

            b.position_at_end(entry);
            b.build_br(outer_h, {b.build_iconst_i32(0), b.build_iconst_i32(0)});

            b.position_at_end(outer_h);
            b.build_br_if(b.build_slt(oi, n), inner_h, {b.build_iconst_i32(0), oa}, done, {});

            b.position_at_end(inner_h);
            b.build_br_if(b.build_slt(ij, m), inner_b, {}, outer_latch, {});

            b.position_at_end(inner_b);
            Value* contrib = opt_style ? b.build_shl(ij, b.build_iconst_i32(1))
                                       : b.build_mul(ij, b.build_iconst_i32(2));
            if (buggy) contrib = b.build_add(contrib, b.build_iconst_i32(3));
            b.build_br(inner_h, {b.build_add(ij, b.build_iconst_i32(1)), b.build_add(ia, contrib)});

            b.position_at_end(outer_latch);
            b.build_br(outer_h, {b.build_add(oi, b.build_iconst_i32(1)), ia});

            b.position_at_end(done);
            b.build_ret(oa);
            fn->rebuild_cfg_predecessors();
            return fn;
        };

        Function* f_unopt = build_nested("nested_unopt", false, false);
        Function* f_opt   = build_nested("nested_opt",   true,  false);
        Function* f_buggy = build_nested("nested_buggy", true,  true);

        EquivOptions opts;
        opts.loop_unroll_bound = 1;
        CHECK_EQ(verify_mir_equivalence(*f_unopt, *f_opt, opts).status, EquivStatus::ProvenEquivalent);
        CHECK_EQ(verify_mir_equivalence(*f_unopt, *f_buggy, opts).status, EquivStatus::Counterexample);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
