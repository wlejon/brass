#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/verify/smt_equiv.hpp>

using namespace brass;

namespace {

size_t count_guards(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard) count++;
        }
    }
    return count;
}

} // namespace

TEST_CASE("SMT Guard Elim - Discharges 2D Strided Indexing Bounds (i + stride * j < len) (#70)") {
    Module mod("smt_2d_stride");
    Builder b(mod);
    // fn(i: i32, j: i32, stride: i32, height: i32, len: i32) -> i32
    Function* fn = mod.create_function(
        "index_2d", Type::i32(),
        {Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* i = b.add_block_param(entry, Type::i32());
    Value* j = b.add_block_param(entry, Type::i32());
    Value* stride = b.add_block_param(entry, Type::i32());
    Value* height = b.add_block_param(entry, Type::i32());
    Value* len = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    Value* zero = b.build_iconst_i32(0);
    // Preconditions: 0 <= i < stride, 0 <= j < height, stride * height <= len (no overflow)
    b.build_guard(b.build_sge(i, zero), "deopt");
    b.build_guard(b.build_slt(i, stride), "deopt");
    b.build_guard(b.build_sge(j, zero), "deopt");
    b.build_guard(b.build_slt(j, height), "deopt");
    Value* ovf = b.build_smul_overflow(stride, height);
    b.build_guard(b.build_eq(ovf, zero), "deopt");
    Value* area = b.build_mul(stride, height);
    b.build_guard(b.build_sle(area, len), "deopt");

    // Inner 2D linear index: idx = i + stride * j
    Value* row_off = b.build_mul(stride, j);
    Value* idx = b.build_add(i, row_off);

    // Target bounds checks to discharge: idx >= 0 and idx < len
    b.build_guard(b.build_sge(idx, zero), "deopt");
    b.build_guard(b.build_slt(idx, len), "deopt");
    b.build_ret(idx);
    fn->rebuild_cfg_predecessors();

    CHECK_EQ(count_guards(*fn), 8u);

    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    const uint64_t unsat_checks_before = verify::unsat_sample_checks_performed();
    CHECK(run_smt_guard_elimination(*fn, mod, opts));
    CHECK(verify::unsat_sample_checks_performed() > unsat_checks_before);
    CHECK_EQ(stats.guards_proven_eliminated, 2u);
    CHECK_EQ(count_guards(*fn), 6u);
    CHECK_EQ(fn->guard_elim_certs().size(), 2u);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
    CHECK(!diag.has_errors());
}

TEST_CASE("SMT Guard Elim - Discharges Tiled Loop Bounds (base + k < len) (#70)") {
    Module mod("smt_tiled");
    Builder b(mod);
    // fn(base: i32, k: i32, len: i32) -> i32
    Function* fn = mod.create_function("tiled_idx", Type::i32(), {Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* base = b.add_block_param(entry, Type::i32());
    Value* k = b.add_block_param(entry, Type::i32());
    Value* len = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    Value* zero = b.build_iconst_i32(0);
    Value* tile = b.build_iconst_i32(16);

    // Outer tile condition: 0 <= base, len >= 16, base <= len - 16
    b.build_guard(b.build_sge(base, zero), "deopt");
    b.build_guard(b.build_sge(len, tile), "deopt");
    Value* limit = b.build_sub(len, tile);
    b.build_guard(b.build_sle(base, limit), "deopt");

    // Inner tile condition: 0 <= k < 16
    b.build_guard(b.build_sge(k, zero), "deopt");
    b.build_guard(b.build_slt(k, tile), "deopt");

    // Tiled index: idx = base + k
    Value* idx = b.build_add(base, k);
    b.build_guard(b.build_sge(idx, zero), "deopt");
    b.build_guard(b.build_slt(idx, len), "deopt");
    b.build_ret(idx);
    fn->rebuild_cfg_predecessors();

    auto before_mod = clone_module(mod);
    Function* fn_before = before_mod->get_function("tiled_idx");

    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    CHECK(run_smt_guard_elimination(*fn, mod, opts));
    CHECK_EQ(stats.guards_proven_eliminated, 2u);
    CHECK_EQ(count_guards(*fn), 5u);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
    CHECK_EQ(verify::verify_mir_equivalence(*fn_before, *fn).status,
             verify::EquivStatus::ProvenEquivalent);
}

TEST_CASE("SMT Guard Elim - Discharges Clamped and Power-of-Two Bit-Masked Bounds (#70)") {
    // 1. Clamped index: clamped = min(max(idx, 0), len - 1) under len > 0
    {
        Module mod("smt_clamped");
        Builder b(mod);
        Function* fn = mod.create_function("clamp_idx", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* idx = b.add_block_param(entry, Type::i32());
        Value* len = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* zero = b.build_iconst_i32(0);
        Value* one = b.build_iconst_i32(1);
        b.build_guard(b.build_sgt(len, zero), "deopt");

        Value* hi = b.build_select(b.build_sgt(idx, zero), idx, zero);
        Value* max_idx = b.build_sub(len, one);
        Value* clamped = b.build_select(b.build_slt(hi, max_idx), hi, max_idx);

        b.build_guard(b.build_sge(clamped, zero), "deopt");
        b.build_guard(b.build_slt(clamped, len), "deopt");
        b.build_ret(clamped);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SmtGuardElimStats stats;
        SmtGuardElimOptions opts;
        opts.stats = &stats;
        CHECK(run_smt_guard_elimination(*fn, mod, opts));
        CHECK_EQ(stats.guards_proven_eliminated, 2u);
        CHECK_EQ(count_guards(*fn), 1u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("clamp_idx"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. Power-of-two bit-masked index: (x & (len - 1)) < len when len > 0 and (len & (len - 1)) == 0
    {
        Module mod("smt_pow2_mask");
        Builder b(mod);
        Function* fn = mod.create_function("pow2_mask_idx", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        Value* len = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* zero = b.build_iconst_i32(0);
        Value* one = b.build_iconst_i32(1);
        b.build_guard(b.build_sgt(len, zero), "deopt");
        Value* mask = b.build_sub(len, one);
        Value* pow2_check = b.build_eq(b.build_and(len, mask), zero);
        b.build_guard(pow2_check, "deopt");

        Value* idx = b.build_and(x, mask);
        b.build_guard(b.build_sge(idx, zero), "deopt");
        b.build_guard(b.build_slt(idx, len), "deopt");
        b.build_ret(idx);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SmtGuardElimStats stats;
        SmtGuardElimOptions opts;
        opts.stats = &stats;
        CHECK(run_smt_guard_elimination(*fn, mod, opts));
        CHECK_EQ(stats.guards_proven_eliminated, 2u);
        CHECK_EQ(count_guards(*fn), 2u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("pow2_mask_idx"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("SMT Guard Elim - Soundness: Preserves Guards That Can Fail (#70)") {
    Module mod("smt_keep_unproven");
    Builder b(mod);
    Function* fn = mod.create_function("unproven", Type::i32(), {Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* base = b.add_block_param(entry, Type::i32());
    Value* k = b.add_block_param(entry, Type::i32());
    Value* len = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    Value* zero = b.build_iconst_i32(0);
    Value* tile = b.build_iconst_i32(16);
    b.build_guard(b.build_sge(base, zero), "deopt");
    b.build_guard(b.build_sge(k, zero), "deopt");
    // Buggy bound: k <= 16 (instead of k < 16) when base <= len - 16 allows base + k == len!
    b.build_guard(b.build_sle(k, tile), "deopt");
    b.build_guard(b.build_sge(len, tile), "deopt");
    b.build_guard(b.build_sle(base, b.build_sub(len, tile)), "deopt");

    Value* idx = b.build_add(base, k);
    b.build_guard(b.build_slt(idx, len), "deopt"); // Must NOT be eliminated!
    b.build_ret(idx);
    fn->rebuild_cfg_predecessors();

    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    run_smt_guard_elimination(*fn, mod, opts);
    CHECK_EQ(stats.guards_proven_eliminated, 0u);
    CHECK_EQ(count_guards(*fn), 6u);
}

TEST_CASE("SMT Guard Elim - Inductive Loop IV Bounds, Implied Comparison & Branch Folding (#70)") {
    Module mod("smt_loop_and_branch");
    Builder b(mod);
    // fn(x: i32, len: i32) -> i32
    Function* fn = mod.create_function("branch_and_cmp", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* taken = b.append_block("taken");
    BasicBlock* dead_blk = b.append_block("dead_blk");
    Value* x = b.add_block_param(entry, Type::i32());
    Value* len = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i32(0);
    Value* ten = b.build_iconst_i32(10);
    b.build_guard(b.build_sge(x, ten), "deopt");
    b.build_guard(b.build_slt(x, len), "deopt");
    // Branch condition x > 0 is implied by x >= 10
    Value* br_cond = b.build_sgt(x, zero);
    b.build_br_if(br_cond, taken, {}, dead_blk, {});

    b.position_at_end(taken);
    // Implied comparison len > 10 used in select
    Value* cmp_len = b.build_sgt(len, ten);
    Value* res = b.build_select(cmp_len, x, zero);
    b.build_ret(res);

    b.position_at_end(dead_blk);
    b.build_ret(zero);
    fn->rebuild_cfg_predecessors();

    auto before_mod = clone_module(mod);
    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    CHECK(run_smt_guard_elimination(*fn, mod, opts));
    CHECK_EQ(stats.branches_folded, 1u);
    CHECK_EQ(stats.comparisons_proven_true, 1u);
    CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("branch_and_cmp"), *fn).status,
             verify::EquivStatus::ProvenEquivalent);
}

TEST_CASE("SMT Guard Elim - 2D Strided Overflow Not Bypassed by Unchecked Guard") {
    Module mod("smt_2d_ovf_neg");
    Builder b(mod);
    Function* fn = mod.create_function("strided_2d_ovf", Type::i32(),
                                       {Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* i = b.add_block_param(entry, Type::i32());
    Value* j = b.add_block_param(entry, Type::i32());
    Value* stride = b.add_block_param(entry, Type::i32());
    Value* h = b.add_block_param(entry, Type::i32());
    Value* len = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i32(0);
    b.build_guard(b.build_sge(i, zero), "deopt");
    b.build_guard(b.build_slt(i, stride), "deopt");
    b.build_guard(b.build_sge(j, zero), "deopt");
    b.build_guard(b.build_slt(j, h), "deopt");
    Value* sh = b.build_mul(stride, h);
    b.build_guard(b.build_sle(sh, len), "deopt");
    // Missing smul_overflow(stride, h) == 0: stride * h can wrap negative so idx < len is NOT guaranteed!
    Value* idx = b.build_add(i, b.build_mul(stride, j));
    b.build_guard(b.build_slt(idx, len), "deopt");
    b.build_ret(idx);
    fn->rebuild_cfg_predecessors();

    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    run_smt_guard_elimination(*fn, mod, opts);
    CHECK_EQ(stats.guards_proven_eliminated, 0u);
}

TEST_CASE("SMT Guard Elim - Real Loop IV Sound Positive and Overflow/Unsigned Negative") {
    // 1. Positive: real signed slt loop with step 1 eliminates iv >= 0
    {
        Module mod("smt_real_loop_pos");
        Builder b(mod);
        Function* fn = mod.create_function("loop_pos", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hdr = b.append_block("hdr");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");
        Value* n = b.add_block_param(entry, Type::i32());
        Value* iv = b.add_block_param(hdr, Type::i32());

        b.position_at_end(entry);
        Value* zero = b.build_iconst_i32(0);
        b.build_br(hdr, {zero});

        b.position_at_end(hdr);
        b.build_br_if(b.build_slt(iv, n), body, {}, exit, {});

        b.position_at_end(body);
        b.build_guard(b.build_sge(iv, zero), "deopt");
        Value* next = b.build_add(iv, b.build_iconst_i32(1));
        b.build_br(hdr, {next});

        b.position_at_end(exit);
        b.build_ret(zero);
        fn->rebuild_cfg_predecessors();

        SmtGuardElimStats stats;
        SmtGuardElimOptions opts;
        opts.stats = &stats;
        CHECK(run_smt_guard_elimination(*fn, mod, opts));
        CHECK_EQ(stats.guards_proven_eliminated, 1u);
    }
    // 2. Negative: step 4 can signed-overflow past INT_MAX into negative values!
    {
        Module mod("smt_real_loop_step4_neg");
        Builder b(mod);
        Function* fn = mod.create_function("loop_step4", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hdr = b.append_block("hdr");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");
        Value* n = b.add_block_param(entry, Type::i32());
        Value* iv = b.add_block_param(hdr, Type::i32());

        b.position_at_end(entry);
        Value* zero = b.build_iconst_i32(0);
        b.build_br(hdr, {zero});

        b.position_at_end(hdr);
        b.build_br_if(b.build_slt(iv, n), body, {}, exit, {});

        b.position_at_end(body);
        b.build_guard(b.build_sge(iv, zero), "deopt");
        Value* next = b.build_add(iv, b.build_iconst_i32(4));
        b.build_br(hdr, {next});

        b.position_at_end(exit);
        b.build_ret(zero);
        fn->rebuild_cfg_predecessors();

        SmtGuardElimStats stats;
        SmtGuardElimOptions opts;
        opts.stats = &stats;
        run_smt_guard_elimination(*fn, mod, opts);
        CHECK_EQ(stats.guards_proven_eliminated, 0u);
    }
    // 3. Negative: unsigned ult loop with step 1 does NOT imply iv >=s 0 (iv can pass INT32_MAX)!
    {
        Module mod("smt_real_loop_ult_neg");
        Builder b(mod);
        Function* fn = mod.create_function("loop_ult", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hdr = b.append_block("hdr");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");
        Value* n = b.add_block_param(entry, Type::i32());
        Value* iv = b.add_block_param(hdr, Type::i32());

        b.position_at_end(entry);
        Value* zero = b.build_iconst_i32(0);
        b.build_br(hdr, {zero});

        b.position_at_end(hdr);
        b.build_br_if(b.build_ult(iv, n), body, {}, exit, {});

        b.position_at_end(body);
        b.build_guard(b.build_sge(iv, zero), "deopt");
        Value* next = b.build_add(iv, b.build_iconst_i32(1));
        b.build_br(hdr, {next});

        b.position_at_end(exit);
        b.build_ret(zero);
        fn->rebuild_cfg_predecessors();

        SmtGuardElimStats stats;
        SmtGuardElimOptions opts;
        opts.stats = &stats;
        run_smt_guard_elimination(*fn, mod, opts);
        CHECK_EQ(stats.guards_proven_eliminated, 0u);
    }
}

TEST_CASE("SMT Guard Elim - patchable_const_i32 Treated Symbolic Not Constant") {
    Module mod("smt_patchable_neg");
    Builder b(mod);
    Function* fn = mod.create_function("patchable_guard", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* pconst = b.build_patchable_const_i32("k_limit", 10);
    b.build_guard(b.build_slt(x, pconst), "deopt");
    // Must NOT treat pconst as fixed 10: runtime patching can raise k_limit!
    b.build_guard(b.build_slt(x, b.build_iconst_i32(20)), "deopt");
    b.build_ret(x);
    fn->rebuild_cfg_predecessors();

    SmtGuardElimStats stats;
    SmtGuardElimOptions opts;
    opts.stats = &stats;
    run_smt_guard_elimination(*fn, mod, opts);
    CHECK_EQ(stats.guards_proven_eliminated, 0u);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
