#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/proof_cert.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/verifier.hpp>

using namespace brass;

namespace brass {
inline std::ostream& operator<<(std::ostream& os, BarrierCertKind kind) {
    return os << barrier_cert_kind_name(kind);
}
} // namespace brass

TEST_CASE("Verifier Certs - WriteBarrierElimination Emits Valid Proof Certificates (#68)") {
    Module mod("wbe_certs");
    Builder b(mod);

    Function* alloc_fn = mod.create_function("brass_gc_alloc", Type::gcref(), {Type::i64()});
    (void)alloc_fn;

    Function* fn = mod.create_function("test_wbe", Type::void_type(), {Type::gcref(), Type::gcref()});
    fn->set_requires_barrier_certs(true);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* old_obj = b.add_block_param(entry, Type::gcref());
    Value* ref_val = b.add_block_param(entry, Type::gcref());
    b.position_at_end(entry);

    // 1. Young nursery allocation store (YoungGenBumpAlloc)
    Value* young_obj = b.build_call("brass_gc_alloc", Type::gcref(), {b.build_iconst_i64(64)});
    Instruction* st_young = b.build_store(Type::gcref(), young_obj, 8, ref_val);
    b.build_write_barrier(young_obj, ref_val);

    // 2. Non-pointer scalar store (NonRefStore)
    Value* int_val = b.build_iconst_i64(12345);
    Instruction* st_nonref = b.build_store(Type::i64(), old_obj, 16, int_val);
    b.build_write_barrier(old_obj, int_val);

    // 3. Constant 0 store (ConstantOrNullRef)
    Value* zero_val = b.build_iconst_i64(0);
    Instruction* st_null = b.build_store(Type::i64(), old_obj, 24, zero_val);
    b.build_write_barrier(old_obj, zero_val);

    // 4. First barriered store to old_obj + redundant second store in same epoch (RedundantSameEpoch)
    b.build_store(Type::gcref(), old_obj, 32, ref_val);
    b.build_write_barrier(old_obj, ref_val);
    Instruction* st_red = b.build_store(Type::gcref(), old_obj, 40, ref_val);
    b.build_write_barrier(old_obj, ref_val);

    b.build_ret(nullptr);
    fn->rebuild_cfg_predecessors();

    WriteBarrierElimination wbe;
    CHECK(wbe.run_on_function(*fn));
    CHECK_EQ(wbe.stats().total_eliminated(), 4u);

    CHECK_EQ(st_young->barrier_cert(), BarrierCertKind::YoungGenBumpAlloc);
    CHECK_EQ(st_nonref->barrier_cert(), BarrierCertKind::NonRefStore);
    CHECK_EQ(st_null->barrier_cert(), BarrierCertKind::ConstantOrNullRef);
    CHECK_EQ(st_red->barrier_cert(), BarrierCertKind::RedundantSameEpoch);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
    CHECK(!diag.has_errors());
}

TEST_CASE("Verifier Certs - Rejects Unsound Write-Barrier Elision Certificates (#68)") {
    Module mod("wbe_bad");
    Builder b(mod);
    mod.create_function("brass_gc_alloc", Type::gcref(), {Type::i64()});

    // Case 1: YoungGenBumpAlloc invalidated by an intervening safepoint
    {
        Function* fn = mod.create_function("bad_young_across_safepoint", Type::void_type(), {Type::gcref()});
        fn->set_requires_barrier_certs(true);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* ref_val = b.add_block_param(entry, Type::gcref());
        b.position_at_end(entry);

        Value* young_obj = b.build_call("brass_gc_alloc", Type::gcref(), {b.build_iconst_i64(64)});
        b.build_safepoint(); // May promote young_obj to tenured space!
        Instruction* st = b.build_store(Type::gcref(), young_obj, 8, ref_val);
        st->set_barrier_cert(BarrierCertKind::YoungGenBumpAlloc);
        b.build_ret(nullptr);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }

    // Case 2: NonRefStore claimed on a gcref store
    {
        Function* fn = mod.create_function("bad_nonref_on_gcref", Type::void_type(), {Type::gcref(), Type::gcref()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* obj = b.add_block_param(entry, Type::gcref());
        Value* ref_val = b.add_block_param(entry, Type::gcref());
        b.position_at_end(entry);

        Instruction* st = b.build_store(Type::gcref(), obj, 8, ref_val);
        st->set_barrier_cert(BarrierCertKind::NonRefStore);
        b.build_ret(nullptr);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }

    // Case 3: Uncertified gcref store when requires_barrier_certs() is true
    {
        Function* fn = mod.create_function("missing_barrier_cert", Type::void_type(), {Type::gcref(), Type::gcref()});
        fn->set_requires_barrier_certs(true);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* obj = b.add_block_param(entry, Type::gcref());
        Value* ref_val = b.add_block_param(entry, Type::gcref());
        b.position_at_end(entry);

        b.build_store(Type::gcref(), obj, 8, ref_val);
        b.build_ret(nullptr);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }

    // Case 4: RedundantSameEpoch claimed after a safepoint cleared the epoch
    {
        Function* fn = mod.create_function("bad_redundant_epoch", Type::void_type(), {Type::gcref(), Type::gcref()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* obj = b.add_block_param(entry, Type::gcref());
        Value* ref_val = b.add_block_param(entry, Type::gcref());
        b.position_at_end(entry);

        b.build_store(Type::gcref(), obj, 8, ref_val);
        b.build_write_barrier(obj, ref_val);
        b.build_safepoint();
        Instruction* st2 = b.build_store(Type::gcref(), obj, 16, ref_val);
        st2->set_barrier_cert(BarrierCertKind::RedundantSameEpoch);
        b.build_ret(nullptr);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }
}

TEST_CASE("Verifier Certs - BoundsCheckElimination Emits Valid GuardElimCerts (#68)") {
    Module mod("bce_certs");
    Builder b(mod);
    Function* fn = mod.create_function("check_guards", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    // 1. Dominated guard: x < c10 implies x <= c10
    Value* c10 = b.build_iconst_i32(10);
    Value* lt10 = b.build_slt(x, c10);
    b.build_guard(lt10, "deopt");
    Value* le10 = b.build_sle(x, c10);
    b.build_guard(le10, "deopt");

    // 2. NonNegativeMask guard: (x & 7) < 8
    Value* masked = b.build_and(x, b.build_iconst_i32(7));
    Value* mask_lt8 = b.build_slt(masked, b.build_iconst_i32(8));
    b.build_guard(mask_lt8, "deopt");

    b.build_ret(masked);
    fn->rebuild_cfg_predecessors();

    CHECK(run_bounds_check_elimination(*fn, mod));
    CHECK(fn->guard_elim_certs().size() >= 2u);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
    CHECK(!diag.has_errors());
}

TEST_CASE("Verifier Certs - Rejects Unsound GuardElimCerts (#68)") {
    // 1. Witness value in a non-dominating sibling block
    {
        Module mod("bad_dom_cert");
        Builder b(mod);
        Function* fn = mod.create_function("non_dom_witness", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* left = b.append_block("left");
        BasicBlock* right = b.append_block("right");
        Value* x = b.add_block_param(entry, Type::i32());

        b.position_at_end(entry);
        Value* c = b.build_slt(x, b.build_iconst_i32(0));
        b.build_br_if(c, left, {}, right, {});

        b.position_at_end(left);
        Value* left_cmp = b.build_slt(x, b.build_iconst_i32(10));
        b.build_guard(left_cmp, "deopt");
        b.build_ret(x);

        b.position_at_end(right);
        b.build_ret(x);
        fn->rebuild_cfg_predecessors();

        // Bogus certificate claiming a guard in 'right' was eliminated using 'left_cmp' (which does not dominate 'right'!)
        GuardElimCert bad_cert;
        bad_cert.eliminated_resume_id = 99;
        bad_cert.block_id = right->id();
        bad_cert.reason = GuardElimCert::Reason::DominatedSameCond;
        bad_cert.witness_inst_or_val_id = left_cmp->id();
        fn->add_guard_elim_cert(bad_cert);

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }

    // 2. Bogus NonNegativeMask certificate whose range does not cover the mask
    {
        Module mod("bad_mask_cert");
        Builder b(mod);
        Function* fn = mod.create_function("bad_mask", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);
        Value* masked = b.build_and(x, b.build_iconst_i32(255));
        b.build_ret(masked);
        fn->rebuild_cfg_predecessors();

        GuardElimCert bad_cert;
        bad_cert.eliminated_resume_id = 42;
        bad_cert.block_id = entry->id();
        bad_cert.reason = GuardElimCert::Reason::NonNegativeMask;
        bad_cert.witness_inst_or_val_id = masked->id();
        bad_cert.bound_min = 0;
        bad_cert.bound_max = 63; // Lies: mask is 255, not <= 63!
        fn->add_guard_elim_cert(bad_cert);

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }

    // 3. Bogus ConstantFolded certificate with constant 0
    {
        Module mod("bad_const_cert");
        Builder b(mod);
        Function* fn = mod.create_function("bad_const", Type::i32(), {});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* zero = b.build_iconst_i32(0);
        b.build_ret(zero);
        fn->rebuild_cfg_predecessors();

        GuardElimCert bad_cert;
        bad_cert.eliminated_resume_id = 7;
        bad_cert.block_id = entry->id();
        bad_cert.reason = GuardElimCert::Reason::ConstantFolded;
        bad_cert.witness_inst_or_val_id = zero->id();
        fn->add_guard_elim_cert(bad_cert);

        DiagnosticReporter diag;
        CHECK(!verify_function(*fn, &diag));
        CHECK(diag.has_errors());
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
