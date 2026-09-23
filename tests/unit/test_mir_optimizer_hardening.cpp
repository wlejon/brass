#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/loop_nest.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include "../../src/mir/slp_analysis.hpp"
#include "../../src/mir/loop_dependence.hpp"
#include <vector>
#include <limits>
#include <cmath>

using namespace brass;

// 1. Alias Analysis Soundness: Entry Block Parameters Default To MayAlias
TEST_CASE("MIR Hardening - Entry Block Parameters Default To MayAlias") {
    Module mod("test_param_may_alias");
    Builder b(mod);
    Function* fn = mod.create_function("test_params", Type::void_type(), {Type::ptr(), Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p1 = b.add_block_param(entry, Type::ptr());
    Value* p2 = b.add_block_param(entry, Type::ptr());

    // Allocate an alloca to compare against
    Value* loc = b.build_alloca(16);

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    AliasAnalysis aa(*fn);

    // By default, distinct entry block parameters without noalias must default to MayAlias
    CHECK_EQ(aa.alias(p1, p2), AliasResult::MayAlias);
    CHECK_EQ(aa.alias(p1, 0, Type::i32(), p2, 0, Type::i32()), AliasResult::MayAlias);

    // An alloca does NOT alias an external parameter
    CHECK_EQ(aa.alias(p1, loc), AliasResult::NoAlias);
    CHECK_EQ(aa.alias(p2, loc), AliasResult::NoAlias);

    // When p1 is annotated as noalias, p1 and p2 must disambiguate to NoAlias
    p1->set_noalias(true);
    CHECK(p1->is_noalias());
    CHECK_EQ(aa.alias(p1, p2), AliasResult::NoAlias);
    p1->set_noalias(false);
    CHECK_EQ(aa.alias(p1, p2), AliasResult::MayAlias);

    // When function-level metadata marks param 0 as noalias, they disambiguate to NoAlias
    fn->set_param_noalias(0, true);
    CHECK(fn->is_param_noalias(0));
    CHECK_EQ(aa.alias(p1, p2), AliasResult::NoAlias);
    fn->set_param_noalias(0, false);
    CHECK_EQ(aa.alias(p1, p2), AliasResult::MayAlias);

    // When function-level metadata marks param 1 as noalias, they disambiguate to NoAlias
    fn->set_param_noalias(1, true);
    CHECK(fn->is_param_noalias(1));
    CHECK_EQ(aa.alias(p1, p2), AliasResult::NoAlias);
    fn->set_param_noalias(1, false);
}

// 2. Alias Analysis Soundness: Overlapping Offset Interval Clobbering
TEST_CASE("MIR Hardening - Overlapping Offset Interval Clobbering") {
    Module mod("test_offset_intervals");
    Builder b(mod);
    Function* fn = mod.create_function("test_intervals", Type::void_type(), {Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* base = b.build_alloca(32);

    // 8-byte write at offset 0 [0, 8)
    Instruction* st_0_i64 = b.build_store(Type::i64(), base, 0, b.build_iconst_i64(100));

    // 4-byte read at offset 4 [4, 8) -> overlaps [0, 8)
    Instruction* ld_4_i32 = b.build_load(Type::i32(), base, 4)->defining_instruction();

    // 8-byte read at offset 8 [8, 16) -> disjoint from [0, 8)
    Instruction* ld_8_i64 = b.build_load(Type::i64(), base, 8)->defining_instruction();

    // 8-byte read at offset 0 [0, 8) -> exact overlap
    Instruction* ld_0_i64 = b.build_load(Type::i64(), base, 0)->defining_instruction();

    // Indexed stores/loads with identical index and scale
    Value* iv = b.build_iconst_i64(1);
    Instruction* st_idx_0 = b.build_store_indexed(Type::i64(), base, iv, 8, 0, b.build_iconst_i64(55));
    Instruction* ld_idx_4 = b.build_load_indexed(Type::i32(), base, iv, 8, 4)->defining_instruction();
    Instruction* ld_idx_8 = b.build_load_indexed(Type::i64(), base, iv, 8, 8)->defining_instruction();

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    AliasAnalysis aa(*fn);

    // Store at [0, 8) clobbers load at [4, 8)
    CHECK(aa.can_clobber(st_0_i64, ld_4_i32));
    CHECK_EQ(aa.alias(base, 0, Type::i64(), base, 4, Type::i32()), AliasResult::MayAlias);

    // Store at [0, 8) DOES NOT clobber load at [8, 16)
    CHECK(!aa.can_clobber(st_0_i64, ld_8_i64));
    CHECK_EQ(aa.alias(base, 0, Type::i64(), base, 8, Type::i64()), AliasResult::NoAlias);

    // Store at [0, 8) clobbers exact match load at [0, 8)
    CHECK(aa.can_clobber(st_0_i64, ld_0_i64));
    CHECK_EQ(aa.alias(base, 0, Type::i64(), base, 0, Type::i64()), AliasResult::MustAlias);

    // Indexed access interval overlap checks
    CHECK(aa.can_clobber(st_idx_0, ld_idx_4));
    CHECK(!aa.can_clobber(st_idx_0, ld_idx_8));
}

// 3. Loop Dependence Analysis: Multi-Byte Element Access Distance
TEST_CASE("MIR Hardening - Loop Dependence Multi-Byte Element Access Distance") {
    Module mod("test_loop_dep_units");
    Function* fn = mod.create_function("dep_test", Type::void_type(), {Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* base = b.add_block_param(entry, Type::ptr());
    Value* iv = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    // Test 4-byte element array (scale = 4):
    // A[i + 1] write vs A[i] read
    Instruction* s_4b = b.build_store_indexed(Type::i32(), base, iv, 4, 4, b.build_iconst_i32(1));
    Instruction* l_4b = b.build_load_indexed(Type::i32(), base, iv, 4, 0)->defining_instruction();

    SubscriptExpr expr_s4{base, iv, 4, 4, 4, true};
    SubscriptExpr expr_l4{base, iv, 4, 0, 4, true};
    ParallelMemAccess ma_s4{s_4b, true, Type::i32(), expr_s4};
    ParallelMemAccess ma_l4{l_4b, false, Type::i32(), expr_l4};

    ParallelDependence dep4 = check_subscript_dependence(ma_s4, ma_l4);
    CHECK(dep4.kind != DependenceKind::None);
    CHECK(dep4.is_loop_carried);
    CHECK_EQ(std::abs(dep4.distance), 1); // Distance is 1 iteration, not 4!

    // Test 8-byte element array (scale = 8):
    // A[i + 1] write vs A[i] read
    Instruction* s_8b = b.build_store_indexed(Type::i64(), base, iv, 8, 8, b.build_iconst_i64(2));
    Instruction* l_8b = b.build_load_indexed(Type::i64(), base, iv, 8, 0)->defining_instruction();

    SubscriptExpr expr_s8{base, iv, 8, 8, 8, true};
    SubscriptExpr expr_l8{base, iv, 8, 0, 8, true};
    ParallelMemAccess ma_s8{s_8b, true, Type::i64(), expr_s8};
    ParallelMemAccess ma_l8{l_8b, false, Type::i64(), expr_l8};

    ParallelDependence dep8 = check_subscript_dependence(ma_s8, ma_l8);
    CHECK(dep8.kind != DependenceKind::None);
    CHECK(dep8.is_loop_carried);
    CHECK_EQ(std::abs(dep8.distance), 1); // Distance is 1 iteration, not 8!

    b.build_ret_void();
}

// 4. SLP Vectorizer Store Bundling: Intervening Writes Barrier
TEST_CASE("MIR Hardening - SLP Vectorizer Preserves Store Ordering On Aliased Writes") {
    Module mod("test_slp_store_barrier");
    Function* fn = mod.create_function("slp_barrier_fn", Type::void_type(), {Type::ptr(), Type::ptr()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::ptr());
    Value* other = b.add_block_param(entry, Type::ptr());
    b.position_at_end(entry);

    Value* v0 = b.build_iconst_i32(10);
    Value* v1 = b.build_iconst_i32(20);
    Value* v2 = b.build_iconst_i32(30);
    Value* v3 = b.build_iconst_i32(40);
    Value* v_inter = b.build_iconst_i32(99);

    // Store arr[0]
    b.build_store(Type::i32(), arr, 0, v0);

    // Intervening store to `other` which may alias `arr`
    b.build_store(Type::i32(), other, 0, v_inter);

    // Consecutive stores arr[1], arr[2], arr[3]
    b.build_store(Type::i32(), arr, 4, v1);
    b.build_store(Type::i32(), arr, 8, v2);
    b.build_store(Type::i32(), arr, 12, v3);

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    SlpOptions opts;
    opts.enable_i32x4 = true;

    std::vector<SlpStoreBundle> bundles = find_slp_store_bundles(*entry, opts);

    // Store bundling starting from arr[0] MUST NOT bundle across the intervening store to `other`
    // (since arr and other may alias, bundling across it would illegally reorder writes).
    bool bundled_across_intervening = false;
    for (const auto& bundle : bundles) {
        if (bundle.stores.size() == 4) {
            bundled_across_intervening = true;
        }
    }
    CHECK(!bundled_across_intervening);
}

// 5. CFG Simplification: Linear Block Merge Updates Successor Predecessors
TEST_CASE("MIR Hardening - CFG Simplification Linear Block Merge Updates Successor Predecessors") {
    Module mod("test_cfg_merge_preds");
    Function* fn = mod.create_function("test_linear_merge", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* P = b.append_block("P");
    BasicBlock* S = b.append_block("S");
    BasicBlock* T1 = b.append_block("T1");
    BasicBlock* T2 = b.append_block("T2");
    BasicBlock* other = b.append_block("other");

    b.position_at_end(entry);
    Value* arg0 = b.add_block_param(entry, Type::i64());
    Value* arg1 = b.add_block_param(entry, Type::i64());
    Value* cond_entry = b.build_slt(arg0, arg1);
    // entry has 2 successors so it cannot merge into P
    b.build_br_if(cond_entry, P, {}, other, {});

    // other block
    b.position_at_end(other);
    b.build_ret(arg1);

    // P is a block with single successor S
    b.position_at_end(P);
    Value* p_val = b.build_add(arg0, arg1);
    b.build_br(S);

    // S has single predecessor P, but multiple successors T1 and T2
    b.position_at_end(S);
    Value* cond = b.build_slt(p_val, b.build_iconst_i64(100));
    b.build_br_if(cond, T1, {p_val}, T2, {p_val});

    // T1
    b.position_at_end(T1);
    Value* t1_val = b.add_block_param(T1, Type::i64());
    b.build_ret(t1_val);

    // T2
    b.position_at_end(T2);
    Value* t2_val = b.add_block_param(T2, Type::i64());
    b.build_ret(t2_val);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // Simplify CFG: P and S must merge into a single linear block (P absorbs S)
    CfgSimplifyOptions opts;
    bool changed = cfg_simplify_function(*fn, opts);
    CHECK(changed);

    // Verify CFG structure and that T1 and T2 predecessors were updated from S to P
    REQUIRE(verify_module(mod, &diag));

    auto has_pred = [](BasicBlock* bb, BasicBlock* target_pred) {
        for (BasicBlock* pred : bb->predecessors()) {
            if (pred == target_pred) return true;
        }
        return false;
    };

    // S has been merged into P; T1 and T2 must now have P as their predecessor, NOT S!
    CHECK(has_pred(T1, P));
    CHECK(has_pred(T2, P));
    CHECK(!has_pred(T1, S));
    CHECK(!has_pred(T2, S));
}

TEST_CASE("MIR Hardening - CFG Simplification renumbers the parameters it keeps") {
    // A loop header (%dead, %i): %dead is never read, so it is dropped and %i
    // moves to index 0. Its recorded index stayed 1, and loop fusion, which
    // reads the latch argument at param_index(), read past the argument list
    // (a fault found by the AArch64 fuzz harness). The verifier now checks
    // that every parameter records its block and index.
    Module mod("test_cfg_param_index");
    Function* fn = mod.create_function("f", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hdr = b.append_block("hdr");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(hdr, {zero, zero});

    b.position_at_end(hdr);
    Value* dead = b.add_block_param(hdr, Type::i64());
    Value* i = b.add_block_param(hdr, Type::i64());
    b.build_br_if(b.build_slt(i, n), body, {}, exit, {});

    b.position_at_end(body);
    Value* next = b.build_add(i, b.build_iconst_i64(1));
    b.build_br(hdr, {b.build_iconst_i64(7), next});

    b.position_at_end(exit);
    b.build_ret(i);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));
    (void)dead;

    CfgSimplifyOptions opts;
    CHECK(cfg_simplify_function(*fn, opts));
    REQUIRE_EQ(hdr->param_count(), size_t{1});
    CHECK(hdr->param(0) == i);
    CHECK_EQ(i->param_index(), 0u);
    CHECK(verify_module(mod, &diag));
}

// 6. Range Analysis: Wrapping Addition and Subtraction Degrade to Full Range
TEST_CASE("MIR Hardening - Range Analysis Wrapping Addition Degrades To Full Range") {
    // 1. ValueRange addition overflow check
    ValueRange r_max(INT64_MAX - 10, INT64_MAX);
    ValueRange r_add(20, 30);
    ValueRange r_overflow_add = ValueRange::add(r_max, r_add);
    CHECK(r_overflow_add.is_full());
    CHECK_EQ(r_overflow_add.min_val, INT64_MIN);
    CHECK_EQ(r_overflow_add.max_val, INT64_MAX);

    // 2. ValueRange subtraction underflow check
    ValueRange r_min(INT64_MIN, INT64_MIN + 10);
    ValueRange r_sub(20, 30);
    ValueRange r_overflow_sub = ValueRange::sub(r_min, r_sub);
    CHECK(r_overflow_sub.is_full());
    CHECK_EQ(r_overflow_sub.min_val, INT64_MIN);
    CHECK_EQ(r_overflow_sub.max_val, INT64_MAX);

    // 3. Multiplication overflow check (4e9 * 4e9 = 1.6e19 > INT64_MAX ~ 9.22e18)
    ValueRange r_mul1(4000000000LL, 5000000000LL);
    ValueRange r_mul2(4000000000LL, 5000000000LL);
    ValueRange r_overflow_mul = ValueRange::mul(r_mul1, r_mul2);
    CHECK(r_overflow_mul.is_full());

    // 4. Non-overflowing addition computes exact bounds
    ValueRange r1(10, 25);
    ValueRange r2(5, 15);
    ValueRange r_valid_add = ValueRange::add(r1, r2);
    CHECK(!r_valid_add.is_full());
    CHECK_EQ(r_valid_add.min_val, 15);
    CHECK_EQ(r_valid_add.max_val, 40);

    // 5. Non-overflowing subtraction computes exact bounds
    ValueRange r_valid_sub = ValueRange::sub(r1, r2);
    CHECK(!r_valid_sub.is_full());
    CHECK_EQ(r_valid_sub.min_val, -5);
    CHECK_EQ(r_valid_sub.max_val, 20);
}

// 7. Dead Store Elimination: Preserves Stores Preceding Deopt Guards & Traps
TEST_CASE("MIR Hardening - DSE Preserves Stores Preceding Deopt Guards") {
    // 1. Without guard: store 1 followed by store 2 to same address -> store 1 is dead
    {
        Module mod("test_dse_dead");
        Builder b(mod);
        Function* fn = mod.create_function("dse_no_guard", Type::void_type(), {Type::ptr()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* p = b.add_block_param(entry, Type::ptr());
        p->set_noalias(true);
        b.position_at_end(entry);

        Value* v1 = b.build_iconst_i64(10);
        Value* v2 = b.build_iconst_i64(20);
        Instruction* st1 = b.build_store(Type::i64(), p, 0, v1);
        Instruction* st2 = b.build_store(Type::i64(), p, 0, v2);
        (void)st1;
        (void)st2;
        b.build_ret_void();

        fn->rebuild_cfg_predecessors();

        GvnStats stats;
        GvnOptions opts;
        opts.enable_cse = false;
        opts.enable_rle = false;
        opts.enable_dse = true;
        opts.stats = &stats;

        bool changed = gvn_function(*fn, opts);
        CHECK(changed);
        CHECK_EQ(stats.dead_stores_eliminated, 1u);
    }

    // 2. With deopt guard between store 1 and store 2: store 1 MUST be preserved!
    {
        Module mod("test_dse_guard");
        Builder b(mod);
        Function* fn = mod.create_function("dse_with_guard", Type::void_type(), {Type::ptr(), Type::i32()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* p = b.add_block_param(entry, Type::ptr());
        Value* cond = b.add_block_param(entry, Type::i32());
        p->set_noalias(true);
        b.position_at_end(entry);

        Value* v1 = b.build_iconst_i64(10);
        Value* v2 = b.build_iconst_i64(20);
        Instruction* st1 = b.build_store(Type::i64(), p, 0, v1);
        // Deopt guard serves as memory barrier for interpreter state recovery
        b.build_guard(cond, "deopt_trap");
        Instruction* st2 = b.build_store(Type::i64(), p, 0, v2);
        (void)st1;
        (void)st2;
        b.build_ret_void();

        fn->rebuild_cfg_predecessors();

        GvnStats stats;
        GvnOptions opts;
        opts.enable_cse = false;
        opts.enable_rle = false;
        opts.enable_dse = true;
        opts.stats = &stats;

        bool changed = gvn_function(*fn, opts);
        // st1 must NOT be eliminated because guard can deoptimize to runtime/interpreter
        CHECK_EQ(stats.dead_stores_eliminated, 0u);
        (void)changed;
    }
}

// 8. SROA and Allocation Sinking: F32 Zero Generation and Alloca Opcode Verification
TEST_CASE("MIR Hardening - SROA and Allocation Sinking F32 Zero Generation") {
    // 1. SROA uninitialized F32 load promotion
    {
        Module mod("test_sroa_f32");
        Builder b(mod);
        Function* fn = mod.create_function("sroa_f32_zero", Type::f32(), {});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);

        Value* alloc = b.build_alloca(4);
        Value* ld = b.build_load(Type::f32(), alloc, 0);
        b.build_ret(ld);

        fn->rebuild_cfg_predecessors();

        SroaOptions opts;
        SroaStats stats;
        opts.stats = &stats;

        bool changed = sroa_function(*fn, opts);
        CHECK(changed);
        CHECK_EQ(stats.allocations_eliminated, 1u);

        // Verify the promoted value is fconst with Type::f32() and 0.0f
        Instruction* term = entry->terminator();
        REQUIRE(term != nullptr);
        REQUIRE_EQ(term->opcode(), Opcode::ret);
        Value* ret_val = term->operand(0);
        REQUIRE(ret_val != nullptr);
        CHECK_EQ(ret_val->type(), Type::f32());
        REQUIRE(ret_val->is_instruction());
        Instruction* def = ret_val->defining_instruction();
        REQUIRE(def != nullptr);
        CHECK_EQ(def->type(), Type::f32());
        CHECK_EQ(def->imm_f64(), 0.0);
    }

    // 2. Allocation sinking: verify alloca promotion type correctness
    {
        Module mod("test_alloc_sink_f32");
        Builder b(mod);
        Function* fn = mod.create_function("alloc_sink_fn", Type::f32(), {Type::i32()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* then_bb = b.create_block("then");
        BasicBlock* else_bb = b.create_block("else");
        BasicBlock* merge_bb = b.create_block("merge");

        Value* cond = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* alloc = b.build_alloca(4);
        b.build_br_if(cond, then_bb, {}, else_bb, {});

        fn->append_block(then_bb);
        b.position_at_end(then_bb);
        b.build_store(Type::f32(), alloc, 0, b.build_fconst_f32(3.14f));
        Value* v_then = b.build_load(Type::f32(), alloc, 0);
        b.build_br(merge_bb, {v_then});

        fn->append_block(else_bb);
        b.position_at_end(else_bb);
        Value* v_else = b.build_fconst_f32(1.0f);
        b.build_br(merge_bb, {v_else});

        fn->append_block(merge_bb);
        b.position_at_end(merge_bb);
        Value* phi = b.add_block_param(merge_bb, Type::f32());
        b.build_ret(phi);

        fn->rebuild_cfg_predecessors();

        AllocationSinkingOptions opts;
        sink_allocations(*fn, opts);
        CHECK(verify_function(*fn));
    }
}
