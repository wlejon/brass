#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/inline_transform.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <limits>
#include <cmath>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Phase 2 - GVN Load-to-Load Elimination Offset Bug") {
    Module mod("test_gvn_offsets");
    Builder b(mod);

    Function* fn = mod.create_function("gvn_offset_fn", Type::i64(), {Type::ptr()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* base = b.add_block_param(entry, Type::ptr());

    Value* c8 = b.build_iconst_i64(8);
    Value* p8 = b.build_add(base, c8);
    Value* c16 = b.build_iconst_i64(16);
    Value* p16 = b.build_add(base, c16);

    // Load from base + 8 (offset 0)
    Value* l1 = b.build_load(Type::i64(), p8, 0);
    // Load from base + 16 (offset 0)
    Value* l2 = b.build_load(Type::i64(), p16, 0);
    // Redundant second load from base + 8 (offset 0)
    Value* l3 = b.build_load(Type::i64(), p8, 0);

    Value* sum1 = b.build_add(l1, l2);
    Value* sum2 = b.build_add(sum1, l3);
    b.build_ret(sum2);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    GvnStats stats;
    GvnOptions opts;
    opts.enable_rle = true;
    opts.stats = &stats;

    bool changed = gvn_function(*fn, opts);
    CHECK(changed);
    // l3 should be eliminated (matching l1 at offset 8), but l2 (at offset 16) must NOT be eliminated!
    CHECK_EQ(stats.loads_eliminated, 1U);

    // Verify JIT execution
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    alignas(16) uint64_t mem[4] = {0, 42, 100, 0}; // mem[1] at +8 is 42, mem[2] at +16 is 100
    RuntimeValue res = jit.invoke("gvn_offset_fn", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(mem))});
    // l1 = 42, l2 = 100, l3 = 42 -> 42 + 100 + 42 = 184
    CHECK_EQ(res.as_i64(), 184);
}

TEST_CASE("Phase 2 - Loop Vectorizer Trip Count Guard Underflow (N=1, W=4)") {
    Module mod("test_vec_underflow");
    Builder b(mod);

    Function* fn = mod.create_function("vec_guard_fn", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    Value* c1 = b.build_fconst_f32(1.0f);

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* va = b.build_load_indexed(Type::f32(), pa, i, 4, 0);
    Value* vb = b.build_add(va, c1);
    b.build_store_indexed(Type::f32(), pb, i, 4, 0, vb);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    DominatorTree dom(*fn);
    LoopVectorizeOptions vec_opts;
    bool changed = loop_vectorize_pass(*fn, dom, vec_opts);
    CHECK(changed);

    DiagnosticReporter vdiag;
    bool vok = verify_module(mod, &vdiag);
    if (!vok) {
        std::cerr << "Vectorizer verify error:\n" << vdiag.format_all() << "\n";
    }
    REQUIRE(vok);

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    alignas(16) float a[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    alignas(16) float out[8] = {-999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f};

    // Test with N = 1 (less than vector width W = 4)
    jit.invoke("vec_guard_fn", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out)),
        RuntimeValue::from_i64(1)
    });

    // out[0] should be 10.0 + 1.0 = 11.0f
    CHECK_EQ(out[0], 11.0f);
    // Canaries out[1..7] must NOT be overwritten!
    for (int k = 1; k < 8; ++k) {
        CHECK_EQ(out[k], -999.0f);
    }
}

TEST_CASE("Phase 2 - Write Barrier Elimination Preserves Derived GC Pointers") {
    Module mod("test_wbe_derived");
    Builder b(mod);

    Function* fn = mod.create_function("wbe_derived_fn", Type::void_type(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* old_obj = b.add_block_param(entry, Type::gcref());

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* young_alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    // Compute derived young pointer via pointer arithmetic
    Value* c8 = b.build_iconst_i64(8);
    Value* derived_gc = b.build_add(young_alloc, c8);

    // Plain integer value
    Value* int_val = b.build_iconst_i64(42);

    // Write barrier 1: on non-pointer int_val (should be eliminated)
    b.build_write_barrier(old_obj, int_val);

    // Write barrier 2: on derived young pointer (must NOT be eliminated!)
    b.build_write_barrier(old_obj, derived_gc);

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);
    CHECK(changed);

    // Write barrier on int_val is eliminated, but barrier on derived_gc is preserved
    CHECK_EQ(wbe.stats().eliminated_non_pointer, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 1U);
}

TEST_CASE("Phase 2 - SROA Skipping Overlapping Byte Offsets") {
    Module mod("test_sroa_overlap");
    Builder b(mod);

    Function* fn = mod.create_function("sroa_overlap_fn", Type::i32(), {});
    b.set_function(fn);
    b.append_block("entry");

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    // Store 8-byte i64 at offset 0 (bytes 0..7)
    Value* val64 = b.build_iconst_i64(0x1122334455667788LL);
    b.build_store(Type::i64(), obj, 0, val64);

    // Load 4-byte i32 at offset 4 (bytes 4..7, overlapping with offset 0!)
    Value* val32 = b.build_load(Type::i32(), obj, 4);
    b.build_ret(val32);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    SroaOptions opts;
    bool changed = sroa_function(*fn, opts);
    // SROA must reject promoting overlapping byte ranges!
    CHECK(!changed);
}

TEST_CASE("Phase 2 - SCCP Handling NaN and Overflow Float-to-Int Conversions Safely") {
    // 1. NaN conversion should safely produce Bottom (overdefined) without host UB or folding
    {
        Module mod("test_sccp_nan");
        Builder b(mod);
        Function* fn = mod.create_function("fn_nan", Type::i32(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* nan_val = b.build_fconst_f64(std::numeric_limits<double>::quiet_NaN());
        Value* conv = b.build_fptosi_i32(nan_val);
        b.build_ret(conv);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        SccpOptions opts;
        SccpStats stats;
        opts.stats = &stats;
        sccp_function(*fn, opts);
        CHECK_EQ(stats.constants_propagated, 0U);
    }

    // 2. Overflow f64 conversion (1e20) safely yields Bottom without host UB
    {
        Module mod("test_sccp_overflow");
        Builder b(mod);
        Function* fn = mod.create_function("fn_over", Type::i32(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* over_val = b.build_fconst_f64(1e20);
        Value* conv = b.build_fptosi_i32(over_val);
        b.build_ret(conv);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        SccpOptions opts;
        SccpStats stats;
        opts.stats = &stats;
        sccp_function(*fn, opts);
        CHECK_EQ(stats.constants_propagated, 0U);
    }

    // 3. Overflow f32 conversion (1e20f) safely yields Bottom without host UB
    {
        Module mod("test_sccp_f32_overflow");
        Builder b(mod);
        Function* fn = mod.create_function("fn_f32_over", Type::i32(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* over_val = b.build_fconst_f32(1e20f);
        Value* conv = b.build_fptosi_i32_f32(over_val);
        b.build_ret(conv);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        SccpOptions opts;
        SccpStats stats;
        opts.stats = &stats;
        sccp_function(*fn, opts);
        CHECK_EQ(stats.constants_propagated, 0U);
    }

    // 4. Overflow f64 to i64 conversion (1e30) safely yields Bottom
    {
        Module mod("test_sccp_i64_overflow");
        Builder b(mod);
        Function* fn = mod.create_function("fn_i64_over", Type::i64(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* over_val = b.build_fconst_f64(1e30);
        Value* conv = b.build_fptosi_i64(over_val);
        b.build_ret(conv);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        SccpOptions opts;
        SccpStats stats;
        opts.stats = &stats;
        sccp_function(*fn, opts);
        CHECK_EQ(stats.constants_propagated, 0U);
    }

    // 5. Valid normal conversion folds cleanly to iconst
    {
        Module mod("test_sccp_valid");
        Builder b(mod);
        Function* fn = mod.create_function("fn_valid", Type::i32(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* valid_val = b.build_fconst_f64(12345.0);
        Value* conv = b.build_fptosi_i32(valid_val);
        b.build_ret(conv);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        SccpOptions opts;
        SccpStats stats;
        opts.stats = &stats;
        bool changed = sccp_function(*fn, opts);
        CHECK(changed);
        CHECK_EQ(stats.constants_propagated, 1U);

        Interpreter interp;
        RuntimeValue iv = interp.run(*fn, {});
        CHECK_EQ(iv.as_i32(), 12345);
    }
}

TEST_CASE("Phase 2 - Invoke Inlining Normal and Exception Edge Wiring") {
    Module mod("test_invoke_inline");
    Builder b(mod);

    // callee: (x: i32) -> x * 2 + 1
    Function* callee = mod.create_function("callee_fn", Type::i32(), {Type::i32()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    Value* cx = b.add_block_param(c_entry, Type::i32());
    Value* two = b.build_iconst_i32(2);
    Value* prod = b.build_mul(cx, two);
    Value* one = b.build_iconst_i32(1);
    Value* res = b.build_add(prod, one);
    b.build_ret(res);
    callee->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*callee));

    // caller: invoke @callee_fn(%arg), normal_bb, unwind_bb
    Function* caller = mod.create_function("caller_fn", Type::i32(), {Type::i32()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* arg = b.add_block_param(entry, Type::i32());
    Instruction* inv = b.build_invoke("callee_fn", Type::i32(), {arg}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    Value* hundred = b.build_iconst_i32(100);
    Value* final_res = b.build_add(inv->result(), hundred);
    b.build_ret(final_res);

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* exc_trunc = b.build_trunc_i32(exc);
    b.build_ret(exc_trunc);

    caller->rebuild_cfg_predecessors();
    REQUIRE(verify_module(mod));

    // Inline the invoke call site
    InlineResult in_res = inline_call_site(*caller, inv, *callee);
    CHECK(in_res.success);

    DiagnosticReporter diag;
    bool ok_ver = verify_function(*caller, &diag);
    if (!ok_ver) {
        std::cerr << diag.format_all() << "\n";
    }
    REQUIRE(ok_ver);

    // Test execution via Interpreter: arg = 5 -> (5 * 2 + 1) + 100 = 111
    Interpreter interp;
    RuntimeValue iv = interp.run(*caller, {RuntimeValue::from_i32(5)});
    CHECK_EQ(iv.as_i32(), 111);
}

TEST_CASE("Phase 2 - Loop Dependence Distinct Bases Alias Analysis Check") {
    Module mod("test_loop_dep_alias");
    Builder b(mod);

    Function* fn = mod.create_function("loop_dep_fn", Type::void_type(), {Type::ptr(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* ptr_table = b.add_block_param(entry, Type::ptr());
    b.add_block_param(entry, Type::i64());

    Value* p1 = b.build_load(Type::ptr(), ptr_table, 0);
    Value* p2 = b.build_load(Type::ptr(), ptr_table, 8);

    // Both accesses are on unknown loaded pointers p1 and p2 (which MayAlias)
    AliasAnalysis aa(*fn);
    CHECK(aa.alias(p1, p2) == AliasResult::MayAlias);

    // Parallel dependence analysis between p1 and p2 must not assume NoAlias!
    ParallelMemAccess a1;
    a1.is_store = true;
    a1.expr.base = p1;
    a1.expr.is_valid = true;
    a1.expr.stride = 4;
    a1.expr.offset = 0;

    ParallelMemAccess a2;
    a2.is_store = false;
    a2.expr.base = p2;
    a2.expr.is_valid = true;
    a2.expr.stride = 4;
    a2.expr.offset = 0;

    ParallelDependence dep = check_subscript_dependence(a1, a2, &aa);
    // Because p1 and p2 MayAlias, dep.dir must be Any and loop-carried!
    CHECK(dep.dir == DependenceDir::Any);
    CHECK(dep.is_loop_carried);
}
