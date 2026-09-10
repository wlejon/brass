#include "test_framework.hpp"
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_analysis.hpp>

using namespace brass;

TEST_CASE("Subscript Expression Analysis") {
    Module mod("subscript_test");
    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* bb = b.append_block("entry");
    Value* base = b.add_block_param(bb, Type::ptr());
    Value* iv = b.add_block_param(bb, Type::i64());
    b.position_at_end(bb);

    // Test 1: load_indexed
    Value* vli = b.build_load_indexed(Type::i64(), base, iv, 8, 16);
    Instruction* li = vli->defining_instruction();
    SubscriptExpr expr;
    bool ok = parse_subscript_expression(li, iv, expr);
    CHECK(ok);
    CHECK_EQ(expr.base, base);
    CHECK_EQ(expr.stride, 8);
    CHECK_EQ(expr.offset, 16);
    CHECK(expr.is_valid);

    // Test 2: store_indexed
    Value* val = b.build_iconst_i64(42);
    Instruction* si = b.build_store_indexed(Type::i64(), base, iv, 4, 0, val);
    ok = parse_subscript_expression(si, iv, expr);
    CHECK(ok);
    CHECK_EQ(expr.base, base);
    CHECK_EQ(expr.stride, 4);
    CHECK_EQ(expr.offset, 0);

    // Test 3: GEP + load
    Value* scaled = b.build_mul(iv, b.build_iconst_i64(8));
    Value* off = b.build_add(scaled, b.build_iconst_i64(24));
    Value* addr = b.build_add(base, off);
    Value* vld = b.build_load(Type::i64(), addr, 0);
    Instruction* ld = vld->defining_instruction();
    ok = parse_subscript_expression(ld, iv, expr);
    CHECK(ok);
    CHECK_EQ(expr.base, base);
    CHECK_EQ(expr.stride, 8);
    CHECK_EQ(expr.offset, 24);

    b.build_ret_void();
}

TEST_CASE("Distance and Direction Vectors") {
    Module mod("dist_test");
    Function* fn = mod.create_function("dist_fn", Type::void_type(), {Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* bb = b.append_block("entry");
    Value* baseA = b.add_block_param(bb, Type::ptr());
    Value* iv = b.add_block_param(bb, Type::i64());
    b.position_at_end(bb);
    Value* val = b.build_iconst_i64(1);

    // a1: store A[i * 8 + 8]
    Instruction* s1 = b.build_store_indexed(Type::i64(), baseA, iv, 8, 8, val);
    // a2: load A[i * 8 + 0]
    Value* vl1 = b.build_load_indexed(Type::i64(), baseA, iv, 8, 0);
    Instruction* l1 = vl1->defining_instruction();
    // a3: load A[i * 8 + 8]
    Value* vl2 = b.build_load_indexed(Type::i64(), baseA, iv, 8, 8);
    Instruction* l2 = vl2->defining_instruction();

    SubscriptExpr expr_store{baseA, iv, 8, 8, 8, true};
    SubscriptExpr expr_load_carried{baseA, iv, 8, 0, 8, true};
    SubscriptExpr expr_load_indep{baseA, iv, 8, 8, 8, true};

    ParallelMemAccess ma_store{s1, true, Type::i64(), expr_store};
    ParallelMemAccess ma_load_carried{l1, false, Type::i64(), expr_load_carried};
    ParallelMemAccess ma_load_independent{l2, false, Type::i64(), expr_load_indep};

    // Store followed by load at same iteration -> loop-independent (distance 0)
    ParallelDependence dep_indep = check_subscript_dependence(ma_store, ma_load_independent);
    CHECK(dep_indep.kind != DependenceKind::None);
    CHECK(!dep_indep.is_loop_carried);
    CHECK_EQ(dep_indep.distance, 0);
    CHECK(dep_indep.dir == DependenceDir::Equal);

    // Store A[i+1] and load A[i]: diff = 8, stride = 8 -> distance = -1 or 1 (loop-carried)
    ParallelDependence dep_carried = check_subscript_dependence(ma_store, ma_load_carried);
    CHECK(dep_carried.kind != DependenceKind::None);
    CHECK(dep_carried.is_loop_carried);
    CHECK_NE(dep_carried.distance, 0);
    CHECK(dep_carried.dir == DependenceDir::Forward || dep_carried.dir == DependenceDir::Backward);

    b.build_ret_void();
}

TEST_CASE("DOALL Loop Detection") {
    Module mod("doall_test");
    Function* fn = mod.create_function("vector_add", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::ptr());
    Value* c = b.add_block_param(entry, Type::ptr());
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* va = b.build_load_indexed(Type::i64(), a, iv, 8, 0);
    Value* vc = b.build_load_indexed(Type::i64(), c, iv, 8, 0);
    Value* sum = b.build_add(va, vc);
    b.build_store_indexed(Type::i64(), out, iv, 8, 0, sum);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    REQUIRE_EQ(la.top_level_loops().size(), 1u);

    ParallelLoopInfo pli;
    ParallelLoopOptions opts;
    bool ok = analyze_parallel_loop(*fn, *la.top_level_loops()[0], dom, pli, opts);
    CHECK(ok);
    CHECK(pli.kind == LoopParallelKind::DOALL);
    CHECK(pli.is_parallelizable());
    CHECK(!pli.has_reduction);
}

TEST_CASE("Loop Carried Dependence Rejection") {
    Module mod("recurrence_test");
    Function* fn = mod.create_function("fib_like", Type::void_type(), {
        Type::ptr(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* start = b.build_iconst_i64(1);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {start});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    // arr[i] = arr[i - 1] + 10 (loop carried RAW dependence)
    Value* prev = b.build_load_indexed(Type::i64(), arr, iv, 8, -8);
    Value* plus10 = b.build_add(prev, b.build_iconst_i64(10));
    b.build_store_indexed(Type::i64(), arr, iv, 8, 0, plus10);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    REQUIRE_EQ(la.top_level_loops().size(), 1u);

    ParallelLoopInfo pli;
    ParallelLoopStats stats;
    ParallelLoopOptions opts;
    opts.stats = &stats;
    bool ok = analyze_parallel_loop(*fn, *la.top_level_loops()[0], dom, pli, opts);
    CHECK(!ok);
    CHECK(pli.kind == LoopParallelKind::Sequential);
    CHECK(stats.loops_rejected_carried_dependence > 0u);
}

TEST_CASE("Reduction Detection (Sum, Product, Min, Max)") {
    Module mod("red_test");
    Function* fn = mod.create_function("sum_reduce", Type::i64(), {
        Type::ptr(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {zero, zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* acc = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, body, {}, exit, {acc});

    fn->append_block(body);
    b.position_at_end(body);
    Value* val = b.build_load_indexed(Type::i64(), arr, iv, 8, 0);
    Value* next_acc = b.build_add(acc, val);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv, next_acc});

    fn->append_block(exit);
    b.position_at_end(exit);
    Value* final_res = b.add_block_param(exit, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    REQUIRE_EQ(la.top_level_loops().size(), 1u);

    ParallelLoopInfo pli;
    ParallelLoopStats stats;
    ParallelLoopOptions opts;
    opts.stats = &stats;
    bool ok = analyze_parallel_loop(*fn, *la.top_level_loops()[0], dom, pli, opts);
    CHECK(ok);
    CHECK(pli.kind == LoopParallelKind::Reduction);
    CHECK(pli.has_reduction);
    CHECK(pli.reduction_kind == runtime::ReductionKind::SumI64);
    CHECK_EQ(stats.reduction_loops_found, 1u);
}
