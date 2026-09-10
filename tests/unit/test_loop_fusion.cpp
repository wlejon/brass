#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/printer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

namespace {
LoopInfo* find_loop(const LoopAnalysis& la, BasicBlock* hdr) {
    for (const auto& loop : la.top_level_loops()) {
        if (loop->header() == hdr) return loop.get();
    }
    return nullptr;
}
} // namespace

TEST_CASE("Loop Fusion - Detect and Fuse Adjacent Congruent Loops") {
    Module mod("test_fusion_adjacent");
    Builder b(mod);

    // fn(n: i64) -> i64
    // Loop 1: acc1 += i (i = 0..n)
    // Loop 2: acc2 += i * 2 (i = 0..n)
    // ret acc1 + acc2
    Function* fn = mod.create_function("sum_two_loops", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    b.build_br(l1_hdr, {zero, zero});

    // l1_hdr(i1, acc1)
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* acc1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {acc1});

    // l1_body
    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* next_acc1 = b.build_add(acc1, i1);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1, next_acc1});

    // l1_exit(final_acc1)
    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    Value* final_acc1 = b.add_block_param(l1_exit, Type::i64());
    b.build_br(l2_hdr, {zero, zero});

    // l2_hdr(i2, acc2)
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* acc2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {acc2});

    // l2_body
    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* mul_i2 = b.build_mul(i2, two);
    Value* next_acc2 = b.build_add(acc2, mul_i2);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2, next_acc2});

    // l2_exit(final_acc2)
    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    Value* final_acc2 = b.add_block_param(l2_exit, Type::i64());
    Value* total = b.build_add(final_acc1, final_acc2);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    CHECK_EQ(la.top_level_loops().size(), 2);

    LoopFusionOptions options;
    LoopFusionStats stats;
    options.stats = &stats;

    LoopInfo* l1 = find_loop(la, l1_hdr);
    LoopInfo* l2 = find_loop(la, l2_hdr);
    REQUIRE(l1 != nullptr);
    REQUIRE(l2 != nullptr);

    CHECK(can_fuse_loops(*fn, *l1, *l2, dom, options));
    bool fused = fuse_loops(*fn, *l1, *l2, dom, options);
    CHECK(fused);
    CHECK_EQ(stats.loops_fused, 1);

    // After fusion, verify function passes verifier
    DiagnosticReporter diag;
    bool ok = verify_function(*fn, &diag);
    if (!ok) std::cerr << "VERIFY ERROR:\n" << diag.format_all() << std::endl;
    CHECK(ok);

    // Re-run loop analysis: now exactly 1 loop!
    fn->rebuild_cfg_predecessors();
    DominatorTree dom_after(*fn);
    LoopAnalysis la_after(*fn, dom_after);
    CHECK_EQ(la_after.top_level_loops().size(), 1);

    // Execute with JIT: n = 10
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t)>("sum_two_loops");
    REQUIRE(fn_ptr != nullptr);
    CHECK_EQ(fn_ptr(10), 135);

    // Execute with Interpreter: n = 10
    // acc1 = 0..9 sum = 45
    // acc2 = 2 * (0..9) = 90
    // total = 135
    Interpreter interp;
    auto res = interp.run(*fn, {RuntimeValue::from_i64(10)});
    CHECK_EQ(res.as_i64(), 135);
}

TEST_CASE("Loop Fusion - Reject Non-Congruent Domains") {
    Module mod("test_fusion_domain_mismatch");
    Builder b(mod);

    Function* fn = mod.create_function("domain_mismatch", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n1 = b.add_block_param(entry, Type::i64());
    Value* n2 = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(l1_hdr, {zero});

    // Loop 1 iterates to n1
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n1);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    // Exit 1 leads to Loop 2
    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero});

    // Loop 2 iterates to n2
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n2);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2});

    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    b.build_ret(zero);

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    CHECK_EQ(la.top_level_loops().size(), 2);

    LoopFusionOptions options;
    LoopFusionStats stats;
    options.stats = &stats;

    LoopInfo* l1 = find_loop(la, l1_hdr);
    LoopInfo* l2 = find_loop(la, l2_hdr);
    REQUIRE(l1 != nullptr);
    REQUIRE(l2 != nullptr);

    // n1 != n2, so domains are not congruent!
    CHECK(!can_fuse_loops(*fn, *l1, *l2, dom, options));
    CHECK(stats.rejected_domain_mismatch > 0);
}

TEST_CASE("Loop Fusion - Reject Unsafe Cross-Iteration Dependency") {
    Module mod("test_fusion_dependency_safety");
    Builder b(mod);

    // Loop 1 writes A[i]
    // Loop 2 reads A[i + 1] (anti-dependence: reading ahead)
    Function* fn = mod.create_function("raw_hazard", Type::void_type(), {Type::gcref(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::gcref());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* val = b.build_iconst_i64(42);
    b.build_br(l1_hdr, {zero});

    // Loop 1: arr[i1] = 42
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    b.build_store_indexed(Type::i64(), arr, i1, 8, val);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero});

    // Loop 2: read arr[i2 + 1] (not i2!)
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* i2_plus_one = b.build_add(i2, one);
    b.build_load_indexed(Type::i64(), arr, i2_plus_one, 8);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2});

    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    CHECK_EQ(la.top_level_loops().size(), 2);

    LoopFusionOptions options;
    LoopFusionStats stats;
    options.stats = &stats;

    LoopInfo* l1 = find_loop(la, l1_hdr);
    LoopInfo* l2 = find_loop(la, l2_hdr);
    REQUIRE(l1 != nullptr);
    REQUIRE(l2 != nullptr);

    // Must reject due to non-IV or non-zero distance dependence!
    CHECK(!can_fuse_loops(*fn, *l1, *l2, dom, options));
    CHECK(stats.rejected_dependencies > 0);
}
