#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

TEST_CASE("Loop Distribution - Split Mixed Loop and Enable Vectorization") {
    Module mod("test_dist_vec");
    Builder b(mod);

    // fn(src: gcref, dst: gcref, n: i64)
    // for i = 0..n:
    //   v = load_indexed src, i
    //   res = mul v, 3
    //   store_indexed dst, i, res
    //   call @dummy_side_effect(res)
    Function* fn = mod.create_function("mixed_loop", Type::void_type(), {Type::gcref(), Type::gcref(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* src = b.add_block_param(entry, Type::gcref());
    Value* dst = b.add_block_param(entry, Type::gcref());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* three = b.build_iconst_i32(3);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* loaded = b.build_load_indexed(Type::i32(), src, i, 4);
    Value* multiplied = b.build_mul(loaded, three);
    b.build_store_indexed(Type::i32(), dst, i, 4, multiplied);

    // Unvectorizable call that previously blocked vectorization
    b.build_call("dummy_side_effect", Type::void_type(), {multiplied});

    Value* next_i = b.build_add(i, one);
    b.build_br(hdr, {next_i});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    CHECK_EQ(la.top_level_loops().size(), 1);

    // Before distribution, LoopVectorizer must reject the loop due to the call
    LoopVectorizeOptions vec_opts;
    CHECK(!loop_vectorize_pass(*fn, dom, vec_opts));

    // Distribute loop
    LoopDistributionOptions dist_opts;
    LoopDistributionStats stats;
    dist_opts.stats = &stats;

    CHECK(can_distribute_loop(*fn, *la.top_level_loops()[0], dom, dist_opts));
    bool distributed = distribute_loop(*fn, *la.top_level_loops()[0], dom, dist_opts);
    CHECK(distributed);
    CHECK_EQ(stats.loops_distributed, 1);

    CHECK(verify_function(*fn));

    // Now there should be 2 loops
    fn->rebuild_cfg_predecessors();
    DominatorTree dom_after(*fn);
    LoopAnalysis la_after(*fn, dom_after);
    CHECK_EQ(la_after.top_level_loops().size(), 2);

    // Downstream LoopVectorizer can now vectorize Loop 1!
    bool vec_ok = loop_vectorize_pass(*fn, dom_after, vec_opts);
    CHECK(vec_ok);
    CHECK(verify_function(*fn));

    // Verify Loop 1 now contains SIMD operations
    bool has_vload = false, has_vmul = false, has_vstore = false;
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::vload) has_vload = true;
            if (inst->opcode() == Opcode::vmul) has_vmul = true;
            if (inst->opcode() == Opcode::vstore) has_vstore = true;
        }
    }
    CHECK(has_vload);
    CHECK(has_vmul);
    CHECK(has_vstore);
}

TEST_CASE("Loop Distribution - Reject Pure Vectorizable and Pure Scalar Loops") {
    Module mod("test_dist_reject");
    Builder b(mod);

    // Pure vectorizable loop (no unvectorizable statements)
    Function* fn_vec = mod.create_function("pure_vec", Type::void_type(), {Type::gcref(), Type::i64()});
    b.set_function(fn_vec);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::gcref());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* c = b.build_iconst_i32(10);
    b.build_br(hdr, {zero});

    fn_vec->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn_vec->append_block(body);
    b.position_at_end(body);
    Value* v = b.build_load_indexed(Type::i32(), arr, i, 4);
    Value* res = b.build_add(v, c);
    b.build_store_indexed(Type::i32(), arr, i, 4, res);
    Value* next_i = b.build_add(i, one);
    b.build_br(hdr, {next_i});

    fn_vec->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn_vec->rebuild_cfg_predecessors();
    DominatorTree dom(*fn_vec);
    LoopAnalysis la(*fn_vec, dom);

    LoopDistributionOptions dist_opts;
    LoopDistributionStats stats;
    dist_opts.stats = &stats;

    // Pure vectorizable loop should be rejected by distribution
    CHECK(!can_distribute_loop(*fn_vec, *la.top_level_loops()[0], dom, dist_opts));
    CHECK(stats.rejected_pure_vectorizable > 0);
}
