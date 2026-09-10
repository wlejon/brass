#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential PEA - Loop with Conditional Early Exit") {
    Module mod("diff_pea_early_exit");
    Builder b(mod);

    // func @loop_search(%n: i64, %target: i64) -> i64
    Function* fn = mod.create_function("loop_search", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* next_iter = b.append_block("next_iter");
    BasicBlock* found_exit = b.append_block("found_exit");
    BasicBlock* not_found_exit = b.append_block("not_found_exit");

    Value* n = b.add_block_param(entry, Type::i64());
    Value* target = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    // Temporary object allocated per iteration
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, i_param);
    Value* c10 = b.build_iconst_i64(10);
    Value* scaled = b.build_mul(i_param, c10);
    b.build_store(Type::i64(), alloc, 8, scaled);

    Value* loop_done = b.build_sge(i_param, n);
    b.build_br_if(loop_done, not_found_exit, body);

    // body
    b.position_at_end(body);
    Value* is_target = b.build_eq(i_param, target);
    b.build_br_if(is_target, found_exit, next_iter);

    // next_iter
    b.position_at_end(next_iter);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i_param, one);
    b.build_br(header, {next_i});

    // found_exit: reads fields from escaping object
    b.position_at_end(found_exit);
    Value* f0 = b.build_load(Type::i64(), alloc, 0);
    Value* f8 = b.build_load(Type::i64(), alloc, 8);
    Value* c100 = b.build_iconst_i64(100);
    Value* part0 = b.build_mul(f0, c100);
    Value* found_res = b.build_add(part0, f8);
    b.build_ret(found_res);

    // not_found_exit: returns -1
    b.position_at_end(not_found_exit);
    Value* minus_one = b.build_iconst_i64(-1);
    b.build_ret(minus_one);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Optimize with Allocation Sinking
    AllocationSinkingOptions opts;
    bool changed = sink_allocations(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // Verify JIT vs Interpreter across diverse inputs
    assert_diff(mod, "loop_search", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(4)});
    assert_diff(mod, "loop_search", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(0)});
    assert_diff(mod, "loop_search", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(9)});
    assert_diff(mod, "loop_search", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(15)});
    assert_diff(mod, "loop_search", {RuntimeValue::from_i64(0), RuntimeValue::from_i64(0)});
}

TEST_CASE("Differential PEA - Multi-Field Loop Accumulator with Threshold Break") {
    Module mod("diff_pea_threshold");
    Builder b(mod);

    // func @loop_acc(%limit: i64, %threshold: i64) -> i64
    Function* fn = mod.create_function("loop_acc", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* loop_cont = b.append_block("loop_cont");
    BasicBlock* exit_early = b.append_block("exit_early");
    BasicBlock* exit_normal = b.append_block("exit_normal");

    Value* limit = b.add_block_param(entry, Type::i64());
    Value* threshold = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, zero); // sum
    b.build_store(Type::i64(), alloc, 8, zero); // sumsq
    b.build_store(Type::i64(), alloc, 16, zero); // count

    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* done = b.build_sge(i_param, limit);
    b.build_br_if(done, exit_normal, body);

    // body
    b.position_at_end(body);
    Value* s = b.build_load(Type::i64(), alloc, 0);
    Value* sq = b.build_load(Type::i64(), alloc, 8);
    Value* cnt = b.build_load(Type::i64(), alloc, 16);

    Value* new_s = b.build_add(s, i_param);
    Value* i_sq = b.build_mul(i_param, i_param);
    Value* new_sq = b.build_add(sq, i_sq);
    Value* one = b.build_iconst_i64(1);
    Value* new_cnt = b.build_add(cnt, one);

    b.build_store(Type::i64(), alloc, 0, new_s);
    b.build_store(Type::i64(), alloc, 8, new_sq);
    b.build_store(Type::i64(), alloc, 16, new_cnt);

    Value* over_thresh = b.build_sgt(new_s, threshold);
    b.build_br_if(over_thresh, exit_early, loop_cont);

    b.position_at_end(loop_cont);
    Value* next_i = b.build_add(i_param, one);
    b.build_br(header, {next_i});

    // exit_early: returns sum * 1000 + count
    b.position_at_end(exit_early);
    Value* es = b.build_load(Type::i64(), alloc, 0);
    Value* ec = b.build_load(Type::i64(), alloc, 16);
    Value* c1000 = b.build_iconst_i64(1000);
    Value* es_scaled = b.build_mul(es, c1000);
    Value* early_res = b.build_add(es_scaled, ec);
    b.build_ret(early_res);

    // exit_normal: returns sum + sumsq + count
    b.position_at_end(exit_normal);
    Value* ns = b.build_load(Type::i64(), alloc, 0);
    Value* nsq = b.build_load(Type::i64(), alloc, 8);
    Value* nc = b.build_load(Type::i64(), alloc, 16);
    Value* n_sum = b.build_add(ns, nsq);
    Value* norm_res = b.build_add(n_sum, nc);
    b.build_ret(norm_res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    AllocationSinkingOptions opts;
    bool changed = sink_allocations(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "loop_acc", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(15)});
    assert_diff(mod, "loop_acc", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(1000)});
    assert_diff(mod, "loop_acc", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(5)});
    assert_diff(mod, "loop_acc", {RuntimeValue::from_i64(0), RuntimeValue::from_i64(10)});
    assert_diff(mod, "loop_acc", {RuntimeValue::from_i64(20), RuntimeValue::from_i64(30)});
}

TEST_CASE("Differential PEA - Multi-Branch Loop with Conditional Escape") {
    Module mod("diff_pea_branch_escape");
    Builder b(mod);

    // func @branch_loop(%n: i64) -> i64
    Function* fn = mod.create_function("branch_loop", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* even_bb = b.append_block("even_bb");
    BasicBlock* odd_bb = b.append_block("odd_bb");
    BasicBlock* do_even = b.append_block("do_even");
    BasicBlock* merge_bb = b.append_block("merge_bb");
    BasicBlock* exit_bb = b.append_block("exit_bb");

    Value* n = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, zero);
    b.build_store(Type::i64(), alloc, 8, zero);

    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* is_done = b.build_sge(i_param, n);
    b.build_br_if(is_done, exit_bb, even_bb);

    // even_bb
    b.position_at_end(even_bb);
    Value* two = b.build_iconst_i64(2);
    Value* rem = b.build_smod(i_param, two);
    Value* is_even = b.build_eq(rem, zero);
    b.build_br_if(is_even, do_even, odd_bb);

    b.position_at_end(do_even);
    Value* old_0 = b.build_load(Type::i64(), alloc, 0);
    Value* add_even = b.build_add(old_0, i_param);
    b.build_store(Type::i64(), alloc, 0, add_even);
    b.build_br(merge_bb);

    // odd_bb
    b.position_at_end(odd_bb);
    Value* old_8 = b.build_load(Type::i64(), alloc, 8);
    Value* add_odd = b.build_add(old_8, i_param);
    b.build_store(Type::i64(), alloc, 8, add_odd);
    b.build_br(merge_bb);

    // merge_bb
    b.position_at_end(merge_bb);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i_param, one);
    b.build_br(header, {next_i});

    // exit_bb: returns f0 * 1000 + f8
    b.position_at_end(exit_bb);
    Value* f0 = b.build_load(Type::i64(), alloc, 0);
    Value* f8 = b.build_load(Type::i64(), alloc, 8);
    Value* c1000 = b.build_iconst_i64(1000);
    Value* res0 = b.build_mul(f0, c1000);
    Value* total = b.build_add(res0, f8);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    AllocationSinkingOptions opts;
    bool changed = sink_allocations(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "branch_loop", {RuntimeValue::from_i64(0)});
    assert_diff(mod, "branch_loop", {RuntimeValue::from_i64(1)});
    assert_diff(mod, "branch_loop", {RuntimeValue::from_i64(6)});
    assert_diff(mod, "branch_loop", {RuntimeValue::from_i64(11)});
    assert_diff(mod, "branch_loop", {RuntimeValue::from_i64(25)});
}
