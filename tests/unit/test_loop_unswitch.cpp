#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

TEST_CASE("Loop Unswitch - Invariant Condition Flag") {
    Module mod("test_loop_unswitch_flag");
    Builder b(mod);

    Function* fn = mod.create_function("unswitch_flag", Type::i64(), {Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* flag = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* path_true = b.create_block("path_true");
    BasicBlock* path_false = b.create_block("path_false");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* ten = b.build_iconst_i64(10);
    Value* twenty = b.build_iconst_i64(20);

    b.build_br(loop_hdr, {zero, zero});

    // loop_hdr: i, acc
    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    // loop_body: test invariant flag
    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    b.build_br_if(flag, path_true, {}, path_false, {});

    // path_true: add 10
    fn->append_block(path_true);
    b.position_at_end(path_true);
    Value* acc_t = b.build_add(acc, ten);
    b.build_br(latch, {acc_t});

    // path_false: add 20
    fn->append_block(path_false);
    b.position_at_end(path_false);
    Value* acc_f = b.build_add(acc, twenty);
    b.build_br(latch, {acc_f});

    // latch: takes next acc
    fn->append_block(latch);
    b.position_at_end(latch);
    Value* acc_next = b.add_block_param(latch, Type::i64());
    Value* i_next = b.build_add(i, one);
    b.build_br(loop_hdr, {i_next, acc_next});

    // exit
    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    CHECK_EQ(la.top_level_loops().size(), 1);

    LoopInfo* loop = la.top_level_loops()[0].get();
    CHECK(loop->is_loop_invariant(flag));

    LoopUnswitchOptions opts;
    bool unswitched = unswitch_loop(*fn, *loop, dom, opts);
    CHECK(unswitched);
    CHECK(verify_function(*fn));

    // Evaluate via interpreter: flag = 1 => 5 * 10 = 50
    Interpreter interp;
    std::vector<RuntimeValue> args_t = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)};
    RuntimeValue res_t = interp.run(*fn, args_t);
    CHECK_EQ(res_t.as_i64(), 50);

    // Evaluate via interpreter: flag = 0 => 5 * 20 = 100
    std::vector<RuntimeValue> args_f = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
    RuntimeValue res_f = interp.run(*fn, args_f);
    CHECK_EQ(res_f.as_i64(), 100);
}

TEST_CASE("Loop Unswitch - Loop Accumulators and Exit Values") {
    Module mod("test_loop_unswitch_accum");
    Builder b(mod);

    Function* fn = mod.create_function("unswitch_accum", Type::i64(), {Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* count = b.add_block_param(entry, Type::i64());
    Value* mode = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_work = b.create_block("loop_work");
    BasicBlock* mode_a = b.create_block("mode_a");
    BasicBlock* mode_b = b.create_block("mode_b");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    Value* three = b.build_iconst_i64(3);

    b.build_br(loop_hdr, {zero, zero, one});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_hdr, Type::i64());
    Value* prod = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_work, {}, exit_bb, {sum, prod});

    fn->append_block(loop_work);
    b.position_at_end(loop_work);
    b.build_br_if(mode, mode_a, {}, mode_b, {});

    fn->append_block(mode_a);
    b.position_at_end(mode_a);
    Value* sum_a = b.build_add(sum, two);
    Value* prod_a = b.build_mul(prod, two);
    b.build_br(latch, {sum_a, prod_a});

    fn->append_block(mode_b);
    b.position_at_end(mode_b);
    Value* sum_b = b.build_add(sum, three);
    Value* prod_b = b.build_mul(prod, three);
    b.build_br(latch, {sum_b, prod_b});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* next_sum = b.add_block_param(latch, Type::i64());
    Value* next_prod = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_sum, next_prod});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* out_sum = b.add_block_param(exit_bb, Type::i64());
    Value* out_prod = b.add_block_param(exit_bb, Type::i64());
    Value* total = b.build_add(out_sum, out_prod);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    bool changed = unswitch_loops_in_function(*fn);
    CHECK(changed);
    CHECK(verify_function(*fn));

    // N=3, mode=1:
    // sum: 0 + 2 + 2 + 2 = 6
    // prod: 1 * 2 * 2 * 2 = 8
    // total = 14
    Interpreter interp;
    RuntimeValue res_a = interp.run(*fn, {RuntimeValue::from_i64(3), RuntimeValue::from_i32(1)});
    CHECK_EQ(res_a.as_i64(), 14);

    // N=3, mode=0:
    // sum: 0 + 3 + 3 + 3 = 9
    // prod: 1 * 3 * 3 * 3 = 27
    // total = 36
    RuntimeValue res_b = interp.run(*fn, {RuntimeValue::from_i64(3), RuntimeValue::from_i32(0)});
    CHECK_EQ(res_b.as_i64(), 36);
}

TEST_CASE("Loop Unswitch - Nested Loop Unswitching") {
    Module mod("test_loop_unswitch_nested");
    Builder b(mod);

    Function* fn = mod.create_function("nested_unswitch", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* N = b.add_block_param(entry, Type::i64());
    Value* M = b.add_block_param(entry, Type::i64());
    Value* flag = b.add_block_param(entry, Type::i32());

    BasicBlock* outer_hdr = b.create_block("outer_hdr");
    BasicBlock* outer_body = b.create_block("outer_body");
    BasicBlock* inner_hdr = b.create_block("inner_hdr");
    BasicBlock* inner_body = b.create_block("inner_body");
    BasicBlock* inner_t = b.create_block("inner_t");
    BasicBlock* inner_f = b.create_block("inner_f");
    BasicBlock* inner_latch = b.create_block("inner_latch");
    BasicBlock* outer_latch = b.create_block("outer_latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* five = b.build_iconst_i64(5);
    Value* seven = b.build_iconst_i64(7);

    b.build_br(outer_hdr, {zero, zero});

    fn->append_block(outer_hdr);
    b.position_at_end(outer_hdr);
    Value* i = b.add_block_param(outer_hdr, Type::i64());
    Value* outer_acc = b.add_block_param(outer_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, outer_body, {}, exit_bb, {outer_acc});

    fn->append_block(outer_body);
    b.position_at_end(outer_body);
    b.build_br(inner_hdr, {zero, outer_acc});

    fn->append_block(inner_hdr);
    b.position_at_end(inner_hdr);
    Value* j = b.add_block_param(inner_hdr, Type::i64());
    Value* inner_acc = b.add_block_param(inner_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, M);
    b.build_br_if(cond_j, inner_body, {}, outer_latch, {inner_acc});

    fn->append_block(inner_body);
    b.position_at_end(inner_body);
    b.build_br_if(flag, inner_t, {}, inner_f, {});

    fn->append_block(inner_t);
    b.position_at_end(inner_t);
    Value* acc_t = b.build_add(inner_acc, five);
    b.build_br(inner_latch, {acc_t});

    fn->append_block(inner_f);
    b.position_at_end(inner_f);
    Value* acc_f = b.build_add(inner_acc, seven);
    b.build_br(inner_latch, {acc_f});

    fn->append_block(inner_latch);
    b.position_at_end(inner_latch);
    Value* next_inner_acc = b.add_block_param(inner_latch, Type::i64());
    Value* next_j = b.build_add(j, one);
    b.build_br(inner_hdr, {next_j, next_inner_acc});

    fn->append_block(outer_latch);
    b.position_at_end(outer_latch);
    Value* next_outer_acc = b.add_block_param(outer_latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(outer_hdr, {next_i, next_outer_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    bool ok = unswitch_loops_in_function(*fn);
    CHECK(ok);

    CHECK(verify_function(*fn));

    // Check with interpreter: N=2, M=3, flag=1 => 2 * 3 * 5 = 30
    Interpreter interp;
    RuntimeValue r1 = interp.run(*fn, {RuntimeValue::from_i64(2), RuntimeValue::from_i64(3), RuntimeValue::from_i32(1)});
    CHECK_EQ(r1.as_i64(), 30);

    // N=2, M=3, flag=0 => 2 * 3 * 7 = 42
    RuntimeValue r0 = interp.run(*fn, {RuntimeValue::from_i64(2), RuntimeValue::from_i64(3), RuntimeValue::from_i32(0)});
    CHECK_EQ(r0.as_i64(), 42);
}

TEST_CASE("Loop Unswitch - Size Budget Cutoff Check") {
    Module mod("test_loop_unswitch_budget");
    Builder b(mod);

    Function* fn = mod.create_function("budget_fn", Type::i64(), {Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* flag = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* path_t = b.create_block("path_t");
    BasicBlock* path_f = b.create_block("path_f");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    // Add several instructions to increase size
    Value* v1 = b.build_add(acc, one);
    Value* v2 = b.build_add(v1, one);
    Value* v3 = b.build_add(v2, one);
    (void)v3;
    b.build_br_if(flag, path_t, {}, path_f, {});

    fn->append_block(path_t);
    b.position_at_end(path_t);
    b.build_br(latch, {v1});

    fn->append_block(path_f);
    b.position_at_end(path_f);
    b.build_br(latch, {v2});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* next_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    LoopInfo* loop = la.top_level_loops()[0].get();

    // With very small instruction budget cutoff (e.g. 2 instructions):
    LoopUnswitchOptions small_opts;
    small_opts.max_loop_instructions = 2;
    bool rejected = unswitch_loop(*fn, *loop, dom, small_opts);
    CHECK(!rejected);

    // With generous instruction budget (e.g. 100):
    LoopUnswitchOptions generous_opts;
    generous_opts.max_loop_instructions = 100;
    bool accepted = unswitch_loop(*fn, *loop, dom, generous_opts);
    CHECK(accepted);
    CHECK(verify_function(*fn));
}
