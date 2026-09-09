#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

TEST_CASE("Jump Threading - Parameter-based Diamond Join") {
    Module mod("test_jt_diamond");
    Builder b(mod);

    Function* fn = mod.create_function("diamond_threading", Type::i64(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());

    BasicBlock* left = b.create_block("left");
    BasicBlock* right = b.create_block("right");
    BasicBlock* join_bb = b.create_block("join_bb");
    BasicBlock* path_true = b.create_block("path_true");
    BasicBlock* path_false = b.create_block("path_false");

    b.build_br_if(x, left, {}, right, {});

    // left passes 1 to join_bb
    fn->append_block(left);
    b.position_at_end(left);
    Value* one_32 = b.build_iconst_i32(1);
    b.build_br(join_bb, {one_32});

    // right passes 0 to join_bb
    fn->append_block(right);
    b.position_at_end(right);
    Value* zero_32 = b.build_iconst_i32(0);
    b.build_br(join_bb, {zero_32});

    // join_bb: tests flag
    fn->append_block(join_bb);
    b.position_at_end(join_bb);
    Value* flag = b.add_block_param(join_bb, Type::i32());
    b.build_br_if(flag, path_true, {}, path_false, {});

    // path_true
    fn->append_block(path_true);
    b.position_at_end(path_true);
    Value* ret111 = b.build_iconst_i64(111);
    b.build_ret(ret111);

    // path_false
    fn->append_block(path_false);
    b.position_at_end(path_false);
    Value* ret222 = b.build_iconst_i64(222);
    b.build_ret(ret222);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    bool threaded = run_jump_threading(*fn);
    CHECK(threaded);
    CHECK(verify_function(*fn));

    // Left should now branch directly to path_true
    CHECK_EQ(left->terminator()->branch_target().block, path_true);
    // Right should now branch directly to path_false
    CHECK_EQ(right->terminator()->branch_target().block, path_false);

    // Execution check
    Interpreter interp;
    RuntimeValue res_t = interp.run(*fn, {RuntimeValue::from_i32(1)});
    CHECK_EQ(res_t.as_i64(), 111);

    RuntimeValue res_f = interp.run(*fn, {RuntimeValue::from_i32(0)});
    CHECK_EQ(res_f.as_i64(), 222);
}

TEST_CASE("Jump Threading - Icmp Condition with Incoming Constant") {
    Module mod("test_jt_icmp");
    Builder b(mod);

    Function* fn = mod.create_function("icmp_threading", Type::i64(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* selector = b.add_block_param(entry, Type::i32());

    BasicBlock* path_a = b.create_block("path_a");
    BasicBlock* path_b = b.create_block("path_b");
    BasicBlock* check_bb = b.create_block("check_bb");
    BasicBlock* target_ten = b.create_block("target_ten");
    BasicBlock* target_other = b.create_block("target_other");

    b.build_br_if(selector, path_a, {}, path_b, {});

    // path_a passes 10 to check_bb
    fn->append_block(path_a);
    b.position_at_end(path_a);
    Value* ten = b.build_iconst_i64(10);
    b.build_br(check_bb, {ten});

    // path_b passes 20 to check_bb
    fn->append_block(path_b);
    b.position_at_end(path_b);
    Value* twenty = b.build_iconst_i64(20);
    b.build_br(check_bb, {twenty});

    // check_bb: compares val == 10
    fn->append_block(check_bb);
    b.position_at_end(check_bb);
    Value* val = b.add_block_param(check_bb, Type::i64());
    Value* expected_ten = b.build_iconst_i64(10);
    Value* is_ten = b.build_eq(val, expected_ten);
    b.build_br_if(is_ten, target_ten, {}, target_other, {});

    // target_ten returns 100
    fn->append_block(target_ten);
    b.position_at_end(target_ten);
    Value* ret100 = b.build_iconst_i64(100);
    b.build_ret(ret100);

    // target_other returns 200
    fn->append_block(target_other);
    b.position_at_end(target_other);
    Value* ret200 = b.build_iconst_i64(200);
    b.build_ret(ret200);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    bool threaded = run_jump_threading(*fn);
    CHECK(threaded);
    CHECK(verify_function(*fn));

    Interpreter interp;
    RuntimeValue r_a = interp.run(*fn, {RuntimeValue::from_i32(1)});
    CHECK_EQ(r_a.as_i64(), 100);

    RuntimeValue r_b = interp.run(*fn, {RuntimeValue::from_i32(0)});
    CHECK_EQ(r_b.as_i64(), 200);
}

TEST_CASE("Jump Threading - Loop Latch Jump Threading") {
    Module mod("test_jt_latch");
    Builder b(mod);

    // fn(count, skip_fast) -> i64
    Function* fn = mod.create_function("latch_threading", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* check_step = b.create_block("check_step");
    BasicBlock* step_small = b.create_block("step_small");
    BasicBlock* step_large = b.create_block("step_large");
    BasicBlock* latch_join = b.create_block("latch_join");
    BasicBlock* loop_continue = b.create_block("loop_continue");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    Value* five = b.build_iconst_i64(5);

    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, check_step, {}, exit_bb, {sum});

    fn->append_block(check_step);
    b.position_at_end(check_step);
    Value* is_even = b.build_slt(i, five);
    b.build_br_if(is_even, step_small, {}, step_large, {});

    // step_small adds 1 and passes flag=1 to latch_join
    fn->append_block(step_small);
    b.position_at_end(step_small);
    Value* s1 = b.build_add(sum, one);
    Value* flag1 = b.build_iconst_i32(1);
    b.build_br(latch_join, {s1, flag1});

    // step_large adds 2 and passes flag=0 to latch_join
    fn->append_block(step_large);
    b.position_at_end(step_large);
    Value* s2 = b.build_add(sum, two);
    Value* flag0 = b.build_iconst_i32(0);
    b.build_br(latch_join, {s2, flag0});

    // latch_join: checks flag
    fn->append_block(latch_join);
    b.position_at_end(latch_join);
    Value* cur_sum = b.add_block_param(latch_join, Type::i64());
    Value* f = b.add_block_param(latch_join, Type::i32());
    b.build_br_if(f, loop_continue, {cur_sum}, loop_continue, {cur_sum});

    fn->append_block(loop_continue);
    b.position_at_end(loop_continue);
    Value* final_sum = b.add_block_param(loop_continue, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, final_sum});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    bool changed = run_jump_threading(*fn);
    CHECK(changed);
    CHECK(verify_function(*fn));

    Interpreter interp_latch;
    RuntimeValue r = interp_latch.run(*fn, {RuntimeValue::from_i64(10)});
    // First 5 iterations add 1 (5), next 5 iterations add 2 (10) => total 15
    CHECK_EQ(r.as_i64(), 15);
}
