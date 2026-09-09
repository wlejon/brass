#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential - Loop Unswitch with Mode Flag") {
    Module mod("diff_unswitch_mode");
    Builder b(mod);

    // func @compute_mode(%n: i64, %mode: i32, %scale: i64) -> i64
    Function* fn = mod.create_function("compute_mode", Type::i64(),
        {Type::i64(), Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* mode = b.add_block_param(entry, Type::i32());
    Value* scale = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* mode_fast = b.create_block("mode_fast");
    BasicBlock* mode_slow = b.create_block("mode_slow");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* seven = b.build_iconst_i64(7);
    Value* thirteen = b.build_iconst_i64(13);

    b.build_br(loop_hdr, {zero, zero});

    // Header
    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    // Body: invariant branch on mode
    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* is_fast = b.build_eq(mode, b.build_iconst_i32(1));
    b.build_br_if(is_fast, mode_fast, {}, mode_slow, {});

    // Fast path: acc = acc + scale * 7 + i
    fn->append_block(mode_fast);
    b.position_at_end(mode_fast);
    Value* term_fast = b.build_mul(scale, seven);
    Value* term_fast_i = b.build_add(term_fast, i);
    Value* acc_f = b.build_add(acc, term_fast_i);
    b.build_br(latch, {acc_f});

    // Slow path: acc = acc + scale * 13 - i
    fn->append_block(mode_slow);
    b.position_at_end(mode_slow);
    Value* term_slow = b.build_mul(scale, thirteen);
    Value* term_slow_i = b.build_sub(term_slow, i);
    Value* acc_s = b.build_add(acc, term_slow_i);
    b.build_br(latch, {acc_s});

    // Latch
    fn->append_block(latch);
    b.position_at_end(latch);
    Value* next_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    // Exit
    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_acc = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_acc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Optimize with loop unswitch
    LoopUnswitchOptions unsw_opts;
    unswitch_loops_in_module(mod, unsw_opts);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "compute_mode", {RuntimeValue::from_i64(10), RuntimeValue::from_i32(1), RuntimeValue::from_i64(3)});
    assert_diff(mod, "compute_mode", {RuntimeValue::from_i64(10), RuntimeValue::from_i32(0), RuntimeValue::from_i64(3)});
    assert_diff(mod, "compute_mode", {RuntimeValue::from_i64(25), RuntimeValue::from_i32(1), RuntimeValue::from_i64(5)});
    assert_diff(mod, "compute_mode", {RuntimeValue::from_i64(25), RuntimeValue::from_i32(0), RuntimeValue::from_i64(5)});
}

TEST_CASE("Differential - Jump Threading Diamond Join") {
    Module mod("diff_threading");
    Builder b(mod);

    Function* fn = mod.create_function("diamond_calc", Type::i64(),
        {Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* selector = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* branch_left = b.create_block("branch_left");
    BasicBlock* branch_right = b.create_block("branch_right");
    BasicBlock* diamond_join = b.create_block("diamond_join");
    BasicBlock* path_positive = b.create_block("path_positive");
    BasicBlock* path_negative = b.create_block("path_negative");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* ten = b.build_iconst_i64(10);
    Value* fifty = b.build_iconst_i64(50);

    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* is_left = b.build_slt(i, ten);
    b.build_br_if(is_left, branch_left, {}, branch_right, {});

    // branch_left forwards flag=1 and acc_add=10
    fn->append_block(branch_left);
    b.position_at_end(branch_left);
    Value* flag1 = b.build_iconst_i32(1);
    b.build_br(diamond_join, {acc, flag1});

    // branch_right forwards flag=0 and acc_add=50
    fn->append_block(branch_right);
    b.position_at_end(branch_right);
    Value* flag0 = b.build_iconst_i32(0);
    b.build_br(diamond_join, {acc, flag0});

    // diamond_join tests flag
    fn->append_block(diamond_join);
    b.position_at_end(diamond_join);
    Value* in_acc = b.add_block_param(diamond_join, Type::i64());
    Value* in_flag = b.add_block_param(diamond_join, Type::i32());
    b.build_br_if(in_flag, path_positive, {in_acc}, path_negative, {in_acc});

    fn->append_block(path_positive);
    b.position_at_end(path_positive);
    Value* p_acc = b.add_block_param(path_positive, Type::i64());
    Value* next_pos = b.build_add(p_acc, ten);
    b.build_br(latch, {next_pos});

    fn->append_block(path_negative);
    b.position_at_end(path_negative);
    Value* n_acc = b.add_block_param(path_negative, Type::i64());
    Value* next_neg = b.build_add(n_acc, fifty);
    b.build_br(latch, {next_neg});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* acc_latch = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, acc_latch});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_acc = b.add_block_param(exit_bb, Type::i64());
    Value* sel_64 = b.build_sext_i64(selector);
    Value* result = b.build_add(final_acc, sel_64);
    b.build_ret(result);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Optimize with jump threading
    JumpThreadingOptions jt_opts;
    jump_thread_module(mod, jt_opts);
    cfg_simplify_module(mod);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "diamond_calc", {RuntimeValue::from_i64(15), RuntimeValue::from_i32(3)});
    assert_diff(mod, "diamond_calc", {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)});
    assert_diff(mod, "diamond_calc", {RuntimeValue::from_i64(25), RuntimeValue::from_i32(-7)});
}

TEST_CASE("Differential - End-to-End Pipeline with Unswitch, Threading, and Block Layout") {
    Module mod("diff_pipeline_e2e");
    Builder b(mod);

    // Multi-stage kernel computing polynomial reduction with nested invariant control
    Function* fn = mod.create_function("pipeline_kernel", Type::i64(),
        {Type::i64(), Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* N = b.add_block_param(entry, Type::i64());
    Value* flag = b.add_block_param(entry, Type::i32());
    Value* bias = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* path_fast = b.create_block("path_fast");
    BasicBlock* path_slow = b.create_block("path_slow");
    BasicBlock* join_stage = b.create_block("join_stage");
    BasicBlock* stage_even = b.create_block("stage_even");
    BasicBlock* stage_odd = b.create_block("stage_odd");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    Value* four = b.build_iconst_i64(4);

    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, N);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    b.build_br_if(flag, path_fast, {}, path_slow, {});

    fn->append_block(path_fast);
    b.position_at_end(path_fast);
    Value* f_val = b.build_mul(i, four);
    Value* fast_code = b.build_iconst_i32(1);
    b.build_br(join_stage, {f_val, fast_code});

    fn->append_block(path_slow);
    b.position_at_end(path_slow);
    Value* s_val = b.build_mul(i, two);
    Value* slow_code = b.build_iconst_i32(0);
    b.build_br(join_stage, {s_val, slow_code});

    fn->append_block(join_stage);
    b.position_at_end(join_stage);
    Value* stage_v = b.add_block_param(join_stage, Type::i64());
    Value* code = b.add_block_param(join_stage, Type::i32());
    b.build_br_if(code, stage_even, {stage_v}, stage_odd, {stage_v});

    fn->append_block(stage_even);
    b.position_at_end(stage_even);
    Value* ev_in = b.add_block_param(stage_even, Type::i64());
    Value* ev_res = b.build_add(ev_in, bias);
    b.build_br(latch, {ev_res});

    fn->append_block(stage_odd);
    b.position_at_end(stage_odd);
    Value* od_in = b.add_block_param(stage_odd, Type::i64());
    Value* od_res = b.build_sub(od_in, bias);
    b.build_br(latch, {od_res});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* term = b.add_block_param(latch, Type::i64());
    Value* next_acc = b.build_add(acc, term);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Full optimization pipeline with unswitch and jump threading
    LoopOptOptions opts;
    opts.enable_loop_unswitch = true;
    opts.enable_jump_threading = true;
    opts.enable_trace_layout = true;
    optimize_module(mod, opts);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "pipeline_kernel", {RuntimeValue::from_i64(10), RuntimeValue::from_i32(1), RuntimeValue::from_i64(5)});
    assert_diff(mod, "pipeline_kernel", {RuntimeValue::from_i64(10), RuntimeValue::from_i32(0), RuntimeValue::from_i64(5)});
    assert_diff(mod, "pipeline_kernel", {RuntimeValue::from_i64(20), RuntimeValue::from_i32(1), RuntimeValue::from_i64(11)});
    assert_diff(mod, "pipeline_kernel", {RuntimeValue::from_i64(20), RuntimeValue::from_i32(0), RuntimeValue::from_i64(11)});
}
