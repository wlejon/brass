#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>

using namespace brass;
using namespace brass::runtime;

TEST_CASE("OSR Runtime - Mid-Flight Tier-Up from Interpreter to Native OSR") {
    Module mod("runtime_osr_mod");

    // func @sum_to_n(%n: i64) -> i64
    // b0:
    //   jump b1(0, 0)
    // b1(%i: i64, %sum: i64):
    //   %cond = cmp.slt %i, %n
    //   br %cond, b2, b3
    // b2:
    //   %sum_next = add %sum, %i
    //   %i_next = add %i, 1
    //   jump b1(%i_next, %sum_next)
    // b3:
    //   ret %sum
    Function* fn = mod.create_function("sum_to_n", Type::i64(), {Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* sum_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* sum_next = b.build_add(sum_val, i_val);
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, sum_next});

    b.position_at_end(b3);
    b.build_ret(sum_val);

    OsrCoordinator& coord = OsrCoordinator::instance();
    coord.clear_cache();
    coord.reset_stats();
    coord.set_enabled(true);
    coord.set_threshold(50); // Tier-up on 50th backedge

    Interpreter interp;
    interp.set_module(&mod);

    uint64_t n_val = 500;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(static_cast<int64_t>(n_val))});

    uint64_t expected = (n_val - 1) * n_val / 2; // 0 + 1 + ... + 499 = 124750
    CHECK_EQ(res.as_i64(), static_cast<int64_t>(expected));
    CHECK(coord.total_osr_migrations() > 0);

    const TieringFeedback& fb = TieringRegistry::instance().get_or_create("sum_to_n");
    CHECK(fb.backedge_count() >= 50);

    coord.set_enabled(false);
    coord.set_threshold(BACKEDGE_OSR_THRESHOLD);
    coord.clear_cache();
}

TEST_CASE("OSR Runtime - OSR Disabled Executes Pure Interpreter") {
    Module mod("pure_interp_mod");

    Function* fn = mod.create_function("pure_loop", Type::i64(), {Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* sum_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* sum_next = b.build_add(sum_val, i_val);
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, sum_next});

    b.position_at_end(b3);
    b.build_ret(sum_val);

    OsrCoordinator& coord = OsrCoordinator::instance();
    coord.clear_cache();
    coord.reset_stats();
    coord.set_enabled(false);

    Interpreter interp;
    interp.set_module(&mod);

    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(200)});
    CHECK_EQ(res.as_i64(), 199 * 200 / 2);
    CHECK_EQ(coord.total_osr_migrations(), 0);
}

TEST_CASE("OSR Runtime - Bi-Directional Deoptimization Resume back to Interpreter") {
    Module mod("deopt_osr_mod");

    // func @loop_with_guard(%n: i64, %deopt_at: i64) -> i64
    // b0:
    //   jump b1(0, 0)
    // b1(%i: i64, %acc: i64):
    //   %cond = cmp.slt %i, %n
    //   br %cond, b2, b3
    // b2:
    //   %not_deopt = cmp.slt %i, %deopt_at
    //   guard %not_deopt, "", [%i, %acc] (resume_id = 1)
    //   %acc_next = add %acc, 10
    //   %i_next = add %i, 1
    //   jump b1(%i_next, %acc_next)
    // b_resume (resume_id = 1):
    //   %acc_next_slow = add %acc, 10
    //   %i_next_slow = add %i, 1
    //   jump b1(%i_next_slow, %acc_next_slow)
    // b3:
    //   ret %acc
    Function* fn = mod.create_function("loop_with_guard", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b_resume = b.append_block("b_resume");
    BasicBlock* b3 = b.append_block("b3");

    fn->add_resume_point(1, b_resume);

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    Value* deopt_at = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* acc_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* not_deopt = b.build_slt(i_val, deopt_at);
    Instruction* g = b.build_guard(not_deopt, "", {i_val, acc_val});
    g->set_resume_id(1);
    Value* acc_next = b.build_add(acc_val, b.build_iconst_i64(10));
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, acc_next});

    b.position_at_end(b_resume);
    Value* acc_next_slow = b.build_add(acc_val, b.build_iconst_i64(10));
    Value* i_next_slow = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next_slow, acc_next_slow});

    b.position_at_end(b3);
    b.build_ret(acc_val);

    OsrCoordinator& coord = OsrCoordinator::instance();
    coord.clear_cache();
    coord.reset_stats();
    coord.set_enabled(true);
    coord.set_threshold(30); // OSR tier up at iteration 30

    Interpreter interp;
    interp.set_module(&mod);

    // Total iterations 100, guard fails at iteration 60
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(100), RuntimeValue::from_i64(60)});

    // 100 iterations * 10 = 1000
    CHECK_EQ(res.as_i64(), 1000);
    CHECK(coord.total_osr_migrations() > 0);
    CHECK(coord.total_native_deopts() > 0);

    coord.set_enabled(false);
    coord.set_threshold(BACKEDGE_OSR_THRESHOLD);
    coord.clear_cache();
}
