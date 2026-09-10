#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/runtime/osr_coordinator.hpp>

using namespace brass;
using namespace brass::runtime;

TEST_CASE("OSR Differential - Accumulator Loop: Pure Interpreter vs Full JIT vs OSR JIT") {
    Module mod("diff_acc_mod");

    Function* fn = mod.create_function("diff_acc", Type::i64(), {Type::i64()});
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
    Value* acc_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* step = b.build_add(i_val, b.build_iconst_i64(1));
    Value* acc_next = b.build_add(acc_val, step);
    b.build_br(b1, {step, acc_next});

    b.position_at_end(b3);
    b.build_ret(acc_val);

    int64_t n_iters = 300;

    // 1. Pure Interpreter (OSR disabled)
    OsrCoordinator::instance().clear_cache();
    OsrCoordinator::instance().reset_stats();
    OsrCoordinator::instance().set_enabled(false);

    Interpreter interp_pure;
    interp_pure.set_module(&mod);
    RuntimeValue res_pure = interp_pure.run(*fn, {RuntimeValue::from_i64(n_iters)});

    // 2. Full JIT AOT
    HostEngine engine;
    auto compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);
    RuntimeValue res_jit = compiled->invoke("diff_acc", {RuntimeValue::from_i64(n_iters)});

    // 3. OSR JIT (Tier-up mid-flight at iteration 40)
    OsrCoordinator::instance().clear_cache();
    OsrCoordinator::instance().reset_stats();
    OsrCoordinator::instance().set_enabled(true);
    OsrCoordinator::instance().set_threshold(40);

    Interpreter interp_osr;
    interp_osr.set_module(&mod);
    RuntimeValue res_osr = interp_osr.run(*fn, {RuntimeValue::from_i64(n_iters)});

    // Result Equivalence
    CHECK_EQ(res_pure.as_i64(), res_jit.as_i64());
    CHECK_EQ(res_pure.as_i64(), res_osr.as_i64());
    CHECK(OsrCoordinator::instance().total_osr_migrations() > 0);

    OsrCoordinator::instance().set_enabled(false);
    OsrCoordinator::instance().set_threshold(BACKEDGE_OSR_THRESHOLD);
    OsrCoordinator::instance().clear_cache();
}

TEST_CASE("OSR Differential - Fibonacci Iterative: Pure Interpreter vs Full JIT vs OSR JIT") {
    Module mod("diff_fib_mod");

    // func @fib_iter(%n: i64) -> i64
    Function* fn = mod.create_function("fib_iter", Type::i64(), {Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0), b.build_iconst_i64(1)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* a_val = b.add_param(Type::i64());
    Value* b_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* next_b = b.build_add(a_val, b_val);
    Value* next_i = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {next_i, b_val, next_b});

    b.position_at_end(b3);
    b.build_ret(a_val);

    int64_t fib_n = 45;

    // Pure interpreter
    OsrCoordinator::instance().clear_cache();
    OsrCoordinator::instance().set_enabled(false);
    Interpreter interp_pure;
    interp_pure.set_module(&mod);
    RuntimeValue res_pure = interp_pure.run(*fn, {RuntimeValue::from_i64(fib_n)});

    // Full JIT
    HostEngine engine;
    auto compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);
    RuntimeValue res_jit = compiled->invoke("fib_iter", {RuntimeValue::from_i64(fib_n)});

    // OSR JIT
    OsrCoordinator::instance().clear_cache();
    OsrCoordinator::instance().reset_stats();
    OsrCoordinator::instance().set_enabled(true);
    OsrCoordinator::instance().set_threshold(15);
    Interpreter interp_osr;
    interp_osr.set_module(&mod);
    RuntimeValue res_osr = interp_osr.run(*fn, {RuntimeValue::from_i64(fib_n)});

    CHECK_EQ(res_pure.as_i64(), res_jit.as_i64());
    CHECK_EQ(res_pure.as_i64(), res_osr.as_i64());
    CHECK(OsrCoordinator::instance().total_osr_migrations() > 0);

    OsrCoordinator::instance().set_enabled(false);
    OsrCoordinator::instance().set_threshold(BACKEDGE_OSR_THRESHOLD);
    OsrCoordinator::instance().clear_cache();
}
