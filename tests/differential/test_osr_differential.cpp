#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/vm/fast_interpreter.hpp>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// Runs fn(arg) on a program's fast interpreter with OSR after `threshold`
// backedges until a run has entered the loop's OSR code (the first runs ask
// for it, and it compiles in the background). Every run answers the first
// run's result; the last run's is returned.
RuntimeValue run_until_osr(const Function& fn, int64_t arg, uint64_t threshold, uint64_t& migrations) {
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(threshold);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);

    const RuntimeValue first = interp.run(fn, {RuntimeValue::from_i64(arg)});
    RuntimeValue last = first;
    for (int run = 0; run < 16 && prog.osr().total_osr_migrations() == 0; ++run) {
        CompilePool::shared().wait_owner(&prog.osr());
        last = interp.run(fn, {RuntimeValue::from_i64(arg)});
        CHECK_EQ(last.as_i64(), first.as_i64());
    }
    migrations = prog.osr().total_osr_migrations();
    return last;
}

} // namespace

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

    // 1. Pure Interpreter
    Interpreter interp_pure;
    interp_pure.set_module(&mod);
    RuntimeValue res_pure = interp_pure.run(*fn, {RuntimeValue::from_i64(n_iters)});

    // 2. Full JIT AOT
    HostEngine engine;
    auto compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);
    RuntimeValue res_jit = compiled->invoke("diff_acc", {RuntimeValue::from_i64(n_iters)});

    // 3. OSR JIT (tier-up mid-flight)
    uint64_t migrations = 0;
    RuntimeValue res_osr = run_until_osr(*fn, n_iters, 40, migrations);

    // Result Equivalence
    CHECK_EQ(res_pure.as_i64(), res_jit.as_i64());
    CHECK_EQ(res_pure.as_i64(), res_osr.as_i64());
    CHECK(migrations > 0);
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
    Interpreter interp_pure;
    interp_pure.set_module(&mod);
    RuntimeValue res_pure = interp_pure.run(*fn, {RuntimeValue::from_i64(fib_n)});

    // Full JIT
    HostEngine engine;
    auto compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);
    RuntimeValue res_jit = compiled->invoke("fib_iter", {RuntimeValue::from_i64(fib_n)});

    // OSR JIT
    uint64_t migrations = 0;
    RuntimeValue res_osr = run_until_osr(*fn, fib_n, 15, migrations);

    CHECK_EQ(res_pure.as_i64(), res_jit.as_i64());
    CHECK_EQ(res_pure.as_i64(), res_osr.as_i64());
    CHECK(migrations > 0);
}

#endif
