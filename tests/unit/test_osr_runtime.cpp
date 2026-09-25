// OSR built through the MIR Builder: a loop run by a program's fast
// interpreter moves into its OSR entry mid-loop, a disabled coordinator
// leaves it interpreted, and a guard failing inside the OSR code takes the
// exit the interpreter's guard takes, with the interpreter's result.
#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// Interpreted only (no tier-up by invocation), with OSR after `threshold`
// backedges when `osr` is set.
void init_program(FunctionDispatchTable& prog, bool osr, uint64_t threshold) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(osr);
    prog.osr().set_threshold(threshold);
}

// func @name(%n: i64) -> i64: the sum of 0 .. n-1.
Function* build_sum_loop(Module& mod, const char* name) {
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64()});
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
    return fn;
}

int64_t sum_below(int64_t n) { return (n - 1) * n / 2; }

} // namespace

TEST_CASE("OSR Runtime - Mid-Flight Tier-Up from Interpreter to Native OSR") {
    Module mod("runtime_osr_mod");
    Function* fn = build_sum_loop(mod, "sum_to_n");

    FunctionDispatchTable prog;
    init_program(prog, true, 50);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);

    // The first run asks for the loop's OSR code; the second, once it is
    // compiled, enters it.
    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(500)}).as_i64(), sum_below(500));
    CompilePool::shared().wait_owner(&prog.osr());
    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(5000)}).as_i64(), sum_below(5000));
    CHECK(prog.osr().total_osr_migrations() > 0);

    const TieringFeedback& fb = prog.tiering().get_or_create("sum_to_n");
    CHECK(fb.backedge_count() >= 50);
}

TEST_CASE("OSR Runtime - OSR Disabled Executes Pure Interpreter") {
    Module mod("pure_interp_mod");
    Function* fn = build_sum_loop(mod, "pure_loop");

    FunctionDispatchTable prog;
    init_program(prog, false, 10);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);

    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(200)}).as_i64(), sum_below(200));
    CompilePool::shared().wait_owner(&prog.osr());
    CHECK_EQ(interp.run(*fn, {RuntimeValue::from_i64(2000)}).as_i64(), sum_below(2000));
    CHECK_EQ(prog.osr().total_osr_migrations(), 0);
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
    //   guard %not_deopt, "loop_rest", [%i, %acc, %n] (resume_id = 1)
    //   %acc_next = add %acc, 10
    //   %i_next = add %i, 1
    //   jump b1(%i_next, %acc_next)
    // b3:
    //   ret %acc
    //
    // func @loop_rest(%i, %acc, %n) -> i64: the exit stub, finishing the
    // loop from the failed guard's state: %acc + 10 * (%n - %i).
    Function* rest = mod.create_function("loop_rest", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder rb(*rest);
        rb.position_at_end(rb.append_block("entry"));
        Value* ri = rb.add_param(Type::i64());
        Value* racc = rb.add_param(Type::i64());
        Value* rn = rb.add_param(Type::i64());
        rb.build_ret(rb.build_add(racc, rb.build_mul(rb.build_sub(rn, ri), rb.build_iconst_i64(10))));
    }

    Function* fn = mod.create_function("loop_with_guard", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

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
    Instruction* g = b.build_guard(not_deopt, "loop_rest", {i_val, acc_val, n});
    g->set_resume_id(1);
    Value* acc_next = b.build_add(acc_val, b.build_iconst_i64(10));
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, acc_next});

    b.position_at_end(b3);
    b.build_ret(acc_val);

    FunctionDispatchTable prog;
    init_program(prog, true, 30);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(&mod);

    // Warm-up: no guard failure, the loop's OSR code is asked for.
    const std::vector<RuntimeValue> warm{RuntimeValue::from_i64(200), RuntimeValue::from_i64(1000)};
    CHECK_EQ(interp.run(*fn, warm).as_i64(), 2000);
    CompilePool::shared().wait_owner(&prog.osr());

    // 1000 iterations entering the OSR code early; the guard fails at 600,
    // inside it, and its exit stub finishes the call.
    const uint64_t deopts_before = prog.pipeline().tier2_deopts();
    const std::vector<RuntimeValue> run{RuntimeValue::from_i64(1000), RuntimeValue::from_i64(600)};
    CHECK_EQ(interp.run(*fn, run).as_i64(), 10000);
    CHECK(prog.osr().total_osr_migrations() > 0);
    CHECK(prog.pipeline().tier2_deopts() > deopts_before);
}

#endif
