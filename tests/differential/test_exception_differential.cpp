#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>

using namespace brass;

TEST_CASE("Differential - Exception Unwinding Interpreter vs JIT") {
    Module mod("diff_eh");
    Builder b(mod);

    // callee: func @work(%val: i64) -> i64
    // if %val < 0 throw %val
    // else return %val * 3 + 1
    Function* callee = mod.create_function("work", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    BasicBlock* c_throw = b.append_block("c_throw");
    BasicBlock* c_ok = b.append_block("c_ok");

    b.position_at_end(c_entry);
    Value* val = b.add_block_param(c_entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    Value* is_neg = b.build_slt(val, zero);
    b.build_br_if(is_neg, c_throw, c_ok);

    b.position_at_end(c_throw);
    b.build_throw(val);

    b.position_at_end(c_ok);
    Value* three = b.build_iconst_i64(3);
    Value* one = b.build_iconst_i64(1);
    Value* p = b.build_mul(val, three);
    Value* s = b.build_add(p, one);
    b.build_ret(s);
    callee->rebuild_cfg_predecessors();

    // caller: func @run_work(%x: i64) -> i64
    // try { return work(%x); } catch (%e) { return %e * -10; }
    Function* caller = mod.create_function("run_work", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("work", Type::i64(), {x}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* minus_ten = b.build_iconst_i64(-10);
    Value* catch_res = b.build_mul(exc, minus_ten);
    b.build_ret(catch_res);
    caller->rebuild_cfg_predecessors();

    // Compile in JIT
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));

    typedef int64_t (*RunFn)(int64_t);
    RunFn jit_fn = jit.get_function_ptr<RunFn>("run_work");
    REQUIRE(jit_fn != nullptr);

    Interpreter interp;

    // Test a sweep of positive, negative, and zero values
    std::vector<int64_t> test_inputs = { 0, 1, 5, 10, 42, -1, -5, -12, -99 };

    for (int64_t input : test_inputs) {
        // Run Interpreter
        RuntimeValue interp_res = interp.run(*caller, {RuntimeValue::from_i64(input)});
        int64_t interp_val = interp_res.as_i64();

        // Run JIT
        int64_t jit_val = jit_fn(input);

        CHECK_EQ(interp_val, jit_val);
    }
}
