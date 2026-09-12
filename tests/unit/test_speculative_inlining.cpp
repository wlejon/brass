#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/speculative_inliner.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;
using namespace brass::runtime;
using namespace brass::test;

namespace {

size_t count_opcodes_in_fn(const Function& fn, Opcode op) {
    size_t c = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == op) {
                c++;
            }
        }
    }
    return c;
}

} // namespace

TEST_CASE("SpeculativeInlining - Monomorphic call devirtualization with direct call") {
    Module mod("test_mono_devirt");

    // target_fn: (a: i64) -> a + 10
    Function* target_fn = mod.create_function("target_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(target_fn);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::i64());
        Value* c10 = b.build_iconst_i64(10);
        Value* sum = b.build_add(a, c10);
        b.build_ret(sum);
        target_fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*target_fn));
    }

    // caller_fn: (callee_ptr: ptr, arg: i64) -> call_indirect callee_ptr(arg)
    Function* caller_fn = mod.create_function("caller_fn", Type::i64(), {Type::ptr(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller_fn);
        BasicBlock* entry = b.append_block("entry");
        Value* callee_ptr = b.add_block_param(entry, Type::ptr());
        Value* arg = b.add_block_param(entry, Type::i64());
        Value* res = b.build_call_indirect(callee_ptr, Type::i64(), {arg});
        // Set explicit site_id = 1
        Instruction* call_inst = res->defining_instruction();
        REQUIRE(call_inst != nullptr);
        call_inst->set_site_id(1);
        b.build_ret(res);
        caller_fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller_fn));
    }

    // Register Type Feedback: monomorphic target_fn at site_id 1
    TypeFeedbackVector tfv("caller_fn");
    tfv.record_call_target(1, 0, "target_fn");
    REQUIRE(tfv.find_slot(1)->is_monomorphic());

    // Run speculative devirtualization (inlining disabled)
    SpeculativeInlinerOptions opts;
    opts.enable_inlining = false;
    bool changed = run_speculative_devirtualization(*caller_fn, mod, &tfv, opts);
    CHECK(changed);
    CHECK(verify_function(*caller_fn));

    // Assert IR structure:
    // - No call_indirect
    // - Has func_addr @target_fn
    // - Has eq
    // - Has guard
    // - Has direct call @target_fn
    CHECK_EQ(count_opcodes_in_fn(*caller_fn, Opcode::call_indirect), 0ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller_fn, Opcode::func_addr), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller_fn, Opcode::eq), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller_fn, Opcode::guard), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller_fn, Opcode::call), 1ULL);

    // Verify execution via Interpreter
    Interpreter interp;
    uintptr_t fn_ptr = reinterpret_cast<uintptr_t>(target_fn);
    interp.register_function_pointer(fn_ptr, target_fn);

    RuntimeValue result = interp.run(
        mod,
        "caller_fn",
        {RuntimeValue::from_ptr(fn_ptr), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(result.as_i64(), 15);
    CHECK(!interp.last_deopt().deoptimized);
}

TEST_CASE("SpeculativeInlining - Monomorphic speculative inlining of callee body") {
    Module mod("test_spec_inline");

    // small_leaf: (x: i64, y: i64) -> (x * y) + 7
    Function* small_leaf = mod.create_function("small_leaf", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(small_leaf);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        Value* prod = b.build_mul(x, y);
        Value* c7 = b.build_iconst_i64(7);
        Value* res = b.build_add(prod, c7);
        b.build_ret(res);
        small_leaf->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*small_leaf));
    }

    // caller: (fn_ptr: ptr, a: i64, b: i64) -> call_indirect fn_ptr(a, b)
    Function* caller = mod.create_function("caller_spec", Type::i64(), {Type::ptr(), Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* fn_ptr = b.add_block_param(entry, Type::ptr());
        Value* a = b.add_block_param(entry, Type::i64());
        Value* b_val = b.add_block_param(entry, Type::i64());
        Value* res = b.build_call_indirect(fn_ptr, Type::i64(), {a, b_val});
        Instruction* call_inst = res->defining_instruction();
        REQUIRE(call_inst != nullptr);
        call_inst->set_site_id(1);
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Register Type Feedback
    TypeFeedbackVector tfv("caller_spec");
    tfv.record_call_target(1, 0, "small_leaf");

    // Run speculative inlining (inlining enabled)
    SpeculativeInlinerOptions opts;
    opts.enable_inlining = true;
    bool changed = run_speculative_devirtualization(*caller, mod, &tfv, opts);
    CHECK(changed);
    CHECK(verify_function(*caller));

    // The call should be completely inlined!
    // - No call_indirect
    // - No direct call
    // - Guard is present
    // - Mul and Add from small_leaf are embedded directly
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::call_indirect), 0ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::call), 0ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::guard), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::mul), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::add), 1ULL);

    // Execute via Interpreter
    Interpreter interp;
    uintptr_t leaf_ptr = reinterpret_cast<uintptr_t>(small_leaf);
    interp.register_function_pointer(leaf_ptr, small_leaf);

    RuntimeValue result = interp.run(
        mod,
        "caller_spec",
        {RuntimeValue::from_ptr(leaf_ptr), RuntimeValue::from_i64(3), RuntimeValue::from_i64(4)}
    );
    // (3 * 4) + 7 = 19
    CHECK_EQ(result.as_i64(), 19);
    CHECK(!interp.last_deopt().deoptimized);
}

TEST_CASE("SpeculativeInlining - Deoptimization fallback on unexpected function pointer") {
    Module mod("test_deopt_fallback");

    // expected_fn: (x: i64) -> x + 10
    Function* expected_fn = mod.create_function("expected_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(expected_fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* sum = b.build_add(x, b.build_iconst_i64(10));
        b.build_ret(sum);
        expected_fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*expected_fn));
    }

    // unexpected_fn: (x: i64) -> x * 100
    Function* unexpected_fn = mod.create_function("unexpected_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(unexpected_fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* prod = b.build_mul(x, b.build_iconst_i64(100));
        b.build_ret(prod);
        unexpected_fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*unexpected_fn));
    }

    // caller: (fn_ptr: ptr, x: i64) -> call_indirect fn_ptr(x)
    Function* caller = mod.create_function("caller_deopt", Type::i64(), {Type::ptr(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* fn_ptr = b.add_block_param(entry, Type::ptr());
        Value* x = b.add_block_param(entry, Type::i64());
        Value* res = b.build_call_indirect(fn_ptr, Type::i64(), {x});
        res->defining_instruction()->set_site_id(1);
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Speculate on expected_fn
    TypeFeedbackVector tfv("caller_deopt");
    tfv.record_call_target(1, 0, "expected_fn");

    SpeculativeInlinerOptions opts;
    opts.enable_inlining = true;
    opts.deopt_stub_prefix = "@deopt_slow_call";
    bool changed = run_speculative_devirtualization(*caller, mod, &tfv, opts);
    CHECK(changed);
    CHECK(verify_function(*caller));

    Interpreter interp;
    uintptr_t exp_ptr = reinterpret_cast<uintptr_t>(expected_fn);
    uintptr_t unexp_ptr = reinterpret_cast<uintptr_t>(unexpected_fn);
    interp.register_function_pointer(exp_ptr, expected_fn);
    interp.register_function_pointer(unexp_ptr, unexpected_fn);

    // 1. Invocation with expected pointer -> guard succeeds
    RuntimeValue res_fast = interp.run(
        mod,
        "caller_deopt",
        {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(res_fast.as_i64(), 15);
    CHECK(!interp.last_deopt().deoptimized);

    // 2. Invocation with unexpected pointer -> guard fails and deopts
    interp.set_deopt_handler([](Interpreter&, const DeoptResult& deopt) -> RuntimeValue {
        REQUIRE_EQ(deopt.exit_stub, "@deopt_slow_call");
        return RuntimeValue::from_i64(9999);
    });

    RuntimeValue res_slow = interp.run(
        mod,
        "caller_deopt",
        {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(res_slow.as_i64(), 9999);
    CHECK(interp.last_deopt().deoptimized);
    CHECK_EQ(interp.last_deopt().exit_stub, "@deopt_slow_call");
}

TEST_CASE("SpeculativeInlining - Polymorphic 2-target call splitting dispatch diamond") {
    Module mod("test_poly_diamond");

    // fnA: (x: i64) -> x + 100
    Function* fnA = mod.create_function("fnA", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fnA);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(100)));
        fnA->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fnA));
    }

    // fnB: (x: i64) -> x + 200
    Function* fnB = mod.create_function("fnB", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fnB);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(200)));
        fnB->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fnB));
    }

    // fnC: (x: i64) -> x + 300 (unobserved target)
    Function* fnC = mod.create_function("fnC", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fnC);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(300)));
        fnC->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fnC));
    }

    // caller: (fn_ptr: ptr, x: i64) -> call_indirect fn_ptr(x)
    Function* caller = mod.create_function("caller_poly", Type::i64(), {Type::ptr(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* fn_ptr = b.add_block_param(entry, Type::ptr());
        Value* x = b.add_block_param(entry, Type::i64());
        Value* res = b.build_call_indirect(fn_ptr, Type::i64(), {x});
        res->defining_instruction()->set_site_id(1);
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Polymorphic feedback with 2 targets (Degree 2)
    TypeFeedbackVector tfv("caller_poly");
    tfv.record_call_target(1, 0, "fnA");
    tfv.record_call_target(1, 0, "fnB");
    REQUIRE(tfv.find_slot(1)->is_polymorphic());

    // Run specialization (inlining disabled so direct calls remain visible)
    SpeculativeInlinerOptions opts;
    opts.enable_inlining = false;
    opts.enable_polymorphic = true;
    bool changed = run_speculative_devirtualization(*caller, mod, &tfv, opts);
    CHECK(changed);
    CHECK(verify_function(*caller));

    // Verify CFG dispatch diamond:
    // - 2 direct calls (to fnA and fnB)
    // - 1 fallback call_indirect
    // - 2 conditional branches (br_if)
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::call), 2ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::call_indirect), 1ULL);
    CHECK_EQ(count_opcodes_in_fn(*caller, Opcode::br_if), 2ULL);

    // Verify execution across all 3 targets in Interpreter
    Interpreter interp;
    uintptr_t ptrA = reinterpret_cast<uintptr_t>(fnA);
    uintptr_t ptrB = reinterpret_cast<uintptr_t>(fnB);
    uintptr_t ptrC = reinterpret_cast<uintptr_t>(fnC);
    interp.register_function_pointer(ptrA, fnA);
    interp.register_function_pointer(ptrB, fnB);
    interp.register_function_pointer(ptrC, fnC);

    // Target A
    RuntimeValue resA = interp.run(
        mod,
        "caller_poly",
        {RuntimeValue::from_ptr(ptrA), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(resA.as_i64(), 105);

    // Target B
    RuntimeValue resB = interp.run(
        mod,
        "caller_poly",
        {RuntimeValue::from_ptr(ptrB), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(resB.as_i64(), 205);

    // Fallback: Target C (unobserved target handled by indirect call)
    RuntimeValue resC = interp.run(
        mod,
        "caller_poly",
        {RuntimeValue::from_ptr(ptrC), RuntimeValue::from_i64(5)}
    );
    CHECK_EQ(resC.as_i64(), 305);
}
