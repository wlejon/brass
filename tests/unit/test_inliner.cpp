#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/call_graph.hpp>
#include <brass/mir/devirtualize.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Inliner - Leaf function inlining with arithmetic") {
    Module mod("test_leaf_inline");

    // callee: (a: i64, b: i64) -> (a + b) * 3
    Function* callee = mod.create_function("math_kernel", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(callee);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::i64());
        Value* b_val = b.add_block_param(entry, Type::i64());
        Value* sum = b.build_add(a, b_val);
        Value* three = b.build_iconst_i64(3);
        Value* prod = b.build_mul(sum, three);
        b.build_ret(prod);
        callee->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*callee));
    }

    // caller: (x: i64, y: i64) -> math_kernel(x, y) + 5
    Function* caller = mod.create_function("caller_fn", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        Value* k_res = b.build_call("math_kernel", Type::i64(), {x, y});
        Value* five = b.build_iconst_i64(5);
        Value* res = b.build_add(k_res, five);
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Pre-inlining test via interpreter
    Interpreter interp;
    RuntimeValue pre_val = interp.run(mod, "caller_fn", {RuntimeValue::from_i64(4), RuntimeValue::from_i64(6)});
    CHECK_EQ(pre_val.as_i64(), ((4 + 6) * 3) + 5);

    // CallGraph check
    CallGraph cg(mod);
    CHECK(cg.is_leaf(callee));
    CHECK(!cg.is_leaf(caller));
    CHECK(!cg.is_recursive(caller));
    CHECK(!cg.is_recursive(callee));

    // Run inliner
    bool changed = inline_function(*caller, mod);
    CHECK(changed);

    // Verify caller after inlining
    DiagnosticReporter diag;
    bool ok_ver = verify_function(*caller, &diag);
    if (!ok_ver) {
        std::cerr << diag.format_all() << "\n";
    }
    REQUIRE(ok_ver);

    // Ensure call instruction was eliminated
    size_t call_count = 0;
    for (BasicBlock* bb : caller->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) ++call_count;
        }
    }
    CHECK_EQ(call_count, 0);

    // Post-inlining execution: interpreter and JIT
    RuntimeValue post_val = interp.run(mod, "caller_fn", {RuntimeValue::from_i64(4), RuntimeValue::from_i64(6)});
    CHECK_EQ(post_val.as_i64(), 35);

    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    RuntimeValue jit_val = jit.invoke("caller_fn", {RuntimeValue::from_i64(4), RuntimeValue::from_i64(6)});
    CHECK_EQ(jit_val.as_i64(), 35);
}

TEST_CASE("Inliner - Multiple arguments and multiple return blocks") {
    Module mod("test_multi_ret");

    // callee_max3(a, b, c) -> returns max(a, max(b, c))
    // Uses branches and multiple blocks with ret
    Function* callee = mod.create_function("max3", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(callee);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* a_ge_b = b.create_block("a_ge_b");
        BasicBlock* b_gt_a = b.create_block("b_gt_a");
        BasicBlock* ret_a = b.create_block("ret_a");
        BasicBlock* ret_c1 = b.create_block("ret_c1");
        BasicBlock* ret_b = b.create_block("ret_b");
        BasicBlock* ret_c2 = b.create_block("ret_c2");

        callee->append_block(a_ge_b);
        callee->append_block(b_gt_a);
        callee->append_block(ret_a);
        callee->append_block(ret_c1);
        callee->append_block(ret_b);
        callee->append_block(ret_c2);

        Value* a = b.add_block_param(entry, Type::i64());
        Value* b_val = b.add_block_param(entry, Type::i64());
        Value* c = b.add_block_param(entry, Type::i64());

        Value* cmp_ab = b.build_sge(a, b_val);
        b.build_br_if(cmp_ab, a_ge_b, b_gt_a);

        // a_ge_b: compare a with c
        b.position_at_end(a_ge_b);
        Value* cmp_ac = b.build_sge(a, c);
        b.build_br_if(cmp_ac, ret_a, ret_c1);

        // b_gt_a: compare b with c
        b.position_at_end(b_gt_a);
        Value* cmp_bc = b.build_sge(b_val, c);
        b.build_br_if(cmp_bc, ret_b, ret_c2);

        b.position_at_end(ret_a);
        b.build_ret(a);

        b.position_at_end(ret_c1);
        b.build_ret(c);

        b.position_at_end(ret_b);
        b.build_ret(b_val);

        b.position_at_end(ret_c2);
        b.build_ret(c);

        callee->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*callee));
    }

    // caller: (x, y, z) -> max3(x, y, z) * 2
    Function* caller = mod.create_function("run_max", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        Value* z = b.add_block_param(entry, Type::i64());
        Value* m = b.build_call("max3", Type::i64(), {x, y, z});
        Value* two = b.build_iconst_i64(2);
        Value* res = b.build_mul(m, two);
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Inlining into caller
    bool changed = inline_function(*caller, mod);
    CHECK(changed);

    DiagnosticReporter diag;
    bool ok_ver = verify_function(*caller, &diag);
    if (!ok_ver) {
        std::cerr << diag.format_all() << "\n";
    }
    REQUIRE(ok_ver);

    // Test executions across all branch cases
    Interpreter interp;
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));

    struct TestCase { int64_t x, y, z, expected; };
    std::vector<TestCase> cases = {
        {10, 5, 2, 20},
        {3, 15, 8, 30},
        {1, 4, 25, 50},
        {42, 42, 10, 84},
        {-5, -1, -10, -2}
    };

    for (const auto& tc : cases) {
        std::vector<RuntimeValue> args = {
            RuntimeValue::from_i64(tc.x),
            RuntimeValue::from_i64(tc.y),
            RuntimeValue::from_i64(tc.z)
        };
        RuntimeValue iv = interp.run(mod, "run_max", args);
        RuntimeValue jv = jit.invoke("run_max", args);
        CHECK_EQ(iv.as_i64(), tc.expected);
        CHECK_EQ(jv.as_i64(), tc.expected);
    }
}

TEST_CASE("Inliner - Inlining inside nested loops") {
    Module mod("test_nested_loops_inline");

    // mac_kernel: (acc: i64, a: i64, b: i64) -> acc + (a * b)
    Function* kernel = mod.create_function("mac_kernel", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(kernel);
        BasicBlock* entry = b.append_block("entry");
        Value* acc = b.add_block_param(entry, Type::i64());
        Value* a = b.add_block_param(entry, Type::i64());
        Value* b_val = b.add_block_param(entry, Type::i64());
        Value* prod = b.build_mul(a, b_val);
        Value* res = b.build_add(acc, prod);
        b.build_ret(res);
        kernel->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*kernel));
    }

    // caller: nested loop over i in [0, n) and j in [0, m)
    Function* caller = mod.create_function("matrix_accum", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* outer_hdr = b.create_block("outer_hdr");
        BasicBlock* inner_hdr = b.create_block("inner_hdr");
        BasicBlock* inner_body = b.create_block("inner_body");
        BasicBlock* inner_exit = b.create_block("inner_exit");
        BasicBlock* exit_bb = b.create_block("exit_bb");

        caller->append_block(outer_hdr);
        caller->append_block(inner_hdr);
        caller->append_block(inner_body);
        caller->append_block(inner_exit);
        caller->append_block(exit_bb);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* m = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* one = b.build_iconst_i64(1);
        b.build_br(outer_hdr, {zero, zero}); // (i, acc)

        // outer_hdr: params [i, acc]
        b.position_at_end(outer_hdr);
        Value* i_val = b.add_block_param(outer_hdr, Type::i64());
        Value* acc_outer = b.add_block_param(outer_hdr, Type::i64());
        Value* cond_outer = b.build_slt(i_val, n);
        b.build_br_if(cond_outer, inner_hdr, {zero, acc_outer}, exit_bb, {acc_outer});

        // inner_hdr: params [j, acc_inner]
        b.position_at_end(inner_hdr);
        Value* j_val = b.add_block_param(inner_hdr, Type::i64());
        Value* acc_inner = b.add_block_param(inner_hdr, Type::i64());
        Value* cond_inner = b.build_slt(j_val, m);
        b.build_br_if(cond_inner, inner_body, {}, inner_exit, {acc_inner});

        // inner_body: calls mac_kernel(acc_inner, i_val, j_val)
        b.position_at_end(inner_body);
        Value* new_acc = b.build_call("mac_kernel", Type::i64(), {acc_inner, i_val, j_val});
        Value* next_j = b.build_add(j_val, one);
        b.build_br(inner_hdr, {next_j, new_acc});

        // inner_exit: param [acc_after_inner]
        b.position_at_end(inner_exit);
        Value* acc_after = b.add_block_param(inner_exit, Type::i64());
        Value* next_i = b.build_add(i_val, one);
        b.build_br(outer_hdr, {next_i, acc_after});

        // exit_bb: param [final_acc]
        b.position_at_end(exit_bb);
        Value* final_acc = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(final_acc);

        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    InlinerOptions opts;
    opts.enable_loop_priority = true;
    opts.loop_call_priority_bonus = 4.0;
    bool changed = inline_function(*caller, mod, opts);
    CHECK(changed);

    REQUIRE(verify_function(*caller));

    // Ensure call instruction inside loop body was completely inlined
    size_t call_count = 0;
    for (BasicBlock* bb : caller->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) ++call_count;
        }
    }
    CHECK_EQ(call_count, 0);

    // Compute expected result for n=4, m=5: sum_{i=0..3, j=0..4} (i*j) = (0+1+2+3)*(0+1+2+3+4) = 6 * 10 = 60
    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "matrix_accum", {RuntimeValue::from_i64(4), RuntimeValue::from_i64(5)});
    CHECK_EQ(iv.as_i64(), 60);

    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    RuntimeValue jv = jit.invoke("matrix_accum", {RuntimeValue::from_i64(4), RuntimeValue::from_i64(5)});
    CHECK_EQ(jv.as_i64(), 60);
}

TEST_CASE("Inliner - Void-returning functions") {
    Module mod("test_void_inline");

    // void_store(ptr: ptr, val: i64) -> void
    Function* callee = mod.create_function("void_store", Type::void_type(), {Type::ptr(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(callee);
        BasicBlock* entry = b.append_block("entry");
        Value* p = b.add_block_param(entry, Type::ptr());
        Value* v = b.add_block_param(entry, Type::i64());
        b.build_store(Type::i64(), p, 0, v);
        b.build_ret_void();
        callee->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*callee));
    }

    // caller: stores to ptr and returns value
    Function* caller = mod.create_function("caller_void", Type::i64(), {Type::ptr()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* p = b.add_block_param(entry, Type::ptr());
        Value* c99 = b.build_iconst_i64(99);
        b.build_call("void_store", Type::void_type(), {p, c99});
        Value* loaded = b.build_load(Type::i64(), p, 0);
        b.build_ret(loaded);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    bool changed = inline_function(*caller, mod);
    CHECK(changed);
    REQUIRE(verify_function(*caller));

    size_t call_count = 0;
    for (BasicBlock* bb : caller->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) ++call_count;
        }
    }
    CHECK_EQ(call_count, 0);

    // Test with buffer
    int64_t buf = 0;
    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "caller_void", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(&buf))});
    CHECK_EQ(iv.as_i64(), 99);
    CHECK_EQ(buf, 99);

    buf = 0;
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    RuntimeValue jv = jit.invoke("caller_void", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(&buf))});
    CHECK_EQ(jv.as_i64(), 99);
    CHECK_EQ(buf, 99);
}

TEST_CASE("Inliner - Cycle and recursion safety") {
    Module mod("test_cycles");

    // Mutual recursion: ping(n) and pong(n)
    Function* ping = mod.create_function("ping", Type::i64(), {Type::i64()});
    Function* pong = mod.create_function("pong", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(ping);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* b_base = b.create_block("base");
        BasicBlock* b_rec = b.create_block("rec");
        ping->append_block(b_base);
        ping->append_block(b_rec);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* cond = b.build_sle(n, zero);
        b.build_br_if(cond, b_base, b_rec);

        b.position_at_end(b_base);
        b.build_ret(zero);

        b.position_at_end(b_rec);
        Value* one = b.build_iconst_i64(1);
        Value* n1 = b.build_sub(n, one);
        Value* res = b.build_call("pong", Type::i64(), {n1});
        Value* out = b.build_add(res, one);
        b.build_ret(out);
        ping->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*ping));
    }

    {
        Builder b(mod);
        b.set_function(pong);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* b_base = b.create_block("base");
        BasicBlock* b_rec = b.create_block("rec");
        pong->append_block(b_base);
        pong->append_block(b_rec);

        Value* n = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* cond = b.build_sle(n, zero);
        b.build_br_if(cond, b_base, b_rec);

        b.position_at_end(b_base);
        b.build_ret(zero);

        b.position_at_end(b_rec);
        Value* one = b.build_iconst_i64(1);
        Value* n1 = b.build_sub(n, one);
        Value* res = b.build_call("ping", Type::i64(), {n1});
        Value* out = b.build_add(res, one);
        b.build_ret(out);
        pong->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*pong));
    }

    CallGraph cg(mod);
    CHECK(cg.is_recursive(ping));
    CHECK(cg.is_recursive(pong));

    // Attempt module inlining: must NOT explode or infinitely recurse
    bool changed = inline_module(mod);
    (void)changed;

    REQUIRE(verify_function(*ping));
    REQUIRE(verify_function(*pong));

    // Execution check
    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "ping", {RuntimeValue::from_i64(5)});
    CHECK_EQ(iv.as_i64(), 5);

    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    RuntimeValue jv = jit.invoke("ping", {RuntimeValue::from_i64(5)});
    CHECK_EQ(jv.as_i64(), 5);
}

TEST_CASE("Inliner - Size budget and growth factor cutoffs") {
    Module mod("test_budget");

    // callee: large function with 45 instructions
    Function* large_callee = mod.create_function("large_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(large_callee);
        BasicBlock* entry = b.append_block("entry");
        Value* cur = b.add_block_param(entry, Type::i64());
        for (int i = 0; i < 40; ++i) {
            Value* c = b.build_iconst_i64(i + 1);
            cur = b.build_add(cur, c);
        }
        b.build_ret(cur);
        large_callee->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*large_callee));
    }

    // caller: calls large_fn
    Function* caller = mod.create_function("budget_caller", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* res = b.build_call("large_fn", Type::i64(), {x});
        b.build_ret(res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    InlinerOptions opts;
    opts.leaf_instruction_threshold = 20; // 20 < 45, so large_callee should be rejected!
    opts.enable_loop_priority = false;

    bool changed = inline_function(*caller, mod, opts);
    CHECK(!changed); // Rejected due to budget

    // Now raise threshold so it is accepted
    opts.leaf_instruction_threshold = 100;
    changed = inline_function(*caller, mod, opts);
    CHECK(changed);
    REQUIRE(verify_function(*caller));
}

TEST_CASE("Inliner - Devirtualization of monomorphic patchable_call") {
    Module mod("test_devirt");

    Function* target = mod.create_function("devirt_target", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(target);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* ten = b.build_iconst_i64(10);
        Value* res = b.build_add(x, ten);
        b.build_ret(res);
        target->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*target));
    }

    Function* caller = mod.create_function("devirt_caller", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* c_res = b.build_patchable_call("call_ic_site_0", "devirt_target", Type::i64(), {x});
        b.build_ret(c_res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    // Devirtualize first
    bool devirt_ok = devirtualize_module(mod);
    CHECK(devirt_ok);

    // Caller instruction should now be Opcode::call instead of patchable_call
    bool found_direct_call = false;
    for (BasicBlock* bb : caller->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == "devirt_target") {
                found_direct_call = true;
            }
        }
    }
    CHECK(found_direct_call);

    // Full IPO optimization
    bool ipo_ok = optimize_module_ipo(mod);
    CHECK(ipo_ok);
    REQUIRE(verify_function(*caller));

    // Execution check
    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "devirt_caller", {RuntimeValue::from_i64(32)});
    CHECK_EQ(iv.as_i64(), 42);

    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));
    RuntimeValue jv = jit.invoke("devirt_caller", {RuntimeValue::from_i64(32)});
    CHECK_EQ(jv.as_i64(), 42);
}
