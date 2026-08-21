#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <vector>

using namespace brass;

TEST_CASE("Moving GC Native - Linked List Allocation and Safepoints") {
    Module mod("moving_gc_linked_list");
    mod.add_external_symbol("brass_gc_alloc");
    mod.add_external_symbol("brass_gc_safepoint");

    Function* fn = mod.create_function("test_list_moving", Type::i64(), {});
    {
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask0 = b.build_iconst_i64(0);
        Value* mask2 = b.build_iconst_i64(2); // bit 1 is gcref (offset 8)
        Value* tag1 = b.build_iconst_i32(1);

        // Node 1: val = 10, next = null
        Value* n1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
        Value* v10 = b.build_iconst_i64(10);
        b.build_store(Type::i64(), n1, 0, v10);

        b.build_safepoint();

        // Node 2: val = 20, next = n1
        Value* n2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask2, tag1});
        Value* v20 = b.build_iconst_i64(20);
        b.build_store(Type::i64(), n2, 0, v20);
        b.build_store(Type::gcref(), n2, 8, n1);

        b.build_safepoint();

        // Node 3: val = 30, next = n2
        Value* n3 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask2, tag1});
        Value* v30 = b.build_iconst_i64(30);
        b.build_store(Type::i64(), n3, 0, v30);
        b.build_store(Type::gcref(), n3, 8, n2);

        b.build_safepoint();

        // Node 4: val = 40, next = n3
        Value* n4 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask2, tag1});
        Value* v40 = b.build_iconst_i64(40);
        b.build_store(Type::i64(), n4, 0, v40);
        b.build_store(Type::gcref(), n4, 8, n3);

        b.build_safepoint();

        // Node 5: val = 50, next = n4
        Value* n5 = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask2, tag1});
        Value* v50 = b.build_iconst_i64(50);
        b.build_store(Type::i64(), n5, 0, v50);
        b.build_store(Type::gcref(), n5, 8, n4);

        b.build_safepoint();

        // Read all 5 nodes starting from n5 and accumulate sum
        Value* read_n5_val = b.build_load(Type::i64(), n5, 0);
        Value* acc = read_n5_val;

        Value* read_n4 = b.build_load(Type::gcref(), n5, 8);
        Value* read_n4_val = b.build_load(Type::i64(), read_n4, 0);
        acc = b.build_add(acc, read_n4_val);

        Value* read_n3 = b.build_load(Type::gcref(), read_n4, 8);
        Value* read_n3_val = b.build_load(Type::i64(), read_n3, 0);
        acc = b.build_add(acc, read_n3_val);

        Value* read_n2 = b.build_load(Type::gcref(), read_n3, 8);
        Value* read_n2_val = b.build_load(Type::i64(), read_n2, 0);
        acc = b.build_add(acc, read_n2_val);

        Value* read_n1 = b.build_load(Type::gcref(), read_n2, 8);
        Value* read_n1_val = b.build_load(Type::i64(), read_n1, 0);
        acc = b.build_add(acc, read_n1_val);

        b.build_ret(acc);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    // 1. Interpreter differential baseline
    {
        Interpreter interp(128 * 1024);
        interp.gc().set_stress_mode(true);
        RuntimeValue interp_res = interp.run(mod, "test_list_moving", {});
        CHECK_EQ(interp_res.as_i64(), 150LL);
        CHECK(interp.gc().collection_count() >= 5ULL);
    }

    // 2. Native JIT Execution with Cheney Moving Collector & Stack Walking
    {
        MiniCheneyGC gc(128 * 1024);
        gc.set_stress_mode(true);
        brass_set_active_gc(&gc);

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        brass_set_active_stack_maps(&jit.stack_maps());

        RuntimeValue native_res = jit.invoke("test_list_moving", {});
        CHECK_EQ(native_res.as_i64(), 150LL);
        CHECK(gc.collection_count() >= 5ULL);
    }
}

TEST_CASE("Moving GC Native - Binary Tree Construction and Evacuation") {
    Module mod("moving_gc_tree");
    mod.add_external_symbol("brass_gc_alloc");
    mod.add_external_symbol("brass_gc_safepoint");

    Function* fn = mod.create_function("test_tree_moving", Type::i64(), {});
    {
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");

        Value* sz24 = b.build_iconst_i64(24);
        Value* mask0 = b.build_iconst_i64(0);
        Value* mask6 = b.build_iconst_i64(6); // bit 1 (left at 8) and bit 2 (right at 16)
        Value* tag2 = b.build_iconst_i32(2);

        // Allocate 4 leaves
        Value* l1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag2});
        Value* vl1 = b.build_iconst_i64(100);
        b.build_store(Type::i64(), l1, 0, vl1);

        Value* l2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag2});
        Value* vl2 = b.build_iconst_i64(200);
        b.build_store(Type::i64(), l2, 0, vl2);

        Value* l3 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag2});
        Value* vl3 = b.build_iconst_i64(300);
        b.build_store(Type::i64(), l3, 0, vl3);

        Value* l4 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag2});
        Value* vl4 = b.build_iconst_i64(400);
        b.build_store(Type::i64(), l4, 0, vl4);

        b.build_safepoint();

        // Allocate 2 branch nodes
        Value* b1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask6, tag2});
        Value* vb1 = b.build_iconst_i64(10);
        b.build_store(Type::i64(), b1, 0, vb1);
        b.build_store(Type::gcref(), b1, 8, l1);
        b.build_store(Type::gcref(), b1, 16, l2);

        Value* b2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask6, tag2});
        Value* vb2 = b.build_iconst_i64(20);
        b.build_store(Type::i64(), b2, 0, vb2);
        b.build_store(Type::gcref(), b2, 8, l3);
        b.build_store(Type::gcref(), b2, 16, l4);

        b.build_safepoint();

        // Allocate root
        Value* root = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask6, tag2});
        Value* vr = b.build_iconst_i64(1);
        b.build_store(Type::i64(), root, 0, vr);
        b.build_store(Type::gcref(), root, 8, b1);
        b.build_store(Type::gcref(), root, 16, b2);

        b.build_safepoint();

        // Read all 7 tree nodes through root and accumulate sum
        Value* val_r = b.build_load(Type::i64(), root, 0);
        Value* acc = val_r;

        Value* rb1 = b.build_load(Type::gcref(), root, 8);
        Value* val_b1 = b.build_load(Type::i64(), rb1, 0);
        acc = b.build_add(acc, val_b1);

        Value* rl1 = b.build_load(Type::gcref(), rb1, 8);
        Value* val_l1 = b.build_load(Type::i64(), rl1, 0);
        acc = b.build_add(acc, val_l1);

        Value* rl2 = b.build_load(Type::gcref(), rb1, 16);
        Value* val_l2 = b.build_load(Type::i64(), rl2, 0);
        acc = b.build_add(acc, val_l2);

        Value* rb2 = b.build_load(Type::gcref(), root, 16);
        Value* val_b2 = b.build_load(Type::i64(), rb2, 0);
        acc = b.build_add(acc, val_b2);

        Value* rl3 = b.build_load(Type::gcref(), rb2, 8);
        Value* val_l3 = b.build_load(Type::i64(), rl3, 0);
        acc = b.build_add(acc, val_l3);

        Value* rl4 = b.build_load(Type::gcref(), rb2, 16);
        Value* val_l4 = b.build_load(Type::i64(), rl4, 0);
        acc = b.build_add(acc, val_l4);

        b.build_ret(acc);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    // 1. Interpreter
    {
        Interpreter interp(128 * 1024);
        interp.gc().set_stress_mode(true);
        RuntimeValue interp_res = interp.run(mod, "test_tree_moving", {});
        // 1 + 10 + 100 + 200 + 20 + 300 + 400 = 1031
        CHECK_EQ(interp_res.as_i64(), 1031LL);
    }

    // 2. Native JIT
    {
        MiniCheneyGC gc(128 * 1024);
        gc.set_stress_mode(true);
        brass_set_active_gc(&gc);

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        brass_set_active_stack_maps(&jit.stack_maps());

        RuntimeValue native_res = jit.invoke("test_tree_moving", {});
        CHECK_EQ(native_res.as_i64(), 1031LL);
    }
}

TEST_CASE("Moving GC Native - Cross-Function Call Roots and Callee Registers") {
    Module mod("moving_gc_cross_fn");
    mod.add_external_symbol("brass_gc_alloc");
    mod.add_external_symbol("brass_gc_safepoint");

    // helper_alloc(val: i64) -> gcref
    Function* fn_helper = mod.create_function("helper_alloc", Type::gcref(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_helper);
        BasicBlock* entry = b.append_block("entry");
        Value* v = b.add_block_param(entry, Type::i64());

        Value* sz = b.build_iconst_i64(16);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(3);
        Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz, mask, tag});
        b.build_store(Type::i64(), obj, 0, v);
        b.build_safepoint();
        b.build_ret(obj);

        fn_helper->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_helper));
    }

    // process_pair(x: i64, y: i64) -> i64
    Function* fn_main = mod.create_function("process_pair", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_main);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());

        // Allocate objA with value x
        Value* objA = b.build_call("helper_alloc", Type::gcref(), {x});

        // Allocate objB with value y (while objA is held across the call to helper_alloc!)
        Value* objB = b.build_call("helper_alloc", Type::gcref(), {y});

        // Explicit safepoint while BOTH objA and objB are live!
        b.build_safepoint();

        // Read and add their values
        Value* valA = b.build_load(Type::i64(), objA, 0);
        Value* valB = b.build_load(Type::i64(), objB, 0);
        Value* res = b.build_add(valA, valB);
        b.build_ret(res);

        fn_main->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_main));
    }

    // 1. Interpreter
    {
        Interpreter interp(128 * 1024);
        interp.gc().set_stress_mode(true);
        RuntimeValue interp_res = interp.run(mod, "process_pair", {RuntimeValue::from_i64(1234), RuntimeValue::from_i64(5678)});
        CHECK_EQ(interp_res.as_i64(), 6912LL);
    }

    // 2. Native JIT
    {
        MiniCheneyGC gc(128 * 1024);
        gc.set_stress_mode(true);
        brass_set_active_gc(&gc);

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        brass_set_active_stack_maps(&jit.stack_maps());

        RuntimeValue native_res = jit.invoke("process_pair", {RuntimeValue::from_i64(1234), RuntimeValue::from_i64(5678)});
        CHECK_EQ(native_res.as_i64(), 6912LL);
        CHECK(gc.collection_count() >= 3ULL);
    }
}
