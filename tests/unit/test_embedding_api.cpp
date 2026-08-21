#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/embedding/brass_c_api.h>
#include <vector>
#include <string>

using namespace brass;

TEST_CASE("Embedding API - NaN-Box Value Representation and Operations") {
    // 1. Double values
    HostValue v_dbl1 = HostValue::from_f64(3.1415926535);
    CHECK(v_dbl1.is_f64());
    CHECK(v_dbl1.is_double());
    CHECK(v_dbl1.is_number());
    CHECK(!v_dbl1.is_i32());
    CHECK(!v_dbl1.is_gcref());
    CHECK_EQ(v_dbl1.type(), HostValueType::Double);
    CHECK(std::abs(v_dbl1.as_f64() - 3.1415926535) < 1e-9);

    HostValue v_dbl_neg = HostValue::from_f64(-42.5);
    CHECK(v_dbl_neg.is_f64());
    CHECK_EQ(v_dbl_neg.as_f64(), -42.5);

    HostValue v_dbl_zero = HostValue::from_f64(0.0);
    CHECK(v_dbl_zero.is_f64());
    CHECK_EQ(v_dbl_zero.as_f64(), 0.0);

    // 2. Int32 values
    HostValue v_i1 = HostValue::from_i32(42);
    CHECK(v_i1.is_i32());
    CHECK(v_i1.is_int32());
    CHECK(v_i1.is_number());
    CHECK(!v_i1.is_f64());
    CHECK_EQ(v_i1.type(), HostValueType::Int32);
    CHECK_EQ(v_i1.as_i32(), 42);
    CHECK_EQ(v_i1.as_f64(), 42.0);

    HostValue v_i_neg = HostValue::from_i32(-1000);
    CHECK(v_i_neg.is_i32());
    CHECK_EQ(v_i_neg.as_i32(), -1000);

    // 3. Boolean values
    HostValue v_true = HostValue::from_bool(true);
    HostValue v_false = HostValue::from_bool(false);
    CHECK(v_true.is_bool());
    CHECK(v_false.is_bool());
    CHECK_EQ(v_true.type(), HostValueType::Bool);
    CHECK(v_true.as_bool());
    CHECK(!v_false.as_bool());

    // 4. Null and Undefined
    HostValue v_null = HostValue::null_val();
    HostValue v_undef = HostValue::undefined_val();
    CHECK(v_null.is_null());
    CHECK(!v_null.is_undefined());
    CHECK_EQ(v_null.type(), HostValueType::Null);

    CHECK(v_undef.is_undefined());
    CHECK(!v_undef.is_null());
    CHECK_EQ(v_undef.type(), HostValueType::Undefined);

    // 5. GCRef / Object references
    uintptr_t dummy_heap_addr = 0x00007FFE12345678ULL;
    HostValue v_obj = HostValue::from_gcref(dummy_heap_addr);
    CHECK(v_obj.is_gcref());
    CHECK(v_obj.is_object());
    CHECK(!v_obj.is_f64());
    CHECK_EQ(v_obj.type(), HostValueType::GCRef);
    CHECK_EQ(v_obj.as_gcref(), dummy_heap_addr);

    // In-place update
    uintptr_t new_heap_addr = 0x00007FFE87654321ULL;
    v_obj.update_gcref(new_heap_addr);
    CHECK(v_obj.is_gcref());
    CHECK_EQ(v_obj.as_gcref(), new_heap_addr);

    // 6. Equality and String formatting
    CHECK_EQ(v_i1, HostValue::from_i32(42));
    CHECK_NE(v_i1, HostValue::from_i32(43));
    CHECK_EQ(v_null, HostValue::null_val());
    CHECK_EQ(v_undef, HostValue::undefined_val());
    CHECK_EQ(v_null.to_string(), "null");
    CHECK_EQ(v_undef.to_string(), "undefined");
    CHECK_EQ(v_true.to_string(), "true");
}

TEST_CASE("Embedding API - HostEngine In-Memory Compilation and Function Pointers") {
    HostEngine engine;

    // Register a host callback: host_mul(x: i64) -> i64
    auto host_mul = [](int64_t x) -> int64_t {
        return x * 10;
    };
    engine.register_external_symbol("host_mul", reinterpret_cast<void*>(+host_mul));

    Module mod("test_host_engine_mod");
    mod.add_external_symbol("host_mul");

    // func @compute(x: i64, y: i64) -> i64
    //   %t = call @host_mul(%x)
    //   %res = add %t, %y
    //   ret %res
    Function* fn = mod.create_function("compute", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());

        Value* t = b.build_call("host_mul", Type::i64(), {x});
        Value* res = b.build_add(t, y);
        b.build_ret(res);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    std::unique_ptr<CompiledModule> compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);

    // Direct C++ function pointer retrieval
    using ComputeFn = int64_t (*)(int64_t, int64_t);
    ComputeFn compute_ptr = compiled->get_function_ptr<ComputeFn>("compute");
    REQUIRE(compute_ptr != nullptr);

    // Invoke and verify: 5 * 10 + 7 = 57
    int64_t result = compute_ptr(5, 7);
    CHECK_EQ(result, 57LL);

    // Invoke via CompiledModule dynamic invoke
    RuntimeValue dyn_res = compiled->invoke("compute", {RuntimeValue::from_i64(12), RuntimeValue::from_i64(3)});
    CHECK_EQ(dyn_res.as_i64(), 123LL);
}

TEST_CASE("Embedding API - CompiledModule Runtime Patching and Stack Maps") {
    HostEngine engine;

    Module mod("test_patching_mod");

    Function* target_v1 = mod.create_function("target_v1", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(target_v1);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(100)));
        target_v1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*target_v1));
    }

    Function* target_v2 = mod.create_function("target_v2", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(target_v2);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_mul(x, b.build_iconst_i64(3)));
        target_v2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*target_v2));
    }

    Function* main_fn = mod.create_function("main_fn", Type::i64(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(main_fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::i64());

        Value* bias = b.build_patchable_const_i64("test_bias_site", 5);
        Value* biased_x = b.build_add(x, bias);
        Value* call_res = b.build_patchable_call("test_call_site", "target_v1", Type::i64(), {biased_x});
        b.build_ret(call_res);

        main_fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*main_fn));
    }

    std::unique_ptr<CompiledModule> compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);

    using MainFn = int64_t (*)(int64_t);
    MainFn run_main = compiled->get_function_ptr<MainFn>("main_fn");
    REQUIRE(run_main != nullptr);

    // Initial state: bias = 5, target = target_v1 (+100) -> (10 + 5) + 100 = 115
    CHECK_EQ(run_main(10), 115LL);

    // Patch constant to 20: (10 + 20) + 100 = 130
    CHECK(compiled->patch_constant("test_bias_site", int64_t(20)));
    CHECK_EQ(run_main(10), 130LL);

    // Patch call to target_v2 (*3): (10 + 20) * 3 = 90
    void* target_v2_addr = compiled->get_symbol_address("target_v2");
    REQUIRE(target_v2_addr != nullptr);
    CHECK(compiled->patch_call("test_call_site", target_v2_addr));
    CHECK_EQ(run_main(10), 90LL);

    // Patch call back to target_v1 via string: (10 + 20) + 100 = 130
    CHECK(compiled->patch_call("test_call_site", "target_v1"));
    CHECK_EQ(run_main(10), 130LL);

    // Verify stack map access
    const ModuleStackMap& maps = compiled->stack_maps();
    CHECK(!maps.functions().empty());
}

TEST_CASE("Embedding API - End-to-End Host Moving Cheney GC Proof with Root Relocation") {
    // Setup HostGC in STRESS MODE (collects on every allocation and safepoint)
    HostGC host_gc(256 * 1024);
    host_gc.set_stress_mode(true);

    HostEngine engine;
    engine.register_host_gc(&host_gc);

    Module mod("moving_gc_host_proof");
    mod.add_external_symbol("host_gc_alloc");
    mod.add_external_symbol("host_gc_safepoint");

    // Construct a binary tree of GC objects in JIT code:
    // struct Node { int64_t val; Node* left; Node* right; }
    // pointer_mask: bits 1 and 2 (offsets 8 and 16) are GC references.
    Function* fn = mod.create_function("build_and_sum_tree", Type::i64(), {});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);

        Value* sz24 = b.build_iconst_i64(24);
        Value* mask0 = b.build_iconst_i64(0);
        Value* mask6 = b.build_iconst_i64(6); // bit 1 (left) and bit 2 (right)
        Value* tag = b.build_iconst_i32(1);

        // Leaf 1: val = 100
        Value* l1 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask0, tag});
        b.build_store(Type::i64(), l1, 0, b.build_iconst_i64(100));

        // Leaf 2: val = 200
        Value* l2 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask0, tag});
        b.build_store(Type::i64(), l2, 0, b.build_iconst_i64(200));

        // Leaf 3: val = 300
        Value* l3 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask0, tag});
        b.build_store(Type::i64(), l3, 0, b.build_iconst_i64(300));

        // Leaf 4: val = 400
        Value* l4 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask0, tag});
        b.build_store(Type::i64(), l4, 0, b.build_iconst_i64(400));

        // Explicit Host Safepoint Trigger (relocates all leaves in stack frame)
        b.build_safepoint();

        // Branch 1: val = 10, left = l1, right = l2
        Value* b1 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask6, tag});
        b.build_store(Type::i64(), b1, 0, b.build_iconst_i64(10));
        b.build_store(Type::gcref(), b1, 8, l1);
        b.build_store(Type::gcref(), b1, 16, l2);

        // Branch 2: val = 20, left = l3, right = l4
        Value* b2 = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask6, tag});
        b.build_store(Type::i64(), b2, 0, b.build_iconst_i64(20));
        b.build_store(Type::gcref(), b2, 8, l3);
        b.build_store(Type::gcref(), b2, 16, l4);

        // Explicit Host Safepoint Trigger
        b.build_safepoint();

        // Root Node: val = 1, left = b1, right = b2
        Value* root = b.build_call("host_gc_alloc", Type::gcref(), {sz24, mask6, tag});
        b.build_store(Type::i64(), root, 0, b.build_iconst_i64(1));
        b.build_store(Type::gcref(), root, 8, b1);
        b.build_store(Type::gcref(), root, 16, b2);

        // Explicit Host Safepoint Trigger while the entire 7-node tree is live
        b.build_safepoint();

        // Read values through relocated pointers from root:
        // sum = root.val + b1.val + l1.val + l2.val + b2.val + l3.val + l4.val
        Value* vr = b.build_load(Type::i64(), root, 0);
        Value* acc = vr;

        Value* rb1 = b.build_load(Type::gcref(), root, 8);
        Value* vb1 = b.build_load(Type::i64(), rb1, 0);
        acc = b.build_add(acc, vb1);

        Value* rl1 = b.build_load(Type::gcref(), rb1, 8);
        Value* vl1 = b.build_load(Type::i64(), rl1, 0);
        acc = b.build_add(acc, vl1);

        Value* rl2 = b.build_load(Type::gcref(), rb1, 16);
        Value* vl2 = b.build_load(Type::i64(), rl2, 0);
        acc = b.build_add(acc, vl2);

        Value* rb2 = b.build_load(Type::gcref(), root, 16);
        Value* vb2 = b.build_load(Type::i64(), rb2, 0);
        acc = b.build_add(acc, vb2);

        Value* rl3 = b.build_load(Type::gcref(), rb2, 8);
        Value* vl3 = b.build_load(Type::i64(), rl3, 0);
        acc = b.build_add(acc, vl3);

        Value* rl4 = b.build_load(Type::gcref(), rb2, 16);
        Value* vl4 = b.build_load(Type::i64(), rl4, 0);
        acc = b.build_add(acc, vl4);

        b.build_ret(acc);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    std::unique_ptr<CompiledModule> compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);

    using TreeFn = int64_t (*)();
    TreeFn run_tree = compiled->get_function_ptr<TreeFn>("build_and_sum_tree");
    REQUIRE(run_tree != nullptr);

    // 1 + 10 + 100 + 200 + 20 + 300 + 400 = 1031
    int64_t total_sum = run_tree();
    CHECK_EQ(total_sum, 1031LL);

    // Verify multiple collections occurred and live roots were relocated without corruption
    CHECK(host_gc.collection_count() >= 7ULL);
}

TEST_CASE("Embedding API - C-Callable ABI Layer Verification") {
    brass_engine_t* engine = brass_engine_create();
    REQUIRE(engine != nullptr);

    brass_gc_t* gc = brass_host_gc_create(128 * 1024);
    REQUIRE(gc != nullptr);
    brass_host_gc_set_stress_mode(gc, 1);
    CHECK_EQ(brass_host_gc_get_stress_mode(gc), 1);

    brass_engine_register_gc(engine, gc);

    // NaN-Box C API checks
    brass_value_t v_f64 = brass_value_from_f64(2.71828);
    CHECK(brass_value_is_f64(v_f64));
    CHECK(!brass_value_is_i32(v_f64));
    CHECK(std::abs(brass_value_as_f64(v_f64) - 2.71828) < 1e-5);

    brass_value_t v_i32 = brass_value_from_i32(777);
    CHECK(brass_value_is_i32(v_i32));
    CHECK_EQ(brass_value_as_i32(v_i32), 777);

    brass_value_t v_bool = brass_value_from_bool(1);
    CHECK(brass_value_is_bool(v_bool));
    CHECK_EQ(brass_value_as_bool(v_bool), 1);

    brass_value_t v_null = brass_value_null();
    CHECK(brass_value_is_null(v_null));

    brass_value_t v_undef = brass_value_undefined();
    CHECK(brass_value_is_undefined(v_undef));

    // Allocate value in GC via C API
    brass_value_t obj_val = brass_host_gc_allocate_value(gc, 16, 0, 1);
    CHECK(brass_value_is_gcref(obj_val));
    CHECK_NE(brass_value_as_gcref(obj_val), 0ULL);

    brass_host_gc_destroy(gc);
    brass_engine_destroy(engine);
}
