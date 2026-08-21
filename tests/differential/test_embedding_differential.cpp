#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/embedding/embedding.hpp>
#include <random>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("Embedding Differential - NaN-Box Random Fuzz and Tag Invariant Stress") {
    std::mt19937_64 rng(0x123456789ABCDEFULL);
    std::uniform_real_distribution<double> dbl_dist(-1e12, 1e12);
    std::uniform_int_distribution<int32_t> i32_dist(INT32_MIN, INT32_MAX);
    std::uniform_int_distribution<uint64_t> addr_dist(0x1000ULL, 0x00007FFFFFFFFFFFULL);

    constexpr size_t ITERATIONS = 10000;

    for (size_t i = 0; i < ITERATIONS; ++i) {
        // 1. Double fuzzing
        double d = dbl_dist(rng);
        HostValue vd = HostValue::from_f64(d);
        CHECK(vd.is_f64());
        CHECK(vd.is_double());
        CHECK(!vd.is_i32());
        CHECK(!vd.is_bool());
        CHECK(!vd.is_gcref());
        CHECK_EQ(vd.as_f64(), d);

        // 2. Int32 fuzzing
        int32_t val_i = i32_dist(rng);
        HostValue vi = HostValue::from_i32(val_i);
        CHECK(vi.is_i32());
        CHECK(!vi.is_f64());
        CHECK_EQ(vi.as_i32(), val_i);

        // 3. GCRef fuzzing and relocation update
        uintptr_t addr = static_cast<uintptr_t>(addr_dist(rng) & ~static_cast<uint64_t>(7));
        HostValue vg = HostValue::from_gcref(addr);
        CHECK(vg.is_gcref());
        CHECK(!vg.is_f64());
        CHECK(!vg.is_i32());
        CHECK_EQ(vg.as_gcref(), addr);

        uintptr_t new_addr = static_cast<uintptr_t>(addr_dist(rng) & ~static_cast<uint64_t>(7));
        vg.update_gcref(new_addr);
        CHECK(vg.is_gcref());
        CHECK_EQ(vg.as_gcref(), new_addr);
    }
}

TEST_CASE("Embedding Differential - Deep Nested Frame GC Relocation") {
    // Stress test deep stack frames with live GCRefs across recursive calls
    HostGC host_gc(256 * 1024);
    host_gc.set_stress_mode(true);

    HostEngine engine;
    engine.register_host_gc(&host_gc);

    Module mod("deep_stack_gc_mod");
    mod.add_external_symbol("host_gc_alloc");
    mod.add_external_symbol("host_gc_safepoint");

    // helper_alloc(val: i64) -> gcref
    Function* fn_alloc = mod.create_function("helper_alloc", Type::gcref(), {Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_alloc);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* v = b.add_block_param(entry, Type::i64());

        Value* sz = b.build_iconst_i64(16);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(1);
        Value* obj = b.build_call("host_gc_alloc", Type::gcref(), {sz, mask, tag});
        b.build_store(Type::i64(), obj, 0, v);
        b.build_safepoint();
        b.build_ret(obj);

        fn_alloc->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_alloc));
    }

    // recursive_sum(depth: i64, acc_val: i64) -> i64
    Function* fn_rec = mod.create_function("recursive_sum", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn_rec);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* depth = b.add_block_param(entry, Type::i64());
        Value* acc = b.add_block_param(entry, Type::i64());

        Value* zero = b.build_iconst_i64(0);
        Value* is_zero = b.build_eq(depth, zero);

        BasicBlock* base_case = b.append_block("base_case");
        b.position_at_end(base_case);
        Value* base_obj = b.build_call("helper_alloc", Type::gcref(), {acc});
        b.build_safepoint();
        Value* base_res = b.build_load(Type::i64(), base_obj, 0);
        b.build_ret(base_res);

        BasicBlock* rec_case = b.append_block("rec_case");
        b.position_at_end(rec_case);
        Value* one = b.build_iconst_i64(1);
        Value* next_depth = b.build_sub(depth, one);
        Value* obj_curr = b.build_call("helper_alloc", Type::gcref(), {depth});

        // Recurse while obj_curr is held live on stack
        Value* sub_sum = b.build_call("recursive_sum", Type::i64(), {next_depth, acc});
        b.build_safepoint();

        // Read obj_curr back after return (must have been relocated safely)
        Value* read_depth = b.build_load(Type::i64(), obj_curr, 0);
        Value* total = b.build_add(sub_sum, read_depth);
        b.build_ret(total);

        b.position_at_end(entry);
        b.build_br_if(is_zero, base_case, rec_case);

        fn_rec->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_rec));
    }

    std::unique_ptr<CompiledModule> compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);

    using RecFn = int64_t (*)(int64_t, int64_t);
    RecFn run_rec = compiled->get_function_ptr<RecFn>("recursive_sum");
    REQUIRE(run_rec != nullptr);

    // Sum from 1..10 + acc(50) = 55 + 50 = 105
    int64_t result = run_rec(10, 50);
    CHECK_EQ(result, 105LL);
    CHECK(host_gc.collection_count() >= 20ULL);
}

TEST_CASE("Embedding Differential - Dynamic Cyclic Graph with Moving Cheney GC") {
    HostGC host_gc(256 * 1024);
    host_gc.set_stress_mode(true);

    HostEngine engine;
    engine.register_host_gc(&host_gc);

    Module mod("cyclic_graph_mod");
    mod.add_external_symbol("host_gc_alloc");
    mod.add_external_symbol("host_gc_safepoint");

    // Builds two mutually referencing nodes:
    // A -> B and B -> A
    Function* fn = mod.create_function("test_cycle", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* vA = b.add_block_param(entry, Type::i64());
        Value* vB = b.add_block_param(entry, Type::i64());

        Value* sz16 = b.build_iconst_i64(16);
        Value* mask2 = b.build_iconst_i64(2); // field 1 is gcref
        Value* tag = b.build_iconst_i32(1);

        Value* nodeA = b.build_call("host_gc_alloc", Type::gcref(), {sz16, mask2, tag});
        b.build_store(Type::i64(), nodeA, 0, vA);

        Value* nodeB = b.build_call("host_gc_alloc", Type::gcref(), {sz16, mask2, tag});
        b.build_store(Type::i64(), nodeB, 0, vB);
        b.build_store(Type::gcref(), nodeB, 8, nodeA); // B -> A
        b.build_store(Type::gcref(), nodeA, 8, nodeB); // A -> B (cycle established)

        // Explicit safepoints with cycle live
        b.build_safepoint();
        b.build_safepoint();

        // Traverse: A -> B -> A and read values
        Value* b_from_a = b.build_load(Type::gcref(), nodeA, 8);
        Value* a_from_b = b.build_load(Type::gcref(), b_from_a, 8);

        Value* val_a = b.build_load(Type::i64(), a_from_b, 0);
        Value* val_b = b.build_load(Type::i64(), b_from_a, 0);

        Value* res = b.build_add(val_a, val_b);
        b.build_ret(res);

        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    std::unique_ptr<CompiledModule> compiled = engine.compile(mod);
    REQUIRE(compiled != nullptr);

    using CycleFn = int64_t (*)(int64_t, int64_t);
    CycleFn run_cycle = compiled->get_function_ptr<CycleFn>("test_cycle");
    REQUIRE(run_cycle != nullptr);

    int64_t sum = run_cycle(1234, 5678);
    CHECK_EQ(sum, 6912LL);
    CHECK(host_gc.collection_count() >= 4ULL);
}
