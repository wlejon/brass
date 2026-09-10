#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/generational_gc.hpp>
#include <vector>

using namespace brass;

TEST_CASE("Differential - Generational GC vs Mini Cheney GC Linked List Churn") {
    // Semi-space for Cheney: 128 KB
    MiniCheneyGC cheney_gc(128 * 1024);
    // Generational: 32 KB nursery, 32 KB survivor, 128 KB tenured
    GenerationalGC gen_gc(32 * 1024, 32 * 1024, 128 * 1024);

    uintptr_t cheney_head = 0;
    uintptr_t gen_head = 0;

    std::vector<uintptr_t*> cheney_roots = { &cheney_head };
    std::vector<uintptr_t*> gen_roots = { &gen_head };

    constexpr int ITERATIONS = 50;
    for (int i = 0; i < ITERATIONS; ++i) {
        // Allocate persistent list node: payload 16 bytes (val: i64, next: gcref)
        // pointer_mask: bit 1 (offset 8) is gcref -> 1ULL << 1 = 2
        uintptr_t c_node = cheney_gc.allocate(16, 2ULL, 1);
        cheney_gc.write_field(c_node, 0, static_cast<uint64_t>(i * 10));
        cheney_gc.write_field(c_node, 1, cheney_head);
        cheney_head = c_node;

        uintptr_t g_node = gen_gc.allocate(16, 2ULL, 1);
        gen_gc.write_field(g_node, 0, static_cast<uint64_t>(i * 10));
        gen_gc.write_field(g_node, 1, gen_head);
        // Write barrier on list link
        gen_gc.write_barrier(g_node, gen_head);
        gen_head = g_node;

        // Churn: allocate transient throwaway garbage
        for (int churn = 0; churn < 20; ++churn) {
            uintptr_t c_garbage = cheney_gc.allocate(32, 0, 99);
            cheney_gc.write_field(c_garbage, 0, 0xDEADBEEF);

            uintptr_t g_garbage = gen_gc.allocate(32, 0, 99);
            gen_gc.write_field(g_garbage, 0, 0xDEADBEEF);
        }

        // Trigger collection periodically
        if (i % 10 == 0) {
            cheney_gc.collect(cheney_roots);
            gen_gc.collect(gen_roots);
        }
    }

    // Final collection
    cheney_gc.collect(cheney_roots);
    gen_gc.collect(gen_roots);

    CHECK(gen_gc.minor_collections() > 0);

    // Verify both lists produce identical length and values
    uintptr_t cur_c = cheney_head;
    uintptr_t cur_g = gen_head;
    int verified_nodes = 0;

    while (cur_c != 0 && cur_g != 0) {
        CHECK(cheney_gc.is_valid_object(cur_c));
        CHECK(gen_gc.is_valid_object(cur_g));

        uint64_t val_c = cheney_gc.read_field(cur_c, 0);
        uint64_t val_g = gen_gc.read_field(cur_g, 0);
        CHECK_EQ(val_c, val_g);

        cur_c = cheney_gc.read_field(cur_c, 1);
        cur_g = gen_gc.read_field(cur_g, 1);
        verified_nodes++;
    }

    CHECK_EQ(cur_c, 0ULL);
    CHECK_EQ(cur_g, 0ULL);
    CHECK_EQ(verified_nodes, ITERATIONS);
}

TEST_CASE("Differential - Old-to-Young mutation across Minor Collections") {
    MiniCheneyGC cheney_gc(64 * 1024);
    GenerationalGC gen_gc(32 * 1024, 32 * 1024, 64 * 1024);
    gen_gc.set_tenuring_threshold(1);

    // Create long-lived root object in both GCs
    uintptr_t c_root = cheney_gc.allocate(16, 2ULL, 1);
    cheney_gc.write_field(c_root, 0, 1000ULL);
    cheney_gc.write_field(c_root, 1, 0ULL);

    uintptr_t g_root = gen_gc.allocate(16, 2ULL, 1);
    gen_gc.write_field(g_root, 0, 1000ULL);
    gen_gc.write_field(g_root, 1, 0ULL);

    std::vector<uintptr_t*> c_roots = { &c_root };
    std::vector<uintptr_t*> g_roots = { &g_root };

    // Advance generation: g_root becomes tenured
    gen_gc.collect(g_roots);
    cheney_gc.collect(c_roots);
    CHECK(gen_gc.is_in_tenured(g_root));

    // Now allocate new young leaf in nursery
    uintptr_t c_leaf = cheney_gc.allocate(16, 0ULL, 2);
    cheney_gc.write_field(c_leaf, 0, 9999ULL);
    cheney_gc.write_field(c_root, 1, c_leaf);

    uintptr_t g_leaf = gen_gc.allocate(16, 0ULL, 2);
    gen_gc.write_field(g_leaf, 0, 9999ULL);
    gen_gc.write_field(g_root, 1, g_leaf);
    gen_gc.write_barrier(g_root, g_leaf); // Critical: old points to young!

    // Scavenge with ONLY roots (leaf is not directly rooted!)
    gen_gc.collect(g_roots);
    cheney_gc.collect(c_roots);

    // Both should preserve leaf through root's reference
    uintptr_t c_leaf_after = cheney_gc.read_field(c_root, 1);
    uintptr_t g_leaf_after = gen_gc.read_field(g_root, 1);

    REQUIRE_NE(c_leaf_after, 0ULL);
    REQUIRE_NE(g_leaf_after, 0ULL);

    CHECK(cheney_gc.is_valid_object(c_leaf_after));
    CHECK(gen_gc.is_valid_object(g_leaf_after));

    CHECK_EQ(cheney_gc.read_field(c_leaf_after, 0), 9999ULL);
    CHECK_EQ(gen_gc.read_field(g_leaf_after, 0), 9999ULL);
}

TEST_CASE("Differential - Interpreter MIR execution with Write Barrier") {
    Module mod("diff_mir_wb");
    Builder b(mod);

    // Function: allocates an object, writes fields with write_barrier, returns field 0
    Function* fn = mod.create_function("wb_test", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag1 = b.build_iconst_i32(1);

    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag1});
    Value* val = b.build_iconst_i64(4242);
    b.build_store(Type::i64(), obj, 0, val);
    b.build_write_barrier(obj, val);

    Value* loaded = b.build_load(Type::i64(), obj, 0);
    b.build_ret(loaded);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // Execute on Interpreter
    Interpreter interp_cheney;
    interp_cheney.set_module(&mod);
    auto res_cheney = interp_cheney.run(*fn, {});
    CHECK_EQ(res_cheney.as_i64(), 4242LL);

    GenerationalGC gen_gc(32 * 1024, 32 * 1024, 64 * 1024);
    Interpreter interp_gen;
    interp_gen.set_module(&mod);
    interp_gen.set_generational_gc(&gen_gc);
    auto res_gen = interp_gen.run(*fn, {});
    CHECK_EQ(res_gen.as_i64(), 4242LL);
}
