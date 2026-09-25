// The same object graphs built on two heaps, one collected only by young
// (minor) collections, whose old-to-young references are found through the
// write barrier's cards, the other only by full collections, which trace
// everything and promote every survivor: both must keep identical graphs.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/heap.hpp>
#include <vector>

using namespace brass;

namespace {

gc::HeapConfig small_config(uint8_t tenure_age) {
    gc::HeapConfig config;
    config.eden_bytes = 64 * 1024;
    config.survivor_bytes = 32 * 1024;
    config.tenure_age = tenure_age;
    config.poison = true;
    config.read_environment = false;
    return config;
}

} // namespace

TEST_CASE("Differential - Minor-collected vs full-collected heap: linked list churn") {
    gc::Heap full_gc(small_config(2));
    gc::Heap gen_gc(small_config(2));

    uint64_t full_head = 0;
    uint64_t gen_head = 0;
    full_gc.add_root(&full_head);
    gen_gc.add_root(&gen_head);

    constexpr int ITERATIONS = 50;
    for (int i = 0; i < ITERATIONS; ++i) {
        // Allocate persistent list node: payload 16 bytes (val: i64, next: gcref)
        // pointer_mask: bit 1 (offset 8) is gcref -> 1ULL << 1 = 2
        uintptr_t f_node = full_gc.allocate_masked(16, 2ULL, 1);
        full_gc.store(f_node, 0, static_cast<uint64_t>(i * 10));
        full_gc.store(f_node, 1, full_head);
        full_head = f_node;

        uintptr_t g_node = gen_gc.allocate_masked(16, 2ULL, 1);
        gen_gc.store(g_node, 0, static_cast<uint64_t>(i * 10));
        gen_gc.store(g_node, 1, gen_head);  // store() applies the write barrier
        gen_head = g_node;

        // Churn: allocate transient throwaway garbage
        for (int churn = 0; churn < 20; ++churn) {
            uintptr_t f_garbage = full_gc.allocate_masked(32, 0, 99);
            full_gc.store(f_garbage, 0, 0xDEADBEEF);

            uintptr_t g_garbage = gen_gc.allocate_masked(32, 0, 99);
            gen_gc.store(g_garbage, 0, 0xDEADBEEF);
        }

        // Trigger collection periodically
        if (i % 10 == 0) {
            full_gc.collect(gc::CollectionKind::Full);
            gen_gc.collect(gc::CollectionKind::Minor);
        }
    }

    // Final collection
    full_gc.collect(gc::CollectionKind::Full);
    gen_gc.collect(gc::CollectionKind::Minor);

    CHECK(gen_gc.stats().minor_collections > 0);
    CHECK_EQ(gen_gc.stats().full_collections, 0u);
    CHECK(full_gc.stats().full_collections > 0);
    CHECK(gen_gc.stats().promoted_bytes > 0);  // the list's older nodes are old, the newer young

    // Verify both lists produce identical length and values
    uint64_t cur_f = full_head;
    uint64_t cur_g = gen_head;
    int verified_nodes = 0;

    while (cur_f != 0 && cur_g != 0) {
        CHECK(full_gc.is_valid_object(cur_f));
        CHECK(gen_gc.is_valid_object(cur_g));

        uint64_t val_f = gc::Heap::load(cur_f, 0);
        uint64_t val_g = gc::Heap::load(cur_g, 0);
        CHECK_EQ(val_f, val_g);

        cur_f = gc::Heap::load(cur_f, 1);
        cur_g = gc::Heap::load(cur_g, 1);
        verified_nodes++;
    }

    CHECK_EQ(cur_f, 0ULL);
    CHECK_EQ(cur_g, 0ULL);
    CHECK_EQ(verified_nodes, ITERATIONS);
    full_gc.remove_root(&full_head);
    gen_gc.remove_root(&gen_head);
}

TEST_CASE("Differential - Old-to-Young mutation across Minor Collections") {
    gc::Heap full_gc(small_config(1));
    gc::Heap gen_gc(small_config(1));  // promoted by the first minor collection

    // Create long-lived root object in both heaps
    uint64_t f_root = full_gc.allocate_masked(16, 2ULL, 1);
    full_gc.store(f_root, 0, 1000ULL);
    full_gc.store(f_root, 1, 0ULL);

    uint64_t g_root = gen_gc.allocate_masked(16, 2ULL, 1);
    gen_gc.store(g_root, 0, 1000ULL);
    gen_gc.store(g_root, 1, 0ULL);

    full_gc.add_root(&f_root);
    gen_gc.add_root(&g_root);

    // Advance generation: both roots become old
    gen_gc.collect(gc::CollectionKind::Minor);
    full_gc.collect(gc::CollectionKind::Full);
    CHECK(gen_gc.is_old(g_root));
    CHECK(full_gc.is_old(f_root));

    // Now allocate a new young leaf
    uintptr_t f_leaf = full_gc.allocate_masked(16, 0ULL, 2);
    full_gc.store(f_leaf, 0, 9999ULL);
    full_gc.store(f_root, 1, f_leaf);

    uintptr_t g_leaf = gen_gc.allocate_masked(16, 0ULL, 2);
    gen_gc.store(g_leaf, 0, 9999ULL);
    gen_gc.store(g_root, 1, g_leaf);  // Critical: old points to young, through the barrier
    CHECK(gen_gc.is_young(g_leaf));

    // Collect with ONLY roots (leaf is not directly rooted!)
    gen_gc.collect(gc::CollectionKind::Minor);
    full_gc.collect(gc::CollectionKind::Full);

    // Both should preserve leaf through root's reference
    uint64_t f_leaf_after = gc::Heap::load(f_root, 1);
    uint64_t g_leaf_after = gc::Heap::load(g_root, 1);

    REQUIRE_NE(f_leaf_after, 0ULL);
    REQUIRE_NE(g_leaf_after, 0ULL);
    CHECK_NE(g_leaf_after, g_leaf);  // moved: promoted by this minor collection

    CHECK(full_gc.is_valid_object(f_leaf_after));
    CHECK(gen_gc.is_valid_object(g_leaf_after));

    CHECK_EQ(gc::Heap::load(f_leaf_after, 0), 9999ULL);
    CHECK_EQ(gc::Heap::load(g_leaf_after, 0), 9999ULL);
    full_gc.remove_root(&f_root);
    gen_gc.remove_root(&g_root);
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

    // Execute on an Interpreter with a heap of its own
    Interpreter interp_plain(gc::HeapConfig{});
    interp_plain.set_module(&mod);
    auto res_plain = interp_plain.run(*fn, {});
    CHECK_EQ(res_plain.as_i64(), 4242LL);

    // And on one sharing a heap that collects at every allocation and safepoint
    gc::Heap gen_gc(small_config(2));
    gen_gc.set_stress(gc::StressMode::Alternate);
    Interpreter interp_gen(&gen_gc);
    interp_gen.set_module(&mod);
    auto res_gen = interp_gen.run(*fn, {});
    CHECK_EQ(res_gen.as_i64(), 4242LL);
    CHECK(gen_gc.collection_count() >= 1u);
}
