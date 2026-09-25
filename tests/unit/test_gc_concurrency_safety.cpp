#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/memory_access.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <sstream>
#include <cstddef>

using namespace brass;
using namespace brass::runtime;

namespace {

constexpr uint64_t BRONZE_OBJECT_TAG = 0xFFF1000000000000ULL;
constexpr uint64_t HOST_GCREF_TAG     = 0x7FFD000000000000ULL;
constexpr uint64_t TAG_MASK           = 0xFFFF000000000000ULL;
constexpr uint64_t PAYLOAD_MASK       = 0x0000FFFFFFFFFFFFULL;

std::unique_ptr<Module> create_simple_test_module(const std::string& name) {
    auto mod = std::make_unique<Module>("mod_" + name);
    Function* fn = mod->create_function(name, Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p0 = b.add_block_param(entry, Type::i64());
    Value* c10 = b.build_iconst_i64(10);
    Value* res = b.build_add(p0, c10);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    return mod;
}

// A heap that ignores BRASS_GC_* (these tests place objects by generation).
gc::HeapConfig plain_config() {
    gc::HeapConfig config;
    config.read_environment = false;
    return config;
}

uint8_t& card_of(gc::Heap& heap, uintptr_t old_address) {
    return heap.card_table_base()[(old_address - heap.old_base()) >> gc::kCardShift];
}
bool card_dirty(gc::Heap& heap, uintptr_t old_address) { return card_of(heap, old_address) == gc::kCardDirty; }
void clean_card(gc::Heap& heap, uintptr_t old_address) { card_of(heap, old_address) = gc::kCardClean; }

uintptr_t allocate_old(gc::Heap& heap, size_t bytes, uint64_t mask, uint32_t tag) {
    return heap.allocate(bytes, gc::mask_layout(mask, tag), gc::kAllocOld);
}

} // namespace

// ============================================================================
// 1. Write Barrier NaN-Tag Masking
// ============================================================================

TEST_CASE("GC Concurrency Safety - Write barrier card marking with tagged young pointers") {
    gc::Heap heap(plain_config());
    gc::HeapScope bind(heap);

    uintptr_t old_obj = allocate_old(heap, 256, 0, 10);
    REQUIRE(old_obj != 0);
    REQUIRE(heap.is_old(old_obj));

    uintptr_t young_obj = heap.allocate_masked(64, 0, 20);
    REQUIRE(young_obj != 0);
    REQUIRE(heap.is_young(young_obj));

    clean_card(heap, old_obj);
    CHECK(!card_dirty(heap, old_obj));

    // Test 1: Raw young pointer marks card
    heap.write_barrier(old_obj, young_obj);
    CHECK(card_dirty(heap, old_obj));
    clean_card(heap, old_obj);

    // Test 2: Bronze object tagged young pointer marks card
    uint64_t bronze_tagged_young = BRONZE_OBJECT_TAG | young_obj;
    heap.write_barrier(old_obj, bronze_tagged_young);
    CHECK(card_dirty(heap, old_obj));
    clean_card(heap, old_obj);

    // Test 3: HostValue GCRef tagged young pointer marks card
    uint64_t host_tagged_young = HOST_GCREF_TAG | young_obj;
    heap.write_barrier(old_obj, host_tagged_young);
    CHECK(card_dirty(heap, old_obj));
    clean_card(heap, old_obj);

    // Test 4: brass's default barrier (the thread's heap) with Bronze tag marks card
    brass_default_gc_write_barrier(old_obj, static_cast<uintptr_t>(bronze_tagged_young));
    CHECK(card_dirty(heap, old_obj));
    clean_card(heap, old_obj);

    // Test 5: brass's default barrier with Host tag marks card
    brass_default_gc_write_barrier(old_obj, static_cast<uintptr_t>(host_tagged_young));
    CHECK(card_dirty(heap, old_obj));
    clean_card(heap, old_obj);

    // Test 6: Non-pointer tagged value (e.g. tagged int32 42) must NOT mark card
    uint64_t tagged_int = 0x7FF900000000002AULL;
    heap.write_barrier(old_obj, tagged_int);
    CHECK(!card_dirty(heap, old_obj));

    // Test 7: Another old object written into old object must NOT mark card
    uintptr_t another_old_obj = allocate_old(heap, 256, 0, 11);
    REQUIRE(heap.is_old(another_old_obj));
    heap.write_barrier(old_obj, another_old_obj);
    CHECK(!card_dirty(heap, old_obj));

    // Test 8: Young object written into young object marks no card (only old->young)
    uintptr_t young_obj2 = heap.allocate_masked(64, 0, 21);
    REQUIRE(heap.is_young(young_obj2));
    heap.write_barrier(young_obj, young_obj2);
    CHECK(!card_dirty(heap, old_obj));
    CHECK(!card_dirty(heap, another_old_obj));
}

// ============================================================================
// 2. Generational GC Tagged Pointer Scavenge & Evacuate Safety
// ============================================================================

TEST_CASE("GC Concurrency Safety - Minor scavenge evacuates tagged roots preserving tag bits") {
    gc::Heap heap(plain_config());

    // Allocate two young objects
    uintptr_t young1 = heap.allocate_masked(64, 0, 1);
    uintptr_t young2 = heap.allocate_masked(64, 0, 2);
    REQUIRE(young1 != 0);
    REQUIRE(young2 != 0);
    REQUIRE(heap.is_young(young1));
    REQUIRE(heap.is_young(young2));

    // Write distinctive payloads
    *reinterpret_cast<uint64_t*>(young1) = 0x1122334455667788ULL;
    *reinterpret_cast<uint64_t*>(young2) = 0xAABBCCDDEEFF0011ULL;

    // Root 1: Bronze tagged object pointer
    uint64_t root1 = BRONZE_OBJECT_TAG | young1;
    // Root 2: HostValue GCRef tagged pointer
    uint64_t root2 = HOST_GCREF_TAG | young2;
    // Root 3: Duplicate reference to young1 with different tag
    uint64_t root3 = HOST_GCREF_TAG | young1;

    heap.add_root(&root1);
    heap.add_root(&root2);
    heap.add_root(&root3);

    // Run minor collection (evacuate to a survivor space)
    heap.collect(gc::CollectionKind::Minor);

    // Check Root 1: Tag preserved, pointer updated into the young generation
    CHECK((root1 & TAG_MASK) == BRONZE_OBJECT_TAG);
    uintptr_t new_ptr1 = root1 & PAYLOAD_MASK;
    CHECK(new_ptr1 != young1);
    CHECK(heap.is_young(new_ptr1));
    CHECK(*reinterpret_cast<uint64_t*>(new_ptr1) == 0x1122334455667788ULL);

    // Check Root 2: Tag preserved, pointer updated into the young generation
    CHECK((root2 & TAG_MASK) == HOST_GCREF_TAG);
    uintptr_t new_ptr2 = root2 & PAYLOAD_MASK;
    CHECK(new_ptr2 != young2);
    CHECK(heap.is_young(new_ptr2));
    CHECK(*reinterpret_cast<uint64_t*>(new_ptr2) == 0xAABBCCDDEEFF0011ULL);

    // Check Root 3: Pointing to same object as root1, with HOST_GCREF_TAG preserved
    CHECK((root3 & TAG_MASK) == HOST_GCREF_TAG);
    uintptr_t new_ptr3 = root3 & PAYLOAD_MASK;
    CHECK(new_ptr3 == new_ptr1);

    // Run full collection: promotes to the old generation, preserving tags
    heap.collect(gc::CollectionKind::Full);

    CHECK((root1 & TAG_MASK) == BRONZE_OBJECT_TAG);
    uintptr_t tenured_ptr1 = root1 & PAYLOAD_MASK;
    CHECK(heap.is_old(tenured_ptr1));
    CHECK(*reinterpret_cast<uint64_t*>(tenured_ptr1) == 0x1122334455667788ULL);

    CHECK((root2 & TAG_MASK) == HOST_GCREF_TAG);
    uintptr_t tenured_ptr2 = root2 & PAYLOAD_MASK;
    CHECK(heap.is_old(tenured_ptr2));
    CHECK(*reinterpret_cast<uint64_t*>(tenured_ptr2) == 0xAABBCCDDEEFF0011ULL);

    CHECK((root3 & TAG_MASK) == HOST_GCREF_TAG);
    CHECK((root3 & PAYLOAD_MASK) == tenured_ptr1);

    heap.remove_root(&root1);
    heap.remove_root(&root2);
    heap.remove_root(&root3);
}

TEST_CASE("GC Concurrency Safety - Card roots and Cheney scan with tagged pointers in fields") {
    gc::Heap heap(plain_config());

    // An old parent whose word 0 is a reference (pointer_mask = 1)
    uintptr_t tenured_parent = allocate_old(heap, 256, 1ULL, 5);
    REQUIRE(heap.is_old(tenured_parent));
    // Rooted: an unreachable old object is only reclaimed by a full collection,
    // but the test's reference should not rely on that.
    uint64_t parent_root = tenured_parent;
    heap.add_root(&parent_root);

    // Allocate young child
    uintptr_t young_child = heap.allocate_masked(64, 0, 6);
    REQUIRE(heap.is_young(young_child));
    *reinterpret_cast<uint64_t*>(young_child) = 0x9988776655443322ULL;

    // Store tagged young pointer into the old parent's field 0
    uint64_t tagged_child = BRONZE_OBJECT_TAG | young_child;
    heap.store(tenured_parent, 0, tagged_child);

    // Verify card is dirty
    CHECK(card_dirty(heap, tenured_parent));

    // Collect minor: parent is old, child referenced only via dirty card
    heap.collect(gc::CollectionKind::Minor);
    CHECK(parent_root == tenured_parent);  // old objects never move

    // Verify the parent's field 0 has the evacuated child with tag intact
    uint64_t updated_child_val = gc::Heap::load(tenured_parent, 0);
    CHECK((updated_child_val & TAG_MASK) == BRONZE_OBJECT_TAG);
    uintptr_t updated_child_ptr = updated_child_val & PAYLOAD_MASK;
    CHECK(updated_child_ptr != young_child);
    CHECK(heap.is_young(updated_child_ptr));
    CHECK(*reinterpret_cast<uint64_t*>(updated_child_ptr) == 0x9988776655443322ULL);

    // Card should still be dirty because survivor is young
    CHECK(card_dirty(heap, tenured_parent));
    heap.remove_root(&parent_root);
}

// ============================================================================
// 3. Float Read/Write Width Handling (f32 vs f64)
// ============================================================================

TEST_CASE("GC Concurrency Safety - Float read_memory and write_memory width safety") {
    gc::Heap heap(plain_config());

    // Memory buffer with canaries surrounding float fields
    struct alignas(8) TestBuffer {
        uint32_t canary_before = 0xAAAAAAAA;
        float f32_val          = 1.5f;
        uint32_t canary_middle = 0xBBBBBBBB;
        double f64_val         = 3.141592653589793;
        uint32_t canary_after  = 0xCCCCCCCC;
    };

    TestBuffer buf;
    uintptr_t base = reinterpret_cast<uintptr_t>(&buf);

    // 1. Read f32: must return f32 value without reading 8 bytes
    RuntimeValue read_f32 = read_memory(heap, base, offsetof(TestBuffer, f32_val), Type::f32());
    CHECK(read_f32.is_f32());
    CHECK(read_f32.as_f32() == 1.5f);

    // 2. Read f64: must return f64 value
    RuntimeValue read_f64 = read_memory(heap, base, offsetof(TestBuffer, f64_val), Type::f64());
    CHECK(read_f64.is_f64());
    CHECK(read_f64.as_f64() == 3.141592653589793);

    // 3. Write f32: must write exactly 4 bytes without corrupting canary_middle
    write_memory(heap, base, offsetof(TestBuffer, f32_val), Type::f32(), RuntimeValue::from_f32(99.25f));
    CHECK(buf.f32_val == 99.25f);
    CHECK(buf.canary_before == 0xAAAAAAAA);
    CHECK(buf.canary_middle == 0xBBBBBBBB); // MUST NOT be overwritten by 8-byte write!

    // Verify reading back written f32
    RuntimeValue read_f32_back = read_memory(heap, base, offsetof(TestBuffer, f32_val), Type::f32());
    CHECK(read_f32_back.is_f32());
    CHECK(read_f32_back.as_f32() == 99.25f);

    // 4. Write f64: writes 8 bytes
    write_memory(heap, base, offsetof(TestBuffer, f64_val), Type::f64(), RuntimeValue::from_f64(2.718281828459045));
    CHECK(buf.f64_val == 2.718281828459045);
    CHECK(buf.canary_middle == 0xBBBBBBBB);
    CHECK(buf.canary_after == 0xCCCCCCCC);

    // Verify reading back written f64
    RuntimeValue read_f64_back = read_memory(heap, base, offsetof(TestBuffer, f64_val), Type::f64());
    CHECK(read_f64_back.is_f64());
    CHECK(read_f64_back.as_f64() == 2.718281828459045);
}

// ============================================================================
// 4. Concurrency Thread Safety in TieringRegistry
// ============================================================================

TEST_CASE("GC Concurrency Safety - TieringRegistry multi-threaded concurrency safety") {
    auto& registry = TieringRegistry::instance();
    registry.clear();

    constexpr size_t NUM_THREADS = 8;
    constexpr size_t ITERATIONS = 500;
    constexpr size_t NUM_FUNCTIONS = 16;

    std::atomic<bool> start_signal{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (size_t i = 0; i < ITERATIONS; ++i) {
                // Concurrently get and mutate thread-specific feedback
                std::string fn_name = "thread_fn_" + std::to_string(t) + "_" + std::to_string(i % NUM_FUNCTIONS);
                auto& fb = registry.get_feedback(fn_name);
                fb.record_invocation();
                if ((i % 10) == 0) {
                    fb.record_backedge(static_cast<uint32_t>(i % 3));
                }
                if ((i % 50) == 0) {
                    fb.record_deopt(static_cast<uint32_t>(i % 2));
                }

                // Concurrently get and query shared function feedback to test multi-thread contention on map
                std::string shared_fn = "shared_fn_" + std::to_string(i % 4);
                auto& shared_fb = registry.get_feedback(shared_fn);
                (void)shared_fb;

                // Periodic check of find_feedback
                const auto* found = registry.find_feedback(fn_name);
                CHECK(found != nullptr);
                const auto* found_shared = registry.find_feedback(shared_fn);
                CHECK(found_shared != nullptr);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& th : threads) {
        th.join();
    }

    // Verify all thread-specific functions exist and have recorded stats
    uint64_t total_invocations = 0;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (size_t f = 0; f < NUM_FUNCTIONS; ++f) {
            std::string fn_name = "thread_fn_" + std::to_string(t) + "_" + std::to_string(f);
            const auto* fb = registry.find_feedback(fn_name);
            REQUIRE(fb != nullptr);
            total_invocations += fb->invocation_count();
            CHECK(fb->invocation_count() > 0);
        }
    }

    CHECK_EQ(total_invocations, NUM_THREADS * ITERATIONS);

    // Verify shared functions also exist
    for (size_t s = 0; s < 4; ++s) {
        std::string shared_fn = "shared_fn_" + std::to_string(s);
        CHECK(registry.find_feedback(shared_fn) != nullptr);
    }

    // Verify dump_stats executes cleanly
    std::ostringstream oss;
    registry.dump_stats(oss);
    CHECK(!oss.str().empty());

    registry.clear();
}

// ============================================================================
// 5. Tier 2 Stack Map Registration in MultiTierPipeline
// ============================================================================

TEST_CASE("GC Concurrency Safety - Tier 2 stack map registration in MultiTierPipeline") {
    auto& pipeline = MultiTierPipeline::instance();
    pipeline.initialize();
    REQUIRE(pipeline.is_initialized());

    // Initially active stack maps pointer should point to pipeline's active stack maps
    CHECK(brass_get_active_stack_maps() == &pipeline.active_stack_maps());

    auto mod = create_simple_test_module("tier2_stack_map_test_fn");
    Function* fn = mod->get_function("tier2_stack_map_test_fn");
    REQUIRE(fn != nullptr);

    FunctionHandle handle("tier2_stack_map_test_fn", fn);
    CodeInstaller installer;

    CodeInstallResult res = installer.install_tier2(handle, *mod, "tier2_stack_map_test_fn");
    REQUIRE(res.success);
    REQUIRE(handle.has_native_entry());
    CHECK_EQ(handle.tier(), TierLevel::Tier2_Optimized);

    // Verify function stack map was registered in MultiTierPipeline's active_stack_maps
    const auto* registered_map = pipeline.active_stack_maps().find_function_by_name("tier2_stack_map_test_fn");
    CHECK(registered_map != nullptr);

    // Verify active stack maps pointer is still set to pipeline's active stack maps
    CHECK(brass_get_active_stack_maps() == &pipeline.active_stack_maps());

    pipeline.shutdown();
}
