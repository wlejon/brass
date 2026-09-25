#include "test_framework.hpp"
#include <brass/gc/heap.hpp>
#include <brass/core/arena.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <vector>
#include <string>
#include <memory>

using namespace brass;

namespace {

// A heap that ignores BRASS_GC_* (these tests count collections).
gc::HeapConfig plain_config() {
    gc::HeapConfig config;
    config.read_environment = false;
    // The 100KB allocations below stand for large, directly-tenured objects.
    config.large_object_bytes = 64 * 1024;
    return config;
}

// Visits `slots` (duplicates included) as roots for the scope's lifetime.
class SlotRoots {
public:
    SlotRoots(gc::Heap& heap, std::vector<uint64_t*> slots) : heap_(heap), slots_(std::move(slots)) {
        id_ = heap_.add_root_source([this](gc::Tracer& tracer) {
            for (uint64_t* slot : slots_) tracer.visit(slot);
        });
    }
    ~SlotRoots() { heap_.remove_root_source(id_); }
    SlotRoots(const SlotRoots&) = delete;
    SlotRoots& operator=(const SlotRoots&) = delete;

private:
    gc::Heap& heap_;
    std::vector<uint64_t*> slots_;
    gc::Heap::RootSourceId id_ = 0;
};

} // namespace

// =============================================================================
// Task 1: Generational GC Scavenge Re-Evacuation and Object Duplication Bug
// =============================================================================

TEST_CASE("Phase 4 - Generational GC: generation queries classify young, old and foreign addresses") {
    gc::Heap gc(plain_config());

    uintptr_t obj = gc.allocate_masked(24, 0, 10);
    REQUIRE(obj != 0);

    // A fresh small object is young (collected by a minor collection)
    CHECK(gc.is_young(obj));
    CHECK(!gc.is_old(obj));
    CHECK(gc.contains(obj));

    // A large allocation goes straight to the old generation
    uintptr_t tenured = gc.allocate_masked(100 * 1024, 0, 20);
    REQUIRE(tenured != 0);
    CHECK(gc.is_old(tenured));
    CHECK(!gc.is_young(tenured));

    // Addresses outside the heap are neither
    CHECK(!gc.contains(0));
    CHECK(!gc.is_young(0));
    CHECK(!gc.is_old(0));
    CHECK(!gc.contains(0x12345678));
    CHECK(!gc.is_valid_object(0x12345678));
}

TEST_CASE("Phase 4 - Generational GC: Multiple and duplicate roots do not duplicate young objects") {
    gc::Heap gc(plain_config());

    // Allocate young object with data
    uintptr_t orig_obj = gc.allocate_masked(24, 0, 42);
    REQUIRE(orig_obj != 0);
    gc.store(orig_obj, 0, 0x1122334455667788ULL);
    gc.store(orig_obj, 1, 0x99AABBCCDDEEFF00ULL);

    // Three root pointers pointing to the SAME object, plus a duplicate slot address
    uint64_t r1 = orig_obj;
    uint64_t r2 = orig_obj;
    uint64_t r3 = orig_obj;
    {
        SlotRoots roots(gc, {&r1, &r2, &r3, &r1});
        gc.collect(gc::CollectionKind::Minor);
    }

    CHECK_EQ(gc.stats().minor_collections, 1ULL);
    CHECK(r1 != 0);
    CHECK_NE(r1, orig_obj); // Evacuated from eden

    // All root slots must point to the identical evacuated object (identity preserved!)
    CHECK_EQ(r1, r2);
    CHECK_EQ(r2, r3);

    // Verify object validity and payload integrity
    CHECK(gc.is_valid_object(r1));
    CHECK(gc.is_young(r1));
    CHECK_EQ(gc::Heap::load(r1, 0), 0x1122334455667788ULL);
    CHECK_EQ(gc::Heap::load(r1, 1), 0x99AABBCCDDEEFF00ULL);
}

TEST_CASE("Phase 4 - Generational GC: Card roots deduplication and tenured-to-young identity") {
    gc::Heap gc(plain_config());

    // Allocate an old (large) object with pointer fields 0 and 1
    uintptr_t tenured = gc.allocate_masked(100 * 1024, 0x3ULL, 99);
    REQUIRE(tenured != 0);
    CHECK(gc.is_old(tenured));

    // Allocate a young object
    uintptr_t young = gc.allocate_masked(16, 0, 55);
    REQUIRE(young != 0);
    CHECK(gc.is_young(young));
    gc.store(young, 0, 0xABCDEF0123456789ULL);

    // Old object fields 0 and 1 both point to young object
    gc.store(tenured, 0, young);
    gc.store(tenured, 1, young);

    // Stack root also points to young object
    uint64_t stack_root = young;
    {
        SlotRoots roots(gc, {&stack_root});
        // Minor collection: processes both the root and the dirty cards
        gc.collect(gc::CollectionKind::Minor);
    }

    CHECK_EQ(gc.stats().minor_collections, 1ULL);
    CHECK(gc.is_valid_object(stack_root));
    CHECK(gc.is_young(stack_root));
    CHECK_NE(stack_root, young);

    // Both fields in the old object must point to the EXACT same object as stack_root
    uint64_t f0 = gc::Heap::load(tenured, 0);
    uint64_t f1 = gc::Heap::load(tenured, 1);
    CHECK_EQ(f0, stack_root);
    CHECK_EQ(f1, stack_root);
    CHECK_EQ(gc::Heap::load(f0, 0), 0xABCDEF0123456789ULL);
}

// =============================================================================
// Task 2: Objects with >= 64 Fields (No Undefined 64-bit Shift)
// =============================================================================

TEST_CASE("Phase 4 - Generational GC: Objects with >= 64 fields do not invoke UB shift") {
    // 72 fields = 576 bytes
    constexpr size_t NUM_FIELDS = 72;
    constexpr size_t OBJ_SIZE = NUM_FIELDS * sizeof(uint64_t);

    gc::Heap gc(plain_config());

    // Pointer mask with bits 0 and 1 set
    uint64_t mask = 0x3ULL;
    uintptr_t large_obj = gc.allocate_masked(OBJ_SIZE, mask, 88);
    REQUIRE(large_obj != 0);

    // Allocate young child referenced by field 0
    uintptr_t child = gc.allocate_masked(16, 0, 77);
    REQUIRE(child != 0);
    gc.store(child, 0, 0x42424242ULL);
    gc.store(large_obj, 0, child);

    // Write distinctive values to fields beyond 64 (field 68 and 71): not
    // references under this mask, so a collection must copy them verbatim.
    gc.store(large_obj, 68, 0xDEADBEEFCAFE0068ULL);
    gc.store(large_obj, 71, 0xDEADBEEFCAFE0071ULL);

    uint64_t root = large_obj;
    gc.add_root(&root);

    gc.collect(gc::CollectionKind::Minor);

    CHECK_EQ(gc.stats().minor_collections, 1ULL);
    CHECK(gc.is_valid_object(root));
    CHECK_EQ(gc::Heap::load(root, 68), 0xDEADBEEFCAFE0068ULL);
    CHECK_EQ(gc::Heap::load(root, 71), 0xDEADBEEFCAFE0071ULL);

    uint64_t child_after = gc::Heap::load(root, 0);
    CHECK(gc.is_valid_object(child_after));
    CHECK_EQ(gc::Heap::load(child_after, 0), 0x42424242ULL);

    // A full collection promotes both
    gc.collect(gc::CollectionKind::Full);

    CHECK_EQ(gc.stats().full_collections, 1ULL);
    CHECK(gc.is_valid_object(root));
    CHECK(gc.is_old(root));
    CHECK_EQ(gc::Heap::load(root, 68), 0xDEADBEEFCAFE0068ULL);
    CHECK_EQ(gc::Heap::load(root, 71), 0xDEADBEEFCAFE0071ULL);
    CHECK(gc.is_old(gc::Heap::load(root, 0)));
    CHECK_EQ(gc::Heap::load(gc::Heap::load(root, 0), 0), 0x42424242ULL);
    gc.remove_root(&root);
}

TEST_CASE("Phase 4 - Heap: Objects with >= 64 fields survive a full collection") {
    constexpr size_t NUM_FIELDS = 80;
    constexpr size_t OBJ_SIZE = NUM_FIELDS * sizeof(uint64_t);

    gc::Heap heap(plain_config());

    uintptr_t large_obj = heap.allocate_masked(OBJ_SIZE, 0x1ULL, 0);
    REQUIRE(large_obj != 0);

    heap.store(large_obj, 0, 0);
    heap.store(large_obj, 75, 0xCAFEBABE88776655ULL);

    uint64_t root = large_obj;
    heap.add_root(&root);
    heap.collect(gc::CollectionKind::Full);
    heap.remove_root(&root);

    CHECK(heap.is_valid_object(root));
    CHECK_EQ(gc::Heap::load(root, 75), 0xCAFEBABE88776655ULL);
}

// =============================================================================
// Task 5: Arena Allocator Destructor Cleanup
// =============================================================================

namespace {
struct DestructionTracker {
    int id;
    std::vector<int> data;
    std::vector<int>* order_sink = nullptr;
    int* dtor_counter = nullptr;

    DestructionTracker(int id_, std::vector<int>* sink, int* counter)
        : id(id_), data(16, id_), order_sink(sink), dtor_counter(counter) {}

    ~DestructionTracker() {
        if (dtor_counter) (*dtor_counter)++;
        if (order_sink) order_sink->push_back(id);
    }
};
} // namespace

TEST_CASE("Phase 4 - Arena: Destructors invoked in reverse order on clear and destruction") {
    std::vector<int> order;
    int dtor_count = 0;

    {
        Arena arena(1024);
        CHECK_EQ(arena.cleanup_count(), size_t{0});

        arena.make<DestructionTracker>(1, &order, &dtor_count);
        arena.make<DestructionTracker>(2, &order, &dtor_count);
        arena.make<DestructionTracker>(3, &order, &dtor_count);

        CHECK_EQ(arena.cleanup_count(), size_t{3});
        CHECK_EQ(dtor_count, 0);

        arena.clear();

        CHECK_EQ(arena.cleanup_count(), size_t{0});
        CHECK_EQ(dtor_count, 3);
        // Reverse destruction order (LIFO: 3, 2, 1)
        REQUIRE_EQ(order.size(), size_t{3});
        CHECK_EQ(order[0], 3);
        CHECK_EQ(order[1], 2);
        CHECK_EQ(order[2], 1);
    }

    // Now test destruction via ~Arena()
    order.clear();
    dtor_count = 0;
    {
        Arena arena(1024);
        arena.make<DestructionTracker>(10, &order, &dtor_count);
        arena.make<DestructionTracker>(20, &order, &dtor_count);
    }
    CHECK_EQ(dtor_count, 2);
    REQUIRE_EQ(order.size(), size_t{2});
    CHECK_EQ(order[0], 20);
    CHECK_EQ(order[1], 10);
}

TEST_CASE("Phase 4 - Arena: reset_to_marker cleans up newly constructed objects") {
    std::vector<int> order;
    int dtor_count = 0;

    Arena arena(1024);
    arena.make<DestructionTracker>(1, &order, &dtor_count);
    arena.make<DestructionTracker>(2, &order, &dtor_count);

    Arena::Marker m = arena.get_marker();
    CHECK_EQ(m.cleanup_count, size_t{2});

    arena.make<DestructionTracker>(3, &order, &dtor_count);
    arena.make<DestructionTracker>(4, &order, &dtor_count);
    CHECK_EQ(arena.cleanup_count(), size_t{4});

    arena.reset_to_marker(m);

    CHECK_EQ(arena.cleanup_count(), size_t{2});
    CHECK_EQ(dtor_count, 2);
    // 4 and 3 destroyed in reverse order
    REQUIRE_EQ(order.size(), size_t{2});
    CHECK_EQ(order[0], 4);
    CHECK_EQ(order[1], 3);

    // Clear remaining 1 and 2
    arena.clear();
    CHECK_EQ(dtor_count, 4);
    REQUIRE_EQ(order.size(), size_t{4});
    CHECK_EQ(order[2], 2);
    CHECK_EQ(order[3], 1);
}

TEST_CASE("Phase 4 - Arena: Instruction heap vector memory cleaned up without leak") {
    Arena arena;
    Instruction* inst = arena.make<Instruction>(Opcode::call, Type::i64());
    CHECK(inst != nullptr);

    // Instruction has non-trivial destructor due to std::vector<Value*> operands_
    CHECK(!std::is_trivially_destructible_v<Instruction>);
    CHECK_EQ(arena.cleanup_count(), size_t{1});

    arena.clear();
    CHECK_EQ(arena.cleanup_count(), size_t{0});
}
