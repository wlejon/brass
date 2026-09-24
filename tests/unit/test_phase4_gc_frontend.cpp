#include "test_framework.hpp"
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/core/arena.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <vector>
#include <string>
#include <memory>

using namespace brass;

// =============================================================================
// Task 1: Generational GC Scavenge Re-Evacuation and Object Duplication Bug
// =============================================================================

TEST_CASE("Phase 4 - Generational GC: is_scavenge_source correctly filters survivor_to") {
    // 64 KB nursery, 32 KB survivor, 128 KB tenured
    GenerationalGC gc(64 * 1024, 32 * 1024, 128 * 1024);

    uintptr_t obj = gc.allocate(24, 0, 10);
    REQUIRE(obj != 0);

    // Object is in nursery -> scavenge source
    CHECK(gc.is_in_nursery(obj));
    CHECK(gc.is_young(obj));
    CHECK(gc.is_scavenge_source(obj));

    // Address in tenured is not a scavenge source
    uintptr_t tenured = gc.allocate(100 * 1024, 0, 20); // Large alloc goes to tenured
    REQUIRE(tenured != 0);
    CHECK(gc.is_in_tenured(tenured));
    CHECK(!gc.is_scavenge_source(tenured));

    // Invalid address outside heap is not a scavenge source
    CHECK(!gc.is_scavenge_source(0));
    CHECK(!gc.is_scavenge_source(0x12345678));
}

TEST_CASE("Phase 4 - Generational GC: Multiple and duplicate roots do not duplicate young objects") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 64 * 1024);

    // Allocate young object with data
    uintptr_t orig_obj = gc.allocate(24, 0, 42);
    REQUIRE(orig_obj != 0);
    gc.write_field(orig_obj, 0, 0x1122334455667788ULL);
    gc.write_field(orig_obj, 1, 0x99AABBCCDDEEFF00ULL);

    // Three root pointers pointing to the SAME object, plus a duplicate slot address
    uintptr_t r1 = orig_obj;
    uintptr_t r2 = orig_obj;
    uintptr_t r3 = orig_obj;
    std::vector<uintptr_t*> roots = { &r1, &r2, &r3, &r1 };

    gc.minor_collect(roots);

    CHECK_EQ(gc.minor_collections(), 1ULL);
    CHECK(r1 != 0);
    CHECK_NE(r1, orig_obj); // Evacuated from nursery

    // All root slots must point to the identical evacuated object (identity preserved!)
    CHECK_EQ(r1, r2);
    CHECK_EQ(r2, r3);

    // Verify object validity and payload integrity
    CHECK(gc.is_valid_object(r1));
    CHECK(gc.is_in_survivor(r1));
    CHECK_EQ(gc.read_field(r1, 0), 0x1122334455667788ULL);
    CHECK_EQ(gc.read_field(r1, 1), 0x99AABBCCDDEEFF00ULL);
}

TEST_CASE("Phase 4 - Generational GC: Card roots deduplication and tenured-to-young identity") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 128 * 1024);

    // Allocate a tenured object with pointer fields
    uintptr_t tenured = gc.allocate(100 * 1024, 0x3ULL, 99); // large object -> tenured
    REQUIRE(tenured != 0);
    CHECK(gc.is_in_tenured(tenured));

    // Allocate a young object in nursery
    uintptr_t young = gc.allocate(16, 0, 55);
    REQUIRE(young != 0);
    CHECK(gc.is_in_nursery(young));
    gc.write_field(young, 0, 0xABCDEF0123456789ULL);

    // Tenured object fields 0 and 1 both point to young object
    gc.write_field(tenured, 0, young);
    gc.write_field(tenured, 1, young);

    // Stack root also points to young object
    uintptr_t stack_root = young;
    std::vector<uintptr_t*> roots = { &stack_root };

    // Minor scavenge: processes both stack roots and card roots
    gc.minor_collect(roots);

    CHECK_EQ(gc.minor_collections(), 1ULL);
    CHECK(gc.is_valid_object(stack_root));
    CHECK(gc.is_in_survivor(stack_root));

    // Both fields in tenured must point to the EXACT same object as stack_root
    uintptr_t f0 = gc.read_field(tenured, 0);
    uintptr_t f1 = gc.read_field(tenured, 1);
    CHECK_EQ(f0, stack_root);
    CHECK_EQ(f1, stack_root);
    CHECK_EQ(gc.read_field(f0, 0), 0xABCDEF0123456789ULL);
}

// =============================================================================
// Task 2: Objects with >= 64 Fields (No Undefined 64-bit Shift)
// =============================================================================

TEST_CASE("Phase 4 - Generational GC: Objects with >= 64 fields do not invoke UB shift") {
    // 72 fields = 576 bytes
    constexpr size_t NUM_FIELDS = 72;
    constexpr size_t OBJ_SIZE = NUM_FIELDS * sizeof(uint64_t);

    GenerationalGC gc(64 * 1024, 32 * 1024, 128 * 1024);

    // Pointer mask with bits 0 and 1 set
    uint64_t mask = 0x3ULL;
    uintptr_t large_obj = gc.allocate(OBJ_SIZE, mask, 88);
    REQUIRE(large_obj != 0);

    // Allocate young child referenced by field 0
    uintptr_t child = gc.allocate(16, 0, 77);
    REQUIRE(child != 0);
    gc.write_field(child, 0, 0x42424242ULL);
    gc.write_field(large_obj, 0, child);

    // Write distinctive values to field beyond 64 (field 68 and 71)
    gc.write_field(large_obj, 68, 0xDEADBEEFCAFE0068ULL);
    gc.write_field(large_obj, 71, 0xDEADBEEFCAFE0071ULL);

    uintptr_t root = large_obj;
    std::vector<uintptr_t*> roots = { &root };

    // Minor scavenge - tests lines 370, 386 Cheney scan with f >= 64
    gc.minor_collect(roots);

    CHECK_EQ(gc.minor_collections(), 1ULL);
    CHECK(gc.is_valid_object(root));
    CHECK_EQ(gc.read_field(root, 68), 0xDEADBEEFCAFE0068ULL);
    CHECK_EQ(gc.read_field(root, 71), 0xDEADBEEFCAFE0071ULL);

    uintptr_t child_after = gc.read_field(root, 0);
    CHECK(gc.is_valid_object(child_after));
    CHECK_EQ(gc.read_field(child_after, 0), 0x42424242ULL);

    // Major collect - tests line 526 in Cheney scan within new_tenured with f >= 64
    gc.major_collect(roots);

    CHECK_EQ(gc.major_collections(), 1ULL);
    CHECK(gc.is_valid_object(root));
    CHECK(gc.is_in_tenured(root));
    CHECK_EQ(gc.read_field(root, 68), 0xDEADBEEFCAFE0068ULL);
    CHECK_EQ(gc.read_field(root, 71), 0xDEADBEEFCAFE0071ULL);
}

TEST_CASE("Phase 4 - MiniCheneyGC: Objects with >= 64 fields do not invoke UB shift") {
    constexpr size_t NUM_FIELDS = 80;
    constexpr size_t OBJ_SIZE = NUM_FIELDS * sizeof(uint64_t);

    MiniCheneyGC cheney(64 * 1024);

    uintptr_t large_obj = cheney.allocate(OBJ_SIZE, 0x1ULL);
    REQUIRE(large_obj != 0);

    cheney.write_field(large_obj, 0, 0);
    cheney.write_field(large_obj, 75, 0xCAFEBABE88776655ULL);

    uintptr_t root = large_obj;
    std::vector<uintptr_t*> roots = { &root };

    cheney.collect(roots);

    CHECK(cheney.is_valid_object(root));
    CHECK_EQ(cheney.read_field(root, 75), 0xCAFEBABE88776655ULL);
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
