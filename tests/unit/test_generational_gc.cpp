#include "test_framework.hpp"
#include <brass/gc/generational_gc.hpp>
#include <vector>

using namespace brass;

TEST_CASE("GenerationalGC - Basic allocation in Nursery and field read/write") {
    // 64 KB nursery, 32 KB survivor, 128 KB tenured
    GenerationalGC gc(64 * 1024, 32 * 1024, 128 * 1024);

    CHECK_EQ(gc.minor_collections(), 0ULL);
    CHECK_EQ(gc.major_collections(), 0ULL);
    CHECK_EQ(gc.total_allocations(), 0ULL);

    uintptr_t obj = gc.allocate(24, 0, 101);
    REQUIRE(obj != 0);
    CHECK(gc.is_valid_object(obj));
    CHECK(gc.is_in_nursery(obj));
    CHECK(!gc.is_in_survivor(obj));
    CHECK(!gc.is_in_tenured(obj));
    CHECK_EQ(gc.total_allocations(), 1ULL);

    const GenGcHeader* hdr = gc.get_header(obj);
    REQUIRE(hdr != nullptr);
    CHECK_EQ(hdr->size, 24U);
    CHECK_EQ(hdr->type_tag, 101U);
    CHECK_EQ(hdr->pointer_mask, 0ULL);
    CHECK_EQ(hdr->age, 0U);
    CHECK_EQ(hdr->generation, 0U);

    gc.write_field(obj, 0, 0x1111222233334444ULL);
    gc.write_field(obj, 1, 0x5555666677778888ULL);
    gc.write_field(obj, 2, 0x9999AAAABBBBCCCCULL);

    CHECK_EQ(gc.read_field(obj, 0), 0x1111222233334444ULL);
    CHECK_EQ(gc.read_field(obj, 1), 0x5555666677778888ULL);
    CHECK_EQ(gc.read_field(obj, 2), 0x9999AAAABBBBCCCCULL);
}

TEST_CASE("GenerationalGC - Minor scavenge: evacuation to survivor and age increment") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 64 * 1024);

    uintptr_t obj = gc.allocate(16, 0, 7);
    gc.write_field(obj, 0, 42ULL);
    gc.write_field(obj, 1, 99ULL);

    uintptr_t old_addr = obj;
    uintptr_t root = obj;
    std::vector<uintptr_t*> roots = { &root };

    gc.collect(roots); // Triggers minor scavenge

    CHECK_EQ(gc.minor_collections(), 1ULL);
    uintptr_t new_addr = root;
    CHECK_NE(old_addr, new_addr);
    CHECK(gc.is_valid_object(new_addr));
    CHECK(!gc.is_valid_object(old_addr));

    CHECK(gc.is_in_survivor(new_addr));
    CHECK(!gc.is_in_nursery(new_addr));

    const GenGcHeader* hdr = gc.get_header(new_addr);
    REQUIRE(hdr != nullptr);
    CHECK_EQ(hdr->age, 1U);
    CHECK_EQ(hdr->generation, GEN_YOUNG);

    CHECK_EQ(gc.read_field(new_addr, 0), 42ULL);
    CHECK_EQ(gc.read_field(new_addr, 1), 99ULL);
}

TEST_CASE("GenerationalGC - Tenuring: promotion to tenured space") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 64 * 1024);
    gc.set_tenuring_threshold(2);

    uintptr_t obj = gc.allocate(16, 0, 8);
    gc.write_field(obj, 0, 12345ULL);
    gc.write_field(obj, 1, 67890ULL);

    uintptr_t root = obj;
    std::vector<uintptr_t*> roots = { &root };

    // Scavenge 1: Nursery -> Survivor (age 1)
    gc.collect(roots);
    CHECK(gc.is_in_survivor(root));
    CHECK_EQ(gc.get_header(root)->age, 1U);

    // Scavenge 2: Survivor -> Tenured (age 2 >= threshold 2)
    gc.collect(roots);
    CHECK(gc.is_in_tenured(root));
    CHECK(!gc.is_in_survivor(root));
    CHECK(!gc.is_in_nursery(root));
    CHECK_EQ(gc.get_header(root)->generation, GEN_OLD);
    CHECK(gc.promoted_objects() >= 1);

    CHECK_EQ(gc.read_field(root, 0), 12345ULL);
    CHECK_EQ(gc.read_field(root, 1), 67890ULL);
}

TEST_CASE("GenerationalGC - Old-to-young reference via write barrier & card table") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 64 * 1024);
    gc.set_tenuring_threshold(2); // Promoted to tenured after 2 scavenges

    // Step 1: Allocate object that will become tenured
    // pointer_mask = 1 (field 0 is a GC reference)
    uintptr_t old_obj = gc.allocate(16, 1ULL, 10);
    gc.write_field(old_obj, 0, 0);
    gc.write_field(old_obj, 1, 777ULL);

    uintptr_t root_old = old_obj;
    std::vector<uintptr_t*> roots = { &root_old };
    gc.collect(roots); // Nursery -> Survivor (age 1)
    gc.collect(roots); // Survivor -> Tenured (age 2)

    old_obj = root_old;
    CHECK(gc.is_in_tenured(old_obj));

    // Step 2: Allocate a young object in nursery
    uintptr_t young_obj = gc.allocate(16, 0ULL, 20);
    gc.write_field(young_obj, 0, 8888ULL);
    gc.write_field(young_obj, 1, 9999ULL);
    CHECK(gc.is_in_nursery(young_obj));

    // Step 3: Tenured object points to young object
    gc.write_field(old_obj, 0, young_obj);
    gc.write_barrier(old_obj, young_obj); // DIRTIES CARD TABLE

    CHECK(gc.card_table().is_dirty_addr(old_obj));

    // Step 4: Run minor scavenge with ONLY root_old in roots!
    // young_obj is NOT in roots! It survives solely through the remembered set!
    roots = { &root_old };
    gc.collect(roots);

    // Card table remains dirty because evacuated_young is still in survivor (young generation)
    CHECK(gc.card_table().is_dirty_addr(old_obj));

    // The field in old_obj should now point to young_obj's evacuated location in survivor!
    uintptr_t evacuated_young = gc.read_field(old_obj, 0);
    REQUIRE(evacuated_young != 0);
    CHECK(gc.is_valid_object(evacuated_young));
    CHECK(gc.is_in_survivor(evacuated_young));
    CHECK_EQ(gc.read_field(evacuated_young, 0), 8888ULL);
    CHECK_EQ(gc.read_field(evacuated_young, 1), 9999ULL);

    // One more scavenge promotes evacuated_young to tenured (age 2 >= threshold 2)
    gc.collect(roots);
    CHECK_EQ(gc.minor_collections(), 4ULL);
    uintptr_t promoted_young = gc.read_field(old_obj, 0);
    CHECK(gc.is_in_tenured(promoted_young));
    // Now that the child is also in tenured, the card table is cleaned!
    CHECK(!gc.card_table().is_dirty_addr(old_obj));
}

TEST_CASE("GenerationalGC - Major collection fallback") {
    GenerationalGC gc(32 * 1024, 32 * 1024, 64 * 1024);

    uintptr_t alive = gc.allocate(16, 0, 1);
    uintptr_t dead = gc.allocate(16, 0, 2);
    (void)dead;

    gc.write_field(alive, 0, 555ULL);

    uintptr_t root = alive;
    std::vector<uintptr_t*> roots = { &root };

    gc.collect_major(roots);

    CHECK_EQ(gc.major_collections(), 1ULL);
    CHECK(gc.is_valid_object(root));
    CHECK_EQ(gc.read_field(root, 0), 555ULL);
}
