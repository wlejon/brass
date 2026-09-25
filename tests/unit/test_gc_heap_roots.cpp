// gc::Heap roots that are not plain object starts (derived and conservative
// slots, as the interpreters report them), the interior write barrier, and
// the process-wide default configuration.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <vector>

using namespace brass::gc;

namespace {

HeapConfig small_config() {
    HeapConfig c;
    c.eden_bytes = 64 * 1024;
    c.survivor_bytes = 64 * 1024;
    c.mature_reserve_bytes = size_t{64} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{4} << 20;
    c.read_environment = false;
    c.verify = true;
    return c;
}

const LayoutId kPair = mask_layout(0b01, 41);  // [0] reference, [1..] data

struct Slots {
    std::vector<uint64_t> derived;
    std::vector<uint64_t> conservative;
};

void visit_slots(Tracer& t, void* context) {
    auto* s = static_cast<Slots*>(context);
    for (uint64_t& w : s->derived) t.visit_derived(&w);
    for (uint64_t& w : s->conservative) t.visit_conservative(&w);
}

} // namespace

TEST_CASE("gc::Heap roots - a derived root keeps its object and its offset as the object moves") {
    for (CollectionKind kind : {CollectionKind::Minor, CollectionKind::Full}) {
        Heap h(small_config());
        Slots slots;
        const auto id = h.add_root_source(&visit_slots, &slots);
        const uintptr_t obj = h.allocate(64, kPair);
        reinterpret_cast<uint64_t*>(obj)[5] = 1234;
        slots.derived.push_back(obj + 40);         // interior: word 5
        slots.derived.push_back(obj);              // the start itself
        slots.derived.push_back(0x7FFD000000000000ULL | (obj + 8));  // tagged interior
        h.collect(kind);
        const uintptr_t moved = static_cast<uintptr_t>(slots.derived[1]);
        CHECK(moved != obj);  // young objects move in both kinds
        CHECK(h.is_valid_object(moved));
        CHECK_EQ(slots.derived[0], static_cast<uint64_t>(moved + 40));
        CHECK_EQ(*reinterpret_cast<uint64_t*>(slots.derived[0]), uint64_t{1234});
        CHECK_EQ(slots.derived[2], 0x7FFD000000000000ULL | (moved + 8));
        h.remove_root_source(id);
    }
}

TEST_CASE("gc::Heap roots - a conservative root is followed only when it names an object start") {
    Heap h(small_config());
    Slots slots;
    const auto id = h.add_root_source(&visit_slots, &slots);
    const uintptr_t obj = h.allocate(32, kPair);
    const uintptr_t other = h.allocate(32, kPair);
    slots.conservative.push_back(obj);         // an object: kept and updated
    slots.conservative.push_back(other + 16);  // inside one: left alone, and no root
    slots.conservative.push_back(12345);       // not an address at all
    h.collect(CollectionKind::Minor);
    CHECK(slots.conservative[0] != obj);
    CHECK(h.is_valid_object(static_cast<uintptr_t>(slots.conservative[0])));
    CHECK_EQ(slots.conservative[1], static_cast<uint64_t>(other + 16));
    CHECK_EQ(slots.conservative[2], uint64_t{12345});
    h.remove_root_source(id);
}

TEST_CASE("gc::Heap roots - the interior barrier remembers a store through a derived address") {
    Heap h(small_config());
    uint64_t old = h.allocate(64, mask_layout(~uint64_t{0}, 42), kAllocOld);
    h.add_root(&old);
    for (int i = 0; i < 3; ++i) {
        uint64_t young = h.allocate(16, kPair);
        reinterpret_cast<uint64_t*>(young)[1] = 77 + static_cast<uint64_t>(i);
        const uintptr_t slot = static_cast<uintptr_t>(old) + 8 * (4 + static_cast<uintptr_t>(i));
        *reinterpret_cast<uint64_t*>(slot) = young;
        h.write_barrier_interior(slot, young);  // no object start at hand
    }
    h.collect(CollectionKind::Minor);  // verify mode checks the card invariant too
    for (int i = 0; i < 3; ++i) {
        const uintptr_t young = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(old), 4 + i));
        CHECK(h.is_valid_object(young));
        CHECK_EQ(Heap::load(young, 1), uint64_t{77} + static_cast<uint64_t>(i));
    }
    h.remove_root(&old);
}

TEST_CASE("gc::Heap roots - heaps made without a configuration take the process default") {
    const HeapConfig saved = Heap::default_config();
    HeapConfig forbidden = small_config();
    forbidden.forbid_allocation = true;
    Heap::set_default_config(forbidden);
    {
        Heap h;  // the default: allocation forbidden
        // The inline fast path never admits an object, so every allocation
        // reaches the runtime, which stops the process; collections and
        // safepoints do nothing.
        CHECK(h.allocation_buffer()->top == h.allocation_buffer()->end);
        h.collect(CollectionKind::Full);
        h.safepoint_at(0, 0);
        CHECK_EQ(h.collection_count(), uint64_t{0});
    }
    Heap::set_default_config(saved);
    Heap h;
    CHECK(h.allocate(16, kPair) != 0);
}
