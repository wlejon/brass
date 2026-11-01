// gc::Heap weakness: weak slots, ephemerons (WeakMap semantics), finalizers
// and post-collection hooks, in minor and full collections, with objects
// moving under them.
//
// The stack is read conservatively (docs/gc_contract.md 2.5), so what a test
// means to die or move is held only in heap roots and touched only in frames
// of its own (stack_scrub.hpp).

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <memory>
#include <vector>

using namespace brass::gc;
using brass::test::collect_scrubbed;
using brass::test::from_own_frame;
using brass::test::HeapRoot;
using brass::test::in_own_frame;

namespace {

constexpr uint64_t kCleared = 0x7FFC000000000000ULL;  // an "undefined" word

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

const LayoutId kNode = mask_layout(0b01, 31);

void trace_weak_cell(uintptr_t payload, size_t, Tracer& t) {
    t.visit_weak(reinterpret_cast<uint64_t*>(payload), kCleared);
}

// [0] entry count, then (key, value) pairs.
void trace_ephemeron_table(uintptr_t payload, size_t, Tracer& t) {
    auto* w = reinterpret_cast<uint64_t*>(payload);
    for (uint64_t i = 0; i < w[0]; ++i) t.visit_ephemeron(&w[1 + 2 * i], &w[2 + 2 * i], 0, 0);
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId ephemeron_table_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_ephemeron_table;
        d.name = "ephemeron-table";
        return register_layout(d);
    }();
    return id;
}

uintptr_t node(Heap& h, uint64_t value) {
    uintptr_t n = h.allocate(16, kNode);
    reinterpret_cast<uint64_t*>(n)[1] = value;
    return n;
}

uint64_t word(const HeapRoot<Heap>& r, size_t i) { return Heap::load(static_cast<uintptr_t>(*r.slot), i); }

} // namespace

TEST_CASE("gc::Heap weak - a weak slot follows a surviving target and clears for a dead one") {
    for (CollectionKind kind : {CollectionKind::Minor, CollectionKind::Full}) {
        Heap h(small_config());
        HeapRoot<Heap> cell(h);
        HeapRoot<Heap> strong(h);
        in_own_frame([&] {
            *cell.slot = h.allocate(8, weak_cell_layout());
            *strong.slot = node(h, 5);
            h.store(static_cast<uintptr_t>(*cell.slot), 0, *strong.slot);
        });
        collect_scrubbed(h, kind);
        in_own_frame([&] {
            CHECK_EQ(word(cell, 0), *strong.slot);  // moved together
            CHECK_EQ(word(strong, 1), uint64_t{5});
        });
        *strong.slot = 0;
        collect_scrubbed(h, kind);
        if (kind == CollectionKind::Minor) collect_scrubbed(h, CollectionKind::Minor);
        CHECK_EQ(word(cell, 0), kCleared);
    }
}

TEST_CASE("gc::Heap weak - an old weak cell over a young target clears in a minor collection") {
    HeapConfig config = small_config();
    config.tenure_age = 6;  // the target stays young across the collections below
    Heap h(config);
    HeapRoot<Heap> cell(h);
    HeapRoot<Heap> strong(h);
    in_own_frame([&] {
        *cell.slot = h.allocate(8, weak_cell_layout(), kAllocOld);
        *strong.slot = node(h, 9);
        h.store(static_cast<uintptr_t>(*cell.slot), 0, *strong.slot);  // barrier: old cell, young target
    });
    collect_scrubbed(h, CollectionKind::Minor);
    in_own_frame([&] { CHECK_EQ(word(cell, 0), *strong.slot); });
    collect_scrubbed(h, CollectionKind::Minor);  // the card stayed dirty: still followed
    in_own_frame([&] { CHECK_EQ(word(cell, 0), *strong.slot); });
    *strong.slot = 0;
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(word(cell, 0), kCleared);
}

TEST_CASE("gc::Heap weak - ephemerons keep values only while their keys live") {
    for (CollectionKind kind : {CollectionKind::Minor, CollectionKind::Full}) {
        Heap h(small_config());
        HeapRoot<Heap> table(h);
        HeapRoot<Heap> k1(h);
        HeapRoot<Heap> k2(h);
        in_own_frame([&] {
            *table.slot = h.allocate(8 * 7, ephemeron_table_layout());
            *k1.slot = node(h, 1);
            *k2.slot = node(h, 2);
            const uint64_t k3 = node(h, 3);
            const auto t = static_cast<uintptr_t>(*table.slot);
            reinterpret_cast<uint64_t*>(t)[0] = 3;
            // k1 -> v1 (k1 rooted); k2 -> k3: the value of entry 3 is only
            // reachable through entry 2, whose value keeps k3 alive.
            h.store(t, 1, *k1.slot);
            h.store(t, 2, node(h, 10));
            h.store(t, 3, *k2.slot);
            h.store(t, 4, k3);  // value of k2 is k3
            h.store(t, 5, k3);
            h.store(t, 6, node(h, 30));  // value of k3
        });
        collect_scrubbed(h, kind);
        in_own_frame([&] {
            CHECK_EQ(word(table, 1), *k1.slot);
            CHECK_EQ(Heap::load(static_cast<uintptr_t>(word(table, 2)), 1), uint64_t{10});
            CHECK_EQ(word(table, 3), *k2.slot);
            CHECK_EQ(word(table, 5), word(table, 4));  // entry 3's key is entry 2's value: kept through the fixpoint
            CHECK_EQ(Heap::load(static_cast<uintptr_t>(word(table, 6)), 1), uint64_t{30});
        });

        *k2.slot = 0;  // entry 2 dies, and with its value entry 3's key
        collect_scrubbed(h, kind);
        if (kind == CollectionKind::Minor) collect_scrubbed(h, CollectionKind::Minor);
        in_own_frame([&] {
            CHECK_EQ(word(table, 1), *k1.slot);
            CHECK_EQ(word(table, 3), uint64_t{0});
            CHECK_EQ(word(table, 4), uint64_t{0});
            CHECK_EQ(word(table, 5), uint64_t{0});
            CHECK_EQ(word(table, 6), uint64_t{0});
        });
    }
}

TEST_CASE("gc::Heap weak - a value that refers to its own key does not keep the entry alive") {
    Heap h(small_config());
    HeapRoot<Heap> table(h);
    in_own_frame([&] {
        *table.slot = h.allocate(8 * 3, ephemeron_table_layout());
        const uintptr_t key = node(h, 1);
        const uintptr_t value = node(h, 2);
        h.store(value, 0, key);  // value -> key
        const auto t = static_cast<uintptr_t>(*table.slot);
        reinterpret_cast<uint64_t*>(t)[0] = 1;
        h.store(t, 1, key);
        h.store(t, 2, value);
    });
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_EQ(word(table, 1), uint64_t{0});
    CHECK_EQ(word(table, 2), uint64_t{0});
}

namespace {
int g_finalized = 0;
void count_finalized(void* context) { g_finalized += static_cast<int>(reinterpret_cast<uintptr_t>(context)); }
} // namespace

TEST_CASE("gc::Heap weak - finalizers run once, after the collection that finds the object dead") {
    g_finalized = 0;
    Heap h(small_config());
    HeapRoot<Heap> a(h);
    HeapRoot<Heap> b(h);
    in_own_frame([&] {
        *a.slot = node(h, 1);
        *b.slot = node(h, 2);
        h.add_finalizer(static_cast<uintptr_t>(*a.slot), &count_finalized, reinterpret_cast<void*>(uintptr_t{1}));
        h.add_finalizer(static_cast<uintptr_t>(*b.slot), &count_finalized, reinterpret_cast<void*>(uintptr_t{100}));
    });
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(g_finalized, 0);
    *a.slot = 0;
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(g_finalized, 1);
    collect_scrubbed(h, CollectionKind::Full);  // b promoted and alive
    CHECK_EQ(g_finalized, 1);
    *b.slot = 0;
    collect_scrubbed(h, CollectionKind::Minor);  // old: a minor collection does not decide
    CHECK_EQ(g_finalized, 1);
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_EQ(g_finalized, 101);
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_EQ(g_finalized, 101);
}

TEST_CASE("gc::Heap weak - post-collection hooks see where objects went") {
    Heap h(small_config());
    HeapRoot<Heap> live(h);
    // [0] the live object, [1] the dead one, as the hook last saw them: in
    // the C++ heap, where no stack word holds them.
    auto tracked = std::make_unique<uintptr_t[]>(2);
    in_own_frame([&] {
        *live.slot = node(h, 1);
        tracked[0] = static_cast<uintptr_t>(*live.slot);
        tracked[1] = node(h, 2);
    });
    int runs = 0;
    uintptr_t* t = tracked.get();
    h.add_post_collection_hook([&runs, t](Heap& heap, CollectionKind) {
        ++runs;
        t[0] = heap.survivor_of(t[0]);
        t[1] = t[1] ? heap.survivor_of(t[1]) : 0;
    });
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(runs, 1);
    in_own_frame([&] {
        CHECK(tracked[0] == static_cast<uintptr_t>(*live.slot));
        CHECK(tracked[1] == 0);
    });
    collect_scrubbed(h, CollectionKind::Full);
    in_own_frame([&] { CHECK(tracked[0] == static_cast<uintptr_t>(*live.slot)); });
    CHECK_EQ(h.survivor_of(12345), uintptr_t{12345});  // outside a collection: unchanged
}
