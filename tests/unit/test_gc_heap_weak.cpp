// gc::Heap weakness: weak slots, ephemerons (WeakMap semantics), finalizers
// and post-collection hooks, in minor and full collections, with objects
// moving under them.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <vector>

using namespace brass::gc;

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

} // namespace

TEST_CASE("gc::Heap weak - a weak slot follows a surviving target and clears for a dead one") {
    for (CollectionKind kind : {CollectionKind::Minor, CollectionKind::Full}) {
        Heap h(small_config());
        uint64_t cell = h.allocate(8, weak_cell_layout());
        uint64_t strong = node(h, 5);
        h.add_root(&cell);
        h.add_root(&strong);
        h.store(static_cast<uintptr_t>(cell), 0, strong);
        h.collect(kind);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell), 0), strong);  // moved together
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(strong), 1), uint64_t{5});
        strong = 0;
        h.collect(kind);
        if (kind == CollectionKind::Minor) h.collect(CollectionKind::Minor);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell), 0), kCleared);
        h.remove_root(&cell);
        h.remove_root(&strong);
    }
}

TEST_CASE("gc::Heap weak - an old weak cell over a young target clears in a minor collection") {
    HeapConfig config = small_config();
    config.tenure_age = 6;  // the target stays young across the collections below
    Heap h(config);
    uint64_t cell = h.allocate(8, weak_cell_layout(), kAllocOld);
    h.add_root(&cell);
    uint64_t strong = node(h, 9);
    h.add_root(&strong);
    h.store(static_cast<uintptr_t>(cell), 0, strong);  // barrier: old cell, young target
    h.collect(CollectionKind::Minor);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell), 0), strong);
    h.collect(CollectionKind::Minor);  // the card stayed dirty: still followed
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell), 0), strong);
    strong = 0;
    h.collect(CollectionKind::Minor);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell), 0), kCleared);
    h.remove_root(&cell);
    h.remove_root(&strong);
}

TEST_CASE("gc::Heap weak - ephemerons keep values only while their keys live") {
    for (CollectionKind kind : {CollectionKind::Minor, CollectionKind::Full}) {
        Heap h(small_config());
        uint64_t table = h.allocate(8 * 7, ephemeron_table_layout());
        h.add_root(&table);
        uint64_t k1 = node(h, 1), k2 = node(h, 2), k3 = node(h, 3);
        h.add_root(&k1);
        h.add_root(&k2);
        h.add_root(&k3);
        auto* w = reinterpret_cast<uint64_t*>(table);
        w[0] = 3;
        // k1 -> v1 (k1 rooted); k2 -> k3's key chain; the value of entry 3 is only
        // reachable through entry 2: k3 is kept alive by entry 2's value.
        const uint64_t v1 = node(h, 10);
        w = reinterpret_cast<uint64_t*>(table);
        h.store(static_cast<uintptr_t>(table), 1, k1);
        h.store(static_cast<uintptr_t>(table), 2, v1);
        h.store(static_cast<uintptr_t>(table), 3, k2);
        h.store(static_cast<uintptr_t>(table), 4, k3);   // value of k2 is k3
        const uint64_t v3 = node(h, 30);
        h.store(static_cast<uintptr_t>(table), 5, k3);
        h.store(static_cast<uintptr_t>(table), 6, v3);   // value of k3
        k3 = 0;  // k3 now lives only as k2's value
        h.collect(kind);
        w = reinterpret_cast<uint64_t*>(table);
        CHECK_EQ(w[1], k1);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(w[2]), 1), uint64_t{10});
        CHECK_EQ(w[3], k2);
        CHECK_EQ(w[5], w[4]);  // entry 3's key is entry 2's value: kept through the fixpoint
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(w[6]), 1), uint64_t{30});

        k2 = 0;  // entry 2 dies, and with its value entry 3's key
        h.collect(kind);
        if (kind == CollectionKind::Minor) h.collect(CollectionKind::Minor);
        w = reinterpret_cast<uint64_t*>(table);
        CHECK_EQ(w[1], k1);
        CHECK_EQ(w[3], uint64_t{0});
        CHECK_EQ(w[4], uint64_t{0});
        CHECK_EQ(w[5], uint64_t{0});
        CHECK_EQ(w[6], uint64_t{0});
        h.remove_root(&table);
        h.remove_root(&k1);
        h.remove_root(&k2);
        h.remove_root(&k3);
    }
}

TEST_CASE("gc::Heap weak - a value that refers to its own key does not keep the entry alive") {
    Heap h(small_config());
    uint64_t table = h.allocate(8 * 3, ephemeron_table_layout());
    h.add_root(&table);
    uintptr_t key = node(h, 1);
    uintptr_t value = node(h, 2);
    h.store(value, 0, key);  // value -> key
    auto* w = reinterpret_cast<uint64_t*>(table);
    w[0] = 1;
    h.store(static_cast<uintptr_t>(table), 1, key);
    h.store(static_cast<uintptr_t>(table), 2, value);
    h.collect(CollectionKind::Full);
    w = reinterpret_cast<uint64_t*>(table);
    CHECK_EQ(w[1], uint64_t{0});
    CHECK_EQ(w[2], uint64_t{0});
    h.remove_root(&table);
}

namespace {
int g_finalized = 0;
void count_finalized(void* context) { g_finalized += static_cast<int>(reinterpret_cast<uintptr_t>(context)); }
} // namespace

TEST_CASE("gc::Heap weak - finalizers run once, after the collection that finds the object dead") {
    g_finalized = 0;
    Heap h(small_config());
    uint64_t a = node(h, 1);
    uint64_t b = node(h, 2);
    h.add_root(&a);
    h.add_root(&b);
    h.add_finalizer(static_cast<uintptr_t>(a), &count_finalized, reinterpret_cast<void*>(uintptr_t{1}));
    h.add_finalizer(static_cast<uintptr_t>(b), &count_finalized, reinterpret_cast<void*>(uintptr_t{100}));
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_finalized, 0);
    a = 0;
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_finalized, 1);
    h.collect(CollectionKind::Full);  // b promoted and alive
    CHECK_EQ(g_finalized, 1);
    b = 0;
    h.collect(CollectionKind::Minor);  // old: a minor collection does not decide
    CHECK_EQ(g_finalized, 1);
    h.collect(CollectionKind::Full);
    CHECK_EQ(g_finalized, 101);
    h.collect(CollectionKind::Full);
    CHECK_EQ(g_finalized, 101);
    h.remove_root(&a);
    h.remove_root(&b);
}

TEST_CASE("gc::Heap weak - post-collection hooks see where objects went") {
    Heap h(small_config());
    uint64_t live = node(h, 1);
    h.add_root(&live);
    uintptr_t dead = node(h, 2);
    uintptr_t tracked_live = static_cast<uintptr_t>(live);
    uintptr_t tracked_dead = dead;
    int runs = 0;
    h.add_post_collection_hook([&](Heap& heap, CollectionKind) {
        ++runs;
        tracked_live = heap.survivor_of(tracked_live);
        tracked_dead = tracked_dead ? heap.survivor_of(tracked_dead) : 0;
    });
    h.collect(CollectionKind::Minor);
    CHECK_EQ(runs, 1);
    CHECK_EQ(tracked_live, static_cast<uintptr_t>(live));
    CHECK_EQ(tracked_dead, uintptr_t{0});
    h.collect(CollectionKind::Full);
    CHECK_EQ(tracked_live, static_cast<uintptr_t>(live));
    CHECK_EQ(h.survivor_of(12345), uintptr_t{12345});  // outside a collection: unchanged
    h.remove_root(&live);
}
