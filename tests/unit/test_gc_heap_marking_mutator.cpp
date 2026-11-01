// gc::Heap's marking cycle with a mutator between its steps (incremental
// update, docs/gc_contract.md "Marking Cycle"): the barrier's value range,
// shading by the barrier, by minor collections (through old objects only, so
// a weak target held for a moment in every cycle still dies) and by promotion, grey old
// allocation, each barrier elision rule with a cycle open, ephemerons
// rewritten behind the barrier's back, a full collection asked for inside a
// cycle, and a randomized mutator whose every cycle must pass the
// verification's fresh mark (HeapConfig::verify: a reachable object the
// cycle left unmarked stops the process).
//
// The usual shape: an old object C that only a young object names when the
// cycle starts, so the marker (which traces the old generation only) leaves
// it white, then a mutator step that hides it behind something already
// marked and drops the young path. C survives only if that step shaded it.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

using namespace brass::gc;
using brass::test::collect_scrubbed;
using brass::test::from_own_frame;
using brass::test::HeapRoot;
using brass::test::hide_address;
using brass::test::in_own_frame;
using brass::test::scrub_stack;
using brass::test::set_hidden;
using brass::test::unhide_address;

namespace {

constexpr uint64_t kCleared = 0x7FFC000000000000ULL;

void trace_weak_cell(uintptr_t payload, size_t, Tracer& t) {
    t.visit_weak(reinterpret_cast<uint64_t*>(payload), kCleared);
}

// [0] entry count, then (key, value) pairs.
void trace_table(uintptr_t payload, size_t, Tracer& t) {
    auto* w = reinterpret_cast<uint64_t*>(payload);
    for (uint64_t i = 0; i < w[0]; ++i) t.visit_ephemeron(&w[1 + 2 * i], &w[2 + 2 * i], 0, 0);
}

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.name = "mutator-words";
        return register_layout(d);
    }();
    return id;
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "mutator-weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId table_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_table;
        d.name = "mutator-table";
        return register_layout(d);
    }();
    return id;
}

HeapConfig config(unsigned threads = 1) {
    HeapConfig c;
    c.eden_bytes = size_t{1} << 20;
    c.survivor_bytes = size_t{256} << 10;
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{256} << 20;
    c.large_object_bytes = 16 * 1024;
    c.min_full_threshold_bytes = size_t{1} << 30;  // no full collection but the test's
    c.mark_threads = threads;
    c.parallel_mark_bytes = 0;
    c.read_environment = false;
    c.verify = true;  // every finish compares the cycle's marks with a fresh mark
    return c;
}

uint64_t* slot_of(uintptr_t object, size_t index) { return reinterpret_cast<uint64_t*>(object) + index; }

uint8_t card_of(const Heap& h, uintptr_t object) {
    return h.card_table_base()[(object - h.old_base()) >> kCardShift];
}

void drain(Heap& h) {
    while (!h.mark_slice(UINT64_MAX)) {
    }
}

// An old object C that only young object Y (held by `young_root`) names.
uintptr_t old_behind_young(Heap& h, uint64_t& young_root) {
    const uintptr_t c = h.allocate(16, words_layout(), kAllocOld);
    young_root = h.allocate(16, words_layout());
    h.store(static_cast<uintptr_t>(young_root), 0, c);
    return c;
}

// For the tests that need an object to die. The stack is read
// conservatively, and a collection that marks the old generation (a cycle's
// start and finish, a full collection) keeps every old object a stack word
// names (docs/gc_contract.md 2.5): those tests keep the objects' addresses
// hidden (stack_scrub.hpp), touch them only in frames of their own, and
// scrub the stack before each collection step.

// old_behind_young, made in its own frame; C's address hidden.
BRASS_TEST_NOINLINE uint64_t hidden_old_behind_young(Heap& h, uint64_t* young_root) {
    return hide_address(old_behind_young(h, *young_root));
}

// A new object, its address hidden.
BRASS_TEST_NOINLINE uint64_t hidden_allocate(Heap& h, size_t bytes, LayoutId layout, uint32_t flags = 0) {
    return hide_address(h.allocate(bytes, layout, flags));
}

BRASS_TEST_NOINLINE bool alive(const Heap& h, uint64_t hidden) {
    return h.is_valid_object(static_cast<uintptr_t>(unhide_address(hidden)));
}

BRASS_TEST_FORCEINLINE bool start_scrubbed(Heap& h) {
    scrub_stack();
    return h.start_marking();
}

BRASS_TEST_FORCEINLINE void finish_scrubbed(Heap& h) {
    scrub_stack();
    h.finish_marking();
}

} // namespace

TEST_CASE("gc::Heap marking - the barrier's value range is young, and the whole heap while a cycle is open") {
    Heap h(config());
    CHECK_EQ(h.barrier_value_base(), h.young_base());
    CHECK_EQ(h.barrier_value_span(), h.young_span());
    REQUIRE(h.start_marking());
    CHECK_EQ(h.barrier_value_base(), h.reservation_base());
    CHECK_EQ(h.barrier_value_span(), h.reservation_bytes());
    h.finish_marking();
    CHECK_EQ(h.barrier_value_base(), h.young_base());
    CHECK_EQ(h.barrier_value_span(), h.young_span());

    // With no cycle open an old object's store of an old reference is not the
    // barrier's business: no card.
    uint64_t root = h.allocate(16, words_layout(), kAllocOld);
    h.add_root(&root);
    const uintptr_t c = h.allocate(16, words_layout(), kAllocOld);
    h.collect(CollectionKind::Full);  // every card clean
    h.store(static_cast<uintptr_t>(root), 0, c);
    h.write_barrier_interior(reinterpret_cast<uintptr_t>(slot_of(static_cast<uintptr_t>(root), 1)), c);
    CHECK_EQ(card_of(h, static_cast<uintptr_t>(root)), kCardClean);
    h.remove_root(&root);
}

TEST_CASE("gc::Heap marking - a store during a cycle shades the old reference it stores") {
    // 0: write_barrier (Heap::store); 1: write_barrier_interior; 2: the
    // interior barrier into a large object, past its first card.
    for (int how = 0; how < 3; ++how) {
        Heap h(config());
        uint64_t young = 0;
        uint64_t holder = 0;
        h.add_root(&young);
        h.add_root(&holder);
        const uintptr_t c = old_behind_young(h, young);
        const size_t bytes = how == 2 ? 64 * 1024 : 16;
        const size_t index = how == 2 ? 5000 : 1;
        holder = h.allocate(bytes, words_layout(), kAllocOld);
        const auto o = static_cast<uintptr_t>(holder);
        REQUIRE(h.start_marking());
        drain(h);  // o is black, c white
        if (how == 0) {
            h.store(o, index, c);
        } else {
            *slot_of(o, index) = c;
            h.write_barrier_interior(reinterpret_cast<uintptr_t>(slot_of(o, index)), c);
        }
        young = 0;
        h.finish_marking();
        CHECK(h.is_valid_object(c));
        CHECK_EQ(Heap::load(o, index), uint64_t{c});
        h.remove_root(&young);
        h.remove_root(&holder);
    }
}

TEST_CASE("gc::Heap marking - a minor collection during a cycle keeps what is still held at the finish") {
    Heap h(config());
    HeapRoot<Heap> y1(h), y3(h), root3(h), holder(h), y2(h);
    // Through a live young object.
    const uint64_t c1 = hidden_old_behind_young(h, y1.slot);
    // Through a dirty card: an old holder names young y2, which names c2.
    const uint64_t c2 = hidden_old_behind_young(h, y2.slot);
    in_own_frame([&] {
        *holder.slot = h.allocate(16, words_layout(), kAllocOld);
        h.store(static_cast<uintptr_t>(*holder.slot), 0, *y2.slot);
    });
    *y2.slot = 0;
    // Through a root written after the start (roots carry no barrier).
    const uint64_t c3 = hidden_old_behind_young(h, y3.slot);
    REQUIRE(start_scrubbed(h));
    drain(h);
    set_hidden(root3.slot, c3);
    *y3.slot = 0;
    collect_scrubbed(h, CollectionKind::Minor);
    drain(h);
    // Still held at the finish: the finish's evacuation and root visit reach them.
    finish_scrubbed(h);
    CHECK(alive(h, c1));
    CHECK(alive(h, c2));
    CHECK(alive(h, c3));
    *y1.slot = 0;
    in_own_frame([&] { h.store(static_cast<uintptr_t>(*holder.slot), 0, 0); });
    *root3.slot = 0;
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_FALSE(alive(h, c1));
    CHECK_FALSE(alive(h, c2));
    CHECK_FALSE(alive(h, c3));
}

TEST_CASE("gc::Heap marking - what a root or young object held only across a minor dies with the cycle") {
    // The same paths, gone before the finish. The minor collection does not
    // shade through a root or a young object (the finish visits both again),
    // so what they held only for a moment is not kept: with a cycle open most
    // of the time, shading it would keep a WeakRef's briefly kept target
    // alive in every cycle.
    Heap h(config());
    HeapRoot<Heap> y1(h), y3(h), root3(h), holder(h), y2(h);
    const uint64_t c1 = hidden_old_behind_young(h, y1.slot);
    const uint64_t c2 = hidden_old_behind_young(h, y2.slot);
    in_own_frame([&] {
        *holder.slot = h.allocate(16, words_layout(), kAllocOld);
        h.store(static_cast<uintptr_t>(*holder.slot), 0, *y2.slot);
    });
    *y2.slot = 0;
    const uint64_t c3 = hidden_old_behind_young(h, y3.slot);
    REQUIRE(start_scrubbed(h));
    drain(h);
    set_hidden(root3.slot, c3);
    *y3.slot = 0;
    collect_scrubbed(h, CollectionKind::Minor);
    *y1.slot = 0;
    in_own_frame([&] { h.store(static_cast<uintptr_t>(*holder.slot), 0, 0); });
    *root3.slot = 0;
    finish_scrubbed(h);
    CHECK_FALSE(alive(h, c1));
    CHECK_FALSE(alive(h, c2));
    CHECK_FALSE(alive(h, c3));
}

TEST_CASE("gc::Heap marking - a weak target read back in every cycle still dies") {
    // A host's sweep that reads a weak slot once a cycle and holds the target
    // on a root for a moment (bro's node sweep, through WeakRef.deref and the
    // kept-objects list), with a minor collection inside that moment and a
    // cycle always open. The target must be cleared within a cycle or two.
    for (int minors_before_drop = 0; minors_before_drop < 3; ++minors_before_drop) {
        Heap h(config());
        HeapRoot<Heap> cell(h);
        {
            HeapRoot<Heap> target(h);
            in_own_frame([&] {
                *cell.slot = h.allocate(8, weak_cell_layout(), kAllocOld);
                *target.slot = h.allocate(16, words_layout(), kAllocOld);
                *slot_of(static_cast<uintptr_t>(*cell.slot), 0) = *target.slot;
            });
            collect_scrubbed(h, CollectionKind::Full);  // every card clean, nothing young
            in_own_frame([&] { REQUIRE_EQ(*slot_of(static_cast<uintptr_t>(*cell.slot), 0), *target.slot); });
        }
        HeapRoot<Heap> kept(h);
        int cycles = 0;
        for (; cycles < 6; ++cycles) {
            const bool cleared = from_own_frame([&] { return *slot_of(static_cast<uintptr_t>(*cell.slot), 0) == kCleared; });
            if (cleared) break;
            REQUIRE(start_scrubbed(h));
            // The read, held across minors.
            in_own_frame([&] { *kept.slot = *slot_of(static_cast<uintptr_t>(*cell.slot), 0); });
            (void)hidden_allocate(h, 16, words_layout());
            for (int i = 0; i < minors_before_drop; ++i) collect_scrubbed(h, CollectionKind::Minor);
            *kept.slot = 0;
            (void)h.mark_slice(0);
            collect_scrubbed(h, CollectionKind::Minor);
            drain(h);
            finish_scrubbed(h);
        }
        CHECK(cycles <= 2);
    }
}

TEST_CASE("gc::Heap marking - an unbarriered store into an object made in the cycle survives a minor") {
    // The minor collection cleans the dirty card of an object allocated in
    // the cycle; it shades the old values the card's slots hold, since the
    // finish no longer sees them.
    Heap h(config());
    uint64_t young = 0, frame = 0;
    h.add_root(&young);
    h.add_root(&frame);
    const uintptr_t c = old_behind_young(h, young);
    REQUIRE(h.start_marking());
    frame = h.allocate(16, words_layout(), kAllocOld);
    drain(h);
    *slot_of(static_cast<uintptr_t>(frame), 0) = c;
    young = 0;
    h.collect(CollectionKind::Minor);
    h.finish_marking();
    CHECK(h.is_valid_object(c));
    h.remove_root(&young);
    h.remove_root(&frame);
}

TEST_CASE("gc::Heap marking - the finish visits the roots again as a full collection sees them") {
    // A root source with a remembered set: a minor collection visits its
    // slot only while it names a young object. Written with an old one after
    // the start, only the finish's full root visit can find it.
    Heap h(config());
    uint64_t young = 0;
    uint64_t cell = 0;
    h.add_root(&young);
    h.add_root_source([&cell](Tracer& t) {
        if (t.purpose() != Tracer::Purpose::Minor || t.is_young(&cell)) t.visit(&cell);
    });
    const uintptr_t c = old_behind_young(h, young);
    REQUIRE(h.start_marking());
    drain(h);
    cell = c;
    young = 0;
    h.collect(CollectionKind::Minor);
    h.finish_marking();
    CHECK(h.is_valid_object(c));
    CHECK_EQ(cell, uint64_t{c});
    h.remove_root(&young);
}

TEST_CASE("gc::Heap marking - an object promoted during a cycle is marked as it is copied") {
    HeapConfig cfg = config();
    cfg.tenure_age = 1;
    Heap h(cfg);
    HeapRoot<Heap> root(h);
    REQUIRE(start_scrubbed(h));
    in_own_frame([&] { *root.slot = h.allocate(16, words_layout()); });
    collect_scrubbed(h, CollectionKind::Minor);
    const uint64_t promoted = root.hidden();
    CHECK(from_own_frame([&] { return h.is_old(static_cast<uintptr_t>(*root.slot)); }));
    *root.slot = 0;
    finish_scrubbed(h);
    CHECK(alive(h, promoted));  // dead before the finish, kept: it was marked when promoted
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_FALSE(alive(h, promoted));
}

TEST_CASE("gc::Heap marking - an old object allocated during a cycle is grey and its card dirty") {
    Heap h(config());
    REQUIRE(start_scrubbed(h));
    const uint64_t n = hidden_allocate(h, 16, words_layout(), kAllocOld);
    const uint64_t big = hidden_allocate(h, 64 * 1024, words_layout());  // large: old at once
    in_own_frame([&] {
        CHECK_EQ(card_of(h, static_cast<uintptr_t>(unhide_address(n))), kCardDirty);
        CHECK(h.is_old(static_cast<uintptr_t>(unhide_address(big))));
        CHECK_EQ(card_of(h, static_cast<uintptr_t>(unhide_address(big))), kCardDirty);
    });
    finish_scrubbed(h);
    CHECK(alive(h, n));  // never reachable, yet live to the cycle it was made in
    CHECK(alive(h, big));
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_FALSE(alive(h, n));
    CHECK_FALSE(alive(h, big));

    // Filled with a store the barrier never sees, after the marker scanned
    // it (a frame's arguments, B13): its dirty card covers the store.
    uint64_t young = 0, frame = 0;
    h.add_root(&young);
    h.add_root(&frame);
    const uintptr_t c = old_behind_young(h, young);
    REQUIRE(h.start_marking());
    frame = h.allocate(16, words_layout(), kAllocOld);
    drain(h);
    *slot_of(static_cast<uintptr_t>(frame), 0) = c;
    young = 0;
    h.finish_marking();
    CHECK(h.is_valid_object(c));
    h.remove_root(&young);
    h.remove_root(&frame);
}

TEST_CASE("gc::Heap marking - each barrier elision rule holds with a cycle open") {
    Heap h(config());
    uint64_t young = 0, holder = 0, fresh = 0, big = 0;
    for (uint64_t* r : {&young, &holder, &fresh, &big}) h.add_root(r);
    holder = h.allocate(32, words_layout(), kAllocOld);
    big = h.allocate(64 * 1024, words_layout());
    const auto o = static_cast<uintptr_t>(holder);
    const auto b = static_cast<uintptr_t>(big);

    // Rule 2 (WBE) and Z16: a store into an object allocated young since the
    // last GC point skips the barrier; the finish's evacuation traces it.
    uint64_t y2 = 0;
    h.add_root(&y2);
    const uintptr_t c2 = old_behind_young(h, y2);
    // Rule 3: an earlier barrier for the same object and value covers the
    // second store; rule 1 (a number) has nothing to shade.
    uint64_t y3 = 0;
    h.add_root(&y3);
    const uintptr_t c3 = old_behind_young(h, y3);
    // Card-only bulk writes: remember and remember_range.
    uint64_t y4 = 0, y5 = 0;
    h.add_root(&y4);
    h.add_root(&y5);
    const uintptr_t c4 = old_behind_young(h, y4);
    const uintptr_t c5 = old_behind_young(h, y5);

    REQUIRE(h.start_marking());
    drain(h);

    fresh = h.allocate(16, words_layout());
    *slot_of(static_cast<uintptr_t>(fresh), 0) = c2;

    h.store(o, 0, c3);
    *slot_of(o, 1) = c3;
    h.store(o, 0, 0);
    *slot_of(o, 2) = 42;

    const uint64_t words[2] = {c4, 0};
    std::memcpy(slot_of(o, 3), words, sizeof words[0]);
    h.remember(o);
    *slot_of(b, 6000) = c5;
    h.remember_range(b, reinterpret_cast<uintptr_t>(slot_of(b, 6000)),
                     reinterpret_cast<uintptr_t>(slot_of(b, 6001)));

    y2 = y3 = y4 = y5 = 0;
    h.finish_marking();
    CHECK(h.is_valid_object(c2));
    CHECK(h.is_valid_object(c3));
    CHECK(h.is_valid_object(c4));
    CHECK(h.is_valid_object(c5));
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(fresh), 0), uint64_t{c2});  // promoted, still pointing at c2
    CHECK_EQ(Heap::load(o, 1), uint64_t{c3});
    for (uint64_t* r : {&young, &holder, &fresh, &big, &y2, &y3, &y4, &y5}) h.remove_root(r);
}

TEST_CASE("gc::Heap marking - an ephemeron key rewritten behind the barrier's back is decided by the cycle") {
    for (bool minor_between : {false, true}) {
        Heap h(config());
        HeapRoot<Heap> table(h), key(h), value(h), young(h);
        in_own_frame([&] {
            *table.slot = h.allocate(5 * 8, table_layout(), kAllocOld);
            const auto t = static_cast<uintptr_t>(*table.slot);
            *slot_of(t, 0) = 2;
            *key.slot = h.allocate(16, words_layout(), kAllocOld);
            *value.slot = h.allocate(16, words_layout(), kAllocOld);
            h.store(t, 1, *key.slot);
            h.store(t, 2, *value.slot);
        });
        const uint64_t doomed = hidden_old_behind_young(h, young.slot);
        REQUIRE(start_scrubbed(h));
        drain(h);  // the entry's key was marked when the table was scanned: no entry pending
        in_own_frame([&] {
            const auto t = static_cast<uintptr_t>(*table.slot);
            *slot_of(t, 1) = unhide_address(doomed);  // a bulk copy, remembered by card only
            h.remember(t);
        });
        *young.slot = 0;
        if (minor_between) collect_scrubbed(h, CollectionKind::Minor);
        finish_scrubbed(h);
        // The key dies with nothing else naming it: the entry is cleared, not
        // left naming a freed object (verification after the finish would
        // stop the process on that).
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(*table.slot), 1), uint64_t{0});
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(*table.slot), 2), uint64_t{0});
        CHECK_FALSE(alive(h, doomed));
    }
}

TEST_CASE("gc::Heap marking - a full collection asked for inside a cycle finishes it, then collects whole") {
    Heap h(config());
    HeapRoot<Heap> root(h);
    in_own_frame([&] { *root.slot = h.allocate(16, words_layout(), kAllocOld); });
    const uint64_t x = root.hidden();
    REQUIRE(start_scrubbed(h));
    drain(h);  // x is marked
    *root.slot = 0;
    const uint64_t fulls = h.stats().full_collections;
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_FALSE(h.marking());
    CHECK_EQ(h.stats().full_collections, fulls + 2);
    CHECK_FALSE(alive(h, x));  // the cycle kept it; the whole collection after did not
}

namespace {

// A random mutator over nodes (4 strong words), large arrays, weak cells and
// ephemeron tables, interleaving every kind of store, allocation, minor
// collection and slice with whole cycles. Every finish runs the
// verification's fresh mark; a lost object stops the process.
struct Mutator {
    Heap& h;
    std::mt19937 rng;
    std::vector<uint64_t> roots;
    static constexpr size_t kNodeWords = 4;
    static constexpr size_t kBigWords = 4096;  // 32 KB: a large object
    static constexpr size_t kTableEntries = 4;

    Mutator(Heap& heap, uint32_t seed) : h(heap), rng(seed), roots(48, 0) {
        for (uint64_t& r : roots) h.add_root(&r);
    }
    ~Mutator() {
        for (uint64_t& r : roots) h.remove_root(&r);
    }

    size_t pick(size_t n) { return static_cast<size_t>(rng() % n); }
    static LayoutId layout(uintptr_t o) { return header_of(o)->layout; }
    static size_t strong_words(uintptr_t o) {
        if (layout(o) != words_layout()) return 0;
        return header_of(o)->size / 8;
    }

    uintptr_t make() {
        switch (pick(10)) {
        case 0: return h.allocate(kBigWords * 8, words_layout());
        case 1: {
            const uintptr_t t = h.allocate((1 + 2 * kTableEntries) * 8, table_layout(), pick(2) ? kAllocOld : 0);
            *slot_of(t, 0) = kTableEntries;
            return t;
        }
        case 2: return h.allocate(8, weak_cell_layout(), pick(2) ? kAllocOld : 0);
        case 3:
        case 4: return h.allocate(kNodeWords * 8, words_layout(), kAllocOld);
        default: return h.allocate(kNodeWords * 8, words_layout());
        }
    }

    // A live object: from a root, a few strong steps (and now and then a
    // weak cell's target, as WeakRef.deref() hands one out), or 0.
    uintptr_t reachable() {
        auto o = static_cast<uintptr_t>(roots[pick(roots.size())]);
        for (int steps = static_cast<int>(pick(6)); o != 0 && steps > 0; --steps) {
            if (layout(o) == weak_cell_layout()) {
                const uint64_t target = Heap::load(o, 0);
                if (target == 0 || target == kCleared || pick(2)) break;
                o = static_cast<uintptr_t>(target);
                continue;
            }
            const size_t n = strong_words(o);
            if (n == 0) break;
            const auto next = static_cast<uintptr_t>(Heap::load(o, pick(n)));
            if (next == 0) break;
            o = next;
        }
        return o;
    }

    uint64_t some_value() {
        switch (pick(4)) {
        case 0: return 0;
        case 1: return make();
        default: return reachable();
        }
    }

    // A store of `v` into `slot` of `o` through one of the ways a runtime
    // stores: the object barrier, the interior barrier, or a bulk write
    // remembered by card.
    void store(uintptr_t o, size_t index, uint64_t v) {
        uint64_t* slot = slot_of(o, index);
        switch (pick(4)) {
        case 0:
            h.store(o, index, v);
            return;
        case 1:
        case 2:
            *slot = v;
            h.write_barrier_interior(reinterpret_cast<uintptr_t>(slot), v);
            return;
        default:
            std::memcpy(slot, &v, sizeof v);
            if (pick(2)) h.remember(o);
            else h.remember_range(o, reinterpret_cast<uintptr_t>(slot), reinterpret_cast<uintptr_t>(slot + 1));
            return;
        }
    }

    void mutate() {
        // The value first: making one may collect and move what reachable()
        // would have returned.
        const uint64_t v = some_value();
        const uintptr_t o = reachable();
        if (o == 0) return;
        const LayoutId l = layout(o);
        if (l == weak_cell_layout()) store(o, 0, v);
        else if (l == table_layout()) store(o, 1 + pick(2 * kTableEntries), v);
        else store(o, pick(strong_words(o)), v);
    }

    // The hiding pattern: a reference moved from one object to another and
    // the first slot cleared.
    void move() {
        const uintptr_t a = reachable();
        const uintptr_t b = reachable();
        if (a == 0 || b == 0 || strong_words(a) == 0 || strong_words(b) == 0) return;
        const size_t k = pick(strong_words(a));
        const uint64_t v = Heap::load(a, k);
        store(b, pick(strong_words(b)), v);
        store(a, k, 0);
    }

    void step() {
        const size_t r = pick(100);
        if (r < 30) mutate();
        else if (r < 40) roots[pick(roots.size())] = pick(3) ? reachable() : make();
        else if (r < 52) move();
        else if (r < 60) h.allocate(64, words_layout());  // young garbage: fills eden
        else if (r < 75) {
            if (h.marking()) h.mark_slice(pick(4) == 0 ? 1'000'000 : 0);
        } else if (r < 80) h.collect(CollectionKind::Minor);
        else if (r < 85) {
            if (!h.marking()) h.start_marking();
        } else if (r < 88) {
            if (h.marking()) h.finish_marking();
        } else if (r < 89) h.collect(CollectionKind::Full);
        else mutate();
    }

    // Everything reachable is a valid object after a collection.
    void check_reachable() {
        std::vector<uintptr_t> stack;
        std::unordered_set<uintptr_t> seen;
        for (uint64_t r : roots) {
            if (r) stack.push_back(static_cast<uintptr_t>(r));
        }
        while (!stack.empty()) {
            const uintptr_t o = stack.back();
            stack.pop_back();
            if (!seen.insert(o).second) continue;
            REQUIRE(h.is_valid_object(o));
            for (size_t i = 0; i < strong_words(o); ++i) {
                if (const uint64_t w = Heap::load(o, i)) stack.push_back(static_cast<uintptr_t>(w));
            }
        }
    }
};

} // namespace

TEST_CASE("gc::Heap marking - a randomized mutator between slices, minors and allocations loses nothing") {
    for (unsigned threads : {1u, 4u}) {
        for (uint32_t seed : {11u, 12u, 13u, 14u}) {
            Heap h(config(threads));
            Mutator m(h, seed);
            for (size_t i = 0; i < m.roots.size(); i += 2) m.roots[i] = m.make();
            uint64_t cycles = 0;
            for (int i = 0; i < 4000; ++i) {
                const bool was = h.marking();
                m.step();
                if (was && !h.marking()) ++cycles;
            }
            if (h.marking()) h.finish_marking();
            m.check_reachable();
            h.collect(CollectionKind::Full);
            m.check_reachable();
            CHECK(cycles > 10);
            CHECK(h.stats().minor_collections > 50);
        }
    }
}
