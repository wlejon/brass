// gc::Heap's marking cycle finish with the young generation left in place
// (HeapConfig::finish_promote_all false, the default; docs/gc_contract.md
// "Marking Cycle"): the finish marks young objects with the old ones instead
// of promoting them all, so young survivors that hold the only path to old
// objects keep them, a young object made and linked during the cycle keeps
// what it names and stays remembered, weak cells and ephemerons naming young
// targets are decided by the young marks, finalizers of dead young objects
// run, dead young objects stop being objects, and the young generation does
// not move. Every case runs with the minor collections on one thread and on
// four (ParallelMinor::Always), under verification (the finish's fresh mark
// reaches every young object it must have marked), and a randomized mutator
// checks the same against the promote-all finish.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdint>
#include <iterator>
#include <random>
#include <unordered_set>
#include <vector>

using namespace brass::gc;

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
        d.name = "in-place-words";
        return register_layout(d);
    }();
    return id;
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "in-place-weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId table_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_table;
        d.name = "in-place-table";
        return register_layout(d);
    }();
    return id;
}

HeapConfig config(ParallelMinor minor, bool promote_all = false) {
    HeapConfig c;
    c.eden_bytes = size_t{1} << 20;
    c.survivor_bytes = size_t{512} << 10;
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{256} << 20;
    c.large_object_bytes = 16 * 1024;
    c.min_full_threshold_bytes = size_t{1} << 30;  // no full collection but the test's
    c.tenure_age = 7;                               // survivors stay young for a while
    c.mark_threads = minor == ParallelMinor::Always ? 4 : 1;
    c.parallel_mark_bytes = 0;
    c.parallel_minor = minor;
    c.minor_threads = 4;
    c.finish_promote_all = promote_all;
    c.read_environment = false;
    c.verify = true;  // every finish compares the cycle's marks with a fresh mark
    return c;
}

constexpr ParallelMinor kModes[] = {ParallelMinor::Off, ParallelMinor::Always};

uint64_t* slot_of(uintptr_t object, size_t index) { return reinterpret_cast<uint64_t*>(object) + index; }

uint8_t card_of(const Heap& h, uintptr_t object) {
    return h.card_table_base()[(object - h.old_base()) >> kCardShift];
}

void drain(Heap& h) {
    while (!h.mark_slice(UINT64_MAX)) {
    }
}

// A whole cycle with a mutator-free gap: start, every slice, finish.
void cycle(Heap& h) {
    REQUIRE(h.start_marking());
    drain(h);
    h.finish_marking();
}

uintptr_t young_words(Heap& h, size_t words) { return h.allocate(words * 8, words_layout()); }
uintptr_t old_words(Heap& h, size_t words) { return h.allocate(words * 8, words_layout(), kAllocOld); }

} // namespace

TEST_CASE("gc::Heap finish in place - young survivors holding the only path to old objects keep them") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        uint64_t root = 0, holder = 0;
        h.add_root(&root);
        h.add_root(&holder);
        // root -> young a -> young b -> old c: only young objects name c.
        const uintptr_t c = old_words(h, 2);
        root = young_words(h, 2);
        const uintptr_t b0 = young_words(h, 2);
        h.store(static_cast<uintptr_t>(root), 0, b0);
        h.store(b0, 0, c);
        // holder (old) -> young d -> old e: the path goes through a dirty card.
        holder = old_words(h, 2);
        const uintptr_t e = old_words(h, 2);
        const uintptr_t d0 = young_words(h, 2);
        h.store(d0, 0, e);
        h.store(static_cast<uintptr_t>(holder), 0, d0);
        h.collect(CollectionKind::Minor);  // a, b and d are survivors now
        const auto a = static_cast<uintptr_t>(root);
        REQUIRE(h.is_young(a));
        const auto b = static_cast<uintptr_t>(Heap::load(a, 0));
        const auto d = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holder), 0));
        REQUIRE(h.is_young(d));
        const uint64_t promoted = h.stats().promoted_bytes;

        cycle(h);  // the marker leaves c and e white; only the finish reaches them
        CHECK(h.is_valid_object(c));
        CHECK(h.is_valid_object(e));
        // Nothing moved and nothing was promoted.
        CHECK_EQ(root, uint64_t{a});
        CHECK_EQ(Heap::load(a, 0), uint64_t{b});
        CHECK_EQ(Heap::load(b, 0), uint64_t{c});
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(holder), 0), uint64_t{d});
        CHECK(h.is_young(a));
        CHECK(h.is_young(b));
        CHECK(h.is_young(d));
        CHECK_EQ(h.stats().promoted_bytes, promoted);
        // The holder still remembers its young object.
        CHECK_EQ(card_of(h, static_cast<uintptr_t>(holder)), kCardDirty);
        h.collect(CollectionKind::Minor);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(root), 0)), 0), uint64_t{c});
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holder), 0)), 0), uint64_t{e});

        // Dropped, the next cycle frees them: the young path dies in place.
        root = 0;
        h.store(static_cast<uintptr_t>(holder), 0, 0);
        cycle(h);
        CHECK_FALSE(h.is_valid_object(c));
        CHECK_FALSE(h.is_valid_object(e));
        h.remove_root(&root);
        h.remove_root(&holder);
    }
}

TEST_CASE("gc::Heap finish in place - a dead old object's dirty card keeps nothing alive") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        uint64_t root = 0;
        h.add_root(&root);
        // A mature owner and a large one, each naming a young object and an
        // old one, their cards dirty. Dropped before the cycle, they are dead
        // at its finish: nothing they name may be marked through their cards.
        const uintptr_t owner = old_words(h, 4);
        const uintptr_t large = old_words(h, 4096);  // 32 KB: a large object
        REQUIRE_FALSE(h.is_young(large));
        const uintptr_t target = old_words(h, 2);
        const uintptr_t large_target = old_words(h, 2);
        h.store(owner, 0, young_words(h, 2));
        h.store(owner, 1, target);
        h.store(large, 3000, young_words(h, 2));  // a card past the first
        h.store(large, 3001, large_target);
        h.collect(CollectionKind::Minor);  // the young ones survive through the cards
        REQUIRE(h.is_young(static_cast<uintptr_t>(Heap::load(owner, 0))));
        REQUIRE(h.is_young(static_cast<uintptr_t>(Heap::load(large, 3000))));

        cycle(h);
        CHECK_FALSE(h.is_valid_object(owner));
        CHECK_FALSE(h.is_valid_object(large));
        CHECK_FALSE(h.is_valid_object(target));
        CHECK_FALSE(h.is_valid_object(large_target));
        h.remove_root(&root);
    }
}

TEST_CASE("gc::Heap finish in place - a live owner the marker has not reached yet keeps its young objects") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        uint64_t root = 0;
        h.add_root(&root);
        // root -> old p -> old owner (mature) and old large; each names a
        // young object that names an old one. Finished with no slice between,
        // p, the owner and the large object are still white when the finish
        // reads the dirty cards: the drain reaches them and scans them whole.
        root = old_words(h, 2);
        const auto p = static_cast<uintptr_t>(root);
        const uintptr_t owner = old_words(h, 2);
        const uintptr_t large = old_words(h, 4096);
        const uintptr_t z = old_words(h, 1);
        const uintptr_t large_z = old_words(h, 1);
        h.store(p, 0, owner);
        h.store(p, 1, large);
        const uintptr_t y = young_words(h, 1);
        h.store(y, 0, z);
        h.store(owner, 0, y);
        const uintptr_t large_y = young_words(h, 1);
        h.store(large_y, 0, large_z);
        h.store(large, 3000, large_y);
        h.collect(CollectionKind::Minor);

        REQUIRE(h.start_marking());
        h.finish_marking();
        CHECK(h.is_valid_object(owner));
        CHECK(h.is_valid_object(large));
        CHECK(h.is_valid_object(z));
        CHECK(h.is_valid_object(large_z));
        const auto y1 = static_cast<uintptr_t>(Heap::load(owner, 0));
        const auto large_y1 = static_cast<uintptr_t>(Heap::load(large, 3000));
        CHECK(h.is_young(y1));
        CHECK(h.is_valid_object(y1));
        CHECK(h.is_valid_object(large_y1));
        h.collect(CollectionKind::Minor);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(owner, 0)), 0), uint64_t{z});
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(large, 3000)), 0), uint64_t{large_z});
        h.remove_root(&root);
    }
}

// HeapConfig::rescan_card_ranges: a black large array's dirty cards are
// rescanned by their own ranges (or by the marker's 16 KB slices holding
// them), and every young object stored behind them, at the first card, in the
// middle, across a card boundary and at the last slot, keeps the old object
// it alone names.
TEST_CASE("gc::Heap finish in place - a black large array's dirty ranges are rescanned") {
    constexpr size_t kWords = 8192;  // 64 KB: four of the marker's slices
    constexpr size_t kSlots[] = {1, 2500, 2559, 2560, 5000, kWords - 1};
    for (ParallelMinor mode : kModes) {
        for (bool ranges : {true, false}) {
            HeapConfig c = config(mode);
            c.rescan_card_ranges = ranges;
            Heap h(c);
            uint64_t root = 0;
            h.add_root(&root);
            root = old_words(h, kWords);
            const auto large = static_cast<uintptr_t>(root);
            std::vector<uintptr_t> targets;
            for (size_t i = 0; i < std::size(kSlots); ++i) targets.push_back(old_words(h, 1));
            REQUIRE(h.start_marking());
            drain(h);  // the array black, the targets white and unreachable
            for (size_t i = 0; i < std::size(kSlots); ++i) {
                const uintptr_t y = young_words(h, 1);
                *slot_of(y, 0) = targets[i];  // a young object's store: no barrier
                h.store(large, kSlots[i], y);
            }
            h.finish_marking();
            for (size_t i = 0; i < std::size(kSlots); ++i) {
                CHECK(h.is_valid_object(targets[i]));
                const auto y = static_cast<uintptr_t>(Heap::load(large, kSlots[i]));
                CHECK(h.is_young(y));
                CHECK(h.is_valid_object(y));
            }
            h.collect(CollectionKind::Minor);
            for (size_t i = 0; i < std::size(kSlots); ++i) {
                const auto y = static_cast<uintptr_t>(Heap::load(large, kSlots[i]));
                CHECK_EQ(Heap::load(y, 0), uint64_t{targets[i]});
            }
            // Dropped from the array, the next cycle frees them.
            for (size_t slot : kSlots) h.store(large, slot, 0);
            cycle(h);
            for (uintptr_t t : targets) CHECK_FALSE(h.is_valid_object(t));
            h.remove_root(&root);
        }
    }
}

TEST_CASE("gc::Heap finish in place - a young object made and linked during the cycle") {
    for (ParallelMinor mode : kModes) {
        for (bool minor_between : {false, true}) {
            Heap h(config(mode));
            uint64_t holder = 0, young = 0;
            h.add_root(&holder);
            h.add_root(&young);
            holder = old_words(h, 2);
            // c is old, named only by a young object the program then drops.
            const uintptr_t c = old_words(h, 2);
            young = young_words(h, 2);
            h.store(static_cast<uintptr_t>(young), 0, c);
            REQUIRE(h.start_marking());
            drain(h);  // holder black, c white
            // Made in the cycle and filled with no barrier (WBE rule 2), then
            // stored into the black holder: a young value, so only a card.
            const uintptr_t f = young_words(h, 2);
            *slot_of(f, 0) = c;
            h.store(static_cast<uintptr_t>(holder), 1, f);
            young = 0;
            if (minor_between) h.collect(CollectionKind::Minor);
            h.finish_marking();
            CHECK(h.is_valid_object(c));
            const auto now = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holder), 1));
            CHECK(h.is_young(now));
            CHECK(h.is_valid_object(now));
            CHECK_EQ(Heap::load(now, 0), uint64_t{c});
            CHECK_EQ(card_of(h, static_cast<uintptr_t>(holder)), kCardDirty);
            // The next minor collection still finds it through the card.
            h.collect(CollectionKind::Minor);
            const auto moved = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holder), 1));
            CHECK(h.is_valid_object(moved));
            CHECK_EQ(Heap::load(moved, 0), uint64_t{c});
            h.remove_root(&holder);
            h.remove_root(&young);
        }
    }
}

TEST_CASE("gc::Heap finish in place - weak cells and ephemerons with young targets are decided at the finish") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        uint64_t old_cells = 0, young_cell = 0, live = 0, table = 0, key = 0;
        for (uint64_t* r : {&old_cells, &young_cell, &live, &table, &key}) h.add_root(r);
        // Two old weak cells (held by an old array) and a young one.
        old_cells = old_words(h, 2);
        const uintptr_t cell_live = h.allocate(8, weak_cell_layout(), kAllocOld);
        const uintptr_t cell_dead = h.allocate(8, weak_cell_layout(), kAllocOld);
        h.store(static_cast<uintptr_t>(old_cells), 0, cell_live);
        h.store(static_cast<uintptr_t>(old_cells), 1, cell_dead);
        young_cell = h.allocate(8, weak_cell_layout());
        // An old ephemeron table: one entry's young key is held, the other's
        // is not; each value is an old object named only by its entry.
        table = h.allocate(5 * 8, table_layout(), kAllocOld);
        const auto t = static_cast<uintptr_t>(table);
        *slot_of(t, 0) = 2;
        const uintptr_t v_live = old_words(h, 1);
        const uintptr_t v_dead = old_words(h, 1);

        REQUIRE(h.start_marking());
        drain(h);
        // Young targets, stored during the cycle.
        live = young_words(h, 1);
        const uintptr_t dead = young_words(h, 1);
        h.store(cell_live, 0, live);
        h.store(cell_dead, 0, dead);
        h.store(static_cast<uintptr_t>(young_cell), 0, dead);
        key = young_words(h, 1);
        const uintptr_t key_dead = young_words(h, 1);
        h.store(t, 1, key);
        h.store(t, 2, v_live);
        h.store(t, 3, key_dead);
        h.store(t, 4, v_dead);
        h.finish_marking();

        CHECK_EQ(Heap::load(cell_live, 0), live);  // in place: the same address
        CHECK(h.is_young(static_cast<uintptr_t>(live)));
        CHECK_EQ(Heap::load(cell_dead, 0), kCleared);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(young_cell), 0), kCleared);
        CHECK_FALSE(h.is_valid_object(dead));
        CHECK_EQ(Heap::load(t, 1), key);
        CHECK_EQ(Heap::load(t, 2), uint64_t{v_live});
        CHECK(h.is_valid_object(v_live));
        // The dead young key's entry is cleared. Its value was stored through
        // the barrier during the cycle, so the cycle kept it (floating); the
        // next one, with no entry naming it, does not.
        CHECK_EQ(Heap::load(t, 3), uint64_t{0});
        CHECK_EQ(Heap::load(t, 4), uint64_t{0});
        // Minor collections afterwards move what lived and keep the slots current.
        h.collect(CollectionKind::Minor);
        CHECK_EQ(Heap::load(cell_live, 0), live);
        CHECK_EQ(Heap::load(t, 1), key);
        live = 0;
        h.collect(CollectionKind::Minor);
        CHECK_EQ(Heap::load(cell_live, 0), kCleared);
        cycle(h);
        CHECK_FALSE(h.is_valid_object(v_dead));
        CHECK(h.is_valid_object(v_live));
        for (uint64_t* r : {&old_cells, &young_cell, &live, &table, &key}) h.remove_root(r);
    }
}

namespace {

void count_finalizer(void* context) { ++*static_cast<int*>(context); }

} // namespace

TEST_CASE("gc::Heap finish in place - a dead young object is finalized and stops being an object") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        uint64_t kept = 0;
        h.add_root(&kept);
        int finalized = 0;
        kept = young_words(h, 2);
        const uintptr_t dropped = young_words(h, 2);
        h.add_finalizer(static_cast<uintptr_t>(kept), &count_finalizer, &finalized);
        h.add_finalizer(dropped, &count_finalizer, &finalized);
        // One in the survivor space, one in eden.
        h.collect(CollectionKind::Minor);
        const uintptr_t eden_dropped = young_words(h, 2);
        h.add_finalizer(eden_dropped, &count_finalizer, &finalized);
        CHECK_EQ(finalized, 1);  // the minor found `dropped` dead
        cycle(h);
        CHECK_EQ(finalized, 2);
        CHECK_FALSE(h.is_valid_object(eden_dropped));
        CHECK(h.is_valid_object(static_cast<uintptr_t>(kept)));
        CHECK(h.is_young(static_cast<uintptr_t>(kept)));
        kept = 0;
        h.collect(CollectionKind::Minor);
        CHECK_EQ(finalized, 3);
        h.remove_root(&kept);
    }
}

TEST_CASE("gc::Heap finish in place - finish_promote_all promotes the young generation as before") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode, true));
        uint64_t root = 0;
        h.add_root(&root);
        const uintptr_t c = old_words(h, 1);
        root = young_words(h, 1);
        h.store(static_cast<uintptr_t>(root), 0, c);
        cycle(h);
        CHECK(h.is_old(static_cast<uintptr_t>(root)));
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(root), 0), uint64_t{c});
        CHECK(h.is_valid_object(c));
        h.remove_root(&root);
    }
}

namespace {

// Random graphs of young and old nodes, rewired between slices, minor
// collections and finishes. Each finish runs the verification's fresh mark;
// what the roots reach must be valid objects after every collection.
struct Graph {
    Heap& h;
    std::mt19937 rng;
    std::vector<uint64_t> roots;
    static constexpr size_t kWords = 3;

    Graph(Heap& heap, uint32_t seed) : h(heap), rng(seed), roots(32, 0) {
        for (uint64_t& r : roots) h.add_root(&r);
    }
    ~Graph() {
        for (uint64_t& r : roots) h.remove_root(&r);
    }

    size_t pick(size_t n) { return static_cast<size_t>(rng() % n); }

    uintptr_t make() {
        if (pick(8) == 0) return h.allocate(8, weak_cell_layout(), pick(2) ? kAllocOld : 0);
        return h.allocate(kWords * 8, words_layout(), pick(3) == 0 ? kAllocOld : 0);
    }

    uintptr_t reachable() {
        auto o = static_cast<uintptr_t>(roots[pick(roots.size())]);
        for (int steps = static_cast<int>(pick(5)); o != 0 && steps > 0; --steps) {
            const uint64_t next = Heap::load(o, header_of(o)->layout == words_layout() ? pick(kWords) : 0);
            if (next == 0 || next == kCleared) break;
            o = static_cast<uintptr_t>(next);
        }
        return o;
    }

    void link() {
        const uint64_t v = pick(3) ? reachable() : make();
        const uintptr_t o = reachable();
        if (o == 0) return;
        const size_t index = header_of(o)->layout == words_layout() ? pick(kWords) : 0;
        if (pick(4) == 0 && header_of(o)->layout == words_layout()) {
            *slot_of(o, index) = v;  // a bulk write, remembered by card
            h.remember(o);
        } else {
            h.store(o, index, v);
        }
    }

    void step() {
        const size_t r = pick(100);
        if (r < 40) link();
        else if (r < 50) roots[pick(roots.size())] = pick(2) ? reachable() : make();
        else if (r < 60) (void)h.allocate(48, words_layout());
        else if (r < 75) {
            if (h.marking()) (void)h.mark_slice(pick(3) == 0 ? 1'000'000 : 0);
        } else if (r < 82) h.collect(CollectionKind::Minor);
        else if (r < 88) {
            if (!h.marking()) (void)h.start_marking();
        } else if (r < 92) {
            if (h.marking()) h.finish_marking();
        } else link();
    }

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
            if (header_of(o)->layout != words_layout()) continue;
            for (size_t i = 0; i < kWords; ++i) {
                if (const uint64_t w = Heap::load(o, i)) stack.push_back(static_cast<uintptr_t>(w));
            }
        }
    }
};

} // namespace

TEST_CASE("gc::Heap finish in place - a randomized mutator with young data across finishes loses nothing") {
    for (ParallelMinor mode : kModes) {
        for (bool promote_all : {false, true}) {
            for (uint32_t seed : {21u, 22u, 23u}) {
                Heap h(config(mode, promote_all));
                Graph g(h, seed);
                for (size_t i = 0; i < g.roots.size(); i += 2) g.roots[i] = g.make();
                uint64_t finishes = 0;
                for (int i = 0; i < 3000; ++i) {
                    const bool was = h.marking();
                    g.step();
                    if (was && !h.marking()) {
                        ++finishes;
                        g.check_reachable();
                    }
                }
                if (h.marking()) h.finish_marking();
                g.check_reachable();
                h.collect(CollectionKind::Minor);
                g.check_reachable();
                h.collect(CollectionKind::Full);
                g.check_reachable();
                CHECK(finishes > 10);
            }
        }
    }
}
