// gc::Heap's marking cycle finish with the young generation left in place
// (docs/gc_contract.md "Marking Cycle"): the finish marks young objects with the old ones instead
// of promoting them all, so young survivors that hold the only path to old
// objects keep them, a young object made and linked during the cycle keeps
// what it names and stays remembered, weak cells and ephemerons naming young
// targets are decided by the young marks, finalizers of dead young objects
// run, dead young objects stop being objects, and the young generation does
// not move. Every case runs with the minor collections on one thread and on
// four (ParallelMinor::Always), under verification (the finish's fresh mark
// reaches every young object it must have marked), and a randomized mutator
// checks that what the roots reach survives every finish.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdint>
#include <iterator>
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

HeapConfig config(ParallelMinor minor) {
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
    c.minor_threads = minor == ParallelMinor::Always ? 4 : 1;
    c.read_environment = false;
    c.verify = true;  // every finish compares the cycle's marks with a fresh mark
    return c;
}

// One thread (Auto with minor_threads 1) and four.
constexpr ParallelMinor kModes[] = {ParallelMinor::Auto, ParallelMinor::Always};

uint64_t* slot_of(uintptr_t object, size_t index) { return reinterpret_cast<uint64_t*>(object) + index; }

uint8_t card_of(const Heap& h, uintptr_t object) {
    return h.card_table_base()[(object - h.old_base()) >> kCardShift];
}

void drain(Heap& h) {
    while (!h.mark_slice(UINT64_MAX)) {
    }
}

// A whole cycle with a mutator-free gap: start, every slice, finish. The
// stack is scrubbed before the start and the finish, which read it
// conservatively and keep every object a word of it names
// (docs/gc_contract.md 2.5): the tests that need objects to die keep their
// addresses hidden (stack_scrub.hpp) and touch them in frames of their own.
BRASS_TEST_FORCEINLINE void cycle(Heap& h) {
    scrub_stack();
    REQUIRE(h.start_marking());
    drain(h);
    scrub_stack();
    h.finish_marking();
}

uintptr_t young_words(Heap& h, size_t words) { return h.allocate(words * 8, words_layout()); }
uintptr_t old_words(Heap& h, size_t words) { return h.allocate(words * 8, words_layout(), kAllocOld); }

BRASS_TEST_NOINLINE bool alive(const Heap& h, uint64_t hidden) {
    return h.is_valid_object(static_cast<uintptr_t>(unhide_address(hidden)));
}

BRASS_TEST_NOINLINE uintptr_t addr(uint64_t hidden) { return static_cast<uintptr_t>(unhide_address(hidden)); }

} // namespace

TEST_CASE("gc::Heap finish in place - young survivors holding the only path to old objects keep them") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        HeapRoot<Heap> root(h), holder(h);
        // Hidden: c, e, and the young a, b, d once they have survived.
        uint64_t c = 0, e = 0, a = 0, b = 0, d = 0;
        in_own_frame([&] {
            // root -> young a -> young b -> old c: only young objects name c.
            const uintptr_t cc = old_words(h, 2);
            c = hide_address(cc);
            *root.slot = young_words(h, 2);
            const uintptr_t b0 = young_words(h, 2);
            h.store(static_cast<uintptr_t>(*root.slot), 0, b0);
            h.store(b0, 0, cc);
            // holder (old) -> young d -> old e: the path goes through a dirty card.
            *holder.slot = old_words(h, 2);
            const uintptr_t ee = old_words(h, 2);
            e = hide_address(ee);
            const uintptr_t d0 = young_words(h, 2);
            h.store(d0, 0, ee);
            h.store(static_cast<uintptr_t>(*holder.slot), 0, d0);
        });
        collect_scrubbed(h, CollectionKind::Minor);  // a, b and d are survivors now
        in_own_frame([&] {
            const auto aa = static_cast<uintptr_t>(*root.slot);
            REQUIRE(h.is_young(aa));
            a = hide_address(aa);
            b = hide_address(Heap::load(aa, 0));
            const auto dd = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(*holder.slot), 0));
            REQUIRE(h.is_young(dd));
            d = hide_address(dd);
        });
        const uint64_t promoted = h.stats().promoted_bytes;

        cycle(h);  // the marker leaves c and e white; only the finish reaches them
        CHECK(alive(h, c));
        CHECK(alive(h, e));
        // Nothing moved and nothing was promoted.
        in_own_frame([&] {
            CHECK(*root.slot == addr(a));
            CHECK(Heap::load(addr(a), 0) == addr(b));
            CHECK(Heap::load(addr(b), 0) == addr(c));
            CHECK(Heap::load(static_cast<uintptr_t>(*holder.slot), 0) == addr(d));
            CHECK(h.is_young(addr(a)));
            CHECK(h.is_young(addr(b)));
            CHECK(h.is_young(addr(d)));
            // The holder still remembers its young object.
            CHECK_EQ(card_of(h, static_cast<uintptr_t>(*holder.slot)), kCardDirty);
        });
        CHECK_EQ(h.stats().promoted_bytes, promoted);
        collect_scrubbed(h, CollectionKind::Minor);
        in_own_frame([&] {
            CHECK(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(*root.slot), 0)), 0) == addr(c));
            CHECK(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(*holder.slot), 0)), 0) == addr(e));
        });

        // Dropped, the next cycle frees them: the young path dies in place.
        *root.slot = 0;
        in_own_frame([&] { h.store(static_cast<uintptr_t>(*holder.slot), 0, 0); });
        cycle(h);
        CHECK_FALSE(alive(h, c));
        CHECK_FALSE(alive(h, e));
    }
}

TEST_CASE("gc::Heap finish in place - a dead old object's dirty card keeps nothing alive") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        // A mature owner and a large one, each naming a young object and an
        // old one, their cards dirty. Dropped before the cycle, they are dead
        // at its finish: nothing they name may be marked through their cards.
        uint64_t owner = 0, large = 0, target = 0, large_target = 0;  // hidden
        in_own_frame([&] {
            const uintptr_t o = old_words(h, 4);
            const uintptr_t l = old_words(h, 4096);  // 32 KB: a large object
            REQUIRE_FALSE(h.is_young(l));
            const uintptr_t t = old_words(h, 2);
            const uintptr_t lt = old_words(h, 2);
            h.store(o, 0, young_words(h, 2));
            h.store(o, 1, t);
            h.store(l, 3000, young_words(h, 2));  // a card past the first
            h.store(l, 3001, lt);
            owner = hide_address(o);
            large = hide_address(l);
            target = hide_address(t);
            large_target = hide_address(lt);
        });
        collect_scrubbed(h, CollectionKind::Minor);  // the young ones survive through the cards
        in_own_frame([&] {
            REQUIRE(h.is_young(static_cast<uintptr_t>(Heap::load(addr(owner), 0))));
            REQUIRE(h.is_young(static_cast<uintptr_t>(Heap::load(addr(large), 3000))));
        });

        cycle(h);
        CHECK_FALSE(alive(h, owner));
        CHECK_FALSE(alive(h, large));
        CHECK_FALSE(alive(h, target));
        CHECK_FALSE(alive(h, large_target));
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

// A black large array's dirty cards are rescanned by their own ranges, and
// every young object stored behind them, at the first card, in the middle,
// across a card boundary and at the last slot, keeps the old object it alone
// names.
TEST_CASE("gc::Heap finish in place - a black large array's dirty ranges are rescanned") {
    constexpr size_t kWords = 8192;  // 64 KB: four of the marker's slices
    constexpr size_t kSlots[] = {1, 2500, 2559, 2560, 5000, kWords - 1};
    for (ParallelMinor mode : kModes) {
        {
            Heap h(config(mode));
            HeapRoot<Heap> root(h);
            // The targets' addresses in the C++ heap, where no stack word holds them.
            std::vector<uintptr_t> targets;
            in_own_frame([&] {
                *root.slot = old_words(h, kWords);
                for (size_t i = 0; i < std::size(kSlots); ++i) targets.push_back(old_words(h, 1));
            });
            scrub_stack();
            REQUIRE(h.start_marking());
            drain(h);  // the array black, the targets white and unreachable
            in_own_frame([&] {
                const auto large = static_cast<uintptr_t>(*root.slot);
                for (size_t i = 0; i < std::size(kSlots); ++i) {
                    const uintptr_t y = young_words(h, 1);
                    *slot_of(y, 0) = targets[i];  // a young object's store: no barrier
                    h.store(large, kSlots[i], y);
                }
            });
            scrub_stack();
            h.finish_marking();
            in_own_frame([&] {
                const auto large = static_cast<uintptr_t>(*root.slot);
                for (size_t i = 0; i < std::size(kSlots); ++i) {
                    CHECK(h.is_valid_object(targets[i]));
                    const auto y = static_cast<uintptr_t>(Heap::load(large, kSlots[i]));
                    CHECK(h.is_young(y));
                    CHECK(h.is_valid_object(y));
                }
            });
            collect_scrubbed(h, CollectionKind::Minor);
            in_own_frame([&] {
                const auto large = static_cast<uintptr_t>(*root.slot);
                for (size_t i = 0; i < std::size(kSlots); ++i) {
                    const auto y = static_cast<uintptr_t>(Heap::load(large, kSlots[i]));
                    CHECK_EQ(Heap::load(y, 0), uint64_t{targets[i]});
                }
                // Dropped from the array, the next cycle frees them.
                for (size_t slot : kSlots) h.store(large, slot, 0);
            });
            cycle(h);
            CHECK(from_own_frame([&] {
                for (uintptr_t t : targets) {
                    if (h.is_valid_object(t)) return false;
                }
                return true;
            }));
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
        HeapRoot<Heap> old_cells(h), young_cell(h), live(h), table(h), key(h);
        // Hidden: the old cells and values (held by rooted old objects, so
        // their addresses stay), and the young dead target.
        uint64_t cell_live = 0, cell_dead = 0, v_live = 0, v_dead = 0, dead = 0;
        in_own_frame([&] {
            // Two old weak cells (held by an old array) and a young one.
            *old_cells.slot = old_words(h, 2);
            const uintptr_t cl = h.allocate(8, weak_cell_layout(), kAllocOld);
            const uintptr_t cd = h.allocate(8, weak_cell_layout(), kAllocOld);
            h.store(static_cast<uintptr_t>(*old_cells.slot), 0, cl);
            h.store(static_cast<uintptr_t>(*old_cells.slot), 1, cd);
            cell_live = hide_address(cl);
            cell_dead = hide_address(cd);
            *young_cell.slot = h.allocate(8, weak_cell_layout());
            // An old ephemeron table: one entry's young key is held, the other's
            // is not; each value is an old object named only by its entry.
            *table.slot = h.allocate(5 * 8, table_layout(), kAllocOld);
            *slot_of(static_cast<uintptr_t>(*table.slot), 0) = 2;
            v_live = hide_address(old_words(h, 1));
            v_dead = hide_address(old_words(h, 1));
        });

        scrub_stack();
        REQUIRE(h.start_marking());
        drain(h);
        in_own_frame([&] {
            // Young targets, stored during the cycle.
            const auto t = static_cast<uintptr_t>(*table.slot);
            *live.slot = young_words(h, 1);
            const uintptr_t dd = young_words(h, 1);
            dead = hide_address(dd);
            h.store(addr(cell_live), 0, *live.slot);
            h.store(addr(cell_dead), 0, dd);
            h.store(static_cast<uintptr_t>(*young_cell.slot), 0, dd);
            *key.slot = young_words(h, 1);
            h.store(t, 1, *key.slot);
            h.store(t, 2, addr(v_live));
            h.store(t, 3, young_words(h, 1));  // a key nothing else names
            h.store(t, 4, addr(v_dead));
        });
        scrub_stack();
        h.finish_marking();

        in_own_frame([&] {
            const auto t = static_cast<uintptr_t>(*table.slot);
            CHECK_EQ(Heap::load(addr(cell_live), 0), *live.slot);  // in place: the same address
            CHECK(h.is_young(static_cast<uintptr_t>(*live.slot)));
            CHECK_EQ(Heap::load(addr(cell_dead), 0), kCleared);
            CHECK_EQ(Heap::load(static_cast<uintptr_t>(*young_cell.slot), 0), kCleared);
            CHECK_EQ(Heap::load(t, 1), *key.slot);
            CHECK(Heap::load(t, 2) == addr(v_live));
            // The dead young key's entry is cleared. Its value was stored through
            // the barrier during the cycle, so the cycle kept it (floating); the
            // next one, with no entry naming it, does not.
            CHECK_EQ(Heap::load(t, 3), uint64_t{0});
            CHECK_EQ(Heap::load(t, 4), uint64_t{0});
        });
        CHECK_FALSE(alive(h, dead));
        CHECK(alive(h, v_live));
        // Minor collections afterwards move what lived and keep the slots current.
        collect_scrubbed(h, CollectionKind::Minor);
        in_own_frame([&] {
            CHECK_EQ(Heap::load(addr(cell_live), 0), *live.slot);
            CHECK_EQ(Heap::load(static_cast<uintptr_t>(*table.slot), 1), *key.slot);
        });
        *live.slot = 0;
        collect_scrubbed(h, CollectionKind::Minor);
        CHECK_EQ(Heap::load(addr(cell_live), 0), kCleared);
        cycle(h);
        CHECK_FALSE(alive(h, v_dead));
        CHECK(alive(h, v_live));
    }
}

namespace {

void count_finalizer(void* context) { ++*static_cast<int*>(context); }

} // namespace

TEST_CASE("gc::Heap finish in place - a dead young object is finalized and stops being an object") {
    for (ParallelMinor mode : kModes) {
        Heap h(config(mode));
        HeapRoot<Heap> kept(h);
        int finalized = 0;
        in_own_frame([&] {
            *kept.slot = young_words(h, 2);
            h.add_finalizer(static_cast<uintptr_t>(*kept.slot), &count_finalizer, &finalized);
            h.add_finalizer(young_words(h, 2), &count_finalizer, &finalized);
        });
        // One in the survivor space, one in eden.
        collect_scrubbed(h, CollectionKind::Minor);
        const uint64_t eden_dropped = from_own_frame([&] {
            const uintptr_t o = young_words(h, 2);
            h.add_finalizer(o, &count_finalizer, &finalized);
            return hide_address(o);
        });
        CHECK_EQ(finalized, 1);  // the minor found the first dropped one dead
        cycle(h);
        CHECK_EQ(finalized, 2);
        CHECK_FALSE(alive(h, eden_dropped));
        in_own_frame([&] {
            CHECK(h.is_valid_object(static_cast<uintptr_t>(*kept.slot)));
            CHECK(h.is_young(static_cast<uintptr_t>(*kept.slot)));
        });
        *kept.slot = 0;
        collect_scrubbed(h, CollectionKind::Minor);
        CHECK_EQ(finalized, 3);
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
        {
            for (uint32_t seed : {21u, 22u, 23u}) {
                Heap h(config(mode));
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
