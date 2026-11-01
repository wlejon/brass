// gc::Heap's lazy sweep (docs/gc_contract.md "Lazy Sweep"): a full
// collection decides and settles liveness in its pause and leaves freeing the
// old generation's memory pending. These cover what the pending sweep must
// keep true: the dead are gone to every query at once; the allocator sweeps
// what it takes and never allocates over an unswept object; sweep_slice keeps
// to its budget and finishes; the next cycle's start finishes what is left
// (and its marks start clear); promotion during the sweep survives it; a
// minor collection's card scan never scans a dead object of an unswept block;
// large objects are freed, kept and reused correctly; finalizers, weak slots
// and hooks settle at the finish; OS memory is handed back in slices; an old
// space near its end finishes the sweep rather than collect again; and a
// randomized mutator under verification.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
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

// Counts its calls: an object of this layout that is scanned shows up here.
uint64_t g_spy_traces = 0;
void trace_spy(uintptr_t payload, size_t bytes, Tracer& t) {
    ++g_spy_traces;
    auto* w = reinterpret_cast<uint64_t*>(payload);
    for (size_t i = 0; i < bytes / 8; ++i) t.visit(w + i);
}

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.name = "lazy-sweep-words";
        return register_layout(d);
    }();
    return id;
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "lazy-sweep-weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId spy_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_spy;
        d.name = "lazy-sweep-spy";
        return register_layout(d);
    }();
    return id;
}

HeapConfig config() {
    HeapConfig c;
    c.eden_bytes = size_t{1} << 20;
    c.survivor_bytes = size_t{256} << 10;
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{256} << 20;
    c.large_object_bytes = 16 * 1024;
    c.min_full_threshold_bytes = size_t{1} << 30;  // no full collection but the test's
    c.mark_threads = 1;
    c.read_environment = false;
    c.verify = true;
    return c;
}

uint64_t* slot_of(uintptr_t object, size_t index) { return reinterpret_cast<uint64_t*>(object) + index; }

void sweep_all(Heap& h) {
    while (!h.sweep_slice(UINT64_MAX)) {
    }
}

// `count` old objects of `words` words, every `keep_every`-th kept on the
// returned roots list (the rest garbage once a full collection runs), each
// stamped with its index in word 0.
struct OldSet {
    std::vector<uintptr_t> all;
    std::vector<uint64_t> kept;
};

// In a frame of its own: the stack is read conservatively, and a collection
// that marks the old generation keeps what a stack word names
// (docs/gc_contract.md 2.5). The tests that need an object to die keep its
// address only in the C++ heap (these vectors) or hidden (stack_scrub.hpp),
// touch it in frames of their own, and scrub the stack before collecting.
BRASS_TEST_NOINLINE OldSet make_old(Heap& h, size_t count, size_t words, size_t keep_every) {
    OldSet set;
    set.kept.reserve(count / keep_every + 1);
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t o = h.allocate(words * 8, words_layout(), kAllocOld);
        *slot_of(o, 0) = (i << 1) | 1;  // a number, not a reference
        set.all.push_back(o);
        if (i % keep_every == 0) set.kept.push_back(o);
    }
    return set;
}

void root_all(Heap& h, std::vector<uint64_t>& roots) {
    for (uint64_t& r : roots) h.add_root(&r);
}

void unroot_all(Heap& h, std::vector<uint64_t>& roots) {
    for (uint64_t& r : roots) h.remove_root(&r);
}

BRASS_TEST_FORCEINLINE bool start_scrubbed(Heap& h) {
    scrub_stack();
    return h.start_marking();
}

BRASS_TEST_NOINLINE bool alive(const Heap& h, uint64_t hidden) {
    return h.is_valid_object(static_cast<uintptr_t>(unhide_address(hidden)));
}

// Whether every address listed is (or none is) a valid object.
BRASS_TEST_NOINLINE bool all_valid(const Heap& h, const std::vector<uintptr_t>& objects) {
    for (const uintptr_t o : objects) {
        if (!h.is_valid_object(o)) return false;
    }
    return true;
}

BRASS_TEST_NOINLINE bool none_valid(const Heap& h, const std::vector<uintptr_t>& objects) {
    for (const uintptr_t o : objects) {
        if (h.is_valid_object(o)) return false;
    }
    return true;
}

// Every nonzero root, as addresses.
BRASS_TEST_NOINLINE std::vector<uintptr_t> nonzero(const std::vector<uint64_t>& roots) {
    std::vector<uintptr_t> out;
    for (const uint64_t r : roots) {
        if (r) out.push_back(static_cast<uintptr_t>(r));
    }
    return out;
}

// Moves every `step`-th root (from `first`) into the returned list, zeroing it.
BRASS_TEST_NOINLINE std::vector<uintptr_t> drop_every(std::vector<uint64_t>& roots, size_t first, size_t step) {
    std::vector<uintptr_t> dropped;
    for (size_t i = first; i < roots.size(); i += step) {
        dropped.push_back(static_cast<uintptr_t>(roots[i]));
        roots[i] = 0;
    }
    return dropped;
}

} // namespace

TEST_CASE("gc::Heap lazy sweep - a full collection leaves the sweep pending, and the dead are gone at once") {
    Heap h(config());
    OldSet set = make_old(h, 4000, 6, 3);
    root_all(h, set.kept);
    CHECK_FALSE(h.sweeping());
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    CHECK(h.pressure().sweeping);
    CHECK(h.pressure().unswept_blocks > 0);
    std::unordered_set<uintptr_t> kept(set.kept.begin(), set.kept.end());
    size_t dead = 0;
    for (size_t i = 0; i < set.all.size(); ++i) {
        const uintptr_t o = set.all[i];
        if (kept.count(o)) {
            CHECK(h.is_valid_object(o));
            CHECK_EQ(h.find_object(o + 16), o);
        } else {
            ++dead;
            CHECK_FALSE(h.is_valid_object(o));
            CHECK_EQ(h.find_object(o + 16), uintptr_t{0});
            bool threw = false;
            try {
                h.check_access(o, 0, 8, "lazy sweep test");
            } catch (const std::runtime_error&) {
                threw = true;
            }
            CHECK(threw);
        }
    }
    CHECK(dead > 2000);
    size_t seen = 0;
    h.for_each_object([&](uintptr_t o) {
        if (h.is_old(o)) {
            CHECK(kept.count(o) == 1);
            ++seen;
        }
    });
    CHECK_EQ(seen, kept.size());
    h.verify();
    sweep_all(h);
    CHECK_FALSE(h.sweeping());
    CHECK_EQ(h.pressure().unswept_blocks, size_t{0});
    CHECK(h.sweep_slice(0));  // nothing pending: done at once
    for (uint64_t o : set.kept) CHECK_EQ(Heap::load(static_cast<uintptr_t>(o), 0) & 1, uint64_t{1});
    unroot_all(h, set.kept);
}

TEST_CASE("gc::Heap lazy sweep - swept by many slices it leaves the heap one unbounded slice leaves") {
    // The same program swept by zero-budget slices and by one unbounded
    // slice: the same old bytes in use, committed memory, and the same
    // objects.
    size_t used[2] = {}, committed[2] = {};
    std::vector<uintptr_t> objects[2];
    for (int sliced = 0; sliced < 2; ++sliced) {
        Heap h(config());
        OldSet set = make_old(h, 3000, 5, 4);
        root_all(h, set.kept);
        (void)h.allocate(64 * 1024, words_layout());  // a large object, dead
        h.collect(CollectionKind::Full);
        CHECK(h.sweeping());
        if (sliced == 1) {
            while (!h.sweep_slice(0)) {
            }
        } else {
            sweep_all(h);
        }
        used[sliced] = h.old_used_bytes();
        committed[sliced] = h.committed_bytes();
        h.for_each_object([&](uintptr_t o) { objects[sliced].push_back(o); });
        unroot_all(h, set.kept);
    }
    CHECK_EQ(used[0], used[1]);
    CHECK_EQ(committed[0], committed[1]);
    CHECK_EQ(objects[0].size(), objects[1].size());
}

TEST_CASE("gc::Heap lazy sweep - the allocator sweeps what it takes and reuses the dead's lines") {
    Heap h(config());
    OldSet set = make_old(h, 6000, 31, 8);  // a line each: seven of every eight lines dead
    root_all(h, set.kept);
    uintptr_t highest = 0;
    for (uintptr_t o : set.all) highest = std::max(highest, o);
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    const size_t unswept = h.pressure().unswept_blocks;
    const size_t committed = h.committed_bytes();
    // As many bytes as died: all of it fits in the swept holes, below the
    // highest old object, with no block taken from the frontier.
    std::vector<uint64_t> fresh(2500, 0);
    root_all(h, fresh);
    for (size_t i = 0; i < fresh.size(); ++i) {
        fresh[i] = h.allocate(48, words_layout(), kAllocOld);
        CHECK(fresh[i] < highest);
        *slot_of(static_cast<uintptr_t>(fresh[i]), 1) = set.kept[i % set.kept.size()];
        h.write_barrier(static_cast<uintptr_t>(fresh[i]), set.kept[i % set.kept.size()]);
    }
    CHECK(h.pressure().unswept_blocks < unswept);  // the allocator swept some
    CHECK(h.sweeping());                           // and left the rest
    CHECK_EQ(h.committed_bytes(), committed);
    // Nothing was allocated over an unswept object: every kept object still
    // carries its stamp, every fresh one its link.
    for (uint64_t o : set.kept) CHECK_EQ(Heap::load(static_cast<uintptr_t>(o), 0) & 1, uint64_t{1});
    sweep_all(h);
    for (size_t i = 0; i < fresh.size(); ++i) {
        CHECK(h.is_valid_object(static_cast<uintptr_t>(fresh[i])));
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(fresh[i]), 1), set.kept[i % set.kept.size()]);
    }
    for (uint64_t o : set.kept) CHECK_EQ(Heap::load(static_cast<uintptr_t>(o), 0) & 1, uint64_t{1});
    h.collect(CollectionKind::Full);  // the fresh objects were never marked by the old cycle: kept by this one
    for (uint64_t o : fresh) CHECK(h.is_valid_object(static_cast<uintptr_t>(o)));
    unroot_all(h, fresh);
    unroot_all(h, set.kept);
}

TEST_CASE("gc::Heap lazy sweep - sweep_slice keeps roughly to its budget and finishes") {
    HeapConfig c = config();
    c.verify = false;  // timing: verification walks the heap at each step
    Heap h(c);
    OldSet set = make_old(h, 100000, 31, 5);  // about 800 blocks
    root_all(h, set.kept);
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    using Clock = std::chrono::steady_clock;
    constexpr uint64_t kBudget = 2'000;
    uint64_t slices = 0, worst = 0;
    for (;;) {
        const auto t0 = Clock::now();
        const bool done = h.sweep_slice(kBudget);
        const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        ++slices;
        // The last slice may hand memory back to the OS in one call, whose
        // cost is the system's; every other keeps near the budget.
        if (!done) worst = std::max(worst, ns);
        if (done) break;
        REQUIRE(slices < 100000);
    }
    CHECK(slices > 3);
    // A look at the clock every 8 blocks (about a microsecond each): the
    // overshoot is that, and scheduling noise.
    CHECK(worst < kBudget + 2'000'000);
    CHECK_FALSE(h.sweeping());
    unroot_all(h, set.kept);
}

TEST_CASE("gc::Heap lazy sweep - a cycle's start finishes the pending sweep, and its marks start clear") {
    for (int how = 0; how < 2; ++how) {
        Heap h(config());
        OldSet set = make_old(h, 3000, 4, 3);
        root_all(h, set.kept);
        collect_scrubbed(h, CollectionKind::Full);
        REQUIRE(h.sweeping());
        // Drop more, so the next cycle has dead of its own: had the last
        // cycle's marks survived, these would look marked and live.
        const std::vector<uintptr_t> dropped = drop_every(set.kept, 0, 2);
        if (how == 0) {
            REQUIRE(start_scrubbed(h));
            CHECK_FALSE(h.sweeping());  // finished in the start's pause
            while (!h.mark_slice(UINT64_MAX)) {
            }
            scrub_stack();
            h.finish_marking();
        } else {
            collect_scrubbed(h, CollectionKind::Full);
        }
        CHECK(h.sweeping());  // this cycle's own, pending again
        CHECK(none_valid(h, dropped));
        CHECK(all_valid(h, nonzero(set.kept)));
        sweep_all(h);
        CHECK(none_valid(h, dropped));
        unroot_all(h, set.kept);
    }
}

TEST_CASE("gc::Heap lazy sweep - what a minor collection promotes during the sweep survives it") {
    HeapConfig c = config();
    c.tenure_age = 1;  // every survivor promoted at its first minor collection
    Heap h(c);
    OldSet set = make_old(h, 4000, 6, 2);
    root_all(h, set.kept);
    // Live large objects, more than one allocation's walk looks at, so the
    // large objects promoted below land past the walk's cursor (where it has
    // yet to judge, and would free what is unmarked).
    std::vector<uint64_t> big(40, 0);
    root_all(h, big);
    in_own_frame([&] {
        for (uint64_t& b : big) b = h.allocate(20 * 1024, words_layout());
    });
    collect_scrubbed(h, CollectionKind::Full);
    REQUIRE(h.sweeping());
    std::vector<uint64_t> young(300, 0);
    root_all(h, young);
    in_own_frame([&] {
        for (size_t i = 0; i < young.size(); ++i) {
            // Mature-sized, and every 50th 12 KB: young (below the large-object
            // threshold), promoted into the large-object space.
            const size_t bytes = i % 50 == 0 ? 12 * 1024 : 40;
            young[i] = h.allocate(bytes, words_layout());
            *slot_of(static_cast<uintptr_t>(young[i]), 0) = set.kept[i % set.kept.size()];
        }
    });
    collect_scrubbed(h, CollectionKind::Minor);
    REQUIRE(from_own_frame([&] {
        for (uint64_t o : young) {
            if (!h.is_old(static_cast<uintptr_t>(o))) return false;
        }
        return true;
    }));
    REQUIRE(h.sweeping());
    sweep_all(h);
    in_own_frame([&] {
        for (size_t i = 0; i < young.size(); ++i) {
            const auto o = static_cast<uintptr_t>(young[i]);
            CHECK(h.is_valid_object(o));
            CHECK_EQ(Heap::load(o, 0), set.kept[i % set.kept.size()]);
        }
    });
    // And they are ordinary old objects to the next cycle: kept while held,
    // freed once not.
    collect_scrubbed(h, CollectionKind::Full);
    CHECK(all_valid(h, nonzero(young)));
    const std::vector<uintptr_t> was = drop_every(young, 0, 1);
    collect_scrubbed(h, CollectionKind::Full);
    sweep_all(h);
    CHECK(none_valid(h, was));
    CHECK(all_valid(h, nonzero(big)));
    unroot_all(h, big);
    unroot_all(h, young);
    unroot_all(h, set.kept);
}

TEST_CASE("gc::Heap lazy sweep - a minor collection's card scan never scans a dead object of an unswept block") {
    Heap h(config());
    // Live L and dead D side by side in one card: a store into L dirties the
    // card both start in. D's trace function counts its calls.
    HeapRoot<Heap> live(h);
    const uint64_t dead = from_own_frame([&] {
        *live.slot = h.allocate(16, words_layout(), kAllocOld);
        const uintptr_t d = h.allocate(16, spy_layout(), kAllocOld);
        const auto l = static_cast<uintptr_t>(*live.slot);
        REQUIRE_EQ((d - h.old_base()) >> kCardShift, (l - h.old_base()) >> kCardShift);
        // D names a large object that dies with it: once freed (and its pages
        // reused), D's slot names memory that is no object.
        *slot_of(d, 0) = h.allocate(64 * 1024, words_layout());
        return hide_address(d);
    });
    collect_scrubbed(h, CollectionKind::Full);
    REQUIRE(h.sweeping());
    REQUIRE_FALSE(alive(h, dead));
    g_spy_traces = 0;
    in_own_frame([&] {
        // Dirties the shared card.
        h.store(static_cast<uintptr_t>(*live.slot), 0, h.allocate(16, words_layout()));
    });
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(g_spy_traces, uint64_t{0});
    CHECK(h.sweeping());
    sweep_all(h);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(g_spy_traces, uint64_t{0});
    CHECK(h.is_valid_object(static_cast<uintptr_t>(*live.slot)));
}

TEST_CASE("gc::Heap lazy sweep - large objects: the dead freed, those allocated during the sweep kept, pages reused") {
    Heap h(config());
    constexpr size_t kBig = 40 * 1024;
    // Forty, every fourth dropped: more than one allocation's walk looks at.
    std::vector<uint64_t> big(40, 0);
    root_all(h, big);
    in_own_frame([&] {
        for (uint64_t& b : big) b = h.allocate(kBig, words_layout());
    });
    const std::vector<uintptr_t> dropped = drop_every(big, 0, 4);
    collect_scrubbed(h, CollectionKind::Full);
    REQUIRE(h.sweeping());
    CHECK(none_valid(h, dropped));
    const size_t committed = h.committed_bytes();
    // A large object of the same size: no free run fits yet, so the
    // allocator walks the sweep forward and takes a dead one's pages rather
    // than growing the space.
    HeapRoot<Heap> again(h);
    HeapRoot<Heap> during(h);
    in_own_frame([&] {
        *again.slot = h.allocate(kBig, words_layout());
        CHECK(std::find(dropped.begin(), dropped.end(), static_cast<uintptr_t>(*again.slot)) != dropped.end());
    });
    CHECK(h.committed_bytes() <= committed);
    // Larger than any dead run: the walk's next steps find none, and it lands
    // at the frontier, past the cursor, while the walk is under way: kept by it.
    in_own_frame([&] { *during.slot = h.allocate(5 * kBig, words_layout()); });
    REQUIRE(h.sweeping());
    sweep_all(h);
    CHECK(all_valid(h, nonzero(big)));
    const uint64_t a = again.hidden();
    const uint64_t d = during.hidden();
    CHECK(alive(h, a));
    CHECK(alive(h, d));
    // Their marks were cleared by the walk: dropped, the next cycle frees them.
    *again.slot = 0;
    *during.slot = 0;
    collect_scrubbed(h, CollectionKind::Full);
    CHECK_FALSE(alive(h, a));
    CHECK_FALSE(alive(h, d));
    sweep_all(h);
    unroot_all(h, big);
}

namespace {
uint64_t g_finalized = 0;
void count_finalizer(void*) { ++g_finalized; }
} // namespace

TEST_CASE("gc::Heap lazy sweep - finalizers, weak slots and hooks settle at the finish, before anything is freed") {
    Heap h(config());
    HeapRoot<Heap> cell_live(h), cell_dead(h), target(h);
    // The doomed object's address, for the hook: in the C++ heap, where no
    // stack word holds it.
    auto doomed = std::make_unique<uintptr_t>(0);
    g_finalized = 0;
    in_own_frame([&] {
        *cell_live.slot = h.allocate(8, weak_cell_layout(), kAllocOld);
        *cell_dead.slot = h.allocate(8, weak_cell_layout(), kAllocOld);
        *target.slot = h.allocate(16, words_layout(), kAllocOld);
        *doomed = h.allocate(16, words_layout(), kAllocOld);
        *slot_of(static_cast<uintptr_t>(*cell_live.slot), 0) = *target.slot;
        *slot_of(static_cast<uintptr_t>(*cell_dead.slot), 0) = *doomed;
        h.add_finalizer(*doomed, &count_finalizer, nullptr);
    });
    uintptr_t hook_saw = 1;
    bool hook_ran = false;
    uintptr_t* const doomed_at = doomed.get();
    const auto hook = h.add_post_collection_hook([&hook_saw, &hook_ran, doomed_at](Heap& heap, CollectionKind kind) {
        if (kind != CollectionKind::Full) return;
        hook_ran = true;
        hook_saw = heap.survivor_of(*doomed_at);
    });
    collect_scrubbed(h, CollectionKind::Full);
    REQUIRE(h.sweeping());
    CHECK(hook_ran);
    CHECK_EQ(hook_saw, uintptr_t{0});
    CHECK_EQ(g_finalized, uint64_t{1});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(*cell_dead.slot), 0), kCleared);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(*cell_live.slot), 0), *target.slot);
    // A minor collection during the sweep decides nothing old again.
    collect_scrubbed(h, CollectionKind::Minor);
    sweep_all(h);
    CHECK_EQ(g_finalized, uint64_t{1});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(*cell_live.slot), 0), *target.slot);
    h.remove_post_collection_hook(hook);
}

// The release decommits at most 64 blocks (2 MB) a step, so a run of many MB
// takes many zero-budget slices rather than one past any budget.
TEST_CASE("gc::Heap lazy sweep - excess free blocks go back to the OS in the sweep's slices") {
    {
        HeapConfig c = config();
        c.verify = false;
        c.min_full_threshold_bytes = size_t{1} << 20;  // retained: 64 free blocks, not 32 MB
        Heap h(c);
        // ~15 MB live in a chain (word 1 links the one before), collected while
        // live so the threshold grows past it, then dropped: one collection
        // leaves it all dead in unswept blocks.
        uint64_t chain = 0;
        h.add_root(&chain);
        for (size_t i = 0; i < 60000; ++i) {
            const uintptr_t o = h.allocate(30 * 8, words_layout(), kAllocOld);
            *slot_of(o, 1) = chain;
            chain = o;
        }
        h.collect(CollectionKind::Full);
        sweep_all(h);
        chain = 0;
        h.collect(CollectionKind::Full);
        REQUIRE(h.sweeping());
        const size_t before = h.committed_bytes();
        uint64_t slices = 0;
        while (!h.sweep_slice(0)) ++slices;
        // Over 8 MB released at most 2 MB a step, and a zero-budget slice
        // ends at its first decommit.
        CHECK(h.committed_bytes() + (size_t{8} << 20) < before);
        CHECK(slices >= 4);
        h.remove_root(&chain);
    }
}

TEST_CASE("gc::Heap lazy sweep - an old space near its end finishes the sweep instead of collecting again") {
    HeapConfig c = config();
    c.verify = false;
    c.mature_reserve_bytes = size_t{16} << 20;
    Heap h(c);
    // Fill the mature space nearly to its promotion reserve with garbage,
    // collect, and fill it again: the room is all in unswept blocks.
    const size_t count = (size_t{13} << 20) / 256;
    (void)make_old(h, count, 31, 1);
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    const uint64_t fulls = h.stats().full_collections;
    for (size_t i = 0; i < count; ++i) (void)h.allocate(31 * 8, words_layout(), kAllocOld);
    CHECK_EQ(h.stats().full_collections, fulls);
}

namespace {

// A random mutator over old and young nodes, large arrays and weak cells,
// with full collections, cycles cut into slices, minor collections and sweep
// slices of random budgets interleaved: the pending sweep meets allocation,
// promotion, card scans and the next cycle's start in every order. The heap
// verifies before and after each collection and after each finished sweep,
// so a live object swept (or a dead one scanned) stops the process.
struct Mutator {
    Heap& h;
    std::mt19937 rng;
    std::vector<uint64_t> roots;
    static constexpr size_t kNodeWords = 4;

    Mutator(Heap& heap, uint32_t seed) : h(heap), rng(seed), roots(40, 0) {
        for (uint64_t& r : roots) h.add_root(&r);
    }
    ~Mutator() {
        for (uint64_t& r : roots) h.remove_root(&r);
    }

    size_t pick(size_t n) { return static_cast<size_t>(rng() % n); }
    static bool is_node(uintptr_t o) { return header_of(o)->layout == words_layout(); }

    uintptr_t make() {
        switch (pick(12)) {
        case 0: return h.allocate(4096 * 8, words_layout());          // large
        case 1: return h.allocate(1500 * 8, words_layout());          // young, promoted large
        case 2: return h.allocate(8, weak_cell_layout(), pick(2) ? kAllocOld : 0);
        case 3:
        case 4:
        case 5: return h.allocate(kNodeWords * 8, words_layout(), kAllocOld);
        default: return h.allocate(kNodeWords * 8, words_layout());
        }
    }

    uintptr_t reachable() {
        auto o = static_cast<uintptr_t>(roots[pick(roots.size())]);
        for (int steps = static_cast<int>(pick(5)); o != 0 && steps > 0; --steps) {
            if (!is_node(o)) {
                const uint64_t t = Heap::load(o, 0);
                if (t == 0 || t == kCleared || pick(2)) break;
                o = static_cast<uintptr_t>(t);
                continue;
            }
            const auto next = static_cast<uintptr_t>(Heap::load(o, pick(kNodeWords)));
            if (next == 0) break;
            o = next;
        }
        return o;
    }

    void mutate() {
        const uint64_t v = pick(3) ? reachable() : make();
        const uintptr_t o = reachable();
        if (o == 0) return;
        const size_t index = is_node(o) ? pick(kNodeWords) : 0;
        if (pick(2)) {
            h.store(o, index, v);
        } else {
            *slot_of(o, index) = v;
            h.write_barrier_interior(reinterpret_cast<uintptr_t>(slot_of(o, index)), v);
        }
    }

    void step() {
        const size_t r = pick(100);
        if (r < 35) mutate();
        else if (r < 47) roots[pick(roots.size())] = pick(3) ? reachable() : make();
        else if (r < 57) (void)h.allocate(64, words_layout());
        else if (r < 70) (void)h.sweep_slice(pick(3) == 0 ? 200'000 : 0);
        else if (r < 76) h.collect(CollectionKind::Minor);
        else if (r < 80) h.collect(CollectionKind::Full);
        else if (r < 84) {
            if (!h.marking()) (void)h.start_marking();
        } else if (r < 92) {
            if (h.marking()) (void)h.mark_slice(pick(2) ? 0 : 1'000'000);
        } else if (r < 95) {
            if (h.marking()) h.finish_marking();
        } else mutate();
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
            if (!is_node(o)) continue;
            const size_t words = std::min<size_t>(kNodeWords, header_of(o)->size / 8);
            for (size_t i = 0; i < words; ++i) {
                if (const uint64_t w = Heap::load(o, i)) stack.push_back(static_cast<uintptr_t>(w));
            }
        }
    }
};

} // namespace

TEST_CASE("gc::Heap lazy sweep - a randomized mutator with the sweep pending loses nothing under verification") {
    for (unsigned threads : {1u, 4u}) {
        for (uint32_t seed : {21u, 22u, 23u, 24u}) {
            HeapConfig c = config();
            c.mark_threads = threads;
            c.parallel_mark_bytes = 0;
            Heap h(c);
            Mutator m(h, seed);
            for (size_t i = 0; i < m.roots.size(); i += 2) m.roots[i] = m.make();
            uint64_t sweeps_seen = 0;
            for (int i = 0; i < 4000; ++i) {
                const bool was = h.sweeping();
                m.step();
                if (was && !h.sweeping()) ++sweeps_seen;
                if (i % 500 == 0) m.check_reachable();
            }
            m.check_reachable();
            if (h.marking()) h.finish_marking();
            sweep_all(h);
            m.check_reachable();
            CHECK(sweeps_seen > 10);
        }
    }
}
