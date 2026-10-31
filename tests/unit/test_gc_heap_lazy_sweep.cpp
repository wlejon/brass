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
// space near its end finishes the sweep rather than collect again; the eager
// A/B switch; and a randomized mutator under verification.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace brass::gc;

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

HeapConfig config(bool lazy = true) {
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
    c.lazy_sweep = lazy;
    return c;
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
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

OldSet make_old(Heap& h, size_t count, size_t words, size_t keep_every) {
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

TEST_CASE("gc::Heap lazy sweep - once done it leaves the heap the eager sweep leaves") {
    // The same program on a lazy heap (swept by slices) and an eager one: the
    // same old bytes in use, committed memory, and the same objects.
    size_t used[2] = {}, committed[2] = {};
    std::vector<uintptr_t> objects[2];
    for (int lazy = 0; lazy < 2; ++lazy) {
        Heap h(config(lazy == 1));
        OldSet set = make_old(h, 3000, 5, 4);
        root_all(h, set.kept);
        (void)h.allocate(64 * 1024, words_layout());  // a large object, dead
        h.collect(CollectionKind::Full);
        CHECK_EQ(h.sweeping(), lazy == 1);
        sweep_all(h);
        used[lazy] = h.old_used_bytes();
        committed[lazy] = h.committed_bytes();
        h.for_each_object([&](uintptr_t o) { objects[lazy].push_back(o); });
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
        h.collect(CollectionKind::Full);
        REQUIRE(h.sweeping());
        // Drop more, so the next cycle has dead of its own: had the last
        // cycle's marks survived, these would look marked and live.
        std::vector<uintptr_t> dropped;
        for (size_t i = 0; i < set.kept.size(); i += 2) {
            dropped.push_back(static_cast<uintptr_t>(set.kept[i]));
            set.kept[i] = 0;
        }
        if (how == 0) {
            REQUIRE(h.start_marking());
            CHECK_FALSE(h.sweeping());  // finished in the start's pause
            while (!h.mark_slice(UINT64_MAX)) {
            }
            h.finish_marking();
        } else {
            h.collect(CollectionKind::Full);
        }
        CHECK(h.sweeping());  // this cycle's own, pending again
        for (uintptr_t o : dropped) CHECK_FALSE(h.is_valid_object(o));
        for (uint64_t o : set.kept) {
            if (o) CHECK(h.is_valid_object(static_cast<uintptr_t>(o)));
        }
        sweep_all(h);
        for (uintptr_t o : dropped) CHECK_FALSE(h.is_valid_object(o));
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
    for (uint64_t& b : big) b = h.allocate(20 * 1024, words_layout());
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    std::vector<uint64_t> young(300, 0);
    root_all(h, young);
    for (size_t i = 0; i < young.size(); ++i) {
        // Mature-sized, and every 50th 12 KB: young (below the large-object
        // threshold), promoted into the large-object space.
        const size_t bytes = i % 50 == 0 ? 12 * 1024 : 40;
        young[i] = h.allocate(bytes, words_layout());
        *slot_of(static_cast<uintptr_t>(young[i]), 0) = set.kept[i % set.kept.size()];
    }
    h.collect(CollectionKind::Minor);
    for (uint64_t o : young) REQUIRE(h.is_old(static_cast<uintptr_t>(o)));
    REQUIRE(h.sweeping());
    sweep_all(h);
    for (size_t i = 0; i < young.size(); ++i) {
        const auto o = static_cast<uintptr_t>(young[i]);
        CHECK(h.is_valid_object(o));
        CHECK_EQ(Heap::load(o, 0), set.kept[i % set.kept.size()]);
    }
    // And they are ordinary old objects to the next cycle: kept while held,
    // freed once not.
    h.collect(CollectionKind::Full);
    for (uint64_t o : young) CHECK(h.is_valid_object(static_cast<uintptr_t>(o)));
    std::vector<uintptr_t> was(young.begin(), young.end());
    for (uint64_t& o : young) o = 0;
    h.collect(CollectionKind::Full);
    sweep_all(h);
    for (uintptr_t o : was) CHECK_FALSE(h.is_valid_object(o));
    for (uint64_t b : big) CHECK(h.is_valid_object(static_cast<uintptr_t>(b)));
    unroot_all(h, big);
    unroot_all(h, young);
    unroot_all(h, set.kept);
}

TEST_CASE("gc::Heap lazy sweep - a minor collection's card scan never scans a dead object of an unswept block") {
    Heap h(config());
    // Live L and dead D side by side in one card: a store into L dirties the
    // card both start in. D's trace function counts its calls.
    uint64_t live = h.allocate(16, words_layout(), kAllocOld);
    h.add_root(&live);
    const uintptr_t dead = h.allocate(16, spy_layout(), kAllocOld);
    const auto l = static_cast<uintptr_t>(live);
    REQUIRE_EQ((dead - h.old_base()) >> kCardShift, (l - h.old_base()) >> kCardShift);
    // D names a large object that dies with it: once freed (and its pages
    // reused), D's slot names memory that is no object.
    *slot_of(dead, 0) = h.allocate(64 * 1024, words_layout());
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    REQUIRE_FALSE(h.is_valid_object(dead));
    g_spy_traces = 0;
    uint64_t young = h.allocate(16, words_layout());
    h.store(l, 0, young);  // dirties the shared card
    young = 0;
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_spy_traces, uint64_t{0});
    CHECK(h.sweeping());
    sweep_all(h);
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_spy_traces, uint64_t{0});
    CHECK(h.is_valid_object(l));
    h.remove_root(&live);
}

TEST_CASE("gc::Heap lazy sweep - large objects: the dead freed, those allocated during the sweep kept, pages reused") {
    Heap h(config());
    constexpr size_t kBig = 40 * 1024;
    // Forty, every fourth dropped: more than one allocation's walk looks at.
    std::vector<uint64_t> big(40, 0);
    root_all(h, big);
    for (uint64_t& b : big) b = h.allocate(kBig, words_layout());
    std::vector<uintptr_t> dropped;
    for (size_t i = 0; i < big.size(); i += 4) {
        dropped.push_back(static_cast<uintptr_t>(big[i]));
        big[i] = 0;
    }
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    for (uintptr_t d : dropped) CHECK_FALSE(h.is_valid_object(d));
    const size_t committed = h.committed_bytes();
    // A large object of the same size: no free run fits yet, so the
    // allocator walks the sweep forward and takes a dead one's pages rather
    // than growing the space.
    uint64_t again = h.allocate(kBig, words_layout());
    h.add_root(&again);
    CHECK(std::find(dropped.begin(), dropped.end(), static_cast<uintptr_t>(again)) != dropped.end());
    CHECK(h.committed_bytes() <= committed);
    // Larger than any dead run: the walk's next steps find none, and it lands
    // at the frontier, past the cursor, while the walk is under way: kept by it.
    uint64_t during = h.allocate(5 * kBig, words_layout());
    h.add_root(&during);
    REQUIRE(h.sweeping());
    sweep_all(h);
    for (uint64_t b : big) {
        if (b) CHECK(h.is_valid_object(static_cast<uintptr_t>(b)));
    }
    CHECK(h.is_valid_object(static_cast<uintptr_t>(again)));
    CHECK(h.is_valid_object(static_cast<uintptr_t>(during)));
    // Their marks were cleared by the walk: dropped, the next cycle frees them.
    const auto a = static_cast<uintptr_t>(again);
    const auto d = static_cast<uintptr_t>(during);
    again = during = 0;
    h.collect(CollectionKind::Full);
    CHECK_FALSE(h.is_valid_object(a));
    CHECK_FALSE(h.is_valid_object(d));
    sweep_all(h);
    h.remove_root(&again);
    h.remove_root(&during);
    unroot_all(h, big);
}

namespace {
uint64_t g_finalized = 0;
void count_finalizer(void*) { ++g_finalized; }
} // namespace

TEST_CASE("gc::Heap lazy sweep - finalizers, weak slots and hooks settle at the finish, before anything is freed") {
    Heap h(config());
    uint64_t cell_live = h.allocate(8, weak_cell_layout(), kAllocOld);
    uint64_t cell_dead = h.allocate(8, weak_cell_layout(), kAllocOld);
    uint64_t target = h.allocate(16, words_layout(), kAllocOld);
    for (uint64_t* r : {&cell_live, &cell_dead, &target}) h.add_root(r);
    const uintptr_t doomed = h.allocate(16, words_layout(), kAllocOld);
    *slot_of(static_cast<uintptr_t>(cell_live), 0) = target;
    *slot_of(static_cast<uintptr_t>(cell_dead), 0) = doomed;
    g_finalized = 0;
    h.add_finalizer(doomed, &count_finalizer, nullptr);
    uintptr_t hook_saw = 1;
    bool hook_ran = false;
    const auto hook = h.add_post_collection_hook([&](Heap& heap, CollectionKind kind) {
        if (kind != CollectionKind::Full) return;
        hook_ran = true;
        hook_saw = heap.survivor_of(doomed);
    });
    h.collect(CollectionKind::Full);
    REQUIRE(h.sweeping());
    CHECK(hook_ran);
    CHECK_EQ(hook_saw, uintptr_t{0});
    CHECK_EQ(g_finalized, uint64_t{1});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell_dead), 0), kCleared);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell_live), 0), target);
    // A minor collection during the sweep decides nothing old again.
    h.collect(CollectionKind::Minor);
    sweep_all(h);
    CHECK_EQ(g_finalized, uint64_t{1});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(cell_live), 0), target);
    h.remove_post_collection_hook(hook);
    for (uint64_t* r : {&cell_live, &cell_dead, &target}) h.remove_root(r);
}

// The release decommits at most HeapConfig::release_run_blocks blocks a step
// (0: each free run whole), so a run of many MB takes many zero-budget
// slices rather than one past any budget.
TEST_CASE("gc::Heap lazy sweep - excess free blocks go back to the OS in the sweep's slices") {
    const auto slices_to_release = [](size_t run_blocks) {
        HeapConfig c = config();
        c.verify = false;
        c.min_full_threshold_bytes = size_t{1} << 20;  // retained: 64 free blocks, not 32 MB
        c.release_run_blocks = run_blocks;
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
        CHECK(slices > 1);
        CHECK(h.committed_bytes() + (size_t{8} << 20) < before);
        h.remove_root(&chain);
        return slices;
    };
    const uint64_t whole = slices_to_release(0);
    slices_to_release(64);
    // Over 300 blocks released four at a time: a slice each.
    CHECK(slices_to_release(4) >= whole + 50);
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

TEST_CASE("gc::Heap lazy sweep - lazy_sweep = false and BRASS_GC_EAGER_SWEEP=1 sweep inside the finish") {
    {
        Heap h(config(false));
        OldSet set = make_old(h, 1000, 4, 2);
        root_all(h, set.kept);
        h.collect(CollectionKind::Full);
        CHECK_FALSE(h.sweeping());
        CHECK_FALSE(h.pressure().sweeping);
        unroot_all(h, set.kept);
    }
    {
        set_env("BRASS_GC_EAGER_SWEEP", "1");
        HeapConfig c = config();
        c.read_environment = true;
        Heap h(c);
        set_env("BRASS_GC_EAGER_SWEEP", nullptr);
        h.set_stress(StressMode::None);  // whatever BRASS_GC_STRESS the run has
        OldSet set = make_old(h, 1000, 4, 2);
        root_all(h, set.kept);
        h.collect(CollectionKind::Full);
        CHECK_FALSE(h.sweeping());
        unroot_all(h, set.kept);
    }
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
