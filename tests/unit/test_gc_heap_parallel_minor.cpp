// gc::Heap's parallel minor collection (heap_minor_parallel.cpp): on 2, 4 and
// 8 threads it keeps exactly the object graph the single-threaded copy keeps,
// on random graphs with old-to-young links through mature cards, large
// arrays and per-card objects, under verification; with survivor overflow,
// medium and large promotions, a pending lazy sweep and a marking cycle open;
// weak slots, ephemerons and finalizers settle as on one thread; one young
// object named from thousands of cards scanned by different threads is copied
// once; and the automatic policy copies small collections on one thread.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

using namespace brass::gc;

namespace {

constexpr uint64_t kCleared = 0x7FFC000000000000ULL;

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.name = "parallel-minor-words";
        return register_layout(d);
    }();
    return id;
}

void trace_weak_cell(uintptr_t payload, size_t, Tracer& t) {
    t.visit_weak(reinterpret_cast<uint64_t*>(payload) + 1, kCleared);
}

// [0] id, [1] entry count, then (key, value) pairs.
void trace_ephemeron_table(uintptr_t payload, size_t, Tracer& t) {
    auto* w = reinterpret_cast<uint64_t*>(payload);
    for (uint64_t i = 0; i < w[1]; ++i) t.visit_ephemeron(&w[2 + 2 * i], &w[3 + 2 * i], 0, 0);
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "parallel-minor-weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId ephemeron_table_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_ephemeron_table;
        d.name = "parallel-minor-ephemeron-table";
        return register_layout(d);
    }();
    return id;
}

HeapConfig config(ParallelMinor mode, unsigned threads) {
    HeapConfig c;
    c.eden_bytes = size_t{1} << 20;
    c.survivor_bytes = size_t{128} << 10;  // small: some rounds overflow it
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{256} << 20;
    c.large_object_bytes = 16 * 1024;
    c.min_full_threshold_bytes = size_t{1} << 30;  // no full collection but the test's
    c.parallel_minor = mode;
    c.minor_threads = threads;
    c.mark_threads = std::max(threads, 1u);
    c.parallel_mark_bytes = 0;
    c.read_environment = false;
    c.verify = true;
    return c;
}

struct Options {
    bool cycle = false;  // a marking cycle open through the rounds
    bool lazy = false;   // the old generation's sweep pending at the first round
    bool weak = false;   // weak cells, ephemeron tables, finalizers
    int rounds = 8;
};

struct Outcome {
    std::vector<uint64_t> graph;       // after the rounds' minor collections
    std::vector<uint64_t> graph_full;  // after a full collection
    size_t objects_full = 0;
    uint64_t live_full = 0;
    std::vector<int> weak_cleared;
    std::vector<int> ephemeron_cleared;
    std::vector<uint64_t> finalized;
    uint64_t parallel = 0;
    uint64_t promoted = 0;
    bool swept_pending_at_first = false;
};

void record_finalized(void* list, void* id) {
    static_cast<std::vector<uint64_t>*>(list)->push_back(reinterpret_cast<uintptr_t>(id));
}

constexpr size_t kRoots = 32;
constexpr size_t kOld = 1500;
constexpr size_t kPadded = 300;  // one per card: 504-byte payloads
constexpr size_t kArrays = 3;
constexpr size_t kArrayWords = 8000;  // 64 KB: large, cut into two pieces of card work
constexpr size_t kNodeWords = 6;      // [0] id, then references
constexpr size_t kYoungPerRound = 2500;

class World {
public:
    World(const HeapConfig& c, uint32_t seed) : h(c), rng_(seed) {
        for (uint64_t& r : roots_) h.add_root(&r);
        for (uint64_t& r : old_) h.add_root(&r);
        for (uint64_t& r : padded_) h.add_root(&r);
        for (uint64_t& r : arrays_) h.add_root(&r);
        for (uint64_t& r : holders_) h.add_root(&r);
    }

    Heap h;

    size_t pick(size_t n) { return static_cast<size_t>(rng_() % n); }
    bool chance(unsigned one_in) { return rng_() % one_in == 0; }

    uintptr_t node(size_t words, uint32_t flags = 0) {
        const uintptr_t o = h.allocate(words * 8, words_layout(), flags);
        reinterpret_cast<uint64_t*>(o)[0] = next_id_++;
        return o;
    }

    void build_old(bool lazy) {
        for (size_t i = 0; i < kOld; ++i) old_[i] = node(kNodeWords);
        for (size_t i = 0; i < kOld; ++i) {
            for (size_t k = 1; k < kNodeWords; ++k) {
                if (chance(3)) h.store(static_cast<uintptr_t>(old_[i]), k, old_[pick(kOld)]);
            }
        }
        for (size_t i = 0; i < kPadded; ++i) padded_[i] = node(63);
        for (size_t a = 0; a < kArrays; ++a) arrays_[a] = node(kArrayWords, kAllocOld);
        holders_[0] = node(kArrayWords, kAllocOld);  // weak cells
        holders_[1] = h.allocate((2 + 2 * 400) * 8, ephemeron_table_layout());  // old after the full
        reinterpret_cast<uint64_t*>(holders_[1])[0] = next_id_++;
        h.collect(CollectionKind::Full);
        if (lazy) {
            // Old garbage for a lazy sweep to be pending over: a third of the
            // old nodes unrooted, and with them whatever only they named.
            for (size_t i = 0; i < kOld; i += 3) old_[i] = 0;
            h.collect(CollectionKind::Full);
        }
    }

    // One round: young nodes linked among themselves and from old objects,
    // some old-to-young links cut again, then a minor collection.
    void round(bool weak) {
        std::vector<uintptr_t> young;
        young.reserve(kYoungPerRound);
        for (size_t i = 0; i < kYoungPerRound; ++i) {
            size_t words = kNodeWords;
            if (chance(97)) words = 300;   // medium, promoted into the overflow region
            if (chance(499)) words = 1200; // promoted into the large-object space
            const uintptr_t n = node(words);
            for (size_t k = 1; k < std::min<size_t>(words, kNodeWords); ++k) {
                const unsigned r = static_cast<unsigned>(rng_() % 8);
                uint64_t v = 0;
                if (r < 4 && !young.empty()) v = young[pick(young.size())];
                else if (r == 4 && old_[pick(kOld)] != 0) v = old_[pick(kOld)];
                reinterpret_cast<uint64_t*>(n)[k] = v;
            }
            young.push_back(n);
        }
        auto any_young = [&] { return young[pick(young.size())]; };
        for (size_t i = 0; i < 600; ++i) {
            const uint64_t o = old_[pick(kOld)];
            if (o != 0) h.store(static_cast<uintptr_t>(o), 1 + pick(kNodeWords - 1), any_young());
        }
        for (size_t i = 0; i < 400; ++i) h.store(static_cast<uintptr_t>(arrays_[pick(kArrays)]), 1 + pick(kArrayWords - 1), any_young());
        for (size_t i = 0; i < kPadded; i += 2) h.store(static_cast<uintptr_t>(padded_[i]), 1 + pick(62), any_young());
        for (size_t i = 0; i < 8; ++i) roots_[pick(kRoots)] = any_young();
        // Cut some links: what only they named dies young.
        for (size_t i = 0; i < 300; ++i) {
            const uint64_t o = old_[pick(kOld)];
            if (o != 0) h.store(static_cast<uintptr_t>(o), 1 + pick(kNodeWords - 1), 0);
        }
        for (size_t i = 0; i < 200; ++i) h.store(static_cast<uintptr_t>(arrays_[pick(kArrays)]), 1 + pick(kArrayWords - 1), 0);
        if (weak) weak_round(young);
        h.collect(CollectionKind::Minor);
    }

    void weak_round(const std::vector<uintptr_t>& young) {
        auto any_young = [&] { return young[pick(young.size())]; };
        // Young weak cells, held by an old array, naming young nodes that
        // live or die by the round's links.
        for (size_t i = 0; i < 40 && cells_ < kArrayWords - 1; ++i) {
            const uintptr_t cell = h.allocate(16, weak_cell_layout());
            reinterpret_cast<uint64_t*>(cell)[0] = next_id_++;
            reinterpret_cast<uint64_t*>(cell)[1] = any_young();
            h.store(static_cast<uintptr_t>(holders_[0]), 1 + cells_++, cell);
        }
        // The old ephemeron table's entries: young keys (some kept by a root
        // or link, some not), values that are other entries' keys at times.
        auto* table = reinterpret_cast<uint64_t*>(holders_[1]);
        for (size_t i = 0; i < 40 && table[1] < 400; ++i) {
            const uint64_t e = table[1]++;
            h.store(static_cast<uintptr_t>(holders_[1]), 2 + 2 * e, any_young());
            h.store(static_cast<uintptr_t>(holders_[1]), 3 + 2 * e,
                    chance(3) && e > 0 ? table[2 + 2 * pick(static_cast<size_t>(e))] : any_young());
        }
        for (size_t i = 0; i < 40; ++i) {
            const uintptr_t n = any_young();
            h.add_finalizer(n, &record_finalized, &finalized_,
                            reinterpret_cast<void*>(static_cast<uintptr_t>(reinterpret_cast<uint64_t*>(n)[0])));
        }
    }

    // The graph from the roots as ids: each object's id and size, then each
    // reference slot as the index of its target in the walk (all-ones for
    // null), in a walk whose order depends only on the graph.
    std::vector<uint64_t> graph() {
        std::vector<uint64_t> out;
        std::unordered_map<uintptr_t, uint64_t> index;
        std::vector<uintptr_t> queue;
        auto reach = [&](uint64_t word) -> uint64_t {
            const auto a = static_cast<uintptr_t>(word);
            if (a == 0 || !h.contains(a)) return word == 0 ? ~uint64_t{0} : word;
            auto [it, fresh] = index.emplace(a, index.size());
            if (fresh) queue.push_back(a);
            return it->second;
        };
        for (uint64_t r : roots_) out.push_back(reach(r));
        for (uint64_t r : old_) out.push_back(reach(r));
        for (uint64_t r : padded_) out.push_back(reach(r));
        for (uint64_t r : arrays_) out.push_back(reach(r));
        for (uint64_t r : holders_) out.push_back(reach(r));
        for (size_t q = 0; q < queue.size(); ++q) {
            const uintptr_t o = queue[q];
            CHECK(h.is_valid_object(o));
            const size_t words = header_of(o)->size / 8;
            const auto* w = reinterpret_cast<const uint64_t*>(o);
            out.push_back(words);
            for (size_t i = 0; i < words; ++i) out.push_back(i == 0 ? w[0] : reach(w[i]));
        }
        return out;
    }

    Outcome finish(const Options& o) {
        Outcome out;
        if (o.cycle) h.finish_marking();
        if (o.weak) {
            for (size_t i = 0; i < cells_; ++i) {
                const uintptr_t cell = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holders_[0]), 1 + i));
                out.weak_cleared.push_back(Heap::load(cell, 1) == kCleared);
            }
            const auto* table = reinterpret_cast<const uint64_t*>(holders_[1]);
            for (uint64_t e = 0; e < table[1]; ++e) out.ephemeron_cleared.push_back(table[2 + 2 * e] == 0);
        }
        out.graph = graph();
        out.parallel = h.stats().parallel_evacuations;
        out.promoted = h.stats().promoted_bytes;
        h.collect(CollectionKind::Full);
        out.graph_full = graph();
        h.for_each_object([&](uintptr_t) { ++out.objects_full; });
        out.live_full = h.stats().old_live_bytes;
        out.finalized = finalized_;
        std::sort(out.finalized.begin(), out.finalized.end());
        out.swept_pending_at_first = swept_pending_at_first;
        return out;
    }

    bool swept_pending_at_first = false;

private:
    std::mt19937 rng_;
    uint64_t next_id_ = 1;
    std::vector<uint64_t> roots_ = std::vector<uint64_t>(kRoots, 0);
    std::vector<uint64_t> old_ = std::vector<uint64_t>(kOld, 0);
    std::vector<uint64_t> padded_ = std::vector<uint64_t>(kPadded, 0);
    std::vector<uint64_t> arrays_ = std::vector<uint64_t>(kArrays, 0);
    std::vector<uint64_t> holders_ = std::vector<uint64_t>(2, 0);
    size_t cells_ = 0;
    std::vector<uint64_t> finalized_;
};

Outcome run(ParallelMinor mode, unsigned threads, uint32_t seed, const Options& o) {
    World w(config(mode, threads), seed);
    w.build_old(o.lazy);
    w.swept_pending_at_first = w.h.sweeping();
    if (o.cycle) REQUIRE(w.h.start_marking());
    for (int r = 0; r < o.rounds; ++r) {
        w.round(o.weak);
        if (o.cycle) (void)w.h.mark_slice(20'000);
    }
    return w.finish(o);
}

void check_same(const Outcome& serial, const Outcome& parallel) {
    CHECK(serial.graph == parallel.graph);
    CHECK(serial.graph_full == parallel.graph_full);
    CHECK_EQ(serial.objects_full, parallel.objects_full);
    CHECK_EQ(serial.live_full, parallel.live_full);
    CHECK(serial.weak_cleared == parallel.weak_cleared);
    CHECK(serial.ephemeron_cleared == parallel.ephemeron_cleared);
    CHECK(serial.finalized == parallel.finalized);
}

void compare_all(const Options& o, uint32_t seeds) {
    for (uint32_t seed = 1; seed <= seeds; ++seed) {
        const Outcome serial = run(ParallelMinor::Off, 1, seed, o);
        CHECK_EQ(serial.parallel, uint64_t{0});
        CHECK(serial.promoted > 0);
        for (unsigned threads : {2u, 4u, 8u}) {
            const Outcome parallel = run(ParallelMinor::Always, threads, seed, o);
            // Every evacuation went parallel: the rounds' minors and the
            // full collections' (one or two before the rounds, one after,
            // and the cycle's finish).
            CHECK(parallel.parallel >= static_cast<uint64_t>(o.rounds));
            check_same(serial, parallel);
        }
    }
}

} // namespace

TEST_CASE("gc::Heap parallel minor - random graphs survive exactly as on one thread") {
    compare_all(Options{}, 3);
}

TEST_CASE("gc::Heap parallel minor - with the old generation's lazy sweep pending") {
    Options o;
    o.lazy = true;
    const Outcome serial = run(ParallelMinor::Off, 1, 7, o);
    CHECK(serial.swept_pending_at_first);
    compare_all(o, 2);
}

TEST_CASE("gc::Heap parallel minor - inside a marking cycle") {
    Options o;
    o.cycle = true;
    compare_all(o, 2);
    o.lazy = true;  // the cycle's start completes the pending sweep first
    compare_all(o, 1);
}

TEST_CASE("gc::Heap parallel minor - weak slots, ephemerons and finalizers settle as on one thread") {
    Options o;
    o.weak = true;
    for (uint32_t seed = 1; seed <= 2; ++seed) {
        const Outcome serial = run(ParallelMinor::Off, 1, seed, o);
        // Something died and something lived, or the comparison says little.
        CHECK(std::count(serial.weak_cleared.begin(), serial.weak_cleared.end(), 1) > 0);
        CHECK(std::count(serial.weak_cleared.begin(), serial.weak_cleared.end(), 0) > 0);
        CHECK(std::count(serial.ephemeron_cleared.begin(), serial.ephemeron_cleared.end(), 1) > 0);
        CHECK(std::count(serial.ephemeron_cleared.begin(), serial.ephemeron_cleared.end(), 0) > 0);
        CHECK(!serial.finalized.empty());
        for (unsigned threads : {2u, 4u, 8u}) check_same(serial, run(ParallelMinor::Always, threads, seed, o));
    }
    o.cycle = true;
    const Outcome serial = run(ParallelMinor::Off, 1, 3, o);
    check_same(serial, run(ParallelMinor::Always, 8, 3, o));
}

TEST_CASE("gc::Heap parallel minor - an object named from many cards is copied once") {
    Heap h(config(ParallelMinor::Always, 8));
    // Old objects one per card, a large array, and the young targets they name.
    constexpr size_t kOwners = 4000;
    std::vector<uint64_t> owners(kOwners, 0);
    for (uint64_t& r : owners) h.add_root(&r);
    for (uint64_t& r : owners) r = h.allocate(504, words_layout(), kAllocOld);
    uint64_t array = h.allocate(kArrayWords * 8, words_layout(), kAllocOld);
    h.add_root(&array);
    std::vector<uint64_t> young_list(64, 0);  // young objects that all name one shared one
    for (uint64_t& r : young_list) h.add_root(&r);
    for (int round = 0; round < 30; ++round) {
        const uint64_t targets[3] = {h.allocate(32, words_layout()), h.allocate(32, words_layout()),
                                     h.allocate(32, words_layout())};
        for (size_t i = 0; i < kOwners; ++i) {
            h.store(static_cast<uintptr_t>(owners[i]), 1 + i % 3, targets[i % 3]);
            h.store(static_cast<uintptr_t>(owners[i]), 4 + i % 50, targets[(i / 3) % 3]);
        }
        for (size_t k = 1; k < kArrayWords; k += 3) h.store(static_cast<uintptr_t>(array), k, targets[k % 3]);
        for (uint64_t& y : young_list) {
            y = h.allocate(16, words_layout());
            reinterpret_cast<uint64_t*>(y)[0] = targets[0];  // young into young: no barrier
        }
        h.collect(CollectionKind::Minor);
        // Every slot names the one copy of its target.
        const uint64_t now[3] = {Heap::load(static_cast<uintptr_t>(owners[0]), 1),
                                 Heap::load(static_cast<uintptr_t>(owners[1]), 2),
                                 Heap::load(static_cast<uintptr_t>(owners[2]), 3)};
        for (uint64_t t : now) REQUIRE(h.is_valid_object(static_cast<uintptr_t>(t)));
        CHECK(now[0] != now[1]);
        CHECK(now[1] != now[2]);
        bool same = true;
        for (size_t i = 0; i < kOwners; ++i) {
            same = same && Heap::load(static_cast<uintptr_t>(owners[i]), 1 + i % 3) == now[i % 3];
            same = same && Heap::load(static_cast<uintptr_t>(owners[i]), 4 + i % 50) == now[(i / 3) % 3];
        }
        for (size_t k = 1; k < kArrayWords; k += 3) {
            same = same && Heap::load(static_cast<uintptr_t>(array), k) == now[k % 3];
        }
        for (uint64_t y : young_list) same = same && Heap::load(static_cast<uintptr_t>(y), 0) == now[0];
        CHECK(same);
    }
    CHECK(h.stats().parallel_evacuations >= 30);
}

TEST_CASE("gc::Heap parallel minor - the automatic policy copies small collections on one thread") {
    HeapConfig c = config(ParallelMinor::Auto, 4);
    c.verify = false;
    // Low, so that the test does not depend on how fast this machine copies:
    // the quiet collections expect next to nothing, the others ~1 MB.
    c.parallel_minor_ms = 0.05;
    Heap h(c);
    std::vector<uint64_t> keep(20000, 0);
    for (uint64_t& r : keep) h.add_root(&r);
    // Quiet collections: next to nothing survives and no card is dirty.
    for (int i = 0; i < 20; ++i) {
        for (int k = 0; k < 1000; ++k) (void)h.allocate(32, words_layout());
        h.collect(CollectionKind::Minor);
    }
    CHECK_EQ(h.stats().parallel_evacuations, uint64_t{0});
    // Collections that copy most of what they collect: once the first has
    // shown the survival rate, they go parallel.
    for (int i = 0; i < 6; ++i) {
        for (uint64_t& r : keep) r = h.allocate(48, words_layout());
        h.collect(CollectionKind::Minor);
    }
    CHECK(h.stats().parallel_evacuations >= 4);
    // A collection after which nothing survives again goes back to one thread.
    std::fill(keep.begin(), keep.end(), 0);
    h.collect(CollectionKind::Minor);
    const uint64_t before = h.stats().parallel_evacuations;
    for (int i = 0; i < 10; ++i) {
        for (int k = 0; k < 1000; ++k) (void)h.allocate(32, words_layout());
        h.collect(CollectionKind::Minor);
    }
    CHECK_EQ(h.stats().parallel_evacuations, before);
}
