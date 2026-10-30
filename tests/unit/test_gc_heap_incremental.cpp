// gc::Heap's marking cycle (start_marking, mark_slice, finish_marking): with
// no mutator between the steps, a cycle cut into slices of any size marks
// and sweeps exactly what collect(Full) does, on one thread or several; its
// slices keep roughly to their budget; and the misuses that would lose
// objects until the barrier supports a mutator in between stop the process.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace brass::gc;

namespace {

constexpr uint64_t kCleared = 0x7FFC000000000000ULL;

void trace_weak_cell(uintptr_t payload, size_t, Tracer& t) {
    t.visit_weak(reinterpret_cast<uint64_t*>(payload), kCleared);
}

// [0] entry count, then (key, value) pairs.
void trace_ephemeron_table(uintptr_t payload, size_t, Tracer& t) {
    auto* w = reinterpret_cast<uint64_t*>(payload);
    for (uint64_t i = 0; i < w[0]; ++i) t.visit_ephemeron(&w[1 + 2 * i], &w[2 + 2 * i], 0, 0);
}

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.name = "incremental-words";
        return register_layout(d);
    }();
    return id;
}

LayoutId weak_cell_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_weak_cell;
        d.name = "incremental-weak-cell";
        return register_layout(d);
    }();
    return id;
}

LayoutId ephemeron_table_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Custom;
        d.trace = &trace_ephemeron_table;
        d.name = "incremental-ephemeron-table";
        return register_layout(d);
    }();
    return id;
}

HeapConfig cycle_config(unsigned threads) {
    HeapConfig c;
    c.eden_bytes = size_t{1} << 20;
    c.survivor_bytes = size_t{256} << 10;
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{256} << 20;
    c.large_object_bytes = 16 * 1024;  // the arrays below go straight to the large-object space
    c.min_full_threshold_bytes = size_t{1} << 30;  // no full collection but the test's
    c.mark_threads = threads;
    c.parallel_mark_bytes = 0;  // threads > 1: a parallel cycle at any size
    c.read_environment = false;
    c.verify = true;
    return c;
}

void record_finalized(void* list, void* id) {
    static_cast<std::vector<uintptr_t>*>(list)->push_back(reinterpret_cast<uintptr_t>(id));
}

// What a collection decided, in terms comparable across two heaps built the
// same way: offsets from the reservation, and indices.
struct Outcome {
    uint64_t live_bytes = 0;
    std::vector<uintptr_t> objects;  // every object left, as an offset
    std::vector<int> weak_cleared;   // per weak cell
    std::vector<int> ephemeron_cleared;
    std::vector<uintptr_t> finalized;  // ids, sorted
    uint64_t slices = 0;
};

enum class How {
    Full,         // collect(Full)
    Cycle,        // start, slices of the budget until done, finish
    StartFinish,  // start, then finish with every grey object still to mark
    Partial,      // start, three slices of the budget, finish with the rest
};

constexpr size_t kNodes = 12000;
constexpr size_t kNodeWords = 4;
constexpr size_t kArrays = 12;
constexpr size_t kArrayWords = 6000;  // 48 KB: three 16 KB mark slices each
constexpr size_t kWeakCells = 300;
constexpr size_t kTables = 8;
constexpr size_t kTableEntries = 40;

// A random graph, made old by a first full collection, then with links cut
// at random, collected by `how`. Every random choice comes from one seed, so
// two heaps given the same seed and threads hold the same graph at the same
// offsets.
Outcome run(How how, uint64_t budget_ns, unsigned threads, uint32_t seed) {
    Heap h(cycle_config(threads));
    std::mt19937 rng(seed);
    auto pick = [&](size_t n) { return static_cast<size_t>(rng() % n); };

    // Roots: a few nodes, every array, every weak cell and table (so their
    // contents can be read afterwards).
    std::vector<uint64_t> roots(kArrays + kWeakCells + kTables + 16, 0);
    for (uint64_t& r : roots) h.add_root(&r);
    size_t root_at = 0;

    std::vector<uint64_t> nodes(kNodes, 0);  // a host table: rooted until the cut below
    for (uint64_t& n : nodes) h.add_root(&n);
    for (size_t i = 0; i < kNodes; ++i) nodes[i] = h.allocate(kNodeWords * 8, words_layout());
    for (size_t i = 0; i < kNodes; ++i) {
        for (size_t k = 0; k < kNodeWords; ++k) {
            if (rng() % 4 == 0) continue;  // some slots stay null
            h.store(static_cast<uintptr_t>(nodes[i]), k, nodes[pick(kNodes)]);
        }
    }
    for (size_t a = 0; a < kArrays; ++a) {
        const uintptr_t arr = h.allocate(kArrayWords * 8, words_layout());
        roots[root_at++] = arr;
        for (size_t k = 0; k < kArrayWords; k += 1 + pick(64)) h.store(arr, k, nodes[pick(kNodes)]);
    }
    for (size_t i = 0; i < 16; ++i) roots[root_at++] = nodes[pick(kNodes)];
    const size_t weak_first = root_at;
    for (size_t i = 0; i < kWeakCells; ++i) {
        const uintptr_t cell = h.allocate(8, weak_cell_layout());
        roots[root_at++] = cell;
        h.store(cell, 0, nodes[pick(kNodes)]);
    }
    const size_t table_first = root_at;
    for (size_t t = 0; t < kTables; ++t) {
        const uintptr_t table = h.allocate((1 + 2 * kTableEntries) * 8, ephemeron_table_layout());
        roots[root_at++] = table;
        reinterpret_cast<uint64_t*>(table)[0] = kTableEntries;
        for (size_t e = 0; e < kTableEntries; ++e) {
            h.store(table, 1 + 2 * e, nodes[pick(kNodes)]);
            // Some values are other keys' keys: chains the fixpoint resolves.
            h.store(table, 2 + 2 * e, nodes[pick(kNodes)]);
        }
    }
    std::vector<uintptr_t> finalized;
    h.collect(CollectionKind::Full);  // everything old now; the table of nodes still roots all
    for (size_t i = 0; i < kNodes; i += 7) {
        h.add_finalizer(static_cast<uintptr_t>(nodes[i]), &record_finalized, &finalized,
                        reinterpret_cast<void*>(i));
    }
    // The host table goes; links are cut at random, so some of the graph dies.
    for (uint64_t& n : nodes) h.remove_root(&n);
    for (size_t i = 0; i < kNodes; ++i) {
        for (size_t k = 0; k < kNodeWords; ++k) {
            if (rng() % 3 == 0) h.store(static_cast<uintptr_t>(nodes[i]), k, 0);
        }
    }
    std::fill(nodes.begin(), nodes.end(), 0);

    Outcome out;
    if (how == How::Full) {
        h.collect(CollectionKind::Full);
    } else if (how == How::StartFinish || how == How::Partial) {
        REQUIRE(h.start_marking());
        for (int i = 0; how == How::Partial && i < 3; ++i) CHECK_FALSE(h.mark_slice(budget_ns));
        h.finish_marking();
    } else {
        REQUIRE(h.start_marking());
        CHECK(h.marking());
        CHECK_FALSE(h.start_marking());  // one cycle at a time
        while (!h.mark_slice(budget_ns)) ++out.slices;
        ++out.slices;
        CHECK(h.mark_slice(budget_ns));  // done stays done
        h.finish_marking();
        CHECK_FALSE(h.marking());
    }

    out.live_bytes = h.stats().old_live_bytes;
    const uintptr_t base = h.reservation_base();
    h.for_each_object([&](uintptr_t o) { out.objects.push_back(o - base); });
    std::sort(out.objects.begin(), out.objects.end());
    for (size_t i = 0; i < kWeakCells; ++i) {
        out.weak_cleared.push_back(Heap::load(static_cast<uintptr_t>(roots[weak_first + i]), 0) == kCleared);
    }
    for (size_t t = 0; t < kTables; ++t) {
        const uintptr_t table = static_cast<uintptr_t>(roots[table_first + t]);
        for (size_t e = 0; e < kTableEntries; ++e) out.ephemeron_cleared.push_back(Heap::load(table, 1 + 2 * e) == 0);
    }
    out.finalized = finalized;
    std::sort(out.finalized.begin(), out.finalized.end());
    return out;
}

void check_same(const Outcome& a, const Outcome& b) {
    CHECK_EQ(a.live_bytes, b.live_bytes);
    CHECK_EQ(a.objects.size(), b.objects.size());
    CHECK(a.objects == b.objects);
    CHECK(a.weak_cleared == b.weak_cleared);
    CHECK(a.ephemeron_cleared == b.ephemeron_cleared);
    CHECK(a.finalized == b.finalized);
}

} // namespace

TEST_CASE("gc::Heap incremental - tiny slices mark and sweep exactly what a full collection does") {
    for (unsigned threads : {1u, 4u}) {
        for (uint32_t seed : {1u, 2u, 3u}) {
            const Outcome full = run(How::Full, 0, threads, seed);
            // Something died and something lived, or the comparison says little.
            CHECK(full.live_bytes > 0);
            CHECK(std::count(full.weak_cleared.begin(), full.weak_cleared.end(), 1) > 0);
            CHECK(std::count(full.weak_cleared.begin(), full.weak_cleared.end(), 0) > 0);
            CHECK(!full.finalized.empty());
            CHECK(std::count(full.ephemeron_cleared.begin(), full.ephemeron_cleared.end(), 1) > 0);
            CHECK(std::count(full.ephemeron_cleared.begin(), full.ephemeron_cleared.end(), 0) > 0);

            // A zero budget: every slice stops at its first look at the clock.
            const Outcome tiny = run(How::Cycle, 0, threads, seed);
            check_same(full, tiny);
            CHECK(tiny.slices > 10);
            // Slices long enough to run on the marker's threads.
            const Outcome wide = run(How::Cycle, 5'000'000, threads, seed);
            check_same(full, wide);
            // One slice that drains everything.
            const Outcome whole = run(How::Cycle, UINT64_MAX, threads, seed);
            check_same(full, whole);
            CHECK_EQ(whole.slices, uint64_t{1});
            // No slice at all: finish marks everything.
            check_same(full, run(How::StartFinish, 0, threads, seed));
            // A few slices, then finish with grey work left.
            check_same(full, run(How::Partial, 0, threads, seed));
        }
    }
}

TEST_CASE("gc::Heap incremental - a slice keeps roughly to its budget") {
    HeapConfig c = cycle_config(4);
    Heap h(c);
    // A long list (no parallelism to find), well over a slice to mark.
    uint64_t head = 0;
    h.add_root(&head);
    for (size_t i = 0; i < 200000; ++i) {
        const uintptr_t n = h.allocate(16, words_layout());
        h.store(n, 0, head);
        head = n;
    }
    h.collect(CollectionKind::Full);
    REQUIRE(h.start_marking());
    const uint64_t budget = 100'000;  // 0.1 ms
    uint64_t slices = 0;
    uint64_t longest = 0;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool done = h.mark_slice(budget);
        const auto ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        longest = std::max(longest, ns);
        ++slices;
        if (done) break;
        REQUIRE(slices < 1'000'000);
    }
    h.finish_marking();
    CHECK(slices > 1);
    // Generous for a loaded machine; a slice that ignored its budget would
    // take the whole marking (several milliseconds) in one.
    CHECK(longest < budget + 20'000'000);
    CHECK_EQ(h.stats().full_collections, uint64_t{2});
    CHECK(h.stats().old_live_bytes >= 200000 * 16);
}

#if !defined(_WIN32)

namespace {

// Runs `fn` in a child process; true when the child died of SIGABRT (gc_fatal).
template <typename Fn>
bool aborts(Fn fn) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) dup2(devnull, 2);
        fn();
        _exit(0);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return false;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

} // namespace

TEST_CASE("gc::Heap incremental - misuse outside or inside a cycle is a hard error") {
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.mark_slice(1000);  // no cycle open
    }));
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.finish_marking();  // no cycle open
    }));
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.start_marking();
        h.allocate(16, words_layout());  // nothing allocates black yet
    }));
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.start_marking();
        h.allocate(16, words_layout(), kAllocOld);  // the old slow path too
    }));
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.start_marking();
        h.collect(CollectionKind::Minor);  // minors do not shade yet
    }));
    CHECK(aborts([] {
        Heap h(cycle_config(1));
        h.start_marking();
        h.collect(CollectionKind::Full);
    }));
    // The inline bump path is shut while a cycle is open.
    {
        Heap h(cycle_config(1));
        REQUIRE(h.start_marking());
        CHECK_EQ(h.try_allocate_young(16, words_layout()), uintptr_t{0});
        h.finish_marking();
        CHECK(h.try_allocate_young(16, words_layout()) != 0);
    }
    // The control: the same steps in order do not abort.
    CHECK_FALSE(aborts([] {
        Heap h(cycle_config(1));
        h.start_marking();
        while (!h.mark_slice(0)) {
        }
        h.finish_marking();
        h.allocate(16, words_layout());
    }));
}

#endif
