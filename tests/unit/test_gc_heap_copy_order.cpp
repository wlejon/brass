// gc::Heap: where an evacuation puts what it copies (kCopyDepth). A copy is
// scanned as soon as it is made, so the objects it names land right after it,
// in slot order, whatever order the program allocated them in: into the
// survivor space, into the old generation when promoted, on one thread and on
// several. Deeper than kCopyDepth the copy goes on through the grey stack, and
// the graph is the one the roots name either way (under verification).

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdint>
#include <vector>

using namespace brass::gc;
using brass::test::collect_scrubbed;

namespace {

// [0] id (a number, never a reference: ids are odd), then references.
const LayoutId kNode = mask_layout(~uint64_t{1}, 41);
constexpr size_t kNodeWords = 4;
constexpr size_t kNodeTotal = kHeaderBytes + kNodeWords * 8;

HeapConfig config(unsigned threads, uint8_t tenure_age) {
    HeapConfig c;
    c.eden_bytes = size_t{2} << 20;
    c.survivor_bytes = size_t{2} << 20;
    c.mature_reserve_bytes = size_t{64} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{1} << 30;
    c.tenure_age = tenure_age;
    c.parallel_minor = threads > 1 ? ParallelMinor::Always : ParallelMinor::Auto;
    c.minor_threads = threads;
    c.mark_threads = threads;
    c.read_environment = false;
    c.verify = true;
    return c;
}

uintptr_t node(Heap& h, uint64_t id) {
    const uintptr_t n = h.allocate(kNodeWords * 8, kNode);
    reinterpret_cast<uint64_t*>(n)[0] = id * 2 + 1;
    return n;
}

uint64_t id_of(uintptr_t n) { return Heap::load(n, 0) >> 1; }

// A cluster: root R names A and B; A names A1, which names A2; B names B1.
// Allocated leaves first and with unrelated garbage between them, so that
// allocation order says nothing about the graph.
BRASS_TEST_NOINLINE uintptr_t make_cluster(Heap& h, uint64_t base) {
    const uintptr_t b1 = node(h, base + 5);
    (void)node(h, 0);
    const uintptr_t a2 = node(h, base + 3);
    const uintptr_t b = node(h, base + 4);
    (void)node(h, 0);
    const uintptr_t a1 = node(h, base + 2);
    const uintptr_t a = node(h, base + 1);
    const uintptr_t r = node(h, base);
    h.store(b, 1, b1);
    h.store(a1, 1, a2);
    h.store(a, 1, a1);
    h.store(r, 1, a);
    h.store(r, 2, b);
    return r;
}

// The cluster's nodes A A1 A2 B B1 in the order a depth-first copy places
// them, each right after the one before. R itself is copied as a root is,
// beside the other roots: a root's copy is not scanned as it is made.
bool depth_first_adjacent(uintptr_t r) {
    const uintptr_t a = static_cast<uintptr_t>(Heap::load(r, 1));
    const uintptr_t a1 = static_cast<uintptr_t>(Heap::load(a, 1));
    const uintptr_t a2 = static_cast<uintptr_t>(Heap::load(a1, 1));
    const uintptr_t b = static_cast<uintptr_t>(Heap::load(r, 2));
    const uintptr_t b1 = static_cast<uintptr_t>(Heap::load(b, 1));
    const uintptr_t order[] = {r, a, a1, a2, b, b1};
    for (size_t i = 2; i < 6; ++i) {
        if (order[i] != order[i - 1] + kNodeTotal) return false;
    }
    return true;
}

bool cluster_intact(uintptr_t r, uint64_t base) {
    const uintptr_t a = static_cast<uintptr_t>(Heap::load(r, 1));
    const uintptr_t b = static_cast<uintptr_t>(Heap::load(r, 2));
    const uintptr_t a1 = static_cast<uintptr_t>(Heap::load(a, 1));
    return id_of(r) == base && id_of(a) == base + 1 && id_of(a1) == base + 2 &&
           id_of(static_cast<uintptr_t>(Heap::load(a1, 1))) == base + 3 && id_of(b) == base + 4 &&
           id_of(static_cast<uintptr_t>(Heap::load(b, 1))) == base + 5;
}

struct Clusters {
    explicit Clusters(Heap& heap, size_t count) : h(heap), roots(count, 0) {
        for (uint64_t& r : roots) h.add_root(&r);
    }
    ~Clusters() {
        for (uint64_t& r : roots) h.remove_root(&r);
    }
    BRASS_TEST_NOINLINE void build() {
        for (size_t i = 0; i < roots.size(); ++i) roots[i] = make_cluster(h, 100 + 10 * i);
    }
    size_t adjacent() const {
        size_t n = 0;
        for (uint64_t r : roots) n += depth_first_adjacent(static_cast<uintptr_t>(r));
        return n;
    }
    bool intact() const {
        for (size_t i = 0; i < roots.size(); ++i) {
            if (!cluster_intact(static_cast<uintptr_t>(roots[i]), 100 + 10 * i)) return false;
        }
        return true;
    }
    Heap& h;
    std::vector<uint64_t> roots;
};

} // namespace

TEST_CASE("gc::Heap copy order - a survivor's referents follow it depth first") {
    Heap h(config(1, 2));
    Clusters c(h, 64);
    c.build();
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK(c.intact());
    CHECK(h.is_young(static_cast<uintptr_t>(c.roots[0])));
    CHECK_EQ(c.adjacent(), size_t{64});
    // Promoted by the second: laid out again in the old generation, the same way.
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK(c.intact());
    CHECK(h.is_old(static_cast<uintptr_t>(c.roots[0])));
    CHECK_EQ(c.adjacent(), size_t{64});
}

TEST_CASE("gc::Heap copy order - a full collection's evacuation promotes clusters depth first") {
    Heap h(config(1, 2));
    Clusters c(h, 64);
    c.build();
    collect_scrubbed(h, CollectionKind::Full);
    CHECK(c.intact());
    CHECK(h.is_old(static_cast<uintptr_t>(c.roots[0])));
    CHECK_EQ(c.adjacent(), size_t{64});
}

TEST_CASE("gc::Heap copy order - each thread of a parallel copy keeps its clusters together") {
    Heap h(config(4, 1));
    // Enough clusters that every thread copies some, from roots the calling
    // thread copies and shares out.
    Clusters c(h, 4000);
    c.build();
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK(c.intact());
    CHECK(h.stats().parallel_evacuations == 1);
    CHECK(h.is_old(static_cast<uintptr_t>(c.roots[0])));
    // A thread's region can run out inside a cluster; nearly all are whole.
    CHECK(c.adjacent() * 100 >= c.roots.size() * 95);
}

TEST_CASE("gc::Heap copy order - a chain deeper than the copy depth goes on through the grey stack") {
    Heap h(config(1, 2));
    uint64_t head = 0;
    h.add_root(&head);
    constexpr size_t kLength = 5000;
    brass::test::in_own_frame([&] {
        for (size_t i = 0; i < kLength; ++i) {
            const uintptr_t n = node(h, i);
            h.store(n, 1, head);
            head = n;
        }
    });
    collect_scrubbed(h, CollectionKind::Minor);
    collect_scrubbed(h, CollectionKind::Minor);
    size_t length = 0;
    uint64_t expect = kLength;
    bool ordered = true;
    for (uintptr_t n = static_cast<uintptr_t>(head); n != 0; n = static_cast<uintptr_t>(Heap::load(n, 1))) {
        ordered = ordered && id_of(n) == --expect;
        ++length;
    }
    CHECK_EQ(length, kLength);
    CHECK(ordered);
    h.remove_root(&head);
}
