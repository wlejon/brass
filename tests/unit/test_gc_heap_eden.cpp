// gc::Heap's adaptive eden (HeapConfig::adaptive_eden): a minor collection
// at the whole eden whose survivors are being kept (promoted) shrinks the eden
// the next one runs at to the copy budget over the survival rate; the whole
// eden comes back once the survivors stop living through the next
// collection; a program whose young data dies, even a working set that dies
// in the survivor space, keeps the whole eden, and one that follows a
// build-up gets it back as soon as its data starts dying; a full collection
// that finds the old generation's growth dead restores it, one that finds it
// live does not; and the switch pins it whole. The limit is
// what generated code bumps against (the allocation buffer's end), and an
// empty eden still takes the largest young object.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdlib>
#include <vector>

using namespace brass::gc;

namespace {

constexpr size_t kMB = size_t{1} << 20;
constexpr size_t kNodeTotal = 48;  // a 40-byte payload and its header

HeapConfig eden_config(bool adaptive = true) {
    HeapConfig c;  // the default 16 MB eden, 8 MB survivor spaces, 1 MB copy budget
    c.adaptive_eden = adaptive;
    c.read_environment = false;
    return c;
}

const LayoutId kEdenNode = mask_layout(0b01, 43);

uintptr_t new_node(Heap& h, uint64_t* list) {
    const uintptr_t n = h.allocate(40, kEdenNode);
    if (list) {
        h.store(n, 0, *list);
        *list = n;
    }
    return n;
}

// Allocates nodes until a minor collection runs. With `keep`, every node is
// linked onto *keep (so the collection copies nearly all of eden); otherwise
// each dies at once.
void allocate_until_minor(Heap& h, uint64_t* keep) {
    const uint64_t before = h.stats().minor_collections;
    while (h.stats().minor_collections == before) new_node(h, keep);
}

size_t eden_capacity(const Heap& h) { return h.pressure().eden_capacity; }

// Runs the survival spike: every node kept, until the limit has shrunk.
void shrink_by_spike(Heap& h, uint64_t* keep) {
    allocate_until_minor(h, keep);
    REQUIRE(h.stats().last_survived_bytes > 15 * kMB);
    REQUIRE(eden_capacity(h) < kMB + kMB / 4);
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

} // namespace

TEST_CASE("gc::Heap adaptive eden - a survival spike shrinks the limit, and it recovers once survival falls") {
    Heap h(eden_config());
    const size_t full = eden_capacity(h);
    CHECK_EQ(full, size_t{16} * kMB);
    uint64_t keep = 0;
    h.add_root(&keep);

    // The first spike collection runs at the whole eden (nothing predicted
    // it), copies nearly all of it and promotes what overflows the survivor
    // space: the next runs at about the budget.
    shrink_by_spike(h, &keep);
    CHECK(eden_capacity(h) >= kMB);

    // The next copies the survivor space again, which no eden limit bounds;
    // after it, while the spike lasts, each collection copies about the budget
    // out of eden and as much again out of the survivor space (last time's
    // survivors, aged there to see whether they are kept).
    allocate_until_minor(h, &keep);
    for (int i = 0; i < 4; ++i) {
        allocate_until_minor(h, &keep);
        CHECK(h.stats().last_survived_bytes <= 2 * kMB + kMB / 4);
        CHECK(eden_capacity(h) <= kMB + kMB / 4);
    }

    // The data dies: the survivors aged in the survivor space die with it, so
    // the next collection finds them dead and the whole eden comes back.
    keep = 0;
    int collections = 0;
    while (eden_capacity(h) < full) {
        allocate_until_minor(h, nullptr);
        ++collections;
        REQUIRE(collections <= 2);
    }
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap adaptive eden - a workload whose young data dies keeps the whole eden") {
    Heap h(eden_config());
    const size_t full = eden_capacity(h);
    for (int i = 0; i < 8; ++i) {
        allocate_until_minor(h, nullptr);
        CHECK_EQ(eden_capacity(h), full);
    }
    // Every collection ran at a full eden: 8 of them took 8 edens of allocation.
    CHECK(h.allocated_bytes() >= 8 * (full - kMB));
}

TEST_CASE("gc::Heap adaptive eden - a working set that dies in the survivor space keeps the whole eden") {
    // Eight lists of 512 KB each, the oldest dropped as a new one starts: a
    // 4 MB working set whose data lives 4 MB of allocation. A minor collection
    // at the whole eden copies most of it, well over the budget, but it dies
    // before the next, so a smaller eden would only copy it more often.
    Heap h(eden_config());
    const size_t full = eden_capacity(h);
    std::vector<uint64_t> ring(8, 0);
    for (uint64_t& slot : ring) h.add_root(&slot);
    const size_t per_list = kMB / 2 / kNodeTotal;
    size_t current = 0;
    size_t in_list = 0;
    uint64_t most_copied = 0;
    const uint64_t first = h.stats().minor_collections;
    while (h.stats().minor_collections < first + 8) {
        const uint64_t before = h.stats().minor_collections;
        new_node(h, &ring[current]);
        if (++in_list == per_list) {
            in_list = 0;
            current = (current + 1) % ring.size();
            ring[current] = 0;
        }
        if (h.stats().minor_collections != before) {
            most_copied = std::max(most_copied, h.stats().last_survived_bytes);
            CHECK_EQ(eden_capacity(h), full);
        }
    }
    CHECK(most_copied > 2 * kMB);
    for (uint64_t& slot : ring) h.remove_root(&slot);
}

TEST_CASE("gc::Heap adaptive eden - a working set after a build-up gets the whole eden back") {
    // A build-up of 96 MB shrinks eden; then batches of 3 MB, each live until
    // the next starts. Below 3 MB a batch reads as survival like the build-up
    // did, but a collection's survivors no longer all live through the next
    // (a batch ends within it a third of the time), which shows long before
    // the old generation's garbage brings a full collection.
    Heap h(eden_config());
    const size_t full = eden_capacity(h);
    uint64_t keep = 0;
    uint64_t batch = 0;
    h.add_root(&keep);
    h.add_root(&batch);
    const size_t build_nodes = 96 * kMB / kNodeTotal;
    for (size_t i = 0; i < build_nodes; ++i) new_node(h, &keep);
    REQUIRE(eden_capacity(h) < 2 * kMB);
    const uint64_t fulls = h.stats().full_collections;
    const uint64_t first = h.stats().minor_collections;
    const size_t batch_nodes = 3 * kMB / kNodeTotal;
    uint64_t back_at = 0;
    for (int b = 0; b < 200; ++b) {
        batch = 0;
        for (size_t j = 0; j < batch_nodes; ++j) new_node(h, &batch);
        if (back_at == 0 && eden_capacity(h) == full) back_at = h.stats().minor_collections - first;
    }
    CHECK(back_at > 0);
    CHECK(back_at <= 24);
    CHECK_EQ(h.stats().full_collections, fulls);
    CHECK_EQ(eden_capacity(h), full);
    // 600 MB at the whole eden is about 38 collections; the shrunk start adds
    // its two dozen.
    CHECK(h.stats().minor_collections - first <= 38 + 24 + 8);
    h.remove_root(&batch);
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap adaptive eden - the limit is the allocation buffer's end, and an empty eden takes any young object") {
    Heap h(eden_config());
    uint64_t keep = 0;
    h.add_root(&keep);
    shrink_by_spike(h, &keep);
    const size_t limit = eden_capacity(h);
    // Generated code bumps against `end`: after the collection it sits at the
    // shrunk limit, not at eden's end.
    const Heap::AllocationBuffer* buffer = h.allocation_buffer();
    CHECK_EQ(buffer->end - h.young_base(), limit);

    // A young object larger than the limit still fits an empty eden.
    keep = 0;
    h.collect(CollectionKind::Minor);
    REQUIRE(eden_capacity(h) < kMB + kMB / 4);
    const uintptr_t big = h.allocate(kMB, kLeafLayout);
    CHECK(h.is_young(big));
    CHECK(buffer->end >= buffer->top);
    // The next allocation finds eden past its limit and collects.
    const uint64_t before = h.stats().minor_collections;
    new_node(h, nullptr);
    CHECK_EQ(h.stats().minor_collections, before + 1);
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap adaptive eden - a full collection restores the whole eden once the build-up is dead, not before") {
    Heap h(eden_config());
    const size_t full = eden_capacity(h);
    uint64_t keep = 0;
    h.add_root(&keep);
    shrink_by_spike(h, &keep);
    allocate_until_minor(h, &keep);

    // The data built is still live: the old generation grew by what it
    // holds, and the build-up that shrank eden is taken to go on.
    h.collect(CollectionKind::Full);
    CHECK(eden_capacity(h) < kMB + kMB / 4);

    // Dropped: the next full collection finds the growth since the last one
    // (the young data promoted as it collected) and the build all dead.
    for (int i = 0; i < 4; ++i) allocate_until_minor(h, &keep);
    keep = 0;
    h.collect(CollectionKind::Full);
    CHECK_EQ(eden_capacity(h), full);
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap adaptive eden - adaptive_eden = false and BRASS_GC_FIXED_EDEN=1 keep the whole eden") {
    {
        Heap h(eden_config(/*adaptive=*/false));
        const size_t full = eden_capacity(h);
        uint64_t keep = 0;
        h.add_root(&keep);
        allocate_until_minor(h, &keep);
        CHECK(h.stats().last_survived_bytes > 15 * kMB);
        CHECK_EQ(eden_capacity(h), full);
        h.remove_root(&keep);
    }
    {
        set_env("BRASS_GC_FIXED_EDEN", "1");
        HeapConfig c = eden_config();
        c.read_environment = true;
        Heap h(c);
        set_env("BRASS_GC_FIXED_EDEN", nullptr);
        const size_t full = eden_capacity(h);
        uint64_t keep = 0;
        h.add_root(&keep);
        allocate_until_minor(h, &keep);
        CHECK_EQ(eden_capacity(h), full);
        h.remove_root(&keep);
    }
}
