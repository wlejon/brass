// gc::Heap out of memory: an old generation filled to its reservation fails
// the allocation with std::bad_alloc (after a full collection) instead of
// stopping the process, and the heap goes on once the program lets go.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <new>
#include <vector>

using namespace brass::gc;
using brass::test::from_own_frame;
using brass::test::HeapRoot;
using brass::test::scrub_stack;

namespace {

HeapConfig tiny_old_config() {
    HeapConfig c;
    c.eden_bytes = 64 * 1024;
    c.survivor_bytes = 64 * 1024;
    c.tenure_age = 1;
    c.mature_reserve_bytes = size_t{8} << 20;
    c.large_reserve_bytes = size_t{8} << 20;
    c.min_full_threshold_bytes = size_t{1} << 20;
    c.read_environment = false;
    return c;
}

const LayoutId kOomNode = mask_layout(0b01, 41);

// Links young nodes onto *root until an allocation fails; returns how many
// were linked, or SIZE_MAX if `limit` were linked without a failure.
size_t fill_until_oom(Heap& h, uint64_t* root, size_t limit) {
    for (size_t i = 0; i < limit; ++i) {
        uintptr_t n = 0;
        try {
            n = h.allocate(16, kOomNode);
        } catch (const std::bad_alloc&) {
            return i;
        }
        reinterpret_cast<uint64_t*>(n)[1] = i;
        h.store(n, 0, *root);
        *root = n;
    }
    return SIZE_MAX;
}

size_t chain_length(uint64_t head) {
    size_t len = 0;
    for (uintptr_t n = static_cast<uintptr_t>(head); n != 0; n = static_cast<uintptr_t>(Heap::load(n, 0))) ++len;
    return len;
}

} // namespace

TEST_CASE("gc::Heap OOM - young survivors filling the old generation fail an allocation, and the heap recovers") {
    Heap h(tiny_old_config());
    // The chain's head in a heap root, the filling in frames of its own: the
    // stack is read conservatively, and a stale word naming the old head
    // would keep the whole chain (stack_scrub.hpp).
    HeapRoot<Heap> keep(h);
    const size_t linked = from_own_frame([&] { return fill_until_oom(h, keep.slot, size_t{2} << 20); });
    REQUIRE(linked != SIZE_MAX);
    CHECK(linked > 100000);
    // Everything linked before the failure is intact.
    CHECK_EQ(from_own_frame([&] { return chain_length(*keep.slot); }), linked);
    h.verify();
    CHECK(h.stats().full_collections > 0);

    // Let go: the next collection gives the room back and allocation works.
    *keep.slot = 0;
    scrub_stack();
    const size_t again = from_own_frame([&] { return fill_until_oom(h, keep.slot, 50000); });
    CHECK_EQ(again, SIZE_MAX);
    CHECK_EQ(from_own_frame([&] { return chain_length(*keep.slot); }), size_t{50000});
    h.collect(CollectionKind::Full);
    h.verify();
}

TEST_CASE("gc::Heap OOM - failing again and again stays recoverable") {
    Heap h(tiny_old_config());
    uint64_t keep = 0;
    h.add_root(&keep);
    for (int round = 0; round < 3; ++round) {
        const size_t linked = fill_until_oom(h, &keep, size_t{2} << 20);
        REQUIRE(linked != SIZE_MAX);
        // A failure with the data still held fails again at once.
        CHECK(fill_until_oom(h, &keep, 100000) != SIZE_MAX);
        keep = 0;
    }
    h.verify();
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap OOM - pretenured and large allocations fail with bad_alloc at the reservation") {
    Heap h(tiny_old_config());
    std::vector<uint64_t> held;
    held.reserve(4096);
    bool failed = false;
    for (size_t i = 0; i < 4096 && !failed; ++i) {
        try {
            held.push_back(h.allocate(2048, kOomNode, kAllocOld));
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        if (!failed) h.add_root(&held.back());
    }
    // held never reallocates (reserved above), so the roots stay valid.
    CHECK(failed);
    for (auto& slot : held) h.remove_root(&slot);
    held.clear();
    CHECK(h.allocate(2048, kOomNode, kAllocOld) != 0);

    std::vector<uint64_t> large;
    large.reserve(64);
    failed = false;
    for (size_t i = 0; i < 64 && !failed; ++i) {
        try {
            large.push_back(h.allocate(size_t{1} << 20, kOomNode));
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        if (!failed) h.add_root(&large.back());
    }
    CHECK(failed);
    for (auto& slot : large) h.remove_root(&slot);
    large.clear();
    CHECK(h.allocate(size_t{1} << 20, kOomNode) != 0);
    h.verify();
}
