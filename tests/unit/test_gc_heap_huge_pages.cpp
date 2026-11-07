// gc::Heap's old generation on huge pages (HeapState::commit_block): the
// mature space is advised for huge pages and committed a 2 MB granule at a
// time, while the heap's own accounting (committed_bytes, and so when it
// collects) still counts the 32 KB blocks it uses. Blocks the sweep hands
// back to the OS come back in use, zeroed where an allocation needs them,
// advised again, with what lives beside them untouched.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

using namespace brass::gc;

namespace {

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.name = "huge-pages-words";
        return register_layout(d);
    }();
    return id;
}

HeapConfig config() {
    HeapConfig c;
    c.eden_bytes = size_t{2} << 20;
    c.survivor_bytes = size_t{1} << 20;
    c.mature_reserve_bytes = size_t{256} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{1} << 20;  // retained: 32 free blocks
    c.mark_threads = 1;
    c.read_environment = false;
    c.verify = false;
    return c;
}

void sweep_all(Heap& h) {
    while (!h.sweep_slice(UINT64_MAX)) {
    }
}

constexpr size_t kWords = 30;

// A chain of old objects (word 0 the index as a number, word 1 the one before).
void grow(Heap& h, uint64_t& chain, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t o = h.allocate(kWords * 8, words_layout(), kAllocOld);
        reinterpret_cast<uint64_t*>(o)[0] = (i << 1) | 1;
        h.store(o, 1, chain);
        chain = o;
    }
}

bool chain_intact(uint64_t chain, size_t count) {
    size_t n = 0;
    for (uintptr_t o = static_cast<uintptr_t>(chain); o != 0; o = static_cast<uintptr_t>(Heap::load(o, 1))) {
        if ((Heap::load(o, 0) >> 1) != count - 1 - n) return false;
        ++n;
    }
    return n == count;
}

#if defined(__linux__)
// The VmFlags of the mapping holding `address`: "hg" is MADV_HUGEPAGE.
// Empty when /proc is not readable.
std::string vm_flags_at(uintptr_t address) {
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool inside = false;
    while (std::getline(smaps, line)) {
        uintptr_t lo = 0, hi = 0;
        char dash = 0;
        std::istringstream head(line);
        if (head >> std::hex >> lo >> dash >> hi && dash == '-') {
            inside = address >= lo && address < hi;
            continue;
        }
        if (inside && line.rfind("VmFlags:", 0) == 0) return line;
    }
    return {};
}

bool thp_available() { return std::ifstream("/sys/kernel/mm/transparent_hugepage/enabled").good(); }
#endif

} // namespace

TEST_CASE("gc::Heap huge pages - committing a granule leaves the heap's accounting by blocks") {
    Heap h(config());
    const size_t before = h.committed_bytes();
    uint64_t first = 0;
    uint64_t chain = 0;
    h.add_root(&first);
    h.add_root(&chain);
    // One block's worth: the granule behind it is committed, but the heap
    // counts (and collects by) the block it uses.
    grow(h, first, 4);
    CHECK(h.is_old(static_cast<uintptr_t>(first)));
    CHECK_EQ(h.committed_bytes() - before, size_t{32} << 10);
    // Past one granule into the next: still block by block.
    grow(h, chain, 9000);
    const size_t used = h.committed_bytes() - before;
    CHECK(used > (size_t{2} << 20));
    CHECK(used < (size_t{3} << 20));
    CHECK(chain_intact(chain, 9000));
    CHECK(chain_intact(first, 4));
#if defined(__linux__)
    if (thp_available()) {
        const std::string flags = vm_flags_at(static_cast<uintptr_t>(chain));
        CHECK(flags.empty() || flags.find(" hg") != std::string::npos);  // empty: /proc unreadable
    }
#endif
    h.remove_root(&chain);
    h.remove_root(&first);
}

TEST_CASE("gc::Heap huge pages - blocks given back to the OS are taken again and advised again") {
    Heap h(config());
    uint64_t keep = 0;
    uint64_t chain = 0;
    h.add_root(&keep);
    h.add_root(&chain);
    grow(h, keep, 100);
    // ~15 MB, then dropped: the sweep after it releases most of its blocks.
    grow(h, chain, 60000);
    h.collect(CollectionKind::Full);
    sweep_all(h);
    chain = 0;
    h.collect(CollectionKind::Full);
    const size_t high = h.committed_bytes();
    sweep_all(h);
    const size_t low = h.committed_bytes();
    CHECK(low + (size_t{8} << 20) < high);
    CHECK(chain_intact(keep, 100));
    // Allocated again: the released blocks inside the committed granules are
    // committed once more, read as fresh, and hold what is written to them.
    grow(h, chain, 60000);
    CHECK(h.committed_bytes() > low + (size_t{8} << 20));
    CHECK(chain_intact(chain, 60000));
    CHECK(chain_intact(keep, 100));
#if defined(__linux__)
    if (thp_available()) {
        const std::string flags = vm_flags_at(static_cast<uintptr_t>(chain));
        CHECK(flags.empty() || flags.find(" hg") != std::string::npos);  // empty: /proc unreadable
    }
#endif
    h.collect(CollectionKind::Full);
    sweep_all(h);
    CHECK(chain_intact(chain, 60000));
    CHECK(chain_intact(keep, 100));
    h.remove_root(&chain);
    h.remove_root(&keep);
}
