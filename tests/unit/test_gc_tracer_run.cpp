// gc::Tracer::visit_run and the paths that reach it: a run of slots is
// filtered against the tracer's range (four slots to a compare where the CPU
// has AVX2, runs of 16 slots and up), and must hand the slow path exactly the
// slots a slot-by-slot visit would, in order, for every run length and every
// place a reference can sit in it. Then the same through a collection: an
// object's reference at any word of a long run is traced, and a store through
// a slot's address finds its object (Heap::remember_interior) wherever in
// the object the slot lies.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstdint>
#include <vector>

using namespace brass::gc;

namespace {

HeapConfig config() {
    HeapConfig c;
    c.eden_bytes = size_t{2} << 20;
    c.survivor_bytes = size_t{2} << 20;
    c.mature_reserve_bytes = size_t{64} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{1} << 30;
    c.read_environment = false;
    c.verify = true;
    return c;
}

struct Seen {
    const uint64_t* base = nullptr;
    std::vector<size_t> slots;
};

void record(Tracer& t, uint64_t* slot, uint64_t) {
    auto* seen = static_cast<Seen*>(t.state());
    seen->slots.push_back(static_cast<size_t>(slot - seen->base));
}

// Every payload word a slot.
LayoutId words_layout() {
    LayoutDescriptor d;
    d.kind = LayoutKind::Words;
    d.type_tag = 77;
    d.name = "test.words";
    static const LayoutId id = register_layout(d);
    return id;
}

const LayoutId kLeafPair = mask_layout(0, 78);

} // namespace

TEST_CASE("gc::Tracer visit_run - visits exactly the slots whose address lies in the range") {
    Heap h(config());
    constexpr uintptr_t kLo = uintptr_t{1} << 32;
    constexpr uintptr_t kHi = kLo + (uintptr_t{1} << 20);
    Seen seen;
    Tracer t(h, Tracer::Purpose::Verify, kLo, kHi, &record, nullptr, nullptr, &seen);
    // Words that are not in the range: zero, a number, an address just below
    // and just at the end, and one whose tag bits must be masked off first.
    const uint64_t outside[] = {0, 0x3FF0000000000000ULL, kLo - 8, kHi, (uint64_t{0xFFF1} << 48) | (kHi + 64)};
    const uint64_t inside[] = {kLo, kHi - 8, (uint64_t{0xFFF1} << 48) | (kLo + 4096)};
    for (size_t length = 0; length <= 70; ++length) {
        std::vector<uint64_t> run(length + 2);
        seen.base = run.data() + 1;
        // No reference at all, then one at each position, then one in every
        // third: the words beside the run (in the range) are never visited.
        for (size_t at = 0; at <= length + 1; ++at) {
            run.front() = kLo + 16;
            run.back() = kLo + 24;
            std::vector<size_t> expected;
            for (size_t i = 0; i < length; ++i) {
                const bool in = at == length + 1 ? i % 3 == 1 : i == at;
                run[i + 1] = in ? inside[i % 3] : outside[i % 5];
                if (in) expected.push_back(i);
            }
            seen.slots.clear();
            t.visit_run(run.data() + 1, length);
            CHECK(seen.slots == expected);
        }
    }
}

TEST_CASE("gc::Tracer visit_run - a reference at any word of an old or a young object's run is traced") {
    for (const size_t words : {size_t{1}, size_t{3}, size_t{4}, size_t{15}, size_t{16}, size_t{17}, size_t{31},
                               size_t{64}, size_t{65}, size_t{200}}) {
        for (const uint32_t flags : {uint32_t{0}, kAllocOld}) {
            Heap h(config());
            uint64_t holder = h.allocate(words * 8, words_layout(), flags);
            h.add_root(&holder);
            for (size_t i = 0; i < words; ++i) {
                const uint64_t leaf = h.allocate(16, kLeafPair);
                reinterpret_cast<uint64_t*>(leaf)[1] = 1000 + i;
                // A number in every other slot, so a group of four holds both.
                h.store(static_cast<uintptr_t>(holder), i, (i % 2 == 0 || i + 1 == words) ? leaf : uint64_t{7});
            }
            h.collect(CollectionKind::Minor);
            h.collect(CollectionKind::Minor);
            for (size_t i = 0; i < words; ++i) {
                const uint64_t word = Heap::load(static_cast<uintptr_t>(holder), i);
                if (i % 2 == 0 || i + 1 == words) {
                    CHECK(h.is_valid_object(static_cast<uintptr_t>(word)));
                    CHECK_EQ(Heap::load(static_cast<uintptr_t>(word), 1), uint64_t{1000} + i);
                } else {
                    CHECK_EQ(word, uint64_t{7});
                }
            }
            h.remove_root(&holder);
        }
    }
}

TEST_CASE("gc::Heap barrier - a store through a slot's address finds its object wherever the slot lies") {
    // Mature objects of 1 to 150 words between neighbours, a young reference
    // stored through the address of the first, a middle and the last word:
    // within the object's first 64 granules the start is found in the slot's
    // own word of the block's start bits, past them by the search.
    for (const size_t words : {size_t{1}, size_t{2}, size_t{9}, size_t{63}, size_t{64}, size_t{65}, size_t{150}}) {
        Heap h(config());
        std::vector<uint64_t> olds(12, 0);
        for (uint64_t& o : olds) {
            o = h.allocate(words * 8, words_layout(), kAllocOld);
            h.add_root(&o);
        }
        size_t n = 0;
        for (uint64_t& o : olds) {
            for (const size_t at : {size_t{0}, words / 2, words - 1}) {
                const uint64_t young = h.allocate(16, kLeafPair);
                reinterpret_cast<uint64_t*>(young)[1] = 500 + n;
                const uintptr_t slot = static_cast<uintptr_t>(o) + at * 8;
                *reinterpret_cast<uint64_t*>(slot) = young;
                h.write_barrier_interior(slot, young);
            }
            ++n;
        }
        h.collect(CollectionKind::Minor);  // verification checks every old-to-young slot is remembered
        n = 0;
        for (uint64_t& o : olds) {
            for (const size_t at : {size_t{0}, words / 2, words - 1}) {
                const uintptr_t young = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(o), at));
                CHECK(h.is_valid_object(young));
                CHECK_EQ(Heap::load(young, 1), uint64_t{500} + n);
            }
            ++n;
            h.remove_root(&o);
        }
    }
}
