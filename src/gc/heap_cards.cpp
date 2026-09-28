// The remembered set's half of a minor collection: the dirty cards of the old
// generation, rescanned for references to young objects.
//
// A mature object is remembered by the card of its start, and a dirty card
// rescans every object starting in it. A large object is remembered card by
// card: its first card dirty (kCardDirty) rescans the whole object, any other
// dirty card (and a first card marked kCardDirtyRange) rescans only the slots
// in that card's 512 bytes, so a store into a multi-megabyte array costs one
// card's worth of rescanning rather than the array.
//
// Each card is cleaned before its objects are scanned; the scan re-dirties it
// (Tracer owner, remember_slot) when a slot still names a young object after
// the collection.
//
// Only the regions (kCardRegionShift, one per 32 KB) that a card store marked
// are visited, so the cost follows the remembered set rather than the size of
// the old generation: the walk reads one byte per region, eight at a time.

#include "heap_internal.hpp"

#include <algorithm>
#include <cstring>

namespace brass::gc::detail {

namespace {

struct CardRange {
    uintptr_t object;
    size_t begin;  // payload bytes [begin, end)
    size_t end;
};

// The region bytes (kCardRegionShift) say which blocks may hold a dirty card;
// a region is cleared before its cards are scanned, and a card the scan
// re-dirties marks it again. So the walk reads one byte per block and scans
// only the blocks the mutator wrote to.
void scan_mature_cards(Tracer& t, GcState& g) {
    static_assert((size_t{1} << kCardRegionShift) == kBlockBytes, "a card region is one mature block");
    HeapState& s = g.s;
    const size_t count = s.blocks.size();
    uint8_t* regions = s.regions;
    for (size_t i = 0; i < count; ++i) {
        if ((i & 7) == 0 && i + 8 <= count) {
            uint64_t eight;
            std::memcpy(&eight, regions + i, 8);
            if (eight == 0) {
                i += 7;
                continue;
            }
        }
        if (regions[i] == 0) continue;
        regions[i] = 0;
        BlockMeta& meta = *s.blocks[i];
        if (!meta.in_use) continue;
        const uintptr_t lo = s.block_base(static_cast<uint32_t>(i));
        uint8_t* cards = s.card_of(lo);
        for (size_t c = 0; c < kCardsPerBlock; ++c) {
            if (cards[c] != kCardDirty) continue;
            cards[c] = kCardClean;
            ++g.dirty_cards;
            uint64_t starts = meta.starts[c];  // card c covers start word c
            while (starts) {
                const unsigned bit = ctz64(starts);
                starts &= starts - 1;
                scan_object(t, lo + (c * 64 + bit) * kGranuleBytes, true);
            }
        }
    }
}

// The dirty ranges of every large object, cleaning their cards. Gathered
// before any is scanned: a scan promotes, and a promotion may allocate a
// large object into the table being walked.
// Walked region by region in address order, so an object's first card (whose
// kCardDirty means the whole object, and whose region its whole-object mark
// sets) is met before any other card of it.
void gather_large_ranges(GcState& g, std::vector<CardRange>& out) {
    HeapState& s = g.s;
    constexpr size_t kCardsPerPage = kPageBytes / kCardBytes;
    constexpr size_t kCardsPerRegion = (size_t{1} << kCardRegionShift) / kCardBytes;
    constexpr size_t kPagesPerRegion = (size_t{1} << kCardRegionShift) / kPageBytes;
    uint8_t* regions = s.regions + ((s.large_lo - s.mature_lo) >> kCardRegionShift);
    const size_t region_count = (static_cast<size_t>(s.large_frontier) + kPagesPerRegion - 1) / kPagesPerRegion;
    for (size_t r = 0; r < region_count; ++r) {
        if (regions[r] == 0) continue;
        regions[r] = 0;
        const uintptr_t region_lo = s.large_lo + (static_cast<uintptr_t>(r) << kCardRegionShift);
        uint8_t* cards = s.card_of(region_lo);
        for (size_t c = 0; c < kCardsPerRegion; ++c) {
            if (cards[c] == kCardClean) continue;
            const uintptr_t card_lo = region_lo + c * kCardBytes;
            const size_t page = (card_lo - s.large_lo) / kPageBytes;
            const uint32_t head_plus_one = page < s.large_page_head.size() ? s.large_page_head[page] : 0;
            if (head_plus_one == 0) {  // a card left behind by a freed object
                cards[c] = kCardClean;
                continue;
            }
            const uint32_t head = head_plus_one - 1;
            const uintptr_t object = s.large_object_at(head);
            const uintptr_t header = object - kHeaderBytes;
            const size_t size = header_of(object)->size;
            uint8_t* first = s.card_of(header);
            if (*first == kCardDirty) {
                const auto it = s.large_objects.find(head);
                const size_t pages = it != s.large_objects.end() ? it->second : 0;
                std::memset(first, kCardClean, pages * kCardsPerPage);
                ++g.dirty_cards;
                out.push_back(CardRange{object, 0, size});
                continue;
            }
            cards[c] = kCardClean;
            ++g.dirty_cards;
            const uintptr_t lo = std::max(card_lo, object);
            const uintptr_t hi = std::min(card_lo + kCardBytes, object + size);
            if (hi <= lo) continue;
            const size_t begin = lo - object;
            const size_t end = hi - object;
            if (!out.empty() && out.back().object == object && out.back().end == begin) {
                out.back().end = end;  // adjacent dirty cards: one range
            } else {
                out.push_back(CardRange{object, begin, end});
            }
        }
    }
}

} // namespace

void scan_dirty_cards(Tracer& t, GcState& g) {
    scan_mature_cards(t, g);
    std::vector<CardRange> ranges;
    gather_large_ranges(g, ranges);
    for (const CardRange& r : ranges) {
        if (r.begin == 0 && r.end == header_of(r.object)->size) scan_object(t, r.object, true);
        else scan_object_range(t, r.object, r.begin, r.end);
    }
}

void scan_object_range(Tracer& t, uintptr_t object, size_t begin, size_t end) {
    t.set_owner(object, true);
    const ObjectHeader* h = header_of(object);
    const LayoutDescriptor& d = layout_descriptor(h->layout);
    auto* words = reinterpret_cast<uint64_t*>(object);
    const size_t count = h->size / kGranuleBytes;
    const size_t first = begin / kGranuleBytes;
    const size_t last = std::min(count, (end + kGranuleBytes - 1) / kGranuleBytes);
    switch (d.kind) {
    case LayoutKind::Leaf:
        return;
    case LayoutKind::Words:
        for (size_t i = first; i < last; ++i) t.visit(words + i);
        return;
    case LayoutKind::Mask:
        for (size_t i = first; i < last; ++i) {
            if ((d.mask >> std::min<size_t>(i, 63)) & 1) t.visit(words + i);
        }
        return;
    case LayoutKind::Custom:
        if (d.trace_range) d.trace_range(object, h->size, begin, end, t);
        else d.trace(object, h->size, t);
        return;
    }
}

} // namespace brass::gc::detail
