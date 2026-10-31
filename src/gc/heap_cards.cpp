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
//
// While the last cycle's sweep is pending, a dead object's slots are never
// scanned: a mature card may still be dirty from before the cycle's finish
// (an in-place finish leaves the cards as they were, promote-all cleans them)
// and is shared by the objects starting in it, so an unswept block's dead
// starts are masked off by its marks. A dead large object (unswept, not yet
// freed) has cards of its own, which either finish cleans.
//
// A parallel minor collection (heap_minor_parallel.cpp) gathers the dirty
// cards into a list of pieces first, cleaning them, and its threads scan the
// pieces (gather_dirty_cards, scan_card_work).

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
            // An unswept block (heap_sweep_lazy.cpp) still holds the starts of
            // the objects the last cycle found dead, whose slots may name
            // memory freed since: only its marked starts are objects. Asked
            // per card, since this scan's promotions may sweep the block.
            if (s.block_unswept(meta)) starts &= meta.marks[c];
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
// `clean` false (a marking cycle's in-place finish, collect_dirty_cards): the
// cards and regions are left as they are, and the rest of an object whose
// whole was asked for is skipped rather than cleaned.
void gather_large_ranges(GcState& g, std::vector<CardRange>& out, bool clean = true) {
    HeapState& s = g.s;
    constexpr size_t kCardsPerPage = kPageBytes / kCardBytes;
    constexpr size_t kCardsPerRegion = (size_t{1} << kCardRegionShift) / kCardBytes;
    constexpr size_t kPagesPerRegion = (size_t{1} << kCardRegionShift) / kPageBytes;
    uint8_t* regions = s.regions + ((s.large_lo - s.mature_lo) >> kCardRegionShift);
    const size_t region_count = (static_cast<size_t>(s.large_frontier) + kPagesPerRegion - 1) / kPagesPerRegion;
    uintptr_t whole_end = 0;  // not cleaning: the end of the last object gathered whole
    for (size_t r = 0; r < region_count; ++r) {
        if (regions[r] == 0) continue;
        if (clean) regions[r] = 0;
        const uintptr_t region_lo = s.large_lo + (static_cast<uintptr_t>(r) << kCardRegionShift);
        uint8_t* cards = s.card_of(region_lo);
        for (size_t c = 0; c < kCardsPerRegion; ++c) {
            if (cards[c] == kCardClean) continue;
            const uintptr_t card_lo = region_lo + c * kCardBytes;
            if (card_lo < whole_end) continue;
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
                if (clean) std::memset(first, kCardClean, pages * kCardsPerPage);
                else whole_end = header + pages * kPageBytes;
                ++g.dirty_cards;
                out.push_back(CardRange{object, 0, size});
                continue;
            }
            if (clean) cards[c] = kCardClean;
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

bool scans_ranges(const LayoutDescriptor& d) noexcept {
    return d.kind == LayoutKind::Words || d.kind == LayoutKind::Mask || d.kind == LayoutKind::Leaf ||
           (d.kind == LayoutKind::Custom && d.trace_range != nullptr);
}

// The mature half of gather_dirty_cards: scan_mature_cards' walk, recording
// each dirty card's starts instead of scanning them. No promotion runs yet,
// so no block can be swept under the walk. `clean` as for gather_large_ranges.
void gather_mature_cards(GcState& g, std::vector<CardWork>& out, bool clean) {
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
        if (clean) regions[i] = 0;
        BlockMeta& meta = *s.blocks[i];
        if (!meta.in_use) continue;
        const uintptr_t lo = s.block_base(static_cast<uint32_t>(i));
        uint8_t* cards = s.card_of(lo);
        const bool unswept = s.block_unswept(meta);
        for (size_t c = 0; c < kCardsPerBlock; ++c) {
            if (cards[c] != kCardDirty) continue;
            if (clean) cards[c] = kCardClean;
            ++g.dirty_cards;
            uint64_t starts = meta.starts[c];
            if (unswept) starts &= meta.marks[c];  // only the last cycle's live (scan_mature_cards)
            if (starts != 0) out.push_back(CardWork{lo + c * kCardBytes, 0, 0, starts});
        }
    }
}

// gather_dirty_cards and collect_dirty_cards: the pieces, the cards cleaned
// or not.
void gather_card_work(GcState& g, std::vector<CardWork>& out, bool clean) {
    gather_mature_cards(g, out, clean);
    std::vector<CardRange> ranges;
    gather_large_ranges(g, ranges, clean);
    for (size_t i = 0; i < ranges.size(); ++i) {
        const uintptr_t object = ranges[i].object;
        const size_t size = header_of(object)->size;
        if (!scans_ranges(layout_descriptor(header_of(object)->layout))) {
            // Its range scan is its whole trace (scan_object_range): one piece
            // for all of its dirty cards, which follow one another, or two
            // threads would trace it at once and write the same slots.
            while (i + 1 < ranges.size() && ranges[i + 1].object == object) ++i;
            out.push_back(CardWork{object, 0, size, 0});
            continue;
        }
        // The whole of a Custom object is its full trace, which may visit
        // slots outside the payload (that is why its first card asked for
        // all of it: Heap::remember_slot): one piece. A Words or Mask object
        // has none, so its whole is its ranges.
        const bool whole = ranges[i].begin == 0 && ranges[i].end == size;
        if (whole && layout_descriptor(header_of(object)->layout).kind == LayoutKind::Custom) {
            out.push_back(CardWork{object, 0, size, 0});
            continue;
        }
        for (size_t at = ranges[i].begin; at < ranges[i].end; at += kCardWorkSliceBytes) {
            out.push_back(CardWork{object, at, std::min(ranges[i].end, at + kCardWorkSliceBytes), 0});
        }
    }
}

} // namespace

void gather_dirty_cards(GcState& g, std::vector<CardWork>& out) { gather_card_work(g, out, true); }

void collect_dirty_cards(GcState& g, std::vector<CardWork>& out) { gather_card_work(g, out, false); }

void scan_card_work(Tracer& t, const CardWork& w) {
    if (w.starts != 0) {
        uint64_t starts = w.starts;
        while (starts) {
            const unsigned bit = ctz64(starts);
            starts &= starts - 1;
            scan_object(t, w.object + bit * kGranuleBytes, true);
        }
        return;
    }
    if (w.begin == 0 && w.end == header_of(w.object)->size) scan_object(t, w.object, true);
    else scan_object_range(t, w.object, w.begin, w.end);
}

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
