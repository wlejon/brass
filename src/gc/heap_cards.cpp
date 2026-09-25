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

#include "heap_internal.hpp"

#include <algorithm>
#include <cstring>

namespace brass::gc::detail {

namespace {

constexpr uint64_t kCleanWord = 0x0101010101010101ULL;

struct CardRange {
    uintptr_t object;
    size_t begin;  // payload bytes [begin, end)
    size_t end;
};

void scan_mature_cards(Tracer& t, GcState& g) {
    HeapState& s = g.s;
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        BlockMeta& meta = *s.blocks[i];
        if (!meta.in_use) continue;
        const uintptr_t lo = s.block_base(static_cast<uint32_t>(i));
        uint8_t* cards = s.card_of(lo);
        uint64_t any_dirty = 0;
        for (size_t w = 0; w < kCardsPerBlock / 8; ++w) {
            uint64_t word;
            std::memcpy(&word, cards + w * 8, 8);
            any_dirty |= word ^ kCleanWord;
        }
        if (!any_dirty) continue;
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
void gather_large_ranges(GcState& g, std::vector<CardRange>& out) {
    HeapState& s = g.s;
    constexpr size_t kCardsPerPage = kPageBytes / kCardBytes;
    for (const auto& [head, pages] : s.large_objects) {
        const uintptr_t object = s.large_object_at(head);
        uint8_t* cards = s.card_of(object - kHeaderBytes);
        const size_t count = static_cast<size_t>(pages) * kCardsPerPage;
        uint64_t any_dirty = 0;
        for (size_t w = 0; w < count; w += 8) {
            uint64_t word;
            std::memcpy(&word, cards + w, 8);
            any_dirty |= word ^ kCleanWord;
        }
        if (!any_dirty) continue;
        const size_t size = header_of(object)->size;
        if (cards[0] == kCardDirty) {
            std::memset(cards, kCardClean, count);
            ++g.dirty_cards;
            out.push_back(CardRange{object, 0, size});
            continue;
        }
        const uintptr_t header = object - kHeaderBytes;
        for (size_t c = 0; c < count; ++c) {
            if (cards[c] == kCardClean) continue;
            cards[c] = kCardClean;
            ++g.dirty_cards;
            const uintptr_t lo = std::max(header + c * kCardBytes, object);
            const uintptr_t hi = std::min(header + (c + 1) * kCardBytes, object + size);
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
