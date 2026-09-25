// The two collections.
//
// Minor: every young object reachable from the roots and from the dirty cards
// of the old generation is copied, into the other survivor space while it is
// younger than the tenure age and into the old generation after; the young
// spaces are then empty but for the survivors. Its cost is the roots, the
// dirty cards and the survivors, whatever the old generation's size.
//
// Full: every young survivor is promoted, every reachable old object is
// marked in place (mature objects in the blocks' side bitmaps and their
// lines, large objects in their header), and the old generation is swept:
// unmarked objects' lines and pages are reclaimed without moving anything.
//
// Both trace through the same Tracer with a mode-specific slow path, and both
// settle weak slots, ephemerons and finalizers the same way (heap_weak.cpp).

#include "heap_internal.hpp"

#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/coroutine.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>

namespace brass::gc::detail {

namespace {

constexpr uint8_t kPoisonByte = 0xDB;

GcState& gc_state(Tracer& t) noexcept { return *static_cast<GcState*>(t.state()); }

// ---- minor collection ----

uintptr_t evacuate_young(GcState& g, uintptr_t object) {
    HeapState& s = g.s;
    ObjectHeader* h = header_of(object);
    const size_t total = kHeaderBytes + h->size;
    const unsigned age = std::min<unsigned>(h->age() + 1u, 7u);
    uintptr_t dest = 0;
    bool promoted = false;
    if (age < s.config.tenure_age &&
        s.survivor_top[g.to] + total <= s.survivor_lo[g.to] + s.survivor_bytes) {
        dest = s.survivor_top[g.to];
        s.survivor_top[g.to] += total;
    } else {
        dest = s.old_allocate(total);
        if (dest == 0) gc_fatal("the old generation's reservation is exhausted during a minor collection");
        promoted = true;
    }
    std::memcpy(reinterpret_cast<void*>(dest), h, total);
    auto* nh = reinterpret_cast<ObjectHeader*>(dest);
    uint8_t bits = static_cast<uint8_t>(h->gc_bits & ~(kGcForwarded | kGcAgeMask));
    if (!promoted) bits = static_cast<uint8_t>(bits | (age << kGcAgeShift));
    nh->gc_bits = bits;
    const uintptr_t moved = dest + kHeaderBytes;
    if (promoted) {
        g.promoted_bytes += total;
        s.old_allocated_since_full += total;
    } else {
        s.set_young_start(moved);
    }
    g.copied_bytes += total;
    h->gc_bits = static_cast<uint8_t>(h->gc_bits | kGcForwarded);
    *reinterpret_cast<uintptr_t*>(object) = moved;
    ++s.relocation_epoch;
    g.gray.push_back(moved);
    return moved;
}

void minor_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    GcState& g = gc_state(t);
    if (!g.heap.is_reference_tag(word)) return;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    uintptr_t now = a;
    if (g.in_from(a)) {
        const ObjectHeader* h = header_of(a);
        now = h->forwarded() ? *reinterpret_cast<const uintptr_t*>(a) : evacuate_young(g, a);
        *slot = (word & ~kAddressMask) | now;
    }
    if (t.owner_old() && g.heap.is_young(now)) *g.s.card_of(t.owner()) = kCardDirty;
}

// ---- full collection ----

bool mark_mature(GcState& g, uintptr_t object) {
    HeapState& s = g.s;
    const uint32_t index = s.block_index(object);
    if (index >= s.blocks.size() || !s.blocks[index]->in_use) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "a reference to %p names no block in use of the mature space",
                      reinterpret_cast<void*>(object));
        gc_fatal(msg);
    }
    BlockMeta& meta = *s.blocks[index];
    const uintptr_t lo = s.block_base(index);
    const size_t granule = (object - lo) / kGranuleBytes;
    const uint64_t bit = uint64_t{1} << (granule & 63);
    uint64_t& word = meta.marks[granule >> 6];
    if (word & bit) return false;
    word |= bit;
    const size_t total = kHeaderBytes + header_of(object)->size;
    s.mark_lines(meta, lo, object - kHeaderBytes, total, true);
    g.marked_bytes += total;
    return true;
}

bool mark_large(GcState& g, uintptr_t object) {
    ObjectHeader* h = header_of(object);
    if (h->gc_bits & kGcLargeMarked) return false;
    h->gc_bits = static_cast<uint8_t>(h->gc_bits | kGcLargeMarked);
    g.marked_bytes += kHeaderBytes + h->size;
    return true;
}

// Marks a freshly promoted copy (it lives past this collection's sweep).
void mark_new_old(GcState& g, uintptr_t object) {
    if (g.s.in_mature(object)) mark_mature(g, object);
    else mark_large(g, object);
}

uintptr_t promote_young(GcState& g, uintptr_t object) {
    HeapState& s = g.s;
    ObjectHeader* h = header_of(object);
    const size_t total = kHeaderBytes + h->size;
    const uintptr_t dest = s.old_allocate(total);
    if (dest == 0) gc_fatal("the old generation's reservation is exhausted during a full collection");
    std::memcpy(reinterpret_cast<void*>(dest), h, total);
    auto* nh = reinterpret_cast<ObjectHeader*>(dest);
    nh->gc_bits = static_cast<uint8_t>(h->gc_bits & ~(kGcForwarded | kGcAgeMask));
    const uintptr_t moved = dest + kHeaderBytes;
    h->gc_bits = static_cast<uint8_t>(h->gc_bits | kGcForwarded);
    *reinterpret_cast<uintptr_t*>(object) = moved;
    ++s.relocation_epoch;
    g.copied_bytes += total;
    g.promoted_bytes += total;
    mark_new_old(g, moved);
    g.gray.push_back(moved);
    return moved;
}

void full_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    GcState& g = gc_state(t);
    if (!g.heap.is_reference_tag(word)) return;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    HeapState& s = g.s;
    if (g.heap.is_young(a)) {
        if (!g.in_from(a)) return;
        const ObjectHeader* h = header_of(a);
        const uintptr_t now = h->forwarded() ? *reinterpret_cast<const uintptr_t*>(a) : promote_young(g, a);
        *slot = (word & ~kAddressMask) | now;
        return;
    }
    if (s.in_mature(a)) {
        if (mark_mature(g, a)) g.gray.push_back(a);
    } else if (s.in_large(a)) {
        if (mark_large(g, a)) g.gray.push_back(a);
    }
}

void drain(Tracer& t, GcState& g) {
    while (!g.gray.empty()) {
        const uintptr_t object = g.gray.back();
        g.gray.pop_back();
        scan_object(t, object, g.heap.is_old(object));
    }
}

// ---- the old generation's dirty cards (minor) ----

void scan_dirty_cards(Tracer& t, GcState& g) {
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
            any_dirty |= word ^ 0x0101010101010101ULL;
        }
        if (!any_dirty) continue;
        for (size_t c = 0; c < kCardsPerBlock; ++c) {
            if (cards[c] != kCardDirty) continue;
            cards[c] = kCardClean;
            uint64_t starts = meta.starts[c];  // card c covers start word c
            while (starts) {
                const unsigned bit = ctz64(starts);
                starts &= starts - 1;
                scan_object(t, lo + (c * 64 + bit) * kGranuleBytes, true);
            }
        }
    }
    std::vector<uintptr_t> dirty_large;
    for (const auto& entry : s.large_objects) {
        const uintptr_t object = s.large_lo + static_cast<uintptr_t>(entry.first) * kPageBytes + kHeaderBytes;
        uint8_t* card = s.card_of(object);
        if (*card == kCardDirty) {
            *card = kCardClean;
            dirty_large.push_back(object);
        }
    }
    for (uintptr_t object : dirty_large) scan_object(t, object, true);
}

void run_hooks(GcState& g) {
    HeapState& s = g.s;
    for (size_t i = 0; i < s.hooks.size(); ++i) s.hooks[i].hook(g.heap, g.kind);
}

void clean_all_cards(HeapState& s) {
    for (size_t p = 0; p < s.card_pages_committed.size(); ++p) {
        if (s.card_pages_committed[p]) std::memset(s.cards + p * 4096, kCardClean, 4096);
    }
}

void minor(GcState& g) {
    HeapState& s = g.s;
    Tracer t(g.heap, Tracer::Purpose::Minor, g.heap.young_base(), g.heap.young_base() + g.heap.young_span(),
             &minor_visit, &weak_visit, &ephemeron_visit, &g);
    visit_roots(s, t);
    scan_dirty_cards(t, g);
    trace_to_fixpoint(t, g);
    settle_weakness(g);
    run_hooks(g);

    // Reclaim the evacuated spaces.
    if (s.poison) {
        std::memset(reinterpret_cast<void*>(s.eden_lo), kPoisonByte, g.eden_top - s.eden_lo);
        std::memset(reinterpret_cast<void*>(s.survivor_lo[g.from]), kPoisonByte, g.from_top - s.survivor_lo[g.from]);
    }
    s.clear_young_starts(s.eden_lo, s.eden_hi);
    s.clear_young_starts(s.survivor_lo[g.from], s.survivor_lo[g.from] + s.survivor_bytes);
    s.survivor_top[g.from] = s.survivor_lo[g.from];
    s.from_survivor = g.to;
    s.eden_bytes_retired += g.eden_top - s.eden_lo;
    s.reset_eden();
    s.minor_requested = false;
    if (s.old_allocated_since_full > s.full_threshold) s.full_requested = true;
}

void full(GcState& g) {
    HeapState& s = g.s;
    Tracer t(g.heap, Tracer::Purpose::Full, s.base, s.base + s.reserve_bytes, &full_visit, &weak_visit,
             &ephemeron_visit, &g);
    visit_roots(s, t);
    trace_to_fixpoint(t, g);
    settle_weakness(g);
    run_hooks(g);

    s.sweep_old();
    clean_all_cards(s);
    if (s.poison) {
        std::memset(reinterpret_cast<void*>(s.eden_lo), kPoisonByte, g.eden_top - s.eden_lo);
        std::memset(reinterpret_cast<void*>(s.survivor_lo[g.from]), kPoisonByte, g.from_top - s.survivor_lo[g.from]);
    }
    std::fill(s.young_starts.begin(), s.young_starts.end(), 0);
    s.survivor_top[0] = s.survivor_lo[0];
    s.survivor_top[1] = s.survivor_lo[1];
    s.eden_bytes_retired += g.eden_top - s.eden_lo;
    s.reset_eden();

    const double grown = static_cast<double>(g.marked_bytes) * s.config.growth_factor;
    s.full_threshold = std::max<uint64_t>(s.config.min_full_threshold_bytes, static_cast<uint64_t>(grown));
    s.old_allocated_since_full = 0;
    s.stats.old_live_bytes = g.marked_bytes;
    s.full_requested = false;
    s.minor_requested = false;
}

} // namespace

void visit_roots(HeapState& s, Tracer& t) {
    // Gathered before any root is visited: append_active_coro_roots reads
    // each registered frame's is_done, and once a frame has been evacuated
    // its old copy holds the forwarding address there instead.
    std::vector<uintptr_t*> slots;
    brass_append_generated_frame_roots(s.mutator_fp, s.mutator_ip, slots);
    brass_append_native_frame_roots(slots);
    if (s.coro_frames) runtime::append_active_coro_roots(*s.coro_frames, slots);

    t.set_owner(0, false);
    for (size_t i = 0; i < s.roots.size(); ++i) t.visit(s.roots[i]);
    for (size_t i = 0; i < s.sources.size(); ++i) {
        RootSourceEntry& source = s.sources[i];
        if (source.fn) source.fn(t, source.context);
        else source.callable(t);
        t.set_owner(0, false);
    }
    for (uintptr_t* slot : slots) t.visit_ref(slot);
}

void scan_object(Tracer& t, uintptr_t object, bool old) {
    t.set_owner(object, old);
    const ObjectHeader* h = header_of(object);
    const LayoutDescriptor& d = layout_descriptor(h->layout);
    auto* words = reinterpret_cast<uint64_t*>(object);
    const size_t count = h->size / kGranuleBytes;
    switch (d.kind) {
    case LayoutKind::Leaf:
        return;
    case LayoutKind::Words:
        for (size_t i = 0; i < count; ++i) t.visit(words + i);
        return;
    case LayoutKind::Mask: {
        const size_t low = std::min<size_t>(count, 63);
        uint64_t bits = d.mask & ((uint64_t{1} << low) - 1);
        while (bits) {
            const unsigned i = ctz64(bits);
            bits &= bits - 1;
            t.visit(words + i);
        }
        if ((d.mask >> 63) & 1) {
            for (size_t i = 63; i < count; ++i) t.visit(words + i);
        }
        return;
    }
    case LayoutKind::Custom:
        d.trace(object, h->size, t);
        return;
    }
}

uintptr_t live_address(const GcState& g, uintptr_t a) noexcept {
    const HeapState& s = g.s;
    if (g.heap.is_young(a)) {
        if (!g.in_from(a)) return a;
        const ObjectHeader* h = header_of(a);
        return h->forwarded() ? *reinterpret_cast<const uintptr_t*>(a) : 0;
    }
    if (g.kind == CollectionKind::Minor) return a;
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size()) return a;
        const BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        return ((meta.marks[granule >> 6] >> (granule & 63)) & 1) ? a : 0;
    }
    if (s.in_large(a)) return (header_of(a)->gc_bits & kGcLargeMarked) ? a : 0;
    return a;
}

} // namespace brass::gc::detail

void brass::gc::Tracer::visit_derived(uint64_t* slot) noexcept {
    const uint64_t word = *slot;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    Heap& h = heap();
    if (!h.contains(a) || !h.is_reference_tag(word)) return;
    const uintptr_t base = detail::containing_object(detail::Collector::state(h), a, true);
    if (base == 0) return;
    if (base == a) {
        visit(slot);
        return;
    }
    uint64_t moved = (word & ~kAddressMask) | base;
    visit(&moved);
    *slot = (word & ~kAddressMask) | ((moved & kAddressMask) + (a - base));
}

void brass::gc::Tracer::visit_conservative(uint64_t* slot) noexcept {
    const uint64_t word = *slot;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    Heap& h = heap();
    if (!h.contains(a) || !h.is_reference_tag(word)) return;
    if (detail::containing_object(detail::Collector::state(h), a, true) == a) visit(slot);
}

namespace brass::gc::detail {

void trace_to_fixpoint(Tracer& t, GcState& g) {
    for (;;) {
        drain(t, g);
        if (g.ephemerons.empty()) return;
        bool progress = false;
        std::vector<EphemeronEntry> pending;
        pending.swap(g.ephemerons);
        for (const EphemeronEntry& e : pending) {
            const uint64_t word = *e.key;
            const uintptr_t live = live_address(g, static_cast<uintptr_t>(word & kAddressMask));
            if (!live) {
                g.ephemerons.push_back(e);
                continue;
            }
            *e.key = (word & ~kAddressMask) | live;
            t.set_owner(e.owner, e.owner_old);
            if (g.kind == CollectionKind::Minor && e.owner_old && g.heap.is_young(live)) {
                *g.s.card_of(e.owner) = kCardDirty;
            }
            t.visit(e.value);
            progress = true;
        }
        if (!progress) return;
    }
}

void Collector::collect(Heap& heap, CollectionKind kind) {
    HeapState& s = state(heap);
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    if (s.verify) verify_heap(heap, "before a collection");

    s.collecting = true;
    s.collecting_kind = kind;
    GcState g(s, heap, kind);
    g.from = s.from_survivor;
    g.to = 1 - g.from;
    g.eden_top = heap.alloc_.top;
    g.from_top = s.survivor_top[g.from];
    s.gc_eden_top = g.eden_top;
    s.gc_from_top = g.from_top;
    if (kind == CollectionKind::Minor) minor(g);
    else full(g);
    s.collecting = false;

    if (s.verify) verify_heap(heap, "after a collection");
    coro_lock.reset();

    const uint64_t pause = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    HeapStats& st = s.stats;
    st.last_pause_ns = pause;
    st.promoted_bytes += g.promoted_bytes;
    st.last_survived_bytes = g.copied_bytes;
    if (kind == CollectionKind::Minor) {
        ++st.minor_collections;
        st.minor_pause_ns_total += pause;
        st.minor_pause_ns_max = std::max(st.minor_pause_ns_max, pause);
    } else {
        ++st.full_collections;
        st.full_pause_ns_total += pause;
        st.full_pause_ns_max = std::max(st.full_pause_ns_max, pause);
    }
}

} // namespace brass::gc::detail
