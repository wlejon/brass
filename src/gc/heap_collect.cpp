// The two collections.
//
// Minor: every young object reachable from the roots and from the dirty cards
// of the old generation is copied, into the other survivor space while it is
// younger than the tenure age and into the old generation after; the young
// spaces are then empty but for the survivors. Its cost is the roots, the
// dirty cards and the survivors, whatever the old generation's size.
//
// Full, in three phases. First the young generation is emptied: an
// evacuation exactly like a minor collection's, promoting every survivor
// (whatever a dirty card of a dead old object keeps alive is promoted too,
// and swept below with it). Then every reachable old object is marked in
// place (heap_mark.cpp: mature objects in the blocks' side bitmaps and their
// lines, large objects in their header), on several threads when the old
// generation is large, since nothing moves or is allocated while it runs.
// Last, weak slots, ephemerons and finalizers are settled, the hooks run, and
// the old generation is swept: unmarked objects' lines and pages are
// reclaimed without moving anything.
//
// Every phase traces through a Tracer with a mode-specific slow path, and
// each settles weak slots, ephemerons and finalizers the same way
// (heap_weak.cpp).

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

// ---- the young generation's evacuation (minor, and a full's first phase) ----

uintptr_t evacuate_young(GcState& g, uintptr_t object) {
    HeapState& s = g.s;
    ObjectHeader* h = header_of(object);
    const size_t total = kHeaderBytes + h->size;
    const unsigned age = std::min<unsigned>(h->age() + 1u, 7u);
    uintptr_t dest = 0;
    bool promoted = false;
    const bool young_enough = age < g.tenure_age;
    if (young_enough && s.survivor_top[g.to] + total <= s.survivor_lo[g.to] + s.survivor_bytes) {
        dest = s.survivor_top[g.to];
        s.survivor_top[g.to] += total;
    } else {
        if (young_enough) g.survivor_overflow = true;
        dest = s.old_allocate(total);
        if (dest == 0) gc_fatal("the old generation's reservation is exhausted while promoting a young object");
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
        ++g.promoted_objects;
        s.old_allocated_since_full += total;
    } else {
        s.set_young_start(moved);
    }
    g.copied_bytes += total;
    ++g.copied_objects;
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
    if (t.owner_old() && g.heap.is_young(now)) g.s.remember_slot(t.owner(), slot);
}

void drain(Tracer& t, GcState& g) {
    while (!g.gray.empty()) {
        const uintptr_t object = g.gray.back();
        g.gray.pop_back();
        scan_object(t, object, g.heap.is_old(object));
    }
}

// Copies every young object the roots and the dirty cards reach, and settles
// the weak slots and ephemerons that name young objects. `young` is a Minor
// state; its phases are timed into `clock`'s collection.
void evacuate(GcState& young, PhaseClock& clock) {
    HeapState& s = young.s;
    Tracer t(young.heap, Tracer::Purpose::Minor, young.heap.young_base(),
             young.heap.young_base() + young.heap.young_span(), &minor_visit, &weak_visit, &ephemeron_visit,
             &young);
    visit_roots(s, t);
    clock.lap(kPhaseRoots);
    scan_dirty_cards(t, young);
    clock.lap(kPhaseCards);
    trace_to_fixpoint(t, young);
    clock.lap(kPhaseTrace);
    settle_weakness(young);
    clock.lap(kPhaseWeak);
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

void poison_evacuated(HeapState& s, const GcState& g) {
    if (!s.poison) return;
    std::memset(reinterpret_cast<void*>(s.eden_lo), kPoisonByte, g.eden_top - s.eden_lo);
    std::memset(reinterpret_cast<void*>(s.survivor_lo[g.from]), kPoisonByte, g.from_top - s.survivor_lo[g.from]);
}

void minor(GcState& g) {
    HeapState& s = g.s;
    PhaseClock clock(g);
    g.tenure_age = s.promote_all ? 1u : s.config.tenure_age;
    evacuate(g, clock);
    run_hooks(g);
    clock.lap(kPhaseHooks);

    // Reclaim the evacuated spaces.
    poison_evacuated(s, g);
    s.clear_young_starts(s.eden_lo, g.eden_top);
    s.clear_young_starts(s.survivor_lo[g.from], g.from_top);
    s.survivor_top[g.from] = s.survivor_lo[g.from];
    s.from_survivor = g.to;
    s.eden_bytes_retired += g.eden_top - s.eden_lo;
    s.reset_eden();
    s.minor_requested = false;
    if (s.old_allocated_since_full > s.full_threshold) s.full_requested = true;
    // Promote everything next time while survival stays high: after an
    // overflow, and for as long as half or more of the young data survives
    // each promote-all collection. Short-lived data that merely happens to
    // be live at a collection (a batch being built) is a small fraction of
    // a full eden, so it leaves this mode at once rather than being tenured.
    const uint64_t collected = (g.eden_top - s.eden_lo) + (g.from_top - s.survivor_lo[g.from]);
    s.promote_all = g.survivor_overflow || (s.promote_all && g.copied_bytes * 2 >= collected);
    clock.lap(kPhaseSweep);
}

void full(GcState& g) {
    HeapState& s = g.s;
    PhaseClock clock(g);

    // Phase 1: the young generation, emptied into the old one.
    GcState young(s, g.heap, CollectionKind::Minor);
    young.from = g.from;
    young.to = g.to;
    young.eden_top = g.eden_top;
    young.from_top = g.from_top;
    young.tenure_age = 1;
    evacuate(young, clock);
    g.copied_bytes += young.copied_bytes;
    g.copied_objects += young.copied_objects;
    g.promoted_bytes += young.promoted_bytes;
    g.promoted_objects += young.promoted_objects;
    g.dirty_cards += young.dirty_cards;

    // Phase 2: the old generation, marked in place.
    mark_old_generation(g, clock);

    // Phase 3: weakness, hooks, reclamation.
    settle_weakness(g);
    clock.lap(kPhaseWeak);
    run_hooks(g);
    clock.lap(kPhaseHooks);

    s.sweep_old();
    clean_all_cards(s);
    poison_evacuated(s, g);
    std::fill(s.young_starts.begin(), s.young_starts.end(), 0);
    s.survivor_top[0] = s.survivor_lo[0];
    s.survivor_top[1] = s.survivor_lo[1];
    s.eden_bytes_retired += g.eden_top - s.eden_lo;
    s.reset_eden();
    // promote_all is left as the last minor collection set it: a full
    // collection says nothing about how much of the young data survives, and
    // a program still building long-lived data would otherwise refill the
    // survivor space only to promote it all again at the next overflow.

    const double grown = static_cast<double>(g.marked_bytes) * s.config.growth_factor;
    s.full_threshold = std::max<uint64_t>(s.config.min_full_threshold_bytes, static_cast<uint64_t>(grown));
    s.old_allocated_since_full = 0;
    s.stats.old_live_bytes = g.marked_bytes;
    s.full_requested = false;
    s.minor_requested = false;
    clock.lap(kPhaseSweep);
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
        if (!h->forwarded()) return 0;
        a = *reinterpret_cast<const uintptr_t*>(a);
        // A full collection promoted it first; whether it lives is the mark
        // of its old copy.
        if (g.kind == CollectionKind::Minor || g.heap.is_young(a)) return a;
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
    g.eden_top = heap.alloc_->top;
    g.from_top = s.survivor_top[g.from];
    g.eden_used = g.eden_top - s.eden_lo;
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
    if (s.log) log_collection(g, pause);
    s.trigger = GcTrigger::Explicit;
}

} // namespace brass::gc::detail
