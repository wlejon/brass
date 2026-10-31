// The two collections.
//
// Minor: every young object reachable from the roots and from the dirty cards
// of the old generation is copied, into the other survivor space while it is
// younger than the tenure age and into the old generation after; the young
// spaces are then empty but for the survivors. Its cost is the roots, the
// dirty cards and the survivors, whatever the old generation's size. A large
// one runs on several threads (heap_minor_parallel.cpp), deciding the same.
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
// reclaimed without moving anything. The three run as a marking cycle
// (heap_mark_incremental.cpp: start, slices, finish), back to back here, or
// as separate calls through Heap::start_marking and the rest, with the
// mutator and minor collections running between them; then the young
// generation is not emptied at all but marked where it is at the finish
// (HeapConfig::finish_promote_all evacuates it there instead), and while the
// cycle is open a minor collection shades every old object it reaches or
// promotes (GcState::shade).
//
// Every phase traces through a Tracer with a mode-specific slow path, and
// each settles weak slots, ephemerons and finalizers the same way
// (heap_weak.cpp).

#include "heap_internal.hpp"

#include <brass/gc/fiber.hpp>
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
    // Promoted into the large-object space where a pending sweep has yet to
    // walk: marked, so the walk keeps it (HeapState::note_large_allocated).
    if (promoted) s.note_large_allocated(dest);
    const uintptr_t moved = dest + kHeaderBytes;
    // Promoted during a marking cycle: marked as it is copied, and grey, not
    // black. This collection's scan of it shades its old referents, but
    // through a Minor tracer, and a host's trace function may visit more
    // for a full collection's (a prototype held through an off-heap shape):
    // the marker scans it again as a full collection would.
    if (promoted && g.shade) shade(s, moved);
    if (promoted) {
        g.promoted_bytes += total;
        ++g.promoted_objects;
        s.old_allocated_since_full += total;
        if (!s.in_eden(object)) g.tenured_bytes += total;
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
    } else if (g.shade && !g.heap.is_young(a)) {
        // While a cycle is open the range is the heap, and an old object an
        // old one names is shaded: a dirty card's slot (an unbarriered store
        // into an object allocated in the cycle, or a bulk copy), which this
        // collection cleans and the finish then no longer sees, and a
        // promoted object's. What a root or a young object names is not:
        // the finish visits every root again and scans every live young
        // object, so it reaches what they still hold then. Shading it
        // here would keep alive what they held only for a moment (a
        // WeakRef's target on the kept-objects list until the job's end,
        // say), and with a cycle open most of the time, in every cycle.
        if (t.owner_old()) shade(g.s, a);
        return;
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
    // Large enough to pay for the helpers: the same evacuation on several
    // threads (heap_minor_parallel.cpp).
    if (const unsigned threads = parallel_minor_threads(young); threads > 1) {
        evacuate_parallel(young, clock, threads);
        return;
    }
    const uint64_t began = gc_now_ns();
    // Shading, old references reach minor_visit too: the range is the heap.
    const uintptr_t lo = young.shade ? young.heap.reservation_base() : young.heap.young_base();
    const size_t span = young.shade ? young.heap.reservation_bytes() : young.heap.young_span();
    Tracer t(young.heap, Tracer::Purpose::Minor, lo, lo + span, &minor_visit, &weak_visit, &ephemeron_visit,
             &young);
    visit_roots(s, t);
    clock.lap(kPhaseRoots);
    scan_dirty_cards(t, young);
    clock.lap(kPhaseCards);
    trace_to_fixpoint(t, young);
    clock.lap(kPhaseTrace);
    settle_weakness(young);
    clock.lap(kPhaseWeak);
    note_serial_evacuation(young, gc_now_ns() - began);
}

void run_hooks(GcState& g) {
    HeapState& s = g.s;
    for (size_t i = 0; i < s.hooks.size(); ++i) s.hooks[i].hook(g.heap, g.kind);
}

void clean_all_cards(HeapState& s) {
    const size_t page = os_page_bytes();
    for (size_t p = 0; p < s.card_pages_committed.size(); ++p) {
        if (s.card_pages_committed[p]) std::memset(s.cards + p * page, kCardClean, page);
    }
    std::fill(s.region_bytes.begin(), s.region_bytes.end(), uint8_t{0});
}

// After an in-place finish's marking: the cards of every large object it
// left unmarked, which the sweep frees (every large object is marked or dead
// then: a pending sweep was completed at the cycle's start).
void clean_dead_large_cards(HeapState& s) {
    constexpr size_t kCardsPerPage = kPageBytes / kCardBytes;
    for (const auto& [head, pages] : s.large_objects) {
        const uintptr_t object = s.large_object_at(head);
        if (header_of(object)->gc_bits & kGcLargeMarked) continue;
        std::memset(s.card_of(object - kHeaderBytes), kCardClean, static_cast<size_t>(pages) * kCardsPerPage);
    }
}

// The start bits of the young objects in [lo, hi) after an in-place finish:
// the ones it marked, and no others. A dead one is named by nothing
// reachable, and a lookup by address (containing_object) must not find it,
// since its old referents may be swept. Every marked address is an object's
// start, so the marks are the start bits whether or not eden's were synced
// (HeapState::sync_eden_starts), and the dead objects above the sync point
// are never walked: eden counts as synced to `hi`.
void drop_dead_young_starts(HeapState& s, uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    const size_t first = s.young_bit(lo) >> 6;
    const size_t last = std::min((s.young_bit(hi) + 63) >> 6, s.young_starts.size());
    std::copy(s.young_marks.begin() + static_cast<std::ptrdiff_t>(first),
              s.young_marks.begin() + static_cast<std::ptrdiff_t>(last),
              s.young_starts.begin() + static_cast<std::ptrdiff_t>(first));
    if (s.in_eden(lo)) s.eden_synced = std::max(s.eden_synced, hi);
}

void poison_evacuated(HeapState& s, const GcState& g) {
    if (!s.poison) return;
    std::memset(reinterpret_cast<void*>(s.eden_lo), kPoisonByte, g.eden_top - s.eden_lo);
    std::memset(reinterpret_cast<void*>(s.survivor_lo[g.from]), kPoisonByte, g.from_top - s.survivor_lo[g.from]);
}

void minor(GcState& g) {
    HeapState& s = g.s;
    PhaseClock clock(g);
    // While the adaptive eden is shrunk, survivors age through the survivor
    // space even after an overflow: how many of them live through a second
    // collection is what tells a build-up from a working set
    // (HeapState::adapt_eden_limit), and promote_all would tenure them unseen.
    const bool shrunk = s.eden_limit < s.eden_hi;
    g.tenure_age = s.promote_all && !shrunk ? 1u : s.config.tenure_age;
    g.shade = s.mark_phase == MarkPhase::Marking;
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
    // A parallel collection's buffers left unused ends (survivor_slack):
    // the space held that much less.
    const uint64_t aged = g.from_top - s.survivor_lo[g.from] - s.survivor_slack[g.from];
    s.survivor_slack[g.from] = 0;
    const uint64_t young_collected = (g.eden_top - s.eden_lo) + aged;
    if (young_collected > 0) {
        s.minor_survival = static_cast<double>(g.copied_bytes) / static_cast<double>(young_collected);
    }
    MinorSample sample;
    sample.eden_collected = g.eden_top - s.eden_lo;
    sample.copied = g.copied_bytes;
    sample.aged = aged;
    sample.tenured = g.tenured_bytes;
    sample.tenure_age = g.tenure_age;
    sample.overflow = g.survivor_overflow;
    sample.at_whole_eden = g.eden_limit == s.eden_hi - s.eden_lo;
    s.adapt_eden_limit(sample);
    s.reset_eden();
    s.minor_requested = false;
    if (s.old_allocated_since_full > s.growth_limit()) s.full_requested = true;
    // Promote everything next time while survival stays high: after an
    // overflow, and for as long as half or more of the young data survives
    // each promote-all collection. Short-lived data that merely happens to
    // be live at a collection (a batch being built) is a small fraction of
    // a full eden, so it leaves this mode at once rather than being tenured.
    const uint64_t collected = young_collected;
    s.promote_all = g.survivor_overflow || (s.promote_all && g.copied_bytes * 2 >= collected);
    clock.lap(kPhaseSweep);
}

} // namespace

// A full collection's young generation, emptied into the old one: at its
// start when it runs back to back, at its finish when a mutator ran between
// (heap_mark_incremental.cpp).
void full_evacuate(GcState& g, PhaseClock& clock, bool shade) {
    HeapState& s = g.s;
    GcState young(s, g.heap, CollectionKind::Minor);
    young.from = g.from;
    young.to = g.to;
    young.eden_top = g.eden_top;
    young.from_top = g.from_top;
    young.tenure_age = 1;
    young.shade = shade;
    young.learns_cost = false;  // promoting everything, often shading: not a minor collection's cost
    evacuate(young, clock);
    g.copied_bytes += young.copied_bytes;
    g.copied_objects += young.copied_objects;
    g.promoted_bytes += young.promoted_bytes;
    g.promoted_objects += young.promoted_objects;
    g.dirty_cards += young.dirty_cards;
    g.minor_threads = young.minor_threads;
}

// Phase 3: weakness, hooks, reclamation.
void full_reclaim(GcState& g, PhaseClock& clock) {
    HeapState& s = g.s;
    settle_weakness(g);
    clock.lap(kPhaseWeak);
    run_hooks(g);
    clock.lap(kPhaseHooks);

    // Liveness is settled: the old generation's memory is freed by the lazy
    // sweep from here (all of it now with HeapConfig::lazy_sweep off).
    if (s.young_in_place) {
        // The young generation stays (a cycle's in-place finish), and so do
        // the cards that remember it. A dirty card of a dead mature object is
        // harmless: the card scan reads an unswept block's starts through its
        // marks, and a swept one's dead starts are gone. A dead large object
        // has cards of its own, which the scan would read it through, so
        // they are cleaned; and a dead young object stops being an object,
        // so that nothing finds it again (a conservative root, verification).
        clean_dead_large_cards(s);
        s.begin_sweep(g.marked_bytes);
        drop_dead_young_starts(s, s.eden_lo, g.eden_top);
        drop_dead_young_starts(s, s.survivor_lo[g.from], g.from_top);
    } else {
        // A dead object's card may be dirty; cleaning every card here means
        // no card names a dead object afterwards, since nothing stores into
        // one.
        s.begin_sweep(g.marked_bytes);
        clean_all_cards(s);
        poison_evacuated(s, g);
        std::fill(s.young_starts.begin(), s.young_starts.end(), 0);
        s.survivor_top[0] = s.survivor_lo[0];
        s.survivor_top[1] = s.survivor_lo[1];
        s.survivor_slack[0] = 0;
        s.survivor_slack[1] = 0;
        s.eden_bytes_retired += g.eden_top - s.eden_lo;
    }
    // The adaptive eden goes back to the whole of it when most of what the
    // old generation gained since the last full collection is dead: the
    // program is not building data, so a shrunk eden may be promoting a
    // working set that the whole eden would let die young (reset_eden_limit).
    // While the live old data grows by half the growth or more, the build-up
    // that shrank eden goes on, and the whole eden would copy it in one go.
    // A cycle the program ran inside marked everything it allocated or
    // promoted meanwhile, live or not (grey from the start): that growth says
    // nothing about what lives, and a cycle that promotes fast (a shrunk eden
    // promotes several times what the whole one would) would always read as
    // a build-up, keep eden shrunk and so promote as fast through the next
    // one, each cycle longer than the last (bigheap 1 GB went 1.1, 1.8, 2.5
    // GB marked, its live data 0.8). So only the growth before the cycle's
    // start is judged, against what is marked but for the growth since
    // (HeapConfig::restore_judges_cycle_start). Back to back the two agree.
    uint64_t marked = g.marked_bytes;
    uint64_t growth = s.old_allocated_since_full;
    if (s.config.restore_judges_cycle_start && s.cycle_start_growth <= growth) {
        const uint64_t during = growth - s.cycle_start_growth;
        marked = marked > during ? marked - during : 0;
        growth = s.cycle_start_growth;
    }
    s.cycle_start_growth = 0;
    const uint64_t live_gained = marked > s.stats.old_live_bytes ? marked - s.stats.old_live_bytes : 0;
    // promote_all is otherwise left as the last minor collection set it: a
    // full collection says nothing about how much of the young data
    // survives, and a program still building long-lived data would refill
    // the survivor space only to promote it all again at the next overflow.
    // When the growth was dead, though, the program has stopped building,
    // and the minor collection at the whole eden must age its survivors to
    // measure whether they are kept, not promote them all.
    if (live_gained * 2 < growth) {
        s.reset_eden_limit();
        s.promote_all = false;
    }
    // Eden still holds the young objects an in-place finish kept.
    if (s.young_in_place) s.update_fast_limit();
    else s.reset_eden();

    const double grown = static_cast<double>(g.marked_bytes) * s.config.growth_factor;
    s.full_threshold = std::max<uint64_t>(s.config.min_full_threshold_bytes, static_cast<uint64_t>(grown));
    s.old_allocated_since_full = 0;
    s.stats.old_live_bytes = g.marked_bytes;
    s.full_requested = false;
    s.minor_requested = false;
    clock.lap(kPhaseSweep);
}

void visit_roots(HeapState& s, Tracer& t) {
    // Gathered before any root is visited: append_active_coro_roots reads
    // each registered frame's is_done, and once a frame has been evacuated
    // its old copy holds the forwarding address there instead.
    std::vector<uintptr_t*> slots;
    // The recorded runs of native frames cover the stack from the innermost
    // one that starts at a generated frame to the base; the walk from the
    // collection's own frame stops there rather than visit those frames
    // twice (a deep async recursion records a run per resumed frame).
    const uintptr_t covered_from = brass_native_frames_cover_start();
    if (s.mutator_fp != 0) {
        brass_append_generated_frame_roots(s.mutator_fp, s.mutator_ip, slots, covered_from);
    } else if (s.config.walk_stack_on_host_collection) {
        brass_append_stack_roots_from_here(slots, covered_from);
    }
    brass_append_native_frame_roots(slots);
    // The thread's other stacks (gc::Fiber): each suspended fiber, and each
    // stack waiting on a fiber it resumed, from where it switched away.
    brass_append_switched_stack_roots(slots);
    // The registry's slots hold raw frame addresses, references on any heap.
    std::vector<uintptr_t*> frame_slots;
    if (s.coro_frames) runtime::append_active_coro_roots(*s.coro_frames, frame_slots);

    t.set_owner(0, false);
    for (size_t i = 0; i < s.roots.size(); ++i) t.visit(s.roots[i]);
    for (size_t i = 0; i < s.sources.size(); ++i) {
        RootSourceEntry& source = s.sources[i];
        if (source.fn) source.fn(t, source.context);
        else source.callable(t);
        t.set_owner(0, false);
    }
    for (uintptr_t* slot : slots) t.visit_ref(slot);
    for (uintptr_t* slot : frame_slots) t.visit_address(slot);
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
        // A cycle's in-place finish moved nothing and marked what it reached.
        if (s.young_in_place) return s.young_marked(a) ? a : 0;
        if (!g.in_from(a)) return a;
        const ObjectHeader* h = header_of(a);
        if (!h->forwarded()) return 0;
        a = *reinterpret_cast<const uintptr_t*>(a);
        // A full collection promoted it (at its start, or at a cycle's
        // finish); whether it lives is the mark of its old copy.
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

void brass::gc::Tracer::visit_address(uintptr_t* slot) noexcept {
    if (*slot == 0) return;
    uint64_t word = (static_cast<uint64_t>(*slot) & kAddressMask) | heap().raw_address_tag();
    visit(&word);
    *slot = static_cast<uintptr_t>(word & kAddressMask);
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
                g.s.mark_card(e.owner, kCardDirty);
            }
            t.visit(e.value);
            progress = true;
        }
        if (!progress) return;
    }
}

void begin_collection(GcState& g) {
    HeapState& s = g.s;
    s.collecting = true;
    s.collecting_kind = g.kind;
    g.from = s.from_survivor;
    g.to = 1 - g.from;
    g.eden_top = Collector::buffer(g.heap).top;
    g.from_top = s.survivor_top[g.from];
    g.eden_used = g.eden_top - s.eden_lo;
    g.eden_limit = s.eden_limit - s.eden_lo;
    s.gc_eden_top = g.eden_top;
    s.gc_from_top = g.from_top;
}

void record_collection(const GcState& g, uint64_t pause_ns, uint64_t longest_ns) {
    HeapState& s = g.s;
    HeapStats& st = s.stats;
    st.last_pause_ns = longest_ns;
    st.promoted_bytes += g.promoted_bytes;
    st.last_survived_bytes = g.copied_bytes;
    if (g.minor_threads > 1) ++st.parallel_evacuations;
    if (g.kind == CollectionKind::Minor) {
        ++st.minor_collections;
        st.minor_pause_ns_total += pause_ns;
        st.minor_pause_ns_max = std::max(st.minor_pause_ns_max, longest_ns);
    } else {
        ++st.full_collections;
        st.full_pause_ns_total += pause_ns;
        st.full_pause_ns_max = std::max(st.full_pause_ns_max, longest_ns);
    }
    if (s.log) log_collection(g, pause_ns);
    s.trigger = GcTrigger::Explicit;
}

void Collector::collect(Heap& heap, CollectionKind kind) {
    HeapState& s = state(heap);
    // A full collection asked for while a cycle is open finishes the cycle,
    // which is a full collection. What the cycle marked includes what died
    // since it started (floating garbage), so a caller that wants that gone
    // too (collect(Full), an exhausted reservation, stress) then gets a whole
    // collection; old-generation growth, which only asks for room, does not.
    if (kind == CollectionKind::Full && s.mark_phase == MarkPhase::Marking) {
        const GcTrigger why = s.trigger;
        finish_open_cycle(heap);
        if (why == GcTrigger::OldGrowth) return;
        s.trigger = why;
    }
    if (kind == CollectionKind::Full) {
        collect_full(heap);
        return;
    }
    const uint64_t t0 = gc_now_ns();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    if (s.verify) verify_heap(heap, "before a collection");

    GcState g(s, heap, kind);
    begin_collection(g);
    minor(g);
    s.collecting = false;
    s.old_critical = s.old_in_reserve_swept();

    if (s.verify) verify_heap(heap, "after a collection");
    coro_lock.reset();

    const uint64_t pause = gc_now_ns() - t0;
    record_collection(g, pause, pause);
}

} // namespace brass::gc::detail
