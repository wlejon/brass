// Weak slots, ephemerons and finalizers: the parts of a collection that run
// once strong reachability is known.
//
// A weak slot is recorded when traced and settled at the end: updated if its
// target survived, cleared if not. An ephemeron's value is traced only once
// its key is known alive; keys that become alive by tracing other values are
// found by iterating to a fixpoint (heap_collect.cpp, trace_to_fixpoint), and
// the entries whose keys never did are cleared. A finalizer's object is
// checked the same way and its callback queued for after the collection.
//
// In a minor collection an old owner of a weak slot or ephemeron that still
// names a young object afterwards keeps its card dirty, exactly as for a
// strong slot, so the next minor collection revisits it.

#include "heap_internal.hpp"

namespace brass::gc::detail {

void weak_visit(Tracer& t, uint64_t* slot, uint64_t cleared) {
    if (t.purpose() == Tracer::Purpose::Verify) {
        t.visit(slot);
        return;
    }
    GcState& g = *static_cast<GcState*>(t.state());
    if (!g.heap.is_reference_tag(*slot)) return;
    g.weak.push_back(WeakEntry{slot, cleared, t.owner(), t.owner_old()});
}

void ephemeron_visit(Tracer& t, uint64_t* key, uint64_t* value, uint64_t cleared_key,
                     uint64_t cleared_value) {
    if (t.purpose() == Tracer::Purpose::Verify) {
        t.visit(key);
        t.visit(value);
        return;
    }
    GcState& g = *static_cast<GcState*>(t.state());
    const uint64_t word = *key;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    if (a == 0 || !g.heap.contains(a) || !g.heap.is_reference_tag(word)) {
        t.visit(value);  // not a reference: the key cannot die
        return;
    }
    const uintptr_t live = live_address(g, a);
    if (!live) {
        g.ephemerons.push_back(EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), t.owner_old()});
        return;
    }
    *key = (word & ~kAddressMask) | live;
    if (g.kind == CollectionKind::Minor && t.owner_old() && g.heap.is_young(live)) {
        *g.s.card_of(t.owner()) = kCardDirty;
    }
    t.visit(value);
}

void settle_weakness(GcState& g) {
    HeapState& s = g.s;
    for (const WeakEntry& w : g.weak) {
        const uint64_t word = *w.slot;
        if (!g.heap.is_reference_tag(word)) continue;
        const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
        if (a == 0 || !g.heap.contains(a)) continue;
        const uintptr_t live = live_address(g, a);
        if (!live) {
            *w.slot = w.cleared;
            continue;
        }
        *w.slot = (word & ~kAddressMask) | live;
        if (g.kind == CollectionKind::Minor && w.owner_old && g.heap.is_young(live)) {
            *s.card_of(w.owner) = kCardDirty;
        }
    }
    g.weak.clear();

    for (const EphemeronEntry& e : g.ephemerons) {
        *e.key = e.cleared_key;
        *e.value = e.cleared_value;
    }
    g.ephemerons.clear();

    std::vector<FinalizerEntry> still_young;
    for (FinalizerEntry f : s.young_finalizers) {
        const uintptr_t live = live_address(g, f.object);
        if (!live) {
            s.pending_finalizers.push_back(f);
            continue;
        }
        f.object = live;
        (g.heap.is_old(live) ? s.old_finalizers : still_young).push_back(f);
    }
    s.young_finalizers.swap(still_young);
    if (g.kind == CollectionKind::Full) {
        std::vector<FinalizerEntry> kept;
        for (const FinalizerEntry& f : s.old_finalizers) {
            if (live_address(g, f.object)) kept.push_back(f);
            else s.pending_finalizers.push_back(f);
        }
        s.old_finalizers.swap(kept);
    }
}

} // namespace brass::gc::detail
