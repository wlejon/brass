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
//
// While a marking cycle is open a minor collection also meets old targets;
// an old owner's are handed to the cycle (cycle_note_weak and
// cycle_note_ephemeron), which settles them at its finish with the ones its
// marker met.

#include "heap_internal.hpp"

namespace brass::gc::detail {

void weak_visit(Tracer& t, uint64_t* slot, uint64_t cleared) {
    if (t.purpose() == Tracer::Purpose::Verify) {
        t.visit(slot);
        return;
    }
    GcState& g = *static_cast<GcState*>(t.state());
    if (!g.heap.is_reference_tag(*slot)) return;
    // A minor collection during a marking cycle meets old targets too (its
    // range is the heap). It cannot decide them; the cycle does, by its mark
    // bits at the finish. An old owner's slot goes to the cycle, since its
    // owner may be one the marker scanned before the slot was written. A
    // young owner's or a root's is met again (by a later minor collection,
    // the finish's evacuation or its root visit), so it is left alone.
    if (g.shade && !g.heap.is_young(static_cast<uintptr_t>(*slot & kAddressMask))) {
        if (t.owner_old()) cycle_note_weak(g.s, WeakEntry{slot, cleared, t.owner(), true});
        return;
    }
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
    if (g.shade && !g.heap.is_young(a)) {
        // An old key during a marking cycle (see weak_visit): marked, it is
        // alive. Unmarked, an old owner's entry is the cycle's to decide; its
        // value is traced now only if young, since this collection must move
        // it either way. A young owner's or a root's entry is met again, so
        // its value is traced as a minor collection always has (old keys
        // count as alive), shading it: kept for this cycle at worst.
        if (!is_marked(g.s, a) && t.owner_old()) {
            cycle_note_ephemeron(g.s, EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), true});
            if (g.heap.holds_young(*value)) t.visit(value);
            return;
        }
        t.visit(value);
        return;
    }
    const uintptr_t live = live_address(g, a);
    if (!live) {
        g.ephemerons.push_back(EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), t.owner_old()});
        return;
    }
    *key = (word & ~kAddressMask) | live;
    if (g.kind == CollectionKind::Minor && t.owner_old() && g.heap.is_young(live)) {
        g.s.mark_card(t.owner(), kCardDirty);
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
            s.mark_card(w.owner, kCardDirty);
        }
    }
    g.weak.clear();

    for (const EphemeronEntry& e : g.ephemerons) {
        *e.key = e.cleared_key;
        *e.value = e.cleared_value;
    }
    g.ephemerons.clear();

    // Finalizers, compacted in place (no allocation once the lists have their
    // working-set capacity). A minor collection looks only at the young list:
    // its cost is the finalizable objects allocated since the last one, not
    // every finalizer registered.
    std::vector<FinalizerEntry>& young = s.young_finalizers;
    size_t still_young = 0;
    for (size_t i = 0; i < young.size(); ++i) {
        FinalizerEntry f = young[i];
        const uintptr_t live = live_address(g, f.object);
        if (!live) {
            s.pending_finalizers.push_back(f);
            continue;
        }
        f.object = live;
        if (g.heap.is_old(live)) s.old_finalizers.push_back(f);
        else young[still_young++] = f;
    }
    young.resize(still_young);
    if (g.kind == CollectionKind::Full) {
        std::vector<FinalizerEntry>& old = s.old_finalizers;
        size_t kept = 0;
        for (size_t i = 0; i < old.size(); ++i) {
            if (live_address(g, old[i].object)) old[kept++] = old[i];
            else s.pending_finalizers.push_back(old[i]);
        }
        old.resize(kept);
    }
}

} // namespace brass::gc::detail
