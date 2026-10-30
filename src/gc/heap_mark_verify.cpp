// BRASS_GC_VERIFY's check of a marking cycle: once its marking is done, a
// stop-the-world mark from the roots, into a side set, must reach no object
// the cycle left unmarked. The cycle may mark more (what died while it ran);
// it must not mark less, or the sweep would free a live object. A store the
// barrier never saw, a minor collection that did not shade, an old
// allocation that was not grey: each shows up here as the object it lost,
// before anything is swept.
//
// The fresh mark sees what the cycle's marker sees: a Full tracer over the
// old generation (the young one is empty by now: the finish evacuated it),
// weak slots holding nothing, and an ephemeron's value reached only once its
// key is.

#include "heap_mark.hpp"

#include <cstdio>
#include <unordered_set>
#include <utility>
#include <vector>

namespace brass::gc::detail {

namespace {

struct FreshMark {
    HeapState& s;
    std::unordered_set<uintptr_t> reached;
    std::vector<uintptr_t> stack;
    std::vector<std::pair<uint64_t*, uint64_t*>> ephemerons;  // keys not (yet) reached
};

FreshMark& fresh(Tracer& t) noexcept { return *static_cast<FreshMark*>(t.state()); }

bool old_reference(const HeapState& s, uint64_t word, uintptr_t& a) noexcept {
    if (!s.heap.is_reference_tag(word)) return false;
    a = static_cast<uintptr_t>(word & kAddressMask);
    return s.heap.is_old(a);
}

void fresh_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    (void)slot;
    FreshMark& m = fresh(t);
    uintptr_t a = 0;
    if (!old_reference(m.s, word, a)) return;
    if (m.reached.insert(a).second) m.stack.push_back(a);
}

void fresh_weak(Tracer&, uint64_t*, uint64_t) {}

void fresh_ephemeron(Tracer& t, uint64_t* key, uint64_t* value, uint64_t, uint64_t) {
    FreshMark& m = fresh(t);
    uintptr_t a = 0;
    if (!old_reference(m.s, *key, a) || m.reached.count(a) != 0) {
        t.visit(value);
        return;
    }
    m.ephemerons.emplace_back(key, value);
}

void drain(FreshMark& m, Tracer& t) {
    while (!m.stack.empty()) {
        const uintptr_t o = m.stack.back();
        m.stack.pop_back();
        scan_object(t, o, true);
    }
}

[[noreturn]] void lost(uintptr_t object) {
    const ObjectHeader* h = header_of(object);
    char msg[320];
    std::snprintf(msg, sizeof msg,
                  "heap verification at a marking cycle's finish: object %p (layout %u '%s', %u bytes) is "
                  "reachable from the roots but the cycle left it unmarked (a store no barrier shaded, or "
                  "an object a minor collection or allocation did not shade)",
                  reinterpret_cast<void*>(object), static_cast<unsigned>(h->layout),
                  layout_descriptor(h->layout).name, static_cast<unsigned>(h->size));
    gc_fatal(msg);
}

} // namespace

void verify_cycle_marks(Heap& heap) {
    HeapState& s = Collector::state(heap);
    FreshMark m{s, {}, {}, {}};
    Tracer t(heap, Tracer::Purpose::Full, s.mature_lo, s.large_lo + s.large_reserve, &fresh_visit, &fresh_weak,
             &fresh_ephemeron, &m);
    visit_roots(s, t);
    for (;;) {
        drain(m, t);
        bool progress = false;
        std::vector<std::pair<uint64_t*, uint64_t*>> pending;
        pending.swap(m.ephemerons);
        for (const auto& [key, value] : pending) {
            uintptr_t a = 0;
            if (old_reference(s, *key, a) && m.reached.count(a) == 0) {
                m.ephemerons.emplace_back(key, value);
                continue;
            }
            t.visit(value);
            progress = true;
        }
        if (!progress) break;
    }
    for (uintptr_t a : m.reached) {
        if (!is_marked(s, a)) lost(a);
    }
}

} // namespace brass::gc::detail
