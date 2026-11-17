// PRETENURING (docs/gc_contract.md 2.14): a host's allocation sites that a
// profile found long-lived allocate old (Heap::try_allocate_pretenured), and
// the minor collections check the claim as the program runs. A profile may
// be wrong for this run: a site whose objects die young, allocated old, is
// garbage only full collections take back, and its dirty cards keep alive
// and promote whatever the dead objects held, every minor collection. So
// every kProbeEvery-th object of a site is made young instead, a probe, and
// each minor collection judges the probes it saw (judge_probes): promoted, it
// lived as claimed; copied into the survivor space, it is followed there;
// left behind, it died young. Once more than half a site's probes died
// young, the site is refused: its objects are made young for the rest of the
// run, as with no profile. The probes made before the first minor collection
// (when every object is young) are judged by it, so a claim wrong from the
// start is refused before any of its objects is allocated old.

#include "heap_collection.hpp"

#include <brass/core/atomic_ref.hpp>

namespace brass::gc {

using detail::Collector;
using detail::HeapState;

namespace {

// The probes a site's verdict needs; the probes a site makes between two
// minor collections, at most (the first kProbesPerMinor * kProbeEvery of its
// objects: a verdict costs a collection a header read per probe, and a busy
// site's first eden would otherwise give it thousands); the probes followed
// at once (a site asking past it makes its probe's object old, unfollowed);
// and the count of probes at which a site's tallies are halved, so its
// verdict follows what the program does now: a site turned short-lived is
// refused once its dying probes outnumber its earlier ones weighed, within
// about kHalveAt / kProbesPerMinor (four) minor collections of the change.
constexpr uint32_t kMinProbes = 16;
constexpr uint32_t kProbesPerMinor = 64;
constexpr size_t kMaxProbes = size_t{1} << 16;
constexpr uint32_t kHalveAt = 256;

constexpr auto kRelaxed = std::memory_order_relaxed;

}  // namespace

uintptr_t Heap::try_allocate_pretenured(size_t bytes, LayoutId layout, PretenureSite& site) noexcept {
    HeapState& s = *s_;
    if (AtomicRef<uint32_t>(site.refused).load(kRelaxed) != 0) return try_allocate_young(bytes, layout);
    const uint32_t tick = AtomicRef<uint32_t>(site.tick).fetch_add(1, kRelaxed);
    if (tick % PretenureSite::kProbeEvery == 0 && tick < PretenureSite::kProbeEvery * kProbesPerMinor &&
        s.probes.size() < kMaxProbes) {
        const uintptr_t probe = try_allocate_young(bytes, layout);
        if (probe != 0) {
            AtomicRef<uint32_t>(site.made).fetch_add(1, kRelaxed);
            s.probes.push_back({probe, &site});
        }
        return probe;
    }
    const size_t payload = payload_bytes_for(bytes);
    const size_t total = payload + kHeaderBytes;
    // Young, too, before the first minor collection: the program's start is
    // the first eden, with nothing old yet, and a young generation that holds
    // all of it is traced in one pass with no cards; objects allocated old
    // there are filled with young ones, and the first (and largest) minor
    // collection would rescan their cards besides.
    if (s.stats.minor_collections == 0 || s.collecting || s.full_requested || s.minor_requested || s.old_critical ||
        total > kMaxMediumObjectBytes || s.old_allocated_since_full + total > s.growth_limit()) {
        return try_allocate_young(bytes, layout);
    }
    const uintptr_t header = s.mature_allocate(total);
    if (header == 0) return try_allocate_young(bytes, layout);
    auto* h = reinterpret_cast<ObjectHeader*>(header);
    h->size = static_cast<uint32_t>(payload);
    h->layout = layout;
    h->gc_bits = 0;
    h->host_bits = 0;
    zero_payload(header + kHeaderBytes, payload);
    s.old_allocated_since_full += total;
    s.old_direct_bytes += total;
    const uintptr_t object = header + kHeaderBytes;
    s.pretenured.push_back(object);
    if (s.mark_phase == detail::MarkPhase::Marking) Collector::allocated_old(*this, object);
    else dirty_card(object);
    return object;
}

namespace detail {

uint32_t judge_probes(HeapState& s) noexcept {
    size_t kept = 0;
    uint32_t refusals = 0;
    for (const HeapState::Probe& p : s.probes) {
        // The site's next probes count from here.
        AtomicRef<uint32_t>(p.site->tick).store(0, kRelaxed);
        const uint8_t bits = header_of(p.object)->gc_bits;
        if ((bits & kGcForwarded) != 0) {
            // Copied: into the survivor space it is followed; promoted, it
            // lived as the site was claimed to.
            const uintptr_t to = *reinterpret_cast<const uintptr_t*>(p.object);
            if (s.heap.is_young(to)) s.probes[kept++] = {to, p.site};
            continue;
        }
        // Kept in place by a stack word: alive, no longer followed.
        if ((bits & (kGcPinned | kGcResident)) != 0) continue;
        PretenureSite& site = *p.site;
        AtomicRef<uint32_t> made(site.made);
        AtomicRef<uint32_t> dead(site.dead);
        dead.fetch_add(1, kRelaxed);
        // Halved here, both at once, and not as probes are made: a site
        // making more than kHalveAt probes between two collections would
        // otherwise have its pending probes' deaths weighed against a count
        // halved without them.
        if (made.load(kRelaxed) >= kHalveAt) {
            made.store(made.load(kRelaxed) / 2, kRelaxed);
            dead.store(dead.load(kRelaxed) / 2, kRelaxed);
        }
        const uint32_t m = made.load(kRelaxed);
        const uint32_t d = dead.load(kRelaxed);
        AtomicRef<uint32_t> refused(site.refused);
        if (m >= kMinProbes && d * 2 > m && refused.load(kRelaxed) == 0) {
            refused.store(1, kRelaxed);
            ++refusals;
        }
    }
    s.probes.resize(kept);
    return refusals;
}

}  // namespace detail

}  // namespace brass::gc
