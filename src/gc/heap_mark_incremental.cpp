// The old generation's marking cycle: start, slices, finish, and what keeps
// it sound while the mutator runs between them.
//
// A full collection is one cycle run back to back: start (the young
// generation emptied into the old one, the roots made grey), mark slices
// until the grey work is gone, finish (weakness, hooks, sweep). The cycle's
// state lives on the heap between its steps (MarkCycle: the collection's
// GcState, the marker's workers with their mark stacks and the shared pool),
// so the steps can also be separate calls, Heap::start_marking, mark_slice
// and finish_marking, with the slices given a time budget each and the
// mutator running in between.
//
// With a mutator in between the cycle is incremental-update (an insertion
// barrier): no marked old object may name an unmarked old one except through
// a slot some later step rescans. What keeps that so:
//   - start marks only from the roots (the young generation stays where it
//     is; the marker traces old objects only, and young ones are the minor
//     collections');
//   - the write barrier's value range covers the heap while the cycle is
//     open, so an old object's store of an old reference shades the value
//     (Heap::shade_stored); a young value dirties the card as always, and
//     bulk writes dirty cards (Heap::remember);
//   - a minor collection while the cycle is open traces with the heap as its
//     range and shades every old object an old one names (a dirty card it
//     cleans, an object it promotes) and every object it promotes
//     (GcState::shade); not what a root or a live young object names, which
//     the finish reaches again, so that what they held only for a moment (a
//     WeakRef's kept target) is not kept by every cycle;
//   - an old object the mutator allocates is grey and its card dirty
//     (allocated_old);
//   - finish runs a shading evacuation of the whole young generation, visits
//     every root again as the marker sees them (what a minor's root sources
//     skip), and drains what that left grey before anything is decided.
// The roots are not barriered, which is why finish visits them again.
//
// Shading pushes onto the first worker's stack on the heap's thread, from
// the barrier or a minor collection, and never while the marker's threads
// drain (MarkCycle::draining says when they do): slices and minor
// collections run on the heap's thread between mutator turns, and the
// barrier is the mutator's. So a shade's claim races no marker thread, and
// is a plain store: a slice's helpers are woken through a lock and waited
// for through a release/acquire count (heap_mark_threads.hpp), which order
// it against their claims. An atomic claim's locked instruction was three
// quarters of shade's time (every object a minor collection promotes, every
// value the barrier stores) with a cycle open on a 200 MB heap.
//
// A slice drains the grey work on the marker's threads when its budget is
// large enough to pay for starting them (kParallelSliceNs), on the calling
// thread otherwise; either way every worker's stack and the pool are what it
// drains, so a slice resumes whatever the last one left wherever it left it.
// The ephemeron fixpoint runs in the slices: when the grey work is gone, the
// entries whose keys have since been marked trace their values, and the
// slice goes on while that finds more.

#include "heap_mark.hpp"

#include <brass/runtime/coroutine.hpp>

#include <algorithm>
#include <cstdio>
#include <optional>

namespace brass::gc::detail {

namespace {

// The least slice budget run on several threads: below it, waking the
// parked helpers and waiting for them (heap_mark_threads.hpp; tens of
// microseconds, about 0.1 ms when each drain creates its threads) is too
// much of the slice.
constexpr uint64_t kParallelSliceNs = 200'000;
// StressMode::Incremental: slices a cycle runs before it is finished.
constexpr uint64_t kStressSlicesPerCycle = 8;

} // namespace

struct MarkCycle {
    GcState g;  // the full collection the cycle is
    MarkShared sh;
    MarkWorkers workers;
    bool grey_empty = false;  // the last slice drained everything, and nothing was shaded since
    bool draining = false;    // a slice is draining (on the marker's threads, perhaps)
    // The pauses so far (the public calls'), for the statistics.
    uint64_t pause_ns = 0;
    uint64_t longest_ns = 0;
    uint64_t stress_slices = 0;
    // Why it started: the minor collections inside it log their own.
    GcTrigger trigger;

    MarkCycle(HeapState& s, Heap& heap) : g(s, heap, CollectionKind::Full), sh(s, heap), trigger(s.trigger) {}
};

void MarkCycleDeleter::operator()(MarkCycle* cycle) const noexcept { delete cycle; }

namespace {

MarkCycle& open_cycle(HeapState& s, const char* what) {
    if (!s.cycle || s.cycle->workers.empty()) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s with no marking cycle open", what);
        gc_fatal(msg);
    }
    return *s.cycle;
}

// Sets `bit` in `word`; true when this call set it. Plain stores: see
// "Shading" above.
bool claim_bit(uint64_t& word, uint64_t bit) noexcept {
    if (word & bit) return false;
    word |= bit;
    return true;
}

bool claim_large_bit(uint8_t& bits) noexcept {
    if (bits & kGcLargeMarked) return false;
    bits = static_cast<uint8_t>(bits | kGcLargeMarked);
    return true;
}

// After the grey work is gone: the ephemerons whose keys have been marked
// since they were met trace their values (onto the first worker's stack).
// True when one did, so there may be grey work again. The key is read now:
// the mutator may have rewritten the entry since it was met, and a key that
// is no reference or names no old object (young, say) cannot die here.
bool ephemeron_round(MarkCycle& c) {
    HeapState& s = c.g.s;
    const Heap& heap = c.g.heap;
    MarkWorker& main = *c.workers[0];
    std::vector<EphemeronEntry> pending;
    for (auto& w : c.workers) {
        pending.insert(pending.end(), w->ephemerons.begin(), w->ephemerons.end());
        w->ephemerons.clear();
    }
    bool progress = false;
    for (const EphemeronEntry& e : pending) {
        const uint64_t word = *e.key;
        const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
        if (heap.is_reference_tag(word) && heap.is_old(a) && !is_marked(s, a)) {
            main.ephemerons.push_back(e);
            continue;
        }
        main.tracer.set_owner(e.owner, e.owner_old);
        main.tracer.visit(e.value);
        progress = true;
    }
    return progress;
}

// Grey work until it and the ephemeron fixpoint are done (true) or the
// deadline passes (false).
bool run_slice(MarkCycle& c, uint64_t deadline) {
    c.sh.deadline = deadline;
    const uint64_t now = deadline == kNoDeadline ? 0 : gc_now_ns();
    const bool parallel = c.workers.size() > 1 && (deadline == kNoDeadline || deadline > now + kParallelSliceNs);
    const unsigned run = parallel ? static_cast<unsigned>(c.workers.size()) : 1u;
    for (;;) {
        c.draining = true;
        const bool drained = mark_drain(c.sh, c.workers, run);
        c.draining = false;
        if (!drained) return false;
        if (!ephemeron_round(c)) return true;
        if (deadline != kNoDeadline && gc_now_ns() >= deadline) return false;
    }
}

} // namespace

void shade(HeapState& s, uintptr_t a) noexcept {
    MarkCycle& c = open_cycle(s, "an object shaded");
    if (c.draining) gc_fatal("an object shaded while the marker drains (a trace function stored into the heap)");
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size() || !s.blocks[index]->in_use) return;
        BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        const uint64_t bit = uint64_t{1} << (granule & 63);
        if ((meta.starts[granule >> 6] & bit) == 0) return;
        if (!claim_bit(meta.marks[granule >> 6], bit)) return;
    } else if (s.in_large(a)) {
        const uint32_t head = s.large_head_of(a);
        if (head == UINT32_MAX || s.large_object_at(head) != a) return;
        if (!claim_large_bit(header_of(a)->gc_bits)) return;
    } else {
        return;
    }
    c.workers[0]->stack.push_back(a);
    c.grey_empty = false;
}

void cycle_note_weak(HeapState& s, const WeakEntry& entry) {
    open_cycle(s, "a weak slot noted").workers[0]->weak.push_back(entry);
}

void cycle_note_ephemeron(HeapState& s, const EphemeronEntry& entry) {
    MarkCycle& c = open_cycle(s, "an ephemeron noted");
    c.workers[0]->ephemerons.push_back(entry);
    // Its key may be marked before the next slice looks: the fixpoint must run.
    c.grey_empty = false;
}

void cycle_add_grey(HeapState& s, const std::vector<uintptr_t>& grey) {
    if (grey.empty()) return;
    MarkCycle& c = open_cycle(s, "objects shaded");
    std::vector<uintptr_t>& stack = c.workers[0]->stack;
    stack.insert(stack.end(), grey.begin(), grey.end());
    c.grey_empty = false;
}

void Collector::allocated_old(Heap& heap, uintptr_t object) noexcept {
    shade(state(heap), object);
    heap.dirty_card(object);
}

void Collector::start_cycle(Heap& heap, bool evacuate) {
    HeapState& s = state(heap);
    const uint64_t t0 = gc_now_ns();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    // What the last cycle's lazy sweep has left is finished first, in this
    // pause: the sweep clears the marks it reads, so only then are they all
    // clear for this cycle, and no unswept block is left for the marker to
    // meet (heap_sweep_lazy.cpp).
    s.complete_sweep();
    const uint64_t swept_ns = gc_now_ns() - t0;
    if (s.verify) verify_heap(heap, "before a collection");

    // The mark bitmaps are clear: the last cycle's sweep cleared every mark
    // it kept, and a block or page handed out since starts clear.
    s.cycle.reset(new MarkCycle(s, heap));
    MarkCycle& c = *s.cycle;
    GcState& g = c.g;
    g.start_sweep_ns = swept_ns;
    begin_collection(g);
    PhaseClock clock(g);
    if (evacuate) full_evacuate(g, clock, false);

    c.workers = make_mark_workers(c.sh, mark_thread_count(s));
    visit_roots(s, c.workers[0]->tracer);
    if (!evacuate) {
        // The weak slots and ephemerons the roots hold are met again by the
        // finish's root visit; a root table may be rebuilt by the mutator
        // before then, so these slots are not kept.
        c.workers[0]->weak.clear();
        c.workers[0]->ephemerons.clear();
    }
    clock.lap(kPhaseRoots);

    s.collecting = false;
    s.mark_phase = MarkPhase::Marking;
    // From here to the finish the barrier shades old values. A GC point (this
    // is one), so generated code between two GC points sees one range.
    heap.set_barrier_marking(true);
    s.update_fast_limit();
    g.start_ns = gc_now_ns() - t0;
}

bool Collector::mark_cycle(Heap& heap, uint64_t deadline_ns) {
    HeapState& s = state(heap);
    MarkCycle& c = *s.cycle;
    const uint64_t t0 = gc_now_ns();
    s.collecting = true;
    PhaseClock clock(c.g);
    c.grey_empty = run_slice(c, deadline_ns);
    clock.lap(kPhaseTrace);
    s.collecting = false;
    ++c.g.slices;
    c.g.mark_ns += gc_now_ns() - t0;
    return c.grey_empty;
}

void Collector::finish_cycle(Heap& heap, bool rescan) {
    HeapState& s = state(heap);
    MarkCycle& c = *s.cycle;
    GcState& g = c.g;
    const uint64_t t0 = gc_now_ns();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    s.mark_phase = MarkPhase::Finishing;
    s.collecting = true;
    if (rescan) {
        // The young generation, emptied by a minor collection that shades:
        // every survivor is promoted grey (the marker then scans what it
        // names) and every dirty card shades the old objects it names. Then the
        // roots again through the marker's tracer: a full collection's view
        // of them, which a minor's root sources do not give.
        begin_collection(g);
        PhaseClock clock(g);
        full_evacuate(g, clock, true);
        visit_roots(s, c.workers[0]->tracer);
        clock.lap(kPhaseRoots);
    }
    // The rest of the grey work and the ephemeron fixpoint: what the rescan
    // shaded, what the slices left, and ephemerons whose keys were marked
    // since the last round. Back to back, nothing ran since the last slice
    // reported it all gone, so there is none.
    if (rescan || !c.grey_empty) {
        PhaseClock clock(g);
        c.grey_empty = run_slice(c, kNoDeadline);
        clock.lap(kPhaseTrace);
    }
    // Marking is over: no store needs shading from here (a hook's store, say).
    heap.set_barrier_marking(false);
    if (s.verify) verify_cycle_marks(heap);
    for (auto& w : c.workers) {
        g.weak.insert(g.weak.end(), w->weak.begin(), w->weak.end());
        g.ephemerons.insert(g.ephemerons.end(), w->ephemerons.begin(), w->ephemerons.end());
        g.marked_bytes += w->marked_bytes;
        g.marked_objects += w->marked_objects;
    }
    c.workers.clear();

    PhaseClock clock(g);
    full_reclaim(g, clock);
    s.collecting = false;
    s.old_critical = s.old_in_reserve_swept();
    s.mark_phase = MarkPhase::Idle;
    s.update_fast_limit();
    if (s.verify) verify_heap(heap, "after a collection");
    g.finish_ns = gc_now_ns() - t0;
}

void Collector::end_cycle(Heap& heap, uint64_t pause_ns, uint64_t longest_ns) {
    HeapState& s = state(heap);
    s.trigger = s.cycle->trigger;
    record_collection(s.cycle->g, pause_ns, longest_ns);
    s.cycle.reset();
}

void Collector::collect_full(Heap& heap) {
    const uint64_t t0 = gc_now_ns();
    // Back to back: the young generation emptied at the start, and nothing
    // runs before the finish that it would have to rescan.
    start_cycle(heap, true);
    while (!mark_cycle(heap, kNoDeadline)) {
    }
    finish_cycle(heap, false);
    // One pause: the whole of it, as a collection has always counted.
    const uint64_t pause = gc_now_ns() - t0;
    end_cycle(heap, pause, pause);
}

void Collector::finish_open_cycle(Heap& heap) {
    HeapState& s = state(heap);
    const uint64_t t0 = gc_now_ns();
    finish_cycle(heap, true);
    const uint64_t pause = gc_now_ns() - t0;
    s.cycle->pause_ns += pause;
    s.cycle->longest_ns = std::max(s.cycle->longest_ns, pause);
    end_cycle(heap, s.cycle->pause_ns, s.cycle->longest_ns);
}

void Collector::stress_incremental(Heap& heap) {
    HeapState& s = state(heap);
    if (s.mark_phase == MarkPhase::Idle) {
        // The last cycle's sweep, left pending: most steps sweep a tiny
        // slice of it (the allocation they serve may sweep blocks too), every
        // fourth starts the next cycle, which finishes what is left.
        if (s.sweep.active && (s.stress_counter & 3) != 0) {
            heap.sweep_slice(0);
            return;
        }
        heap.start_marking();
        return;
    }
    // Every fourth step a minor collection inside the cycle, the rest tiny
    // slices (a zero budget: one look at the clock's worth of marking).
    if ((s.stress_counter & 3) == 0) {
        heap.collect_at(CollectionKind::Minor, 0, 0);
        return;
    }
    heap.mark_slice(0);
    if (++s.cycle->stress_slices >= kStressSlicesPerCycle) heap.finish_marking();
}

} // namespace brass::gc::detail

namespace brass::gc {

using detail::Collector;
using detail::HeapState;
using detail::MarkPhase;

namespace {

void note_pause(HeapState& s, uint64_t t0) {
    const uint64_t pause = detail::gc_now_ns() - t0;
    s.cycle->pause_ns += pause;
    s.cycle->longest_ns = std::max(s.cycle->longest_ns, pause);
}

void check_not_collecting(const HeapState& s, const char* what) {
    if (!s.collecting) return;
    char msg[160];
    std::snprintf(msg, sizeof msg, "%s during a collection (a trace function, root source or hook called it)", what);
    detail::gc_fatal(msg);
}

void check_marking(const HeapState& s, const char* what) {
    check_not_collecting(s, what);
    if (s.mark_phase == MarkPhase::Marking) return;
    char msg[160];
    std::snprintf(msg, sizeof msg, "%s with no marking cycle open (Heap::start_marking)", what);
    detail::gc_fatal(msg);
}

} // namespace

bool Heap::start_marking() {
    HeapState& s = *s_;
    check_not_collecting(s, "start_marking");
    if (s.mark_phase != MarkPhase::Idle) return false;
    const uint64_t t0 = detail::gc_now_ns();
    s.trigger = detail::GcTrigger::Incremental;
    Collector::start_cycle(*this, false);
    note_pause(s, t0);
    return true;
}

bool Heap::mark_slice(uint64_t budget_ns) {
    HeapState& s = *s_;
    check_marking(s, "mark_slice");
    const uint64_t t0 = detail::gc_now_ns();
    const uint64_t deadline = budget_ns >= detail::kNoDeadline - t0 ? detail::kNoDeadline : t0 + budget_ns;
    const bool done = Collector::mark_cycle(*this, deadline);
    note_pause(s, t0);
    return done;
}

bool Heap::marking() const noexcept { return s_->mark_phase != MarkPhase::Idle; }

void Heap::finish_marking() {
    HeapState& s = *s_;
    check_marking(s, "finish_marking");
    Collector::finish_open_cycle(*this);
    run_pending_finalizers();
}

void Heap::shade_stored(uint64_t value) noexcept { detail::shade(*s_, static_cast<uintptr_t>(value & kAddressMask)); }

} // namespace brass::gc
