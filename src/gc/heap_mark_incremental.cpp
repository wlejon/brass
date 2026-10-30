// The old generation's marking cycle: start, slices, finish.
//
// A full collection is one cycle run back to back: start (the young
// generation emptied into the old one, the roots made grey), mark slices
// until the grey work is gone, finish (weakness, hooks, sweep). The cycle's
// state lives on the heap between its steps (MarkCycle: the collection's
// GcState, the marker's workers with their mark stacks and the shared pool),
// so the steps can also be separate calls, Heap::start_marking, mark_slice
// and finish_marking, with the slices given a time budget each.
//
// Nothing may run between them yet: a mutator store of an old reference into
// an already scanned object, or a minor collection promoting a young object
// whose old referents are white, would let finish sweep a live object. The
// heap makes either a hard error while a cycle is open (Collector::collect,
// Heap::allocate_at, and the inline allocation path shut by
// update_fast_limit) until the barrier shades and minors mark what they
// promote.
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

// The least slice budget run on several threads: below it, starting and
// joining the threads (tens of microseconds) is too much of the slice.
constexpr uint64_t kParallelSliceNs = 200'000;

} // namespace

struct MarkCycle {
    GcState g;  // the full collection the cycle is
    MarkShared sh;
    MarkWorkers workers;
    bool grey_empty = false;  // the last slice drained everything
    // The pauses so far (the public calls'), for the statistics.
    uint64_t pause_ns = 0;
    uint64_t longest_ns = 0;

    MarkCycle(HeapState& s, Heap& heap) : g(s, heap, CollectionKind::Full), sh(s, heap) {}
};

void MarkCycleDeleter::operator()(MarkCycle* cycle) const noexcept { delete cycle; }

namespace {

// After the grey work is gone: the ephemerons whose keys have been marked
// since they were met trace their values (onto the first worker's stack).
// True when one did, so there may be grey work again.
bool ephemeron_round(MarkCycle& c) {
    HeapState& s = c.g.s;
    MarkWorker& main = *c.workers[0];
    std::vector<EphemeronEntry> pending;
    for (auto& w : c.workers) {
        pending.insert(pending.end(), w->ephemerons.begin(), w->ephemerons.end());
        w->ephemerons.clear();
    }
    bool progress = false;
    for (const EphemeronEntry& e : pending) {
        if (!is_marked(s, static_cast<uintptr_t>(*e.key & kAddressMask))) {
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
        if (!mark_drain(c.sh, c.workers, run)) return false;
        if (!ephemeron_round(c)) return true;
        if (deadline != kNoDeadline && gc_now_ns() >= deadline) return false;
    }
}

} // namespace

void Collector::start_cycle(Heap& heap) {
    HeapState& s = state(heap);
    const uint64_t t0 = gc_now_ns();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    if (s.verify) verify_heap(heap, "before a collection");

    // The mark bitmaps are clear: the last cycle's sweep cleared every mark
    // it kept, and a block or page handed out since starts clear.
    s.cycle.reset(new MarkCycle(s, heap));
    MarkCycle& c = *s.cycle;
    GcState& g = c.g;
    begin_collection(g);
    PhaseClock clock(g);
    full_evacuate(g, clock);

    c.workers = make_mark_workers(c.sh, mark_thread_count(s));
    visit_roots(s, c.workers[0]->tracer);
    clock.lap(kPhaseRoots);

    s.collecting = false;
    s.mark_phase = MarkPhase::Marking;
    s.update_fast_limit();  // shut: every allocation reaches the runtime
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

void Collector::finish_cycle(Heap& heap) {
    HeapState& s = state(heap);
    MarkCycle& c = *s.cycle;
    GcState& g = c.g;
    const uint64_t t0 = gc_now_ns();
    std::optional<runtime::CoroRootsLock> coro_lock;
    if (s.coro_frames) coro_lock.emplace(*s.coro_frames);
    s.mark_phase = MarkPhase::Finishing;
    s.collecting = true;
    // A host that finishes before the slices are done: the rest now.
    if (!c.grey_empty) {
        PhaseClock clock(g);
        c.grey_empty = run_slice(c, kNoDeadline);
        clock.lap(kPhaseTrace);
    }
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
    s.old_critical = s.old_in_reserve();
    s.mark_phase = MarkPhase::Idle;
    s.update_fast_limit();
    if (s.verify) verify_heap(heap, "after a collection");
    g.finish_ns = gc_now_ns() - t0;
}

void Collector::end_cycle(Heap& heap, uint64_t pause_ns, uint64_t longest_ns) {
    HeapState& s = state(heap);
    record_collection(s.cycle->g, pause_ns, longest_ns);
    s.cycle.reset();
}

void Collector::collect_full(Heap& heap) {
    const uint64_t t0 = gc_now_ns();
    start_cycle(heap);
    while (!mark_cycle(heap, kNoDeadline)) {
    }
    finish_cycle(heap);
    // One pause: the whole of it, as a collection has always counted.
    const uint64_t pause = gc_now_ns() - t0;
    end_cycle(heap, pause, pause);
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
    Collector::start_cycle(*this);
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
    const uint64_t t0 = detail::gc_now_ns();
    Collector::finish_cycle(*this);
    note_pause(s, t0);
    Collector::end_cycle(*this, s.cycle->pause_ns, s.cycle->longest_ns);
    run_pending_finalizers();
}

} // namespace brass::gc
