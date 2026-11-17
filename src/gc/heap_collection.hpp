#pragma once

// A collection's side of the heap's private state (heap_internal.hpp): the
// weak slots and ephemerons a trace meets, allocation mementos and the
// pretenured sites' probes, one collection's working state (GcState) and its
// phase clock, and the phases the translation units implementing the
// collections share (evacuation, stack pinning, cards, tracing, weakness,
// logging, verification, marking), down to Collector, the friend of
// gc::Heap they reach it through.

#include "heap_internal.hpp"

namespace brass::gc::detail {

struct WeakEntry {
    uint64_t* slot;
    uint64_t cleared;
    uintptr_t owner;
    bool owner_old;
};

struct EphemeronEntry {
    uint64_t* key;
    uint64_t* value;
    uint64_t cleared_key;
    uint64_t cleared_value;
    uintptr_t owner;
    bool owner_old;
};

// No memento layout (set_memento_layout): never an id, register_layout stops
// short of it.
inline constexpr LayoutId kNoMemento = 0xFFFF;
// A memento: its header and its counter's address.
inline constexpr size_t kMementoBytes = kHeaderBytes + sizeof(uint64_t);
// The memento layout as set_memento_layout left it.
LayoutId memento_layout() noexcept;
// A minor collection's verdicts on the pretenured sites' probes
// (heap_pretenure.cpp): after the evacuation, before the young spaces are
// reclaimed. The sites it refused.
uint32_t judge_probes(HeapState& s) noexcept;

// The counter of the memento allocated right behind young object `object`
// (`total` bytes with its header), or null. Only an object start can be one:
// eden's starts are synced when mementos are on (minor), and a survivor
// space's are the copies'.
inline uint64_t* memento_counter(const HeapState& s, LayoutId layout, uintptr_t object, size_t total) noexcept {
    const uintptr_t m = object + total;
    if (!(s.in_eden(m) || s.in_survivor(0, m) || s.in_survivor(1, m)) || !s.young_start(m)) return nullptr;
    if (header_of(m)->layout != layout) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(*reinterpret_cast<const uint64_t*>(m)));
}

// The bytes each object taken from the grey stack may copy depth first, and
// how deep (heap_collect.cpp, "A copy made just now").
inline constexpr size_t kCopyBudget = 4096;
inline constexpr unsigned kCopyDepth = 16;

// One collection's working state (Tracer::state()).
struct GcState {
    HeapState& s;
    Heap& heap;
    CollectionKind kind;
    int from = 0;               // the survivor space collected from
    int to = 1;                 // the survivor space copied into
    uintptr_t eden_top = 0;     // eden's allocation top when the collection started
    uintptr_t from_top = 0;     // the from-survivor space's top then
    unsigned tenure_age = 1;    // a minor collection promotes survivors that reach this age
    bool survivor_overflow = false;  // a survivor too young to promote did not fit
    // The memento layout (set_memento_layout), or kNoMemento: read once per
    // collection.
    LayoutId memento_layout = kNoMemento;
    // The pretenured sites this minor collection refused (judge_probes), for
    // the log.
    uint32_t sites_refused = 0;
    // A minor collection (or a cycle's finishing evacuation) while a marking
    // cycle is open: its tracer's range is the whole heap, every old object
    // it reaches or promotes is shaded, and a weak slot or ephemeron of an
    // old object naming an old target is left to the cycle (heap_weak.cpp).
    bool shade = false;
    std::vector<uintptr_t> gray;
    // Copies being scanned as they are made, and the bytes left (kCopyBudget).
    unsigned copy_depth = 0;
    size_t copy_budget = 0;
    std::vector<WeakEntry> weak;
    std::vector<EphemeronEntry> ephemerons;  // keys not (yet) known alive
    uint64_t copied_bytes = 0;
    uint64_t promoted_bytes = 0;
    uint64_t tenured_bytes = 0;  // promoted out of the survivor space
    uint64_t marked_bytes = 0;

    // Measurement for the log: objects handled and where the pause went.
    uint64_t copied_objects = 0;
    uint64_t promoted_objects = 0;
    uint64_t marked_objects = 0;
    uint64_t dirty_cards = 0;
    unsigned minor_threads = 1;  // threads the young generation's evacuation copied on
    // parallel_minor_threads' card work, as bytes of copying (kDirtyRegionBytes
    // per dirty region), when it chose one thread: the serial cost is
    // learned per byte of that and of the copying done.
    uint64_t serial_card_bytes = 0;
    bool learns_cost = true;  // false for a full collection's evacuation
    uint64_t eden_used = 0;     // eden allocated when it started (HeapState::eden_allocated)
    uint64_t eden_limit = 0;    // the limit eden_used was allocated under
    uint64_t phase_ns[kGcPhaseCount] = {};
    // A full collection's cycle: the pauses of its start, its mark slices
    // (and how many) and its finish.
    uint64_t start_ns = 0;
    uint64_t start_sweep_ns = 0;  // of start_ns: the last cycle's sweep, finished first
    uint64_t mark_ns = 0;
    uint64_t finish_ns = 0;
    // Of finish_ns, for the log: the young generation's part (an evacuation,
    // or its scan in place), the roots visited again, the grey work drained,
    // and the settling after (weakness, hooks, the sweep's setup).
    uint64_t finish_young_ns = 0;
    uint64_t finish_roots_ns = 0;
    uint64_t finish_drain_ns = 0;
    uint64_t finish_settle_ns = 0;
    // What the finish's drain had to do, for the log: grey entries left when
    // it began (shaded since the last slice), the dirty cards' rescans it
    // pushed, and the old and young objects it marked.
    uint64_t finish_grey = 0;
    uint64_t finish_rescans = 0;
    uint64_t finish_old_marked = 0;
    uint64_t finish_young_marked = 0;
    uint64_t slices = 0;
    uint64_t pinned_objects = 0;  // young objects the stack kept in place

    // The to-space's free fragments (setup_to_space): the space whole, or the
    // gaps between the residents an earlier evacuation left in it. Survivor
    // copies are bumped through them; to_copied_hi is where the copies end.
    std::vector<YoungFragment> to_frags;
    size_t to_frag = 0;
    uintptr_t to_cursor = 0;
    uintptr_t to_limit = 0;
    uintptr_t to_copied_hi = 0;

    GcState(HeapState& state, Heap& h, CollectionKind k) : s(state), heap(h), kind(k) {}

    // Whether `a` (an object's address) lies in the young objects this
    // collection evacuates: eden, the from-space, and the residents an
    // earlier evacuation left in the to-space (a copy made there now does
    // not carry kGcResident).
    [[nodiscard]] bool in_from(uintptr_t a) const noexcept {
        return a - s.eden_lo < eden_top - s.eden_lo ||
               a - s.survivor_lo[from] < from_top - s.survivor_lo[from] ||
               (a - s.survivor_lo[to] < s.gc_to_resident_span &&
                (header_of(a)->gc_bits & kGcResident) != 0);
    }
};

// Monotonic nanoseconds, for phase timing (heap_log.cpp).
uint64_t gc_now_ns() noexcept;

// Times the phases of a collection into GcState::phase_ns, only when logging.
class PhaseClock {
public:
    explicit PhaseClock(GcState& g) noexcept : g_(g), on_(g.s.log), last_(on_ ? gc_now_ns() : 0) {}
    void lap(GcPhase phase) noexcept {
        if (!on_) return;
        const uint64_t now = gc_now_ns();
        g_.phase_ns[phase] += now - last_;
        last_ = now;
    }

private:
    GcState& g_;
    bool on_;
    uint64_t last_;
};

// A full collection's phases around the marking (heap_collect.cpp). Sets up
// `g` for a collection starting now and marks the heap as collecting:
void begin_collection(GcState& g);
// the young generation emptied into the old one, every survivor promoted;
void full_evacuate(GcState& g, PhaseClock& clock);
// last, with the marking done and its weak slots, dead ephemerons and totals
// in `g`: weakness settled, hooks run, the old generation swept, the young
// spaces reset and the next threshold set.
void full_reclaim(GcState& g, PhaseClock& clock);
// A finished collection's statistics and log line: `pause_ns` is all its
// pauses together, `longest_ns` the longest (the same for one pause).
void record_collection(const GcState& g, uint64_t pause_ns, uint64_t longest_ns);

// Where the object at `a` is after this collection so far, or 0 if it is not
// (yet) known to be alive. Addresses the collection does not collect are
// returned unchanged.
uintptr_t live_address(const GcState& g, uintptr_t a) noexcept;
// Visits every root of the heap with `tracer`, the objects the stacks name
// (HeapState::stack_objects) among them.
void visit_roots(HeapState& s, Tracer& tracer);
// The root slots of the generated frames on the thread's stacks, as
// visit_roots visits them: from the frame the collection was asked at (or,
// with walk_stack_on_host_collection, its own), each recorded run of native
// frames, and each switched-out stack. gather_stack_objects walks the same
// frames to learn which words of them the maps describe.
void append_frame_roots(const HeapState& s, std::vector<uintptr_t*>& slots);
// The to-space's fragments for evacuation `g` (begun), and the residents'
// range in_from reads (HeapState::gc_to_resident_span).
void setup_to_space(GcState& g);
// After an evacuation (a minor collection, or a full collection's emptying
// of the young generation): eden and the from-space are free but for the
// pinned objects, which become the residents; eden's fragments and the
// survivor spaces' tops are set from them and the copies.
void reclaim_young(GcState& g);

// ---- stack pinning (heap_pin.cpp) ----
// The most eden the allocation window spans between two slow-path
// allocations (HeapState::update_fast_limit): the most a stack word's object
// is looked for by walking eden's headers.
inline constexpr size_t kPinChunkBytes = 32 * 1024;
// The young object whose payload holds `a` (in eden, or a survivor space),
// or 0: in eden's unsynced part found by walking the headers from the
// nearest start the slow path recorded, not by syncing all of eden.
uintptr_t young_object_containing(HeapState& s, uintptr_t a) noexcept;
// HeapState::stack_objects from every stack of the calling thread (each word
// that, untagged or under a reference tag, names a young object's payload): the
// running stack from here, the callee-saved registers included, and each
// switched-out fiber's from where it switched away, except the words of a
// generated frame's slot area (FunctionStackMap::slot_area_lo/hi: its spill
// slots and locals) and its listed frame-slot roots, in every frame append_frame_roots walks: those
// the maps describe exactly, so a word there no root names is dead, and a
// root is visited and updated by visit_roots. A collection that marks
// the old generation gathers the old objects named too (WithOld), which
// visit_roots then marks; a minor needs only the young.
enum class StackObjects { YoungOnly, WithOld };
void gather_stack_objects(HeapState& s, StackObjects which);
// Evacuation `g` keeps the young ones in place: kGcPinned, onto
// HeapState::pinned (which the caller pushes grey: they are roots).
void pin_stack_objects(GcState& g);
// A full collection's marker after an evacuation: the pinned objects are
// young, outside its range, so their slots are visited here as roots.
void scan_pinned(HeapState& s, Tracer& tracer);
// Calls fn(ctx, lo, hi) for each range of the calling thread's stacks that
// holds frames or saved registers: the running stack from `sp` to its base,
// each switched-out stack from where it switched away (fiber.cpp).
void for_each_thread_stack(uintptr_t sp, void (*fn)(void* ctx, uintptr_t lo, uintptr_t hi), void* ctx);
// Rescans the old generation's dirty cards, cleaning them (heap_cards.cpp).
void scan_dirty_cards(Tracer& tracer, GcState& g);
// One piece of a parallel minor collection's card work: a mature card (its
// first byte at `object`, `starts` the start bits of the objects to scan
// there, nonzero) or a range [begin, end) of a large object's payload
// (`starts` 0; the whole object when begin is 0 and end its size).
inline constexpr size_t kCardWorkSliceBytes = 32 * 1024;
struct CardWork {
    uintptr_t object;
    size_t begin;
    size_t end;
    uint64_t starts;
};
// Every dirty card, cleaned, as card work: gathered before anything is
// copied, so the objects a card names are the ones that were there then (a
// promoted object is scanned once, as grey work, never as well by a card
// some other thread scans), and no scan has to read starts or marks that a
// promotion is writing. Each object is in exactly one piece but for large
// objects whose layout scans ranges, which are cut into pieces of
// kCardWorkSliceBytes at most so that threads share a large array.
void gather_dirty_cards(GcState& g, std::vector<CardWork>& out);
// The same pieces for a marking cycle's in-place finish, with every card left
// dirty: the young objects the slots name stay young, so the cards must go on
// remembering them.
void collect_dirty_cards(GcState& g, std::vector<CardWork>& out);
void scan_card_work(Tracer& tracer, const CardWork& work);
// heap_minor_parallel.cpp: how many threads the young generation's
// evacuation `g` (a Minor state, begun) copies on: 1 below the thresholds
// (HeapConfig::parallel_minor), and 1 when the old frontiers lack the room
// a parallel one's promotion needs without sweeping.
unsigned parallel_minor_threads(GcState& g);
// A serial evacuation `g` took `ns`: folded into the cost per byte of work
// (its copying and GcState::serial_card_bytes) the next estimate uses.
void note_serial_evacuation(GcState& g, uint64_t ns);
// The same of a parallel one, whose wall time per byte is what decides
// whether the next long one is parallel at all.
void note_parallel_evacuation(GcState& g, uint64_t ns);
// The young generation's evacuation (heap_collect.cpp, evacuate) on
// `threads` threads, the calling one among them: roots on the calling
// thread, then the card work and the copying shared out, then the
// ephemeron fixpoint, then weakness settled.
void evacuate_parallel(GcState& g, PhaseClock& clock, unsigned threads);
// Visits the references of the object at `object` (old: it lives in the old generation).
void scan_object(Tracer& tracer, uintptr_t object, bool old);
// Visits the references of old object `object` stored in payload bytes
// [begin, end) (more when its layout cannot trace a range).
void scan_object_range(Tracer& tracer, uintptr_t object, size_t begin, size_t end);
// heap_weak.cpp
void weak_visit(Tracer& tracer, uint64_t* slot, uint64_t cleared);
void ephemeron_visit(Tracer& tracer, uint64_t* key, uint64_t* value, uint64_t cleared_key,
                     uint64_t cleared_value);
// Drains the gray stack, resolving ephemerons until no key becomes alive.
void trace_to_fixpoint(Tracer& tracer, GcState& g);
// Settles weak slots, clears dead ephemerons, queues finalizers of the dead.
void settle_weakness(GcState& g);
// heap_log.cpp: BRASS_GC_LOG. A line per collection, and a summary when the
// heap is destroyed.
void log_collection(const GcState& g, uint64_t pause_ns);
void log_summary(const HeapState& s);
// The line for a lazy sweep once it is done: its time and work, by who did it.
void log_sweep(const HeapState& s);
// heap_verify.cpp
void verify_heap(Heap& heap, const char* when);
// heap_mark_verify.cpp: at a cycle's finish, once its marking is done (the
// young generation empty), marks from the roots afresh into a side set and
// stops the process if it reaches an object the cycle left unmarked (a store
// no barrier shaded). Under verification only.
void verify_cycle_marks(Heap& heap);

// The open marking cycle's side of the mutator and the minor collections
// (heap_mark_incremental.cpp). Each needs a cycle open (a hard error
// otherwise) and none collects or allocates on the heap.
//
// Marks old object `a` and pushes it grey (the cycle's first worker's
// stack), unless it is marked already or is no object's start (a value a
// barrier saw stored in a slot no layout traces). Never while the marker's
// threads drain: shading runs on the heap's thread between slices, and from
// minor collections, which never overlap a slice.
void shade(HeapState& s, uintptr_t a) noexcept;
// A weak slot or ephemeron of an old object, met by a minor collection
// during the cycle, whose target is old: decided at the finish by the mark
// bits, with the ones the marker met (an old object never moves, so the
// slot is still there then).
void cycle_note_weak(HeapState& s, const WeakEntry& entry);
void cycle_note_ephemeron(HeapState& s, const EphemeronEntry& entry);
// The objects a parallel minor collection's threads shaded (marked, with an
// atomic claim, since they shade side by side), pushed grey once they are
// done: onto the first worker's stack, as shade pushes.
void cycle_add_grey(HeapState& s, const std::vector<uintptr_t>& grey);
// Whether old object `a` is marked by the open cycle (heap_mark.cpp).
bool is_marked(const HeapState& s, uintptr_t a) noexcept;
// The payload address of the object whose header or payload holds `a`, or 0.
// With `accept_forwarded`, also a young object a collection already copied
// (its header's size is intact): a derived root's base during a collection.
uintptr_t containing_object(HeapState& s, uintptr_t a, bool accept_forwarded) noexcept;

// The collections (heap_collect.cpp, heap_weak.cpp).
struct Collector {
    static void collect(Heap& heap, CollectionKind kind);
    // The marking cycle (heap_mark_incremental.cpp). A full collection is
    // start_cycle, mark_cycle until it returns true, finish_cycle and
    // end_cycle with nothing in between; Heap::start_marking and the rest
    // run the same steps as separate calls.
    static void collect_full(Heap& heap);
    // The roots' old referents grey; the cycle is Marking, and the mutator
    // may run until finish_cycle. `evacuate`: the young generation emptied
    // first, for a cycle run back to back (collect_full), whose finish then
    // has nothing to evacuate or rescan.
    static void start_cycle(Heap& heap, bool evacuate);
    // Grey work until it is gone, the ephemerons included, or `deadline_ns`
    // (gc_now_ns) passes. True when it is gone.
    static bool mark_cycle(Heap& heap, uint64_t deadline_ns);
    // `rescan` (a mutator ran since the start): the young generation emptied
    // by a shading evacuation and the roots visited again. Then the rest of
    // the grey work and full_reclaim; the cycle is Idle after.
    static void finish_cycle(Heap& heap, bool rescan);
    // Records the finished cycle (record_collection) and drops its state.
    static void end_cycle(Heap& heap, uint64_t pause_ns, uint64_t longest_ns);
    // finish_cycle and end_cycle for a cycle a host started, timed as one of
    // its pauses: finish_marking, and a full collection asked for while the
    // cycle is open.
    static void finish_open_cycle(Heap& heap);
    // An old object the mutator allocated while the cycle is open: grey, and
    // its card dirty.
    static void allocated_old(Heap& heap, uintptr_t object) noexcept;
    // StressMode::Incremental's step at an allocation or safepoint.
    static void stress_incremental(Heap& heap);
    static HeapState& state(const Heap& heap) noexcept { return *heap.s_; }
    static Heap::AllocationBuffer& buffer(Heap& heap) noexcept { return *heap.alloc_; }
    static size_t max_young_total(const Heap& heap) noexcept { return heap.max_young_total_; }
};

} // namespace brass::gc::detail
