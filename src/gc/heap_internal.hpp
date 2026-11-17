#pragma once

// The heap's private state, shared by the translation units that implement
// gc::Heap: heap.cpp (construction, allocation, roots, queries),
// heap_spaces.cpp (the mature blocks and the large-object space),
// heap_sweep_lazy.cpp (their sweep after a marking cycle, a piece at a time),
// heap_collect.cpp (the collections), heap_minor_parallel.cpp (the young
// generation's evacuation on several threads), heap_mark.cpp and
// heap_mark_incremental.cpp (the old generation's marking and its cycle),
// heap_weak.cpp (weak slots, ephemerons, finalizers), heap_verify.cpp
// (verification, walking, poisoning) and heap_mark_verify.cpp (a marking
// cycle's check against a fresh mark). What only the collections share (one
// collection's state, its phases) is heap_collection.hpp.

#include "heap_bits.hpp"

#include <brass/gc/heap.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace brass::gc::detail {

// Reserve, commit, decommit and release address space (virtual_memory.cpp).
// Committed memory reads as zero until written.
size_t os_page_bytes();
void* vm_reserve(size_t bytes, size_t alignment);
void vm_commit(void* address, size_t bytes);
void vm_decommit(void* address, size_t bytes);
void vm_release(void* address, size_t bytes);
// A hint that [address, address + bytes) be backed by huge pages (Linux THP).
void vm_advise_huge(void* address, size_t bytes);
// The mature space's commit granule and the reservation's alignment.
inline constexpr size_t kHugePageBytes = size_t{2} << 20;

[[noreturn]] void gc_fatal(const char* message);

// A young object a parallel minor collection's thread is copying
// (heap_minor_parallel.cpp): claimed with a compare-and-swap of its gc_bits,
// and kGcForwarded once the copy and the forwarding address are written.
// Never set outside a parallel minor collection, and only on from-space
// objects, which are dead once it ends.
inline constexpr uint8_t kGcBusy = 0x08;
static_assert((kGcBusy & (kGcForwarded | kGcPinned | kGcLargeMarked | kGcAgeMask)) == 0,
              "kGcBusy is a gc_bits bit of its own");
// A young object left where it was by an evacuation, because a word of a
// thread's stack named it (heap_pin.cpp): set from that evacuation on, and
// cleared when a later one copies it. A young object never carries kGcPinned
// otherwise, so on a young object kGcPinned means "pinned by the evacuation
// under way": it stays, is a root, and every slot naming it keeps its value.
inline constexpr uint8_t kGcResident = 0x80;
static_assert((kGcResident & (kGcForwarded | kGcPinned | kGcLargeMarked | kGcAgeMask | kGcBusy)) == 0,
              "kGcResident is a gc_bits bit of its own");

// A free run of a young space, [lo, hi): eden's after an evacuation that
// pinned objects in it are the gaps between them, which allocation runs
// through in address order; a to-space's, the gaps between the residents an
// earlier evacuation left there, which copying fills around them.
struct YoungFragment {
    uintptr_t lo;
    uintptr_t hi;
    // Eden's: where the allocation window left it (its objects end there),
    // `lo` until the window has been in it.
    uintptr_t used = 0;
};

inline constexpr size_t kStartWordsPerBlock = kGranulesPerBlock / 64;
inline constexpr size_t kCardsPerBlock = kBlockBytes / kCardBytes;

// Side metadata of one mature block. Bits are indexed by the granule of an
// object's PAYLOAD address within the block.
//
// Between a marking cycle's finish and the block's sweep (heap_sweep_lazy.cpp)
// a block in use is unswept: its sweep_epoch is not the heap's
// (LazySweep::epoch). Its starts then still include the objects the cycle
// found dead, and its marks and line marks are that cycle's: an object of it
// is live exactly when its start is marked. Nothing is allocated into it
// until it is swept, and the sweep clears its marks, so the next cycle's
// marks start clear.
struct BlockMeta {
    uint64_t starts[kStartWordsPerBlock] = {};  // an object's payload begins here
    uint64_t marks[kStartWordsPerBlock] = {};   // reached by the running (or last, while unswept) cycle
    uint8_t line_used[kLinesPerBlock] = {};     // allocation state: some object overlaps the line
    uint8_t line_mark[kLinesPerBlock] = {};     // some reached object overlaps the line
    bool in_use = false;                        // holds (or may hold) objects
    bool committed = false;
    uint32_t sweep_epoch = 0;                   // LazySweep::epoch when last swept or taken free
};

// A bump region over a run of free lines of one block.
struct BumpRegion {
    uintptr_t cursor = 0;
    uintptr_t limit = 0;
    uint32_t block = UINT32_MAX;
    uint32_t next_line = 0;  // where the search for this block's next hole resumes
};

struct RootSourceEntry {
    uint64_t id;
    Heap::RootSourceFn fn;
    void* context;
    std::function<void(Tracer&)> callable;
};

struct HookEntry {
    uint64_t id;
    Heap::PostCollectionHook hook;
};

// A registered finalizer: fn(a, b) once `object` is found dead. Two context
// words so a host whose finalizer is a (pointer, destructor) pair registers it
// by value, with no allocation of its own per object; the one-context form is
// stored as a trampoline (Heap::add_finalizer).
struct FinalizerEntry {
    uintptr_t object;
    void (*fn)(void*, void*);
    void* a;
    void* b;
};

// The phases of a collection the log times.
enum GcPhase : unsigned {
    kPhaseRoots,   // root slots, root sources, generated and native frames
    kPhaseCards,   // the old generation's dirty cards (minor)
    kPhaseTrace,   // the transitive closure, ephemerons included
    kPhaseWeak,    // weak slots, dead ephemerons, finalizer bookkeeping
    kPhaseHooks,   // post-collection hooks
    kPhaseSweep,   // reclaiming: the old generation's sweep, the young spaces' reset
    kGcPhaseCount,
};

// Why a collection runs (the log names it).
enum class GcTrigger : uint8_t {
    Explicit,    // Heap::collect / collect_at
    EdenFull,    // a young allocation found eden full
    OldGrowth,   // old-generation growth passed the full-collection threshold
    Requested,   // request_collection, honoured at an allocation or safepoint
    Stress,      // stress mode
    Exhausted,   // an old allocation found the reservation full
    Incremental, // Heap::start_marking ... finish_marking
};

// Where the old generation's marking cycle is (heap_mark_incremental.cpp).
enum class MarkPhase : uint8_t {
    Idle,       // no cycle: the mark bitmaps are clear
    Marking,    // started: roots are grey, slices drain the grey work
    Finishing,  // in finish_marking: weakness, hooks, sweep
};

// Who did a piece of the lazy sweep (the log's account of it).
enum class SweepBy : uint8_t {
    Slice,      // Heap::sweep_slice
    Allocator,  // an old allocation or promotion that needed a block or pages
    Complete,   // the rest at once: a cycle's start, or an old space near its end
};

// The old generation's sweep after a marking cycle's finish, done a piece at
// a time (heap_sweep_lazy.cpp). The finish decides what is dead (the mark
// bits) and settles everything that depends on it (weak slots, ephemerons,
// finalizers, hooks); freeing the memory is left to the sweep, which runs
// in three phases, each resumable: the large objects (freeing the dead,
// clearing the mark of the rest), the mature blocks (rebuilding each one's
// free lines, clearing its marks), and handing excess free blocks back to the
// OS. It is done by Heap::sweep_slice, by the allocator as it needs blocks
// or pages, and whatever is left when the next cycle starts is finished
// there (so a cycle's marks always start clear).
struct LazySweep {
    bool active = false;  // some of the last cycle's sweep is left
    // A block in use is unswept while its sweep_epoch differs. Each finish
    // bumps it, which makes every block in use unswept at no cost.
    uint32_t epoch = 0;
    // The unswept blocks lie in [low, high): the allocator sweeps up from
    // low (taking the lowest blocks first, as the eager sweep's lists did),
    // slices down from high. Blocks past `high` (taken from the frontier
    // since) were never unswept.
    uint32_t low = 0;
    uint32_t high = 0;
    size_t unswept_blocks = 0;
    // Large objects whose head page is at or past large_cursor (and below the
    // frontier) are unswept: dead unless marked. One allocated there while
    // the phase runs is marked as it is made (note_large_allocated), so the
    // sweep keeps it and clears the mark; one allocated behind is left clear.
    bool large_active = false;
    uint32_t large_cursor = 0;
    // The free blocks to decommit, highest first, planned once the blocks are
    // swept: release_list[release_next...]. Each is checked again before its
    // run is decommitted, since the allocator may have taken it since.
    bool release_active = false;
    bool release_planned = false;
    std::vector<uint32_t> release_list;
    size_t release_next = 0;
    // The cycle's marked bytes: what old_used_bytes() reports until the sweep
    // is done and the line counts are exact again.
    uint64_t live_bytes = 0;

    // For the log line at the end (BRASS_GC_LOG): the collection it sweeps
    // for, and the time and work by who did it. The allocator's share is
    // timed only when logging.
    uint64_t collection_index = 0;
    uint64_t slice_ns = 0;
    uint64_t slice_max_ns = 0;
    uint64_t slices = 0;
    uint64_t allocator_ns = 0;
    uint64_t allocator_blocks = 0;
    uint64_t complete_ns = 0;
    uint64_t blocks_swept = 0;
    uint64_t blocks_freed = 0;
    uint64_t large_freed = 0;
    uint64_t large_freed_bytes = 0;
    uint64_t released_bytes = 0;
};

// A marking cycle's state between its calls (heap_mark_incremental.cpp).
struct MarkCycle;
struct MarkCycleDeleter { void operator()(MarkCycle* cycle) const noexcept; };
// The marker's parked helper threads (heap_mark_threads.hpp).
class MarkThreads;
struct MarkThreadsDeleter { void operator()(MarkThreads* threads) const noexcept; };

// What a minor collection tells the adaptive eden (HeapState::adapt_eden_limit).
struct MinorSample {
    uint64_t eden_collected = 0;  // eden bytes allocated when it started
    uint64_t copied = 0;          // every byte it copied, survivor-space objects too
    uint64_t aged = 0;            // the survivor space's bytes when it started
    uint64_t tenured = 0;         // of those, the bytes promoted: kept two collections
    unsigned tenure_age = 1;      // 2 or more: the survivor space held survivors of the last one
    bool overflow = false;        // the survivor space could not hold what was young enough
    bool at_whole_eden = false;   // it ran at eden_hi, not a shrunk limit
};

struct HeapState {
    Heap& heap;
    HeapConfig config;

    // The reservation: [young | mature | large].
    uintptr_t base = 0;
    size_t reserve_bytes = 0;

    // Young generation: eden, then survivor spaces 0 and 1.
    uintptr_t eden_lo = 0;
    uintptr_t eden_hi = 0;
    uintptr_t survivor_lo[2] = {0, 0};
    uintptr_t survivor_top[2] = {0, 0};
    size_t survivor_bytes = 0;
    int from_survivor = 0;          // holds last minor collection's survivors
    std::vector<uint64_t> young_starts;  // by payload granule from young_lo
    // A marking cycle's in-place finish: the young objects it found reachable, by the same bits as
    // young_starts. While `young_in_place` is set (from the finish's root
    // visit to the end of its settling) the marker marks young objects here as
    // it marks old ones in theirs, and liveness of a young address is this
    // bit (live_address, is_marked). Meaningless otherwise.
    std::vector<uint64_t> young_marks;
    bool young_in_place = false;
    uintptr_t eden_synced = 0;      // eden objects below here have their start bits
    size_t eden_sync_frag = 0;      // the fragment of eden_frags eden_synced lies in
    uint64_t eden_bytes_retired = 0;// eden bytes allocated in earlier cycles

    // ---- stack pinning (heap_pin.cpp) ----
    // The young objects the words of this thread's stacks name, sorted:
    // gathered at the start of each pause that visits the roots and visited
    // as roots by it (visit_roots), dropped at its end.
    std::vector<uintptr_t> stack_objects;
    // The young objects the evacuation under way keeps in place (kGcPinned),
    // sorted, from the evacuation's start to the young spaces' reclaim.
    std::vector<uintptr_t> pinned;
    // What Heap::try_allocate_pretenured made old since the last collection:
    // the next minor promotes what they name (evacuate). Emptied by every
    // collection (a full one may free or move them).
    std::vector<uintptr_t> pretenured;
    // The young probes of pretenured sites (heap_pretenure.cpp) the minor
    // collections are following: each judged by the next, dropped when it is
    // promoted or dead, followed into the survivor space otherwise. A full
    // collection drops them all unjudged.
    struct Probe {
        uintptr_t object;
        PretenureSite* site;
    };
    std::vector<Probe> probes;
    // The young objects the last evacuation kept in place (kGcResident),
    // sorted. Some may have died since (a cycle's in-place finish drops dead
    // young objects without moving anything): a dead one is no object (its
    // start bit is clear), but its bytes stay out of the free fragments.
    std::vector<uintptr_t> residents;
    // Eden's free fragments, ascending (the whole of eden when nothing is
    // pinned). The allocation window lies in eden_frags[eden_frag]; eden is
    // `fresh` (nothing allocated since the reclaim) while the window is at the
    // start of fragment eden_fresh_frag.
    std::vector<YoungFragment> eden_frags;
    size_t eden_frag = 0;
    size_t eden_fresh_frag = 0;
    uintptr_t eden_resident_hi = 0;      // the end of eden's highest resident (eden_lo: none)
    uint64_t eden_free_bytes = 0;        // eden's fragments' bytes at the last build_eden_fragments
    // During a collection: [survivor_lo[to], + this) holds the residents an
    // earlier evacuation left in the to-space (GcState::in_from).
    size_t gc_to_resident_span = 0;
    // Where eden's objects end: the allocation top, or past it the highest
    // resident.
    [[nodiscard]] uintptr_t eden_end() const noexcept;
    // Eden's bytes allocated since the reclaim: the used part of each
    // fragment the window has been in. The residents' bytes are not among
    // them, nor the free fragments under them (eden_end - eden_lo would count
    // the whole of eden below a resident pinned near its end).
    [[nodiscard]] uint64_t eden_allocated() const noexcept;
    // Moves the allocation window on to the next fragment of eden while the
    // current one cannot take `total` bytes (within the adaptive limit, or
    // any fragment while eden is fresh). True when [top, the window's hard
    // end) holds `total`.
    bool eden_make_room(size_t total) noexcept;
    // Eden's fragments from the residents in it (reset_eden then starts the
    // window at the first).
    void build_eden_fragments() noexcept;

    // Mature blocks.
    uintptr_t mature_lo = 0;
    size_t mature_reserve = 0;
    // The end of the mature space's committed prefix (commit_block).
    uintptr_t mature_committed_hi = 0;
    std::vector<std::unique_ptr<BlockMeta>> blocks;  // one per block below the frontier
    std::vector<uint32_t> free_blocks;               // committed, empty
    std::vector<uint32_t> recyclable_blocks;         // in use, with free lines
    BumpRegion small_region;
    BumpRegion medium_region;
    size_t mature_used_lines = 0;

    // Large objects: page runs.
    uintptr_t large_lo = 0;
    size_t large_reserve = 0;
    uint32_t large_frontier = 0;                     // pages ever handed out
    std::map<uint32_t, uint32_t> large_free_runs;    // first page -> page count
    std::unordered_map<uint32_t, uint32_t> large_objects;  // head page -> page count
    std::vector<uint32_t> large_page_head;           // page -> head page + 1 (0: none)
    size_t large_used_bytes = 0;
    // Pages of free runs left committed for the next large object (a growing
    // array frees one run for the next every time it doubles): which pages
    // are committed, and how many bytes of free runs are.
    std::vector<uint8_t> large_page_committed;
    size_t large_retained_bytes = 0;

    // Cards over [mature_lo, large_lo + large_reserve), committed as the
    // spaces grow.
    uint8_t* cards = nullptr;
    size_t cards_bytes = 0;
    std::vector<uint8_t> card_pages_committed;  // per page of the card table

    // Roots, hooks, finalizers.
    std::vector<uint64_t*> roots;
    std::vector<RootSourceEntry> sources;
    std::vector<HookEntry> hooks;
    uint64_t next_id = 1;
    // Every list below keeps its capacity across collections: a program that
    // makes and drops native handles every frame registers and retires
    // thousands a collection, and the bookkeeping allocates nothing once the
    // lists have grown to that working set (settle_weakness compacts them in
    // place).
    std::vector<FinalizerEntry> young_finalizers;
    std::vector<FinalizerEntry> old_finalizers;
    std::vector<FinalizerEntry> pending_finalizers;
    // The batch Heap::collect_at is running. A finalizer may allocate and so
    // collect; that nested collection leaves what it finds dead in
    // pending_finalizers for the outer loop (running_finalizers says one is
    // under way) rather than run it from inside the finalizer.
    std::vector<FinalizerEntry> running_finalizers;
    bool finalizers_running = false;

    // Policy.
    uint64_t old_allocated_since_full = 0;
    uint64_t full_threshold = 0;
    // The growth that runs a full collection (or finishes an open cycle):
    // the threshold, with a quarter of it on top while a cycle marks, so a
    // cycle the host paces has room to finish at its slices rather than in
    // one pause at whichever allocation reached the threshold. A cycle marks
    // while the program promotes, and one started on a threshold left from a
    // heap built faster than it was collected (a boot that built 1 GB in one
    // turn, last collected at 556 MB) could not mark it all at a host's
    // sub-millisecond slices before the plain threshold: the heap then
    // finished it in one pause of whatever marking was left (32 ms). The
    // headroom is memory a cycle that falls behind may use, not where cycles
    // are paced to finish.
    [[nodiscard]] uint64_t growth_limit() const noexcept {
        if (mark_phase != MarkPhase::Marking) return full_threshold;
        return full_threshold + full_threshold / 4;
    }
    // old_allocated_since_full when the open cycle started: what it grew by
    // after that it marks whether live or not (allocated or promoted grey),
    // so the eden's restore test at its finish (full_reclaim) judges only
    // the growth before it.
    uint64_t cycle_start_growth = 0;
    bool minor_requested = false;
    bool full_requested = false;
    uint64_t stress_counter = 0;
    StressMode stress = StressMode::None;
    bool verify = false;
    bool poison = false;
    bool log = false;                 // one line per collection on stderr (heap_log.cpp)
    // The next minor collection promotes every survivor rather than copying it
    // into a survivor space first: set while survival runs high (the last one
    // overflowed the survivor space), when the survivors are mostly on their
    // way to the old generation anyway and copying them twice is waste.
    bool promote_all = false;
    GcTrigger trigger = GcTrigger::Explicit;  // why the next collection runs

    // The collection in progress, if any.
    bool collecting = false;
    CollectionKind collecting_kind = CollectionKind::Minor;
    uintptr_t gc_eden_top = 0;        // eden's allocation top when it started
    uintptr_t gc_from_top = 0;        // the from-survivor space's top when it started
    uintptr_t mutator_fp = 0;         // the generated frame the running allocation or safepoint came from
    uintptr_t mutator_ip = 0;
    // A parallel minor collection's threads are promoting
    // (heap_minor_parallel.cpp): the old spaces' allocators, called under its
    // lock, take only swept blocks, free page runs and the frontiers and never
    // sweep, since the other threads read the blocks' starts and marks as a
    // sweep rewrites them. The collection checked beforehand that the
    // frontiers have room for everything it could promote.
    bool parallel_promotion = false;
    // Bytes of each survivor space below its top that hold no object: the
    // unused ends of a parallel minor collection's per-thread buffers. The
    // adaptive eden's survival sample leaves them out.
    uint64_t survivor_slack[2] = {0, 0};
    // The last minor collection's bytes copied per young byte it collected:
    // what the next one is expected to copy (parallel_minor_threads). A
    // quarter before the first, so that a first collection of a full eden
    // copies on several threads.
    double minor_survival = 0.25;
    // What a minor collection on one thread has been costing, per byte of
    // copying plus kDirtyRegionBytes per old region with dirty cards (a
    // running average of the serial ones, parallel_minor_threads): how long
    // the next would take on one thread.
    double minor_ns_per_byte = 0.4;  // 0.4 ms per MB until one is measured
    // The same of the parallel ones, wall time per byte of that work: the
    // last one's, the lower of the last two (0 until two are measured), and
    // how many collections have gone serial because that was clearly the
    // higher rate (parallel_minor_threads).
    double parallel_minor_last_ns_per_byte = 0.0;
    double parallel_minor_ns_per_byte = 0.0;
    bool minor_ns_measured = false;  // minor_ns_per_byte is a measurement, not its start
    uint32_t parallel_minor_passed_over = 0;

    // The marking cycle, from start_marking (or a full collection's start)
    // to its finish. While it is Marking the mutator runs: the barrier's
    // value range covers the heap (Heap::set_barrier_marking), minor
    // collections shade (GcState::shade) and old allocation is grey. It
    // changes only at a GC point: the cycle's start and finish, each a
    // collection of its own.
    MarkPhase mark_phase = MarkPhase::Idle;
    std::unique_ptr<MarkCycle, MarkCycleDeleter> cycle;
    // Made at the first parallel drain or minor collection (or
    // Heap::start_helper_threads), kept parked until the heap is destroyed.
    std::unique_ptr<MarkThreads, MarkThreadsDeleter> mark_threads;

    HeapStats stats;
    uint64_t relocation_epoch = 0;
    std::vector<uint64_t> reference_tag_bits;  // 65536 bits, empty: every tag
    std::shared_ptr<runtime::CoroFrameRegistry> coro_frames;

    explicit HeapState(Heap& h) : heap(h) {}

    // ---- geometry ----
    [[nodiscard]] bool in_eden(uintptr_t a) const noexcept { return a - eden_lo < eden_hi - eden_lo; }
    [[nodiscard]] bool in_survivor(int which, uintptr_t a) const noexcept { return a - survivor_lo[which] < survivor_bytes; }
    [[nodiscard]] bool in_mature(uintptr_t a) const noexcept { return a - mature_lo < mature_reserve; }
    [[nodiscard]] bool in_large(uintptr_t a) const noexcept { return a - large_lo < large_reserve; }
    [[nodiscard]] uint32_t block_index(uintptr_t a) const noexcept { return static_cast<uint32_t>((a - mature_lo) / kBlockBytes); }
    [[nodiscard]] uintptr_t block_base(uint32_t index) const noexcept { return mature_lo + index * kBlockBytes; }
    [[nodiscard]] size_t young_bit(uintptr_t payload) const noexcept { return (payload - eden_lo) / kGranuleBytes; }
    void set_young_start(uintptr_t payload) noexcept {
        const size_t bit = young_bit(payload);
        young_starts[bit >> 6] |= uint64_t{1} << (bit & 63);
    }
    void clear_young_start(uintptr_t payload) noexcept {
        const size_t bit = young_bit(payload);
        young_starts[bit >> 6] &= ~(uint64_t{1} << (bit & 63));
    }
    [[nodiscard]] bool young_start(uintptr_t payload) const noexcept {
        const size_t bit = young_bit(payload);
        return ((young_starts[bit >> 6] >> (bit & 63)) & 1) != 0;
    }
    [[nodiscard]] bool young_marked(uintptr_t payload) const noexcept {
        const size_t bit = young_bit(payload);
        return ((young_marks[bit >> 6] >> (bit & 63)) & 1) != 0;
    }
    void clear_young_starts(uintptr_t lo, uintptr_t hi) noexcept;
    // Sets the start bits of eden objects allocated since the last sync.
    void sync_eden_starts() noexcept;
    [[nodiscard]] uint8_t* card_of(uintptr_t old_address) const noexcept {
        return cards + ((old_address - mature_lo) >> kCardShift);
    }
    // The payload address of the large object whose run starts at page `head`.
    [[nodiscard]] uintptr_t large_object_at(uint32_t head) const noexcept {
        return large_lo + static_cast<uintptr_t>(head) * kPageBytes + kHeaderBytes;
    }
    // Remembers that `slot` of old object `owner` names a young object: the
    // owner's card for a mature object; for a large object the slot's own
    // card (kCardDirtyRange on the first card, which kCardDirty would make a
    // whole-object rescan), or the whole object when the slot lies outside it.
    void remember_slot(uintptr_t owner, const void* slot) noexcept {
        if (!in_large(owner)) {
            mark_card(owner, kCardDirty);
            return;
        }
        const auto at = reinterpret_cast<uintptr_t>(slot);
        uint8_t* first = card_of(owner);
        if (at - owner >= header_of(owner)->size) {
            mark_card(owner, kCardDirty);
            return;
        }
        uint8_t* card = card_of(at);
        if (card != first) mark_card(at, kCardDirty);
        else if (*card != kCardDirty) mark_card(at, kCardDirtyRange);
    }
    // Sets the card of old address `a` to `value` (kCardDirty or
    // kCardDirtyRange) and marks its region (kCardRegionShift): every card
    // store goes through here or Heap::dirty_card.
    void mark_card(uintptr_t a, uint8_t value) noexcept {
        cards[(a - mature_lo) >> kCardShift] = value;
        regions[(a - mature_lo) >> kCardRegionShift] = 1;
    }
    // One byte per 32 KB of [mature_lo, large_lo + large_reserve).
    uint8_t* regions = nullptr;
    std::vector<uint8_t> region_bytes;

    // ---- spaces (heap_spaces.cpp) ----
    // The header address of `total` bytes (header included) in the mature
    // space, or 0 when the reservation is exhausted. Sets the start bit and
    // the lines' used marks; the caller writes the header.
    uintptr_t mature_allocate(size_t total);
    // As mature_allocate, in the large-object space (total rounded up to pages).
    uintptr_t large_allocate(size_t total);
    // Where an object of `total` bytes that becomes old goes.
    uintptr_t old_allocate(size_t total) {
        return total <= kMaxMediumObjectBytes ? mature_allocate(total) : large_allocate(total);
    }
    void mark_lines(BlockMeta& meta, uintptr_t block_lo, uintptr_t header, size_t total, bool mark) noexcept;
    // Commits (and cleans) the cards covering old addresses [lo, hi).
    void commit_cards(uintptr_t lo, uintptr_t hi);
    void commit_block(uint32_t index);
    bool refill_region(BumpRegion& region, bool whole_block);
    // A free block, swept or fresh from the frontier; UINT32_MAX when none
    // may be had. `holes_will_do` (the small region's refill): kHolesFiled
    // instead when the pending sweep it ran filed a recyclable block.
    static constexpr uint32_t kHolesFiled = UINT32_MAX - 1;
    uint32_t acquire_free_block(bool holes_will_do = false);
    // Frees a large object's pages; returns the pages of the free run they
    // joined.
    uint32_t free_large(uint32_t head, uint32_t pages);
    [[nodiscard]] uint32_t large_head_of(uintptr_t address) const noexcept;

    // ---- the lazy sweep (heap_sweep_lazy.cpp) ----
    LazySweep sweep;
    // At a cycle's finish, once liveness is settled: every block in use
    // becomes unswept, the allocation regions and recyclable list start
    // over, and the large objects' phase begins. `live_bytes`: the cycle's
    // marked bytes.
    void begin_sweep(uint64_t live_bytes);
    // Whether block `meta` still holds the last cycle's dead objects.
    [[nodiscard]] bool block_unswept(const BlockMeta& meta) const noexcept {
        return meta.in_use && meta.sweep_epoch != sweep.epoch;
    }
    // Whether the large object whose run starts at page `head` is one the
    // last cycle found dead and the sweep has not yet freed.
    [[nodiscard]] bool large_dead_pending(uint32_t head) const noexcept;
    // A large object just allocated (its header written) while the large
    // objects' phase runs: marked when the sweep has yet to reach it.
    void note_large_allocated(uintptr_t header) noexcept {
        if (sweep.large_active && in_large(header) && (header - large_lo) / kPageBytes >= sweep.large_cursor) {
            auto* h = reinterpret_cast<ObjectHeader*>(header);
            h->gc_bits = static_cast<uint8_t>(h->gc_bits | kGcLargeMarked);
        }
    }
    // Sweep work until it is all done (true) or `deadline_ns` (gc_now_ns)
    // passes, making some progress however close the deadline.
    bool sweep_work(uint64_t deadline_ns);
    // All of what is left, at once (a cycle's start; a space near its end).
    void complete_sweep();
    // The allocator's share: unswept blocks from the bottom (or the top)
    // until one has free lines (or is free), at most a few; true when one
    // was found.
    bool sweep_blocks_for_allocation(bool from_top = false);
    // Unswept large objects until a free run of `pages` exists (at most a
    // few): true when one was made.
    bool sweep_large_for_allocation(uint32_t pages);
    // Whether the old generation has eaten into the promotion reserve, the
    // pending sweep's free memory counted (it finishes the blocks and large
    // objects first when only they would make the difference).
    bool old_in_reserve_swept();

    // ---- the young generation (heap.cpp) ----
    void reset_eden() noexcept;
    void update_fast_limit() noexcept;
    // The adaptive eden: allocation stops at
    // eden_limit and a minor collection runs there, eden_hi but for one
    // object too large for an empty eden_limit. survival_estimate is the
    // fraction of eden expected to survive the next minor collection; each
    // minor collection raises it to what it saw at once (the next is then
    // bounded) and otherwise halves it, so the limit doubles back toward
    // eden_hi once survival falls, and one odd collection costs a few short
    // ones rather than a long one.
    uintptr_t eden_limit = 0;
    double survival_estimate = 0.0;
    // While eden is shrunk: running averages, over the minor collections
    // that aged survivors, of the survivor-space bytes each started with, of
    // those that died before it, and of the eden it collected, which give
    // the survivors' lifetime in eden bytes (adapt_eden_limit); how many
    // collections they hold; and how many a verdict needs: kFirstDyingNeeded,
    // doubled each time the whole eden it restored shrank again at the first
    // collection that could (after restore_hold of them), so a build-up that
    // only looks like a working set at times does not swing eden back and
    // forth.
    double aged_average = 0.0;
    double dead_average = 0.0;
    double collected_average = 0.0;
    uint32_t dying_measured = 0;
    uint32_t dying_needed = 4;
    bool restored = false;      // the whole eden came back by that test, not yet judged
    uint32_t restore_hold = 0;  // whole-eden collections left that cannot shrink it
    void forget_dying() noexcept;
    // After a minor collection: resets eden_limit.
    void adapt_eden_limit(const MinorSample& m) noexcept;
    // The whole eden, survival forgotten: a new heap, and after a full
    // collection that found most of the old generation's growth dead (the
    // next minor collection measures the whole eden again, so a program that
    // has moved from building data to churning a working set does not keep
    // the small eden the build-up called for).
    void reset_eden_limit() noexcept;
    // A shrunk eden back whole as a marking cycle starts (one the program
    // runs inside promotes marked, so a shrunk eden's promotions, four times
    // the whole one's on bigheap after its boot, are marking work as well as
    // old growth, and the cycle that started on them fell behind), held
    // there for restore_hold collections as after the lifetime test, since the survivor space and
    // the shrunk eden's promotions say little at first; after them the
    // whole eden judges as always. Not a verdict of that test, so a shrink
    // then does not lengthen its next one (restored stays false).
    void restore_eden_for_cycle() noexcept;
    uint64_t old_direct_bytes = 0;  // allocated straight into the old generation

    // ---- the promotion reserve (heap_spaces.cpp) ----
    // A collection copies young survivors into the old generation and cannot
    // stop half way, so the room it may need is kept from the mutator: the
    // mutator's own old allocations stop short of each space's end by what one
    // collection could promote (twice the young generation's live capacity,
    // for fragmentation), and only a collection takes blocks or pages from
    // that margin. Once a collection has had to, the heap is `old_critical`:
    // the next allocation that reaches the runtime runs a full collection
    // first and, if the margin is still breached, fails with std::bad_alloc
    // (which a host turns into its out-of-memory error) instead of risking a
    // collection with nowhere to promote to.
    size_t reserve_mature_blocks = 0;
    size_t reserve_large_bytes = 0;
    bool old_critical = false;
    [[nodiscard]] size_t mature_blocks_available() const noexcept {
        return (mature_reserve / kBlockBytes - blocks.size()) + free_blocks.size();
    }
    // Whether the old generation has eaten into the promotion reserve.
    [[nodiscard]] bool old_in_reserve() const noexcept {
        return mature_blocks_available() < reserve_mature_blocks ||
               large_used_bytes + reserve_large_bytes > large_reserve;
    }
};

// Records, for its lifetime, the generated frame on whose behalf the heap is
// allocating or at a safepoint: any collection meanwhile (including one a
// finalizer's allocation triggers) walks the generated frames from there.
class MutatorFrame {
public:
    MutatorFrame(HeapState& s, uintptr_t fp, uintptr_t ip) noexcept
        : s_(s), fp_(s.mutator_fp), ip_(s.mutator_ip) {
        if (fp != 0 && ip != 0) {
            s.mutator_fp = fp;
            s.mutator_ip = ip;
        }
    }
    ~MutatorFrame() {
        s_.mutator_fp = fp_;
        s_.mutator_ip = ip_;
    }
    MutatorFrame(const MutatorFrame&) = delete;
    MutatorFrame& operator=(const MutatorFrame&) = delete;

private:
    HeapState& s_;
    uintptr_t fp_;
    uintptr_t ip_;
};

} // namespace brass::gc::detail
