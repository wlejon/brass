#pragma once

// The heap's private state, shared by the translation units that implement
// gc::Heap: heap.cpp (construction, allocation, roots, queries),
// heap_spaces.cpp (the mature blocks and the large-object space),
// heap_sweep_lazy.cpp (their sweep after a marking cycle, a piece at a time),
// heap_collect.cpp (the collections), heap_mark.cpp and
// heap_mark_incremental.cpp (the old generation's marking and its cycle),
// heap_weak.cpp (weak slots, ephemerons, finalizers), heap_verify.cpp
// (verification, walking, poisoning) and heap_mark_verify.cpp (a marking
// cycle's check against a fresh mark).

#include <brass/gc/heap.hpp>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace brass::gc::detail {

// Reserve, commit, decommit and release address space (virtual_memory.cpp).
// Committed memory reads as zero until written.
size_t os_page_bytes();
void* vm_reserve(size_t bytes, size_t alignment);
void vm_commit(void* address, size_t bytes);
void vm_decommit(void* address, size_t bytes);
void vm_release(void* address, size_t bytes);

[[noreturn]] void gc_fatal(const char* message);

// Index of the lowest set bit of a nonzero word.
inline unsigned ctz64(uint64_t word) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward64(&index, word);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctzll(word));
#endif
}

inline unsigned popcount64(uint64_t word) noexcept { return static_cast<unsigned>(std::popcount(word)); }

// Index of the highest set bit of a nonzero word.
inline unsigned clz_index64(uint64_t word) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanReverse64(&index, word);
    return static_cast<unsigned>(index);
#else
    return 63u - static_cast<unsigned>(__builtin_clzll(word));
#endif
}

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
    bool lazy = true;     // HeapConfig::lazy_sweep: false sweeps at the finish
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
struct MarkCycleDeleter {
    void operator()(MarkCycle* cycle) const noexcept;
};
// The marker's parked helper threads (heap_mark_threads.hpp).
class MarkThreads;
struct MarkThreadsDeleter {
    void operator()(MarkThreads* threads) const noexcept;
};

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
    uintptr_t eden_synced = 0;      // eden objects below here have their start bits
    uint64_t eden_bytes_retired = 0;// eden bytes allocated in earlier cycles

    // Mature blocks.
    uintptr_t mature_lo = 0;
    size_t mature_reserve = 0;
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

    // The marking cycle, from start_marking (or a full collection's start)
    // to its finish. While it is Marking the mutator runs: the barrier's
    // value range covers the heap (Heap::set_barrier_marking), minor
    // collections shade (GcState::shade) and old allocation is grey. It
    // changes only at a GC point: the cycle's start and finish, each a
    // collection of its own.
    MarkPhase mark_phase = MarkPhase::Idle;
    std::unique_ptr<MarkCycle, MarkCycleDeleter> cycle;
    // Made at the first parallel drain (HeapConfig::mark_thread_pool), kept
    // parked until the heap is destroyed.
    std::unique_ptr<MarkThreads, MarkThreadsDeleter> mark_threads;

    HeapStats stats;
    uint64_t relocation_epoch = 0;
    std::vector<uint64_t> reference_tag_bits;  // 65536 bits, empty: every tag
    std::shared_ptr<runtime::CoroFrameRegistry> coro_frames;

    explicit HeapState(Heap& h) : heap(h) {}

    // ---- geometry ----
    [[nodiscard]] bool in_eden(uintptr_t a) const noexcept { return a - eden_lo < eden_hi - eden_lo; }
    [[nodiscard]] bool in_survivor(int which, uintptr_t a) const noexcept {
        return a - survivor_lo[which] < survivor_bytes;
    }
    [[nodiscard]] bool in_mature(uintptr_t a) const noexcept { return a - mature_lo < mature_reserve; }
    [[nodiscard]] bool in_large(uintptr_t a) const noexcept { return a - large_lo < large_reserve; }
    [[nodiscard]] uint32_t block_index(uintptr_t a) const noexcept {
        return static_cast<uint32_t>((a - mature_lo) / kBlockBytes);
    }
    [[nodiscard]] uintptr_t block_base(uint32_t index) const noexcept { return mature_lo + index * kBlockBytes; }
    [[nodiscard]] size_t young_bit(uintptr_t payload) const noexcept {
        return (payload - eden_lo) / kGranuleBytes;
    }
    void set_young_start(uintptr_t payload) noexcept {
        const size_t bit = young_bit(payload);
        young_starts[bit >> 6] |= uint64_t{1} << (bit & 63);
    }
    [[nodiscard]] bool young_start(uintptr_t payload) const noexcept {
        const size_t bit = young_bit(payload);
        return ((young_starts[bit >> 6] >> (bit & 63)) & 1) != 0;
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
    bool refill_region(BumpRegion& region, bool whole_block);
    uint32_t acquire_free_block();
    // Frees a large object's pages; returns the pages of the free run they
    // joined.
    uint32_t free_large(uint32_t head, uint32_t pages);
    [[nodiscard]] uint32_t large_head_of(uintptr_t address) const noexcept;

    // ---- the lazy sweep (heap_sweep_lazy.cpp) ----
    LazySweep sweep;
    // At a cycle's finish, once liveness is settled: every block in use
    // becomes unswept, the allocation regions and recyclable list start
    // over, and the large objects' phase begins. `live_bytes`: the cycle's
    // marked bytes. With HeapConfig::lazy_sweep off, sweeps all of it now.
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
    // The allocator's share: unswept blocks from the bottom until one has
    // free lines (or is free), at most a few; true when one was found.
    bool sweep_blocks_for_allocation();
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
    // The adaptive eden (HeapConfig::adaptive_eden): allocation stops at
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
    // A minor collection (or a cycle's finishing evacuation) while a marking
    // cycle is open: its tracer's range is the whole heap, every old object
    // it reaches or promotes is shaded, and a weak slot or ephemeron of an
    // old object naming an old target is left to the cycle (heap_weak.cpp).
    bool shade = false;
    std::vector<uintptr_t> gray;
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
    uint64_t eden_used = 0;
    uint64_t eden_limit = 0;    // the limit eden_used was allocated under
    uint64_t phase_ns[kGcPhaseCount] = {};
    // A full collection's cycle: the pauses of its start, its mark slices
    // (and how many) and its finish.
    uint64_t start_ns = 0;
    uint64_t start_sweep_ns = 0;  // of start_ns: the last cycle's sweep, finished first
    uint64_t mark_ns = 0;
    uint64_t finish_ns = 0;
    uint64_t slices = 0;

    GcState(HeapState& state, Heap& h, CollectionKind k) : s(state), heap(h), kind(k) {}

    // Whether `a` lies in the young objects this collection evacuates.
    [[nodiscard]] bool in_from(uintptr_t a) const noexcept {
        return a - s.eden_lo < eden_top - s.eden_lo ||
               a - s.survivor_lo[from] < from_top - s.survivor_lo[from];
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
// the young generation emptied into the old one, every survivor promoted
// (`shade`: during an open cycle, shading what it reaches; GcState::shade);
void full_evacuate(GcState& g, PhaseClock& clock, bool shade);
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
// Visits every root of the heap with `tracer`.
void visit_roots(HeapState& s, Tracer& tracer);
// Rescans the old generation's dirty cards, cleaning them (heap_cards.cpp).
void scan_dirty_cards(Tracer& tracer, GcState& g);
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
