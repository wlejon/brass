#pragma once

// The heap's private state, shared by the translation units that implement
// gc::Heap: heap.cpp (construction, allocation, roots, queries),
// heap_spaces.cpp (the mature blocks and the large-object space),
// heap_collect.cpp (the collections), heap_weak.cpp (weak slots, ephemerons,
// finalizers) and heap_verify.cpp (verification, walking, poisoning).

#include <brass/gc/heap.hpp>
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
struct BlockMeta {
    uint64_t starts[kStartWordsPerBlock] = {};  // an object's payload begins here
    uint64_t marks[kStartWordsPerBlock] = {};   // reached by the running full collection
    uint8_t line_used[kLinesPerBlock] = {};     // allocation state: some object overlaps the line
    uint8_t line_mark[kLinesPerBlock] = {};     // some reached object overlaps the line
    bool in_use = false;                        // holds (or may hold) objects
    bool committed = false;
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

struct FinalizerEntry {
    uintptr_t object;
    void (*fn)(void*);
    void* context;
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
    std::vector<FinalizerEntry> young_finalizers;
    std::vector<FinalizerEntry> old_finalizers;
    std::vector<FinalizerEntry> pending_finalizers;

    // Policy.
    uint64_t old_allocated_since_full = 0;
    uint64_t full_threshold = 0;
    bool minor_requested = false;
    bool full_requested = false;
    uint64_t stress_counter = 0;
    StressMode stress = StressMode::None;
    bool verify = false;
    bool poison = false;

    // The collection in progress, if any.
    bool collecting = false;
    CollectionKind collecting_kind = CollectionKind::Minor;
    uintptr_t gc_eden_top = 0;        // eden's allocation top when it started
    uintptr_t gc_from_top = 0;        // the from-survivor space's top when it started
    uintptr_t mutator_fp = 0;         // the generated frame the running allocation or safepoint came from
    uintptr_t mutator_ip = 0;

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
    // After a full collection's marking: reclaims unmarked mature objects and
    // large objects, rebuilds the free and recyclable block lists. Returns the
    // live old bytes.
    uint64_t sweep_old();
    // Commits (and cleans) the cards covering old addresses [lo, hi).
    void commit_cards(uintptr_t lo, uintptr_t hi);
    bool refill_region(BumpRegion& region, bool whole_block);
    uint32_t acquire_free_block();
    void free_large(uint32_t head, uint32_t pages);
    [[nodiscard]] uint32_t large_head_of(uintptr_t address) const noexcept;

    // ---- the young generation (heap.cpp) ----
    void reset_eden() noexcept;
    void update_fast_limit() noexcept;
    uint64_t old_direct_bytes = 0;  // allocated straight into the old generation
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
    std::vector<uintptr_t> gray;
    std::vector<WeakEntry> weak;
    std::vector<EphemeronEntry> ephemerons;  // keys not (yet) known alive
    uint64_t copied_bytes = 0;
    uint64_t promoted_bytes = 0;
    uint64_t marked_bytes = 0;

    GcState(HeapState& state, Heap& h, CollectionKind k) : s(state), heap(h), kind(k) {}

    // Whether `a` lies in the young objects this collection evacuates.
    [[nodiscard]] bool in_from(uintptr_t a) const noexcept {
        return a - s.eden_lo < eden_top - s.eden_lo ||
               a - s.survivor_lo[from] < from_top - s.survivor_lo[from];
    }
};

// Where the object at `a` is after this collection so far, or 0 if it is not
// (yet) known to be alive. Addresses the collection does not collect are
// returned unchanged.
uintptr_t live_address(const GcState& g, uintptr_t a) noexcept;
// Visits every root of the heap with `tracer`.
void visit_roots(HeapState& s, Tracer& tracer);
// Visits the references of the object at `object` (old: it lives in the old generation).
void scan_object(Tracer& tracer, uintptr_t object, bool old);
// heap_weak.cpp
void weak_visit(Tracer& tracer, uint64_t* slot, uint64_t cleared);
void ephemeron_visit(Tracer& tracer, uint64_t* key, uint64_t* value, uint64_t cleared_key,
                     uint64_t cleared_value);
// Drains the gray stack, resolving ephemerons until no key becomes alive.
void trace_to_fixpoint(Tracer& tracer, GcState& g);
// Settles weak slots, clears dead ephemerons, queues finalizers of the dead.
void settle_weakness(GcState& g);
// heap_verify.cpp
void verify_heap(Heap& heap, const char* when);
// The payload address of the object whose header or payload holds `a`, or 0.
// With `accept_forwarded`, also a young object a collection already copied
// (its header's size is intact): a derived root's base during a collection.
uintptr_t containing_object(HeapState& s, uintptr_t a, bool accept_forwarded) noexcept;

// The collections (heap_collect.cpp, heap_weak.cpp).
struct Collector {
    static void collect(Heap& heap, CollectionKind kind);
    static HeapState& state(const Heap& heap) noexcept { return *heap.s_; }
    static Heap::AllocationBuffer& buffer(Heap& heap) noexcept { return *heap.alloc_; }
    static size_t max_young_total(const Heap& heap) noexcept { return heap.max_young_total_; }
};

} // namespace brass::gc::detail
