#pragma once

// brass's garbage-collected heap: generational, with a copying young
// generation over a mark-region (Immix-style) mature space and a
// large-object space, all inside one reserved address range per heap. One
// heap belongs to one thread at a time; its collections are stop-the-world
// for that thread only. docs/gc_contract.md is the full contract; the short
// version is here, beside each entry point.

#include <brass/gc/object.hpp>
#include <brass/gc/tracer.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace brass::runtime {
class CoroFrameRegistry;
}

namespace brass::gc {

namespace detail {
struct HeapState;
struct Collector;
}

enum class CollectionKind : uint8_t {
    Minor,  // young generation only: survivors copied (aged, then promoted)
    Full,   // everything: young survivors promoted, old generation marked and swept in place
};

enum class StressMode : uint8_t {
    None,
    Minor,      // a minor collection at every allocation and safepoint
    Full,       // a full collection at every allocation and safepoint
    Alternate,  // minor collections, every eighth one full
    // A marking cycle always open: started when none is, then at each
    // allocation and safepoint a tiny mark slice (every fourth one a minor
    // collection instead), the cycle finished every eighth slice.
    Incremental,
};

// Whether minor collections copy on several threads (HeapConfig::parallel_minor).
enum class ParallelMinor : uint8_t {
    Off,     // always on the collecting thread alone
    Auto,    // on several when the collection is large enough to pay for it
    Always,  // on HeapConfig::minor_threads however small (tests, stress)
};

// Allocation flags.
inline constexpr uint32_t kAllocOld = 1u << 0;     // allocate in the old generation (pretenure)
inline constexpr uint32_t kAllocPinned = 1u << 1;  // old, and never moved by any collection
// The payload is not zeroed: for a Leaf layout only (the collector never
// reads it), whose caller writes every byte it later reads. A runtime that
// fills a fresh buffer with its own empty value pays for one pass, not two.
// Any other layout is a fatal error.
inline constexpr uint32_t kAllocUninitialized = 1u << 2;

struct HeapConfig {
    size_t eden_bytes = size_t{16} << 20;
    // Each of the two survivor spaces. Young objects that survive a minor
    // collection but not yet `tenure_age` of them wait here; whatever does
    // not fit is promoted early, and an early-promoted object that then dies
    // stays in the old generation (holding whatever young objects it names)
    // until the next full collection. So this is sized for the young data a
    // program keeps live across one collection, not for the average.
    size_t survivor_bytes = size_t{8} << 20;
    // Minor collections an object survives in a survivor space before it is
    // promoted (1: promoted by its first minor collection; at most 7, since
    // an object's age saturates there).
    uint8_t tenure_age = 2;
    // Survival-adaptive eden: a minor collection's copying costs about its
    // survivors' bytes, so after each one the eden the next may fill is set
    // to `minor_copy_budget_bytes` over the fraction of eden that survived
    // (at least 1 MB, at most `eden_bytes`), when the survivors are data the
    // program keeps (promoted, not dying in the survivor space), and goes
    // back to the whole eden once they start dying young. A burst of
    // survivors then costs several short minor collections instead of one
    // long one, and a program whose young data dies keeps the whole eden
    // (docs/gc_contract.md, "Adaptive eden"). false (or
    // BRASS_GC_FIXED_EDEN=1): every minor collection fills all of eden.
    bool adaptive_eden = true;
    // About 0.5 ms of single-thread copying (1.3-2 MB/ms measured on x64).
    size_t minor_copy_budget_bytes = size_t{1} << 20;
    // Address space reserved for the mature blocks and the large objects;
    // memory is committed as the heap grows.
    size_t mature_reserve_bytes = size_t{4} << 30;
    size_t large_reserve_bytes = size_t{4} << 30;
    // Payloads above this many bytes (or above a quarter of eden) are
    // allocated straight into the large-object space (never below
    // kMinLargeObjectBytes, gc_limits.hpp); smaller ones start young. A large
    // allocation is old at once, so if it dies young it still keeps every
    // young object it names alive until the next full collection: a growing
    // array's backing store must not cross this line while it is short-lived.
    size_t large_object_bytes = size_t{1} << 20;
    // Old-generation growth that triggers a full collection: the larger of
    // this and `growth_factor` times the live old bytes after the last one.
    size_t min_full_threshold_bytes = size_t{32} << 20;
    double growth_factor = 1.0;
    // Threads that mark the old generation in a full collection (the
    // collecting thread among them) once it holds at least
    // `parallel_mark_bytes`; smaller old generations are marked by the
    // collecting thread alone. 0: half the hardware threads, at most 8.
    unsigned mark_threads = 0;
    size_t parallel_mark_bytes = size_t{16} << 20;
    // The helper threads stay parked between drains, so a marking cycle's
    // short slices mark on all of them. false (or BRASS_GC_MARK_POOL=0): each
    // drain creates its threads and joins them.
    bool mark_thread_pool = true;
    // A minor collection (and a marking cycle's finishing evacuation of the
    // young generation) copies on several threads, the marker's parked
    // helpers (docs/gc_contract.md, "Parallel Minor Collection"), when it is
    // expected to take at least `parallel_minor_ms` on one thread (its
    // expected copying and card work at what serial ones have cost); shorter
    // ones copy on the collecting thread alone, since parallel copying costs
    // the program after it more than it saves them. Off (or
    // BRASS_GC_PARALLEL_MINOR=0): every minor collection on one thread.
    // Always (BRASS_GC_PARALLEL_MINOR=always): every one on `minor_threads`
    // threads, however small, for tests and stress.
    ParallelMinor parallel_minor = ParallelMinor::Auto;
    // Threads a parallel minor collection copies with (the collecting
    // thread among them). 0: half the hardware threads, at most 8.
    unsigned minor_threads = 0;
    double parallel_minor_ms = 1.0;
    // A full collection's sweep of the old generation is left for after its
    // pause: done by sweep_slice, by allocation as it needs memory, and
    // whatever is left by the next cycle's start (docs/gc_contract.md, "Lazy
    // Sweep"). false (or BRASS_GC_EAGER_SWEEP=1): swept inside the pause.
    bool lazy_sweep = true;
    // A marking cycle's finish (finish_marking, or a full collection asked
    // for while one is open) leaves the young generation where it is and
    // marks it in place with the old one, so young objects live on and only
    // the dead ones are dropped (docs/gc_contract.md, "Marking Cycle"). true
    // (or BRASS_GC_FINISH_PROMOTE_ALL=1): the finish evacuates the young
    // generation, promoting every survivor, as before.
    bool finish_promote_all = false;
    StressMode stress = StressMode::None;
    bool verify = false;  // verify the heap before and after every collection
    bool poison = false;  // overwrite freed memory with 0xDB after every collection
    // A line on stderr per collection (kind, trigger, pause by phase, bytes
    // copied, promoted and marked) and a summary when the heap is destroyed.
    bool log = false;
    // The high-16-bit tags under which a word slot's low 48 bits are a
    // reference. Empty: every tag. Otherwise exactly the tags listed: a raw
    // gcref has tag 0, so a heap that holds raw gcrefs lists 0, and one whose
    // references are all tagged (NaN-boxed values) leaves it out, so raw
    // pointers and small numbers in its slots and registers are never taken
    // for references.
    std::vector<uint16_t> reference_tags;
    // A collection host code starts (allocate, collect, safepoint without a
    // generated caller) also walks the calling thread's stack for generated
    // frames and visits their stack-map roots, as one started from generated
    // code through brass_gc_alloc does. For a host whose generated code calls
    // host functions that allocate on this heap. The walk passes over the
    // compiled frames on the way by their unwind data on Windows x64 and by
    // their frame pointers elsewhere, where that code must keep them.
    bool walk_stack_on_host_collection = false;
    // BRASS_GC_STRESS (minor|full|alternate|incremental|1), BRASS_GC_VERIFY=1,
    // BRASS_GC_POISON=1, BRASS_GC_LOG=1, BRASS_GC_FIXED_EDEN=1,
    // BRASS_GC_EAGER_SWEEP=1, BRASS_GC_MARK_THREADS=n,
    // BRASS_GC_PARALLEL_MINOR=0|1|always, BRASS_GC_MINOR_THREADS=n and
    // BRASS_GC_OLD_RESERVE_MB=n (both old reservations) override the fields
    // above when set.
    bool read_environment = true;
};

struct HeapStats {
    uint64_t minor_collections = 0;
    uint64_t full_collections = 0;
    uint64_t minor_pause_ns_total = 0;
    uint64_t minor_pause_ns_max = 0;
    uint64_t full_pause_ns_total = 0;
    uint64_t full_pause_ns_max = 0;
    uint64_t last_pause_ns = 0;
    uint64_t promoted_bytes = 0;       // copied into the old generation, all time
    uint64_t last_survived_bytes = 0;  // copied by the last collection
    uint64_t old_live_bytes = 0;       // old generation after the last full collection
    // Young-generation evacuations (minor collections, and full collections'
    // and marking cycles' finishing ones) that copied on several threads.
    uint64_t parallel_evacuations = 0;
    // The finalizers of the objects the last collection found dead, run once
    // it had finished (not part of last_pause_ns), and how long they took.
    uint64_t last_finalizers = 0;
    uint64_t last_finalizer_ns = 0;
};

// How near each generation is to the collection its own trigger would run: a
// minor one when eden is full, a full one when the old generation has grown
// past its threshold. A host that knows when a pause costs nothing (between
// frames) collects ahead of the trigger rather than at an allocation.
struct HeapPressure {
    size_t eden_used = 0;
    // The eden a minor collection runs at: the adaptive limit
    // (HeapConfig::adaptive_eden), at most the eden reserved.
    size_t eden_capacity = 0;
    uint64_t old_growth = 0;      // old-generation bytes added since the last full collection
    uint64_t full_threshold = 0;  // the growth that runs a full collection
    // The last full collection's sweep still to do (Heap::sweep_slice): the
    // mature blocks not yet swept (about a microsecond each), and whether
    // anything at all is left (large objects, memory to hand back).
    size_t unswept_blocks = 0;
    bool sweeping = false;
};

class Heap {
public:
    // With no configuration: the process's default (set_default_config).
    Heap() : Heap(default_config()) {}
    explicit Heap(const HeapConfig& config);
    ~Heap();
    Heap(const Heap&) = delete;
    Heap& operator=(const Heap&) = delete;

    // The configuration of heaps created without one, including those an
    // interpreter creates for itself: HeapConfig{} until a host sets it.
    static HeapConfig default_config();
    static void set_default_config(const HeapConfig& config);

    // ---- The thread's heap -------------------------------------------------
    // The heap generated code and brass's interpreters allocate from on the
    // calling thread (null: none). Prefer HeapScope.
#if defined(_MSC_VER)
    static Heap* current() noexcept;
    static void set_current(Heap* heap) noexcept;
#else
    __attribute__((visibility("default"))) static Heap* current() noexcept;
    __attribute__((visibility("default"))) static void set_current(Heap* heap) noexcept;
#endif

    // ---- Allocation --------------------------------------------------------
    // A zeroed object of at least `bytes` payload bytes (see
    // payload_bytes_for) and layout `layout`. May collect first; every
    // reference the caller holds across the call must be a root. Throws
    // std::bad_alloc when the reservation is exhausted.
    uintptr_t allocate(size_t bytes, LayoutId layout, uint32_t flags = 0, uint8_t host_bits = 0) {
        if (flags == 0) {
            if (const uintptr_t obj = try_allocate_young(bytes, layout, host_bits)) return obj;
        }
        return allocate_at(bytes, layout, flags, host_bits, 0, 0);
    }
    // The bump fast path alone: a zeroed young object when it fits in the
    // allocation buffer, else 0. Never collects, so it needs no roots.
    uintptr_t try_allocate_young(size_t bytes, LayoutId layout, uint8_t host_bits = 0) noexcept {
        const size_t total = payload_bytes_for(bytes) + kHeaderBytes;
        if (total > max_young_total_ || total > static_cast<size_t>(alloc_->end - alloc_->top)) return 0;
        const uintptr_t at = alloc_->top;
        alloc_->top = at + total;
        auto* header = reinterpret_cast<ObjectHeader*>(at);
        header->size = static_cast<uint32_t>(total - kHeaderBytes);
        header->layout = layout;
        header->gc_bits = 0;
        header->host_bits = host_bits;
        zero_payload(at + kHeaderBytes, total - kHeaderBytes);
        return at + kHeaderBytes;
    }
    // Zeroes a fresh payload. Most objects are a few words, and a size known
    // only at run time (brass_gc_alloc's) would otherwise cost a call into the
    // C library's memset per allocation, which is most of an allocation.
    static void zero_payload(uintptr_t payload, size_t bytes) noexcept {
        auto* w = reinterpret_cast<uint64_t*>(payload);
        switch (bytes >> 3) {
            case 6: w[5] = 0; [[fallthrough]];
            case 5: w[4] = 0; [[fallthrough]];
            case 4: w[3] = 0; [[fallthrough]];
            case 3: w[2] = 0; [[fallthrough]];
            case 2: w[1] = 0; [[fallthrough]];
            case 1: w[0] = 0; return;
            default: std::memset(w, 0, bytes); return;
        }
    }
    // The Mask layout (mask_layout) for (pointer_mask, type_tag).
    uintptr_t allocate_masked(size_t bytes, uint64_t pointer_mask, uint32_t type_tag) {
        return allocate(bytes, mask_layout(pointer_mask, type_tag));
    }
    // allocate() on behalf of the generated frame (caller_fp, caller_ip): a
    // collection it triggers walks that frame and the generated frames
    // beneath it through the code's stack maps (both 0: no generated frame).
    uintptr_t allocate_at(size_t bytes, LayoutId layout, uint32_t flags, uint8_t host_bits,
                          uintptr_t caller_fp, uintptr_t caller_ip);

    // The bump region generated code may allocate young objects from inline:
    // write the header at `top`, advance `top` by header + payload, zero the
    // payload. `top` never passes `end`; when the object does not fit, call
    // the runtime (brass_gc_alloc). `end` is pulled back to `top` whenever
    // the runtime needs to see every allocation (stress mode).
    struct AllocationBuffer {
        uintptr_t top = 0;
        uintptr_t end = 0;
    };
    AllocationBuffer* allocation_buffer() noexcept { return alloc_; }
    // Moves the bump region into `buffer` (null: back into the heap's own),
    // carrying over its current top and end. A host whose generated code
    // reaches the region through its own per-thread block keeps it there.
    // The buffer must outlive the binding; the heap reads and writes it for
    // every allocation and collection until rebound.
    void bind_allocation_buffer(AllocationBuffer* buffer) noexcept;

    // ---- Collection --------------------------------------------------------
    void collect(CollectionKind kind = CollectionKind::Full) { collect_at(kind, 0, 0); }
    void collect_at(CollectionKind kind, uintptr_t caller_fp, uintptr_t caller_ip);
    // An opportunity to collect: collects when a collection was requested or
    // in stress mode, otherwise returns at once.
    void safepoint_at(uintptr_t caller_fp, uintptr_t caller_ip);
    void request_collection(CollectionKind kind) noexcept;
    [[nodiscard]] bool in_collection() const noexcept;
    // Hands the young generation's unused pages back to the system: eden
    // above its allocation top and each survivor space above its top. They
    // stay committed, and
    // read as zero when next touched. For a host gone idle (after a minor
    // collection, eden is all of it). Returns the bytes released. Not during
    // a collection.
    size_t release_unused_young() noexcept;

    // ---- Marking cycle -----------------------------------------------------
    // A full collection split into steps a host can spread out: start (the
    // roots' old referents made grey), mark slices of about a budget each,
    // finish (the young generation emptied into the old one, the roots
    // again, the rest of the grey work, then weak slots, ephemerons,
    // finalizers, hooks, sweep). collect(Full) is the same steps back to
    // back. Each step is a pause of its own; the collection is counted and
    // logged once, at finish, its pauses together in full_pause_ns_total and
    // the longest one in full_pause_ns_max and last_pause_ns.
    //
    // The mutator runs between the steps: it may store, allocate and
    // collect. The write barrier shades the old references it stores, minor
    // collections shade the old objects they reach and promote, and old
    // allocation is grey, so finish never sweeps what is live
    // (docs/gc_contract.md, "Marking Cycle"). A full collection asked for
    // while a cycle is open finishes it (and, unless old-generation growth
    // asked, then runs a whole one).
    //
    // Starts a cycle: false, doing nothing, if one is already open.
    bool start_marking();
    // Marks for about `budget_ns` (at least a little, however small);
    // true once the grey work is gone and finish would add none. Calling it
    // with no cycle open stops the process.
    bool mark_slice(uint64_t budget_ns);
    // Whether a cycle is open.
    [[nodiscard]] bool marking() const noexcept;
    // Finishes the open cycle (marking whatever grey work is left first),
    // then runs the finalizers it queued. No cycle open stops the process.
    // The old generation's sweep is left pending (HeapConfig::lazy_sweep).
    void finish_marking();

    // ---- Lazy sweep ----------------------------------------------------------
    // After a full collection (or finish_marking) the dead old objects are
    // decided, their weak slots, finalizers and hooks settled, but their
    // memory is freed afterwards: by these slices, by allocation as it needs
    // blocks or pages, and whatever is left by the next cycle's start, in its
    // pause. A host sweeps where a pause costs nothing (between frames).
    // Sweeps for about `budget_ns` (at least a little, however small); true
    // once nothing is left (at once when nothing was). Not during a collection.
    bool sweep_slice(uint64_t budget_ns);
    // Whether some of the last cycle's sweep is left.
    [[nodiscard]] bool sweeping() const noexcept;

    // ---- Barrier -----------------------------------------------------------
    // Every store of a reference into an object that may be old: `object` is
    // the object whose layout traces the slot written, `value` the word
    // stored. Stores into an object allocated young since the last GC point
    // may skip it (MIR's WriteBarrierElimination relies on this).
    //
    // The value is filtered by its own range (barrier_lo_ / barrier_span_):
    // the young generation while no marking cycle is open, so the barrier is
    // the same two range checks and tag check it always was, and the whole
    // heap while one is, so an old object's store of an old reference also
    // reaches the slow path, which shades the value (marks it grey) instead
    // of dirtying a card (docs/gc_contract.md, "Marking Cycle"). The range
    // changes only at a GC point (the cycle's start and finish).
    void write_barrier(uintptr_t object, uint64_t value) noexcept {
        if (object - old_lo_ < old_span_ &&
            (static_cast<uintptr_t>(value & kAddressMask) - barrier_lo_) < barrier_span_ &&
            is_reference_tag(value)) {
            if (is_young(static_cast<uintptr_t>(value & kAddressMask))) dirty_card(object);
            else shade_stored(value);
        }
    }
    // The barrier for a store through `address`, any address inside an object
    // (a derived pointer): the object is looked up only when the store needs
    // remembering. For stores whose object start is not at hand. Given the
    // slot's own address, a store into a large object remembers only the
    // card holding the slot.
    void write_barrier_interior(uintptr_t address, uint64_t value) noexcept {
        if (address - old_lo_ < old_span_ &&
            (static_cast<uintptr_t>(value & kAddressMask) - barrier_lo_) < barrier_span_ &&
            is_reference_tag(value)) {
            remember_interior(address, value);
        }
    }
    // The barrier for a bulk write (a memcpy of many words) into `object`,
    // which must be an object's payload address: when the object is old, its
    // card is dirtied unconditionally, so the next minor collection rescans
    // every slot of it.
    void remember(uintptr_t object) noexcept {
        if (object - old_lo_ < old_span_) dirty_card(object);
    }
    // As remember, for a bulk write into bytes [begin, end) of `object`: a
    // large object has only the cards covering the range rescanned.
    void remember_range(uintptr_t object, uintptr_t begin, uintptr_t end) noexcept {
        if (object - old_lo_ < old_span_) remember_range_old(object, begin, end);
    }
    // The barrier for an edge the collector reaches only through a host's
    // trace function in a full collection (an object's prototype held by its
    // off-heap shape, say), made when that edge is: while a marking cycle is
    // open, an old object `word` names is shaded, since the object holding
    // the edge may already be marked. Nothing otherwise (a young referent is
    // traced by the finish's evacuation, and no cycle needs anything).
    void shade_reference(uint64_t word) noexcept {
        const auto a = static_cast<uintptr_t>(word & kAddressMask);
        if (a - barrier_lo_ < barrier_span_ && !is_young(a) && is_reference_tag(word)) shade_stored(word);
    }
    // Payload word `index` of `object`, and a store to it with its barrier.
    [[nodiscard]] static uint64_t load(uintptr_t object, size_t index) noexcept {
        return reinterpret_cast<const uint64_t*>(object)[index];
    }
    void store(uintptr_t object, size_t index, uint64_t value) noexcept {
        reinterpret_cast<uint64_t*>(object)[index] = value;
        write_barrier(object, value);
    }
    [[nodiscard]] bool is_young(uintptr_t address) const noexcept { return address - young_lo_ < young_span_; }
    // Whether the slot word `word` is a reference to a young object: its
    // address is young and its tag one of the reference tags.
    [[nodiscard]] bool holds_young(uint64_t word) const noexcept {
        return is_young(static_cast<uintptr_t>(word & kAddressMask)) && is_reference_tag(word);
    }
    [[nodiscard]] bool is_old(uintptr_t address) const noexcept { return address - old_lo_ < old_span_; }
    // Whether `address` lies inside this heap's reservation.
    [[nodiscard]] bool contains(uintptr_t address) const noexcept { return address - heap_lo_ < heap_span_; }
    // The reservation: every object this heap ever holds lies in
    // [reservation_base(), reservation_base() + reservation_bytes()).
    [[nodiscard]] uintptr_t reservation_base() const noexcept { return heap_lo_; }
    [[nodiscard]] size_t reservation_bytes() const noexcept { return heap_span_; }
    [[nodiscard]] bool is_reference_tag(uint64_t word) const noexcept {
        if (!ref_tags_) return true;
        const uint64_t tag = word >> kTagShift;
        return ((ref_tags_[tag >> 6] >> (tag & 63)) & 1) != 0;
    }
    // Whether a raw address (tag 0) in a slot is a reference to this heap.
    // A heap whose reference tags leave 0 out cannot update a raw address
    // when the object moves: what brass itself holds raw (coroutine frames)
    // it allocates in the old generation there, which never moves.
    [[nodiscard]] bool raw_addresses_are_references() const noexcept { return raw_address_tag_ == 0; }
    // The high bits of a tag under which this heap takes a word for a
    // reference (already shifted into place): 0 when raw addresses are
    // references, else the first of the configured tags. A slot brass keeps
    // for a raw address stores it under this tag, so the ordinary visit
    // traces it on every heap; readers mask the tag off (kAddressMask).
    [[nodiscard]] uint64_t raw_address_tag() const noexcept { return raw_address_tag_; }
    // Card of old-generation address `a`: card_table_base()[(a - old_base()) >> kCardShift].
    // Code that dirties a card itself must also set the card's region byte,
    // card_region_base()[(a - old_base()) >> kCardRegionShift] = 1, or a minor
    // collection will not look at the card.
    [[nodiscard]] uint8_t* card_table_base() const noexcept { return cards_; }
    [[nodiscard]] uint8_t* card_region_base() const noexcept { return regions_; }
    [[nodiscard]] uintptr_t old_base() const noexcept { return old_lo_; }
    [[nodiscard]] uintptr_t young_base() const noexcept { return young_lo_; }
    [[nodiscard]] size_t young_span() const noexcept { return young_span_; }
    // The range of stored values the barrier takes its slow path for: the
    // young generation, or the whole heap while a marking cycle is open.
    [[nodiscard]] uintptr_t barrier_value_base() const noexcept { return barrier_lo_; }
    [[nodiscard]] size_t barrier_value_span() const noexcept { return barrier_span_; }

    // ---- Roots -------------------------------------------------------------
    // A slot outside the heap holding a word (a gcref or tagged reference),
    // visited by every collection until removed.
    void add_root(uint64_t* slot);
    void remove_root(uint64_t* slot);

    // A callback that visits a set of roots (a table, an interpreter's
    // frames) at every collection. Returns an id for remove_root_source.
    using RootSourceFn = void (*)(Tracer& tracer, void* context);
    using RootSourceId = uint64_t;
    RootSourceId add_root_source(RootSourceFn fn, void* context);
    RootSourceId add_root_source(std::function<void(Tracer&)> fn);
    void remove_root_source(RootSourceId id) noexcept;

    // ---- Weakness and finalization ------------------------------------------
    // Runs inside every collection once liveness is decided and weak slots
    // are settled, before any memory is reclaimed: the one window in which
    // survivor_of answers. It must not allocate on this heap.
    using PostCollectionHook = std::function<void(Heap&, CollectionKind)>;
    RootSourceId add_post_collection_hook(PostCollectionHook hook);
    void remove_post_collection_hook(RootSourceId id) noexcept;
    // Inside a post-collection hook: where the object at `object` before this
    // collection is now, or 0 if it died. An address this collection did not
    // collect (outside the heap, or old during a minor collection) is
    // returned unchanged.
    [[nodiscard]] uintptr_t survivor_of(uintptr_t object) const noexcept;
    // Calls fn(context) once, after the collection in which `object` is found
    // dead (never with the object itself, which is gone by then). Callbacks
    // run on this thread after the collection completes and may allocate; a
    // collection a callback causes queues what it finds dead for the same
    // run, after the batch under way, rather than running it inside the
    // callback.
    void add_finalizer(uintptr_t object, void (*fn)(void* context), void* context);
    // The same with two context words, fn(a, b): a host whose finalizer is a
    // (data, destructor) pair per object passes both by value instead of
    // allocating a record per object to point `context` at.
    void add_finalizer(uintptr_t object, void (*fn)(void* a, void* b), void* a, void* b);

    // ---- Queries -----------------------------------------------------------
    // Whether `object` is the payload address of an object this heap holds.
    [[nodiscard]] bool is_valid_object(uintptr_t object) const noexcept;
    // The payload address of the object whose header or payload contains
    // `address`, or 0.
    [[nodiscard]] uintptr_t find_object(uintptr_t address) const noexcept;
    // Throws std::runtime_error unless [base + offset, + size) lies inside one
    // object whenever `base` lies inside the heap (the interpreters' checked
    // memory access). `what` names the access in the message.
    void check_access(uintptr_t base, int64_t offset, size_t size, const char* what) const;
    [[nodiscard]] const LayoutDescriptor& layout_of(uintptr_t object) const noexcept {
        return layout_descriptor(header_of(object)->layout);
    }
    [[nodiscard]] uint32_t type_tag_of(uintptr_t object) const noexcept { return layout_of(object).type_tag; }
    [[nodiscard]] static size_t object_size(uintptr_t object) noexcept { return header_of(object)->size; }
    // Whether the object may move in a later collection.
    [[nodiscard]] bool is_movable(uintptr_t object) const noexcept;
    // Every object the heap holds (live or not yet found dead: an object the
    // last full collection found dead is not reported while the sweep that
    // frees it is pending), in no particular order. `fn` must not allocate on
    // this heap.
    void for_each_object(const std::function<void(uintptr_t object)>& fn) const;

    // ---- Modes -------------------------------------------------------------
    void set_stress(StressMode mode) noexcept;
    [[nodiscard]] StressMode stress() const noexcept;
    void set_verify(bool enable) noexcept;
    // Whether collections verify (set_verify, BRASS_GC_VERIFY=1). A root
    // source that skips slots in a minor collection checks the skipped ones
    // when this is on (docs/gc_contract.md, "Roots of a Collection").
    [[nodiscard]] bool verifying() const noexcept;
    void set_poison(bool enable) noexcept;
    // Checks every object and root now; a violation stops the process with a
    // message naming the object and slot.
    void verify();

    // ---- Statistics --------------------------------------------------------
    [[nodiscard]] const HeapStats& stats() const noexcept;
    [[nodiscard]] uint64_t collection_count() const noexcept;
    // Incremented once per object a collection moves: anything keyed on an
    // object's address (an identity hash table) is stale once it changes.
    [[nodiscard]] uint64_t relocation_epoch() const noexcept;
    [[nodiscard]] size_t young_used_bytes() const noexcept;
    [[nodiscard]] HeapPressure pressure() const noexcept;
    [[nodiscard]] size_t old_used_bytes() const noexcept;
    [[nodiscard]] size_t committed_bytes() const noexcept;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;

    // The coroutine frames allocated here (runtime/coroutine.hpp): roots of
    // this heap's collections while suspended.
    runtime::CoroFrameRegistry& coro_frames();

private:
    friend struct detail::Collector;
    friend struct detail::HeapState;

    // The barrier's slow path: a young value dirties the slot's card, an old
    // one (only while marking) is shaded.
    void remember_interior(uintptr_t address, uint64_t value) noexcept;
    // Shades old reference `value` stored while a marking cycle is open
    // (heap_mark_incremental.cpp). Never collects or allocates on the heap.
    void shade_stored(uint64_t value) noexcept;
    // Sets the barrier's value range for the marking phase (at a GC point).
    void set_barrier_marking(bool marking) noexcept {
        barrier_lo_ = marking ? heap_lo_ : young_lo_;
        barrier_span_ = marking ? heap_span_ : young_span_;
    }
    // Runs the finalizers collections have queued (after collect_at and
    // finish_marking), unless a batch is already running.
    void run_pending_finalizers();
    // What the stress mode does at an allocation or safepoint.
    void stress_step(uintptr_t caller_fp, uintptr_t caller_ip);
    void remember_range_old(uintptr_t object, uintptr_t begin, uintptr_t end) noexcept;
    // Dirties the card of old address `a` and marks its region (kCardRegionShift).
    void dirty_card(uintptr_t a) noexcept {
        const uintptr_t off = a - old_lo_;
        cards_[off >> kCardShift] = kCardDirty;
        regions_[off >> kCardRegionShift] = 1;
    }

    AllocationBuffer own_alloc_;
    AllocationBuffer* alloc_ = &own_alloc_;
    size_t max_young_total_ = 0;
    // The barrier's fields first, in one cache line: the value range where
    // the young range sat before it had its own, so the barrier's loads are
    // at the offsets they always were.
    uintptr_t barrier_lo_ = 0;
    uintptr_t barrier_span_ = 0;
    uintptr_t old_lo_ = 0;
    uintptr_t old_span_ = 0;
    uintptr_t heap_lo_ = 0;
    uintptr_t heap_span_ = 0;
    uint8_t* cards_ = nullptr;
    uint8_t* regions_ = nullptr;
    const uint64_t* ref_tags_ = nullptr;
    uint64_t raw_address_tag_ = 0;
    uintptr_t young_lo_ = 0;
    uintptr_t young_span_ = 0;
    std::unique_ptr<detail::HeapState> s_;
};

inline bool Tracer::is_young(const uint64_t* slot) const noexcept { return heap_->holds_young(*slot); }

// Binds `heap` as the calling thread's current heap for the scope's
// lifetime, restoring the previous one after.
class HeapScope {
public:
    explicit HeapScope(Heap& heap) noexcept : previous_(Heap::current()) { Heap::set_current(&heap); }
    explicit HeapScope(Heap* heap) noexcept : previous_(Heap::current()) { Heap::set_current(heap); }
    ~HeapScope() { Heap::set_current(previous_); }
    HeapScope(const HeapScope&) = delete;
    HeapScope& operator=(const HeapScope&) = delete;

private:
    Heap* previous_;
};

} // namespace brass::gc
