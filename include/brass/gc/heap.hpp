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
};

// Allocation flags.
inline constexpr uint32_t kAllocOld = 1u << 0;     // allocate in the old generation (pretenure)
inline constexpr uint32_t kAllocPinned = 1u << 1;  // old, and never moved by any collection

struct HeapConfig {
    size_t eden_bytes = size_t{8} << 20;
    size_t survivor_bytes = size_t{1} << 20;  // each of the two survivor spaces
    // Minor collections an object survives in a survivor space before it is
    // promoted (1: promoted by its first minor collection; at most 7, since
    // an object's age saturates there).
    uint8_t tenure_age = 2;
    // Address space reserved for the mature blocks and the large objects;
    // memory is committed as the heap grows.
    size_t mature_reserve_bytes = size_t{4} << 30;
    size_t large_reserve_bytes = size_t{4} << 30;
    // Payloads above this many bytes are allocated straight into the
    // large-object space (never below kMinLargeObjectBytes, gc_limits.hpp).
    size_t large_object_bytes = 64 * 1024;
    // Old-generation growth that triggers a full collection: the larger of
    // this and `growth_factor` times the live old bytes after the last one.
    size_t min_full_threshold_bytes = size_t{32} << 20;
    double growth_factor = 1.0;
    StressMode stress = StressMode::None;
    bool verify = false;  // verify the heap before and after every collection
    bool poison = false;  // overwrite freed memory with 0xDB after every collection
    // The high-16-bit tags under which a word slot's low 48 bits are a
    // reference. Empty: every tag. Otherwise exactly the tags listed: a raw
    // gcref has tag 0, so a heap that holds raw gcrefs lists 0, and one whose
    // references are all tagged (NaN-boxed values) leaves it out, so raw
    // pointers and small numbers in its slots and registers are never taken
    // for references.
    std::vector<uint16_t> reference_tags;
    // BRASS_GC_STRESS (minor|full|alternate|1), BRASS_GC_VERIFY=1 and
    // BRASS_GC_POISON=1 override the fields above when set.
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
    static Heap* current() noexcept;
    static void set_current(Heap* heap) noexcept;

    // ---- Allocation --------------------------------------------------------
    // A zeroed object of at least `bytes` payload bytes (see
    // payload_bytes_for) and layout `layout`. May collect first; every
    // reference the caller holds across the call must be a root. Throws
    // std::bad_alloc when the reservation is exhausted.
    uintptr_t allocate(size_t bytes, LayoutId layout, uint32_t flags = 0, uint8_t host_bits = 0) {
        const size_t total = payload_bytes_for(bytes) + kHeaderBytes;
        if (flags == 0 && total <= max_young_total_ &&
            total <= static_cast<size_t>(alloc_->end - alloc_->top)) {
            const uintptr_t at = alloc_->top;
            alloc_->top = at + total;
            auto* header = reinterpret_cast<ObjectHeader*>(at);
            header->size = static_cast<uint32_t>(total - kHeaderBytes);
            header->layout = layout;
            header->gc_bits = 0;
            header->host_bits = host_bits;
            std::memset(reinterpret_cast<void*>(at + kHeaderBytes), 0, total - kHeaderBytes);
            return at + kHeaderBytes;
        }
        return allocate_at(bytes, layout, flags, host_bits, 0, 0);
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

    // ---- Barrier -----------------------------------------------------------
    // Every store of a reference into an object that may be old: `object` is
    // the object whose layout traces the slot written, `value` the word
    // stored. Stores into an object allocated young since the last GC point
    // may skip it (MIR's WriteBarrierElimination relies on this).
    void write_barrier(uintptr_t object, uint64_t value) noexcept {
        if (object - old_lo_ < old_span_ &&
            (static_cast<uintptr_t>(value & kAddressMask) - young_lo_) < young_span_ &&
            is_reference_tag(value)) {
            cards_[(object - old_lo_) >> kCardShift] = kCardDirty;
        }
    }
    // The barrier for a store through `address`, any address inside an object
    // (a derived pointer): the object is looked up only when the store needs
    // remembering. For stores whose object start is not at hand.
    void write_barrier_interior(uintptr_t address, uint64_t value) noexcept {
        if (address - old_lo_ < old_span_ &&
            (static_cast<uintptr_t>(value & kAddressMask) - young_lo_) < young_span_ &&
            is_reference_tag(value)) {
            remember_interior(address);
        }
    }
    // The barrier for a bulk write (a memcpy of many words) into `object`,
    // which must be an object's payload address: when the object is old, its
    // card is dirtied unconditionally, so the next minor collection rescans
    // every slot of it.
    void remember(uintptr_t object) noexcept {
        if (object - old_lo_ < old_span_) cards_[(object - old_lo_) >> kCardShift] = kCardDirty;
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
    // Card of old-generation address `a`: card_table_base()[(a - old_base()) >> kCardShift].
    [[nodiscard]] uint8_t* card_table_base() const noexcept { return cards_; }
    [[nodiscard]] uintptr_t old_base() const noexcept { return old_lo_; }
    [[nodiscard]] uintptr_t young_base() const noexcept { return young_lo_; }
    [[nodiscard]] size_t young_span() const noexcept { return young_span_; }

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
    // run on this thread after the collection completes and may allocate.
    void add_finalizer(uintptr_t object, void (*fn)(void* context), void* context);

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
    // Every object the heap holds (live or not yet reclaimed), in no
    // particular order. `fn` must not allocate on this heap.
    void for_each_object(const std::function<void(uintptr_t object)>& fn) const;

    // ---- Modes -------------------------------------------------------------
    void set_stress(StressMode mode) noexcept;
    [[nodiscard]] StressMode stress() const noexcept;
    void set_verify(bool enable) noexcept;
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
    [[nodiscard]] size_t old_used_bytes() const noexcept;
    [[nodiscard]] size_t committed_bytes() const noexcept;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;

    // The coroutine frames allocated here (runtime/coroutine.hpp): roots of
    // this heap's collections while suspended.
    runtime::CoroFrameRegistry& coro_frames();

private:
    friend struct detail::Collector;
    friend struct detail::HeapState;

    void remember_interior(uintptr_t address) noexcept;

    AllocationBuffer own_alloc_;
    AllocationBuffer* alloc_ = &own_alloc_;
    size_t max_young_total_ = 0;
    uintptr_t young_lo_ = 0;
    uintptr_t young_span_ = 0;
    uintptr_t old_lo_ = 0;
    uintptr_t old_span_ = 0;
    uintptr_t heap_lo_ = 0;
    uintptr_t heap_span_ = 0;
    uint8_t* cards_ = nullptr;
    const uint64_t* ref_tags_ = nullptr;
    std::unique_ptr<detail::HeapState> s_;
};

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
