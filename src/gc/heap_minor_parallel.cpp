// The young generation's evacuation on several threads: a minor collection,
// and a full collection's (or a marking cycle's finish's) emptying of the
// young generation, when it is large enough to pay for waking the helpers.
// The serial evacuation is heap_collect.cpp's; this one decides the same
// things (what survives, where it goes, what stays remembered) and leaves the
// heap in the same state but for where in the survivor space and the old
// generation each copy lands.
//
// Work. The dirty cards are gathered first, before anything is copied, and
// cleaned (heap_cards.cpp, gather_dirty_cards): a list of pieces (a mature
// card's object starts, or a range of a large object, large arrays cut into
// 32 KB pieces) that the threads take a few at a time through an atomic
// cursor. Gathering first means a card's objects are the ones that were there
// before any promotion, so a promoted object is scanned exactly once (as
// grey work) and no thread reads the starts or marks a promotion is writing.
// The roots are visited on the calling thread (root sources are host code,
// called on the heap's thread only) before the helpers start; what they copy
// is its grey work, which it shares like any other. Each thread keeps its own
// grey stack of copied objects still to scan, scans depth first, and hands a
// chunk to a shared pool when another thread is idle, as the marker does
// (heap_mark.cpp); the evacuation ends when every thread is idle, the pool is
// empty and the cards are all taken.
//
// Forwarding. A from-space object is claimed by a compare-and-swap of its
// header's gc_bits from unforwarded to kGcBusy; the winner copies it, writes
// the new address into its payload word 0, then publishes kGcForwarded with a
// release store. A thread that meets a forwarded object reads the address
// after an acquire load of the bits; one that meets a busy one waits (a copy
// is a memcpy). So exactly one copy of each object exists, and every slot
// naming it is rewritten to that copy. Each slot is written by one thread
// only: the thread scanning the object that holds it, and every object is
// scanned once (a card's objects by the thread that took the card, a copy by
// the thread whose stack it is on; roots by the calling thread alone).
//
// Copying buffers. A survivor copy goes into its thread's buffer: 16 KB of
// the to-space (less in a small one), taken by an atomic bump of the space's
// top and aligned to 512 bytes, one word of the young start bits, so each
// thread sets the start bits of its own words with plain stores. An object
// that does not fit what is left of the space is promoted, as when the serial
// copy finds the space full (survivor overflow). The unused end of each
// buffer is slack the adaptive eden's survival sample leaves out
// (HeapState::survivor_slack). A promotion goes into its thread's own pair of
// mature regions (holes of a recyclable block, whole free blocks for medium
// objects) or a large object, taken under one lock; a block belongs to one
// region at a time, so its start bits have one writer, stored atomically for
// the other threads' reads. The allocators never sweep while this runs
// (HeapState::parallel_promotion): promotion takes swept blocks, free page
// runs and the frontiers, so it lands only in swept space as the lazy sweep
// requires (a large object ahead of the sweep's cursor still marked as it is
// made), and the block table and the large pages' table are reserved up front
// so that growing them never moves them under a reader. A heap whose
// frontiers lack the room for all it could promote evacuates on one thread
// (parallel_minor_threads), where the allocator may sweep or complete the
// sweep to find it.
//
// Remembering. A slot of an old object that still names a young one once
// visited dirties its card, as in the serial copy, with atomic stores since
// two threads may dirty one card (an old object and a promoted one starting
// in it); a large object's first card goes from clean to kCardDirtyRange by a
// compare-and-swap, so it never overwrites kCardDirty.
//
// During a marking cycle (GcState::shade) the tracers' range is the heap and
// every old object an old object names is shaded, and every promoted object
// is marked as it is copied: mark bits claimed by an atomic or, since the
// threads shade side by side, and the shaded objects pushed grey onto the
// cycle once the threads are done (cycle_add_grey). The weak slots and
// ephemerons of old objects naming old targets go to the cycle the same way.
// An old address in a block or pages made since the collection began is an
// object this collection promoted, which its promoter shaded.
//
// Then, on the calling thread with the helpers parked: the ephemeron
// fixpoint (an entry whose key turned out alive traces its value, and the
// threads drain again), weak slots, ephemerons and finalizers settled
// exactly as in the serial copy (heap_weak.cpp).

#include "heap_internal.hpp"
#include "heap_mark_threads.hpp"

#include <brass/core/atomic_ref.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace brass::gc::detail {

namespace {

constexpr size_t kBufferAlign = 64 * kGranuleBytes;  // one word of young start bits
constexpr size_t kBufferBytes = 16 * 1024;
constexpr size_t kChunk = 128;          // grey entries handed between threads at once
constexpr unsigned kShareEvery = 32;    // objects scanned between looks at whether a thread idles
constexpr size_t kCardsAtOnce = 16;     // card work pieces a thread takes at once


inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_ARM64)
    __yield();
#else
    _mm_pause();
#endif
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

struct ParMinor;

// Lines apart: a thread writes its tracer's owner and its counters for every
// object, and two threads' workers allocated side by side would share a cache
// line otherwise (adjacent-line prefetch counted).
constexpr size_t kApart = 128;

struct alignas(kApart) Worker {
    ParMinor& pm;
    Tracer tracer;
    std::vector<uintptr_t> stack;
    std::vector<WeakEntry> weak;
    std::vector<EphemeronEntry> ephemerons;
    // During a cycle: what this thread shaded, and old owners' weak slots and
    // ephemerons naming old targets, handed to the cycle afterwards.
    std::vector<uintptr_t> shaded;
    std::vector<WeakEntry> cycle_weak;
    std::vector<EphemeronEntry> cycle_ephemerons;
    uintptr_t buffer_cursor = 0;  // the survivor buffer
    uintptr_t buffer_limit = 0;
    BumpRegion small;   // the mature regions it promotes into
    BumpRegion medium;
    uint64_t copied_bytes = 0;
    uint64_t copied_objects = 0;
    uint64_t promoted_bytes = 0;
    uint64_t promoted_objects = 0;
    uint64_t tenured_bytes = 0;
    uint64_t slack = 0;
    bool overflow = false;
    uint64_t cards_done_ns = 0;  // when it found the card work all taken (logging only)

    Worker(ParMinor& p, Heap& heap, uintptr_t lo, uintptr_t hi);
};

struct ParMinor {
    // Read by every thread, written only before they start.
    GcState& g;
    HeapState& s;
    std::vector<CardWork> cards;
    unsigned threads = 1;  // threads draining (the termination count)
    uintptr_t survivor_end = 0;
    size_t buffer_bytes = kBufferBytes;
    // Blocks and large-object pages that existed when the collection began:
    // an old address past them is an object this collection promoted.
    uint32_t block_bound = 0;
    uint32_t page_bound = 0;
    std::vector<std::unique_ptr<Worker>> workers;
    // Written while they run, each on lines of its own.
    alignas(kApart) std::atomic<size_t> card_next{0};
    alignas(kApart) std::atomic<uintptr_t> survivor_top{0};
    alignas(kApart) std::atomic<size_t> pool_size{0};
    std::atomic<unsigned> idle{0};
    alignas(kApart) std::mutex pool_mutex;
    std::vector<std::vector<uintptr_t>> pool;
    // The old spaces' allocators, block lists and frontiers.
    alignas(kApart) std::mutex alloc_mutex;

    explicit ParMinor(GcState& state) : g(state), s(state.s) {}
};

Worker& worker_of(Tracer& t) noexcept { return *static_cast<Worker*>(t.state()); }

// ---- remembering, with stores other threads may race ----

void store_card(uint8_t& card, uint8_t value) noexcept {
    AtomicRef<uint8_t> c(card);
    if (c.load(std::memory_order_relaxed) != value) c.store(value, std::memory_order_relaxed);
}

void mark_region(HeapState& s, uintptr_t a) noexcept {
    store_card(s.regions[(a - s.mature_lo) >> kCardRegionShift], 1);
}

// HeapState::mark_card(a, kCardDirty).
void dirty(HeapState& s, uintptr_t a) noexcept {
    store_card(*s.card_of(a), kCardDirty);
    mark_region(s, a);
}

// HeapState::remember_slot: the first card of a large object goes from clean
// to kCardDirtyRange only, so a kCardDirty another thread stored stays.
void remember(HeapState& s, uintptr_t owner, const void* slot) noexcept {
    if (!s.in_large(owner)) {
        dirty(s, owner);
        return;
    }
    const auto at = reinterpret_cast<uintptr_t>(slot);
    uint8_t* first = s.card_of(owner);
    if (at - owner >= header_of(owner)->size) {
        dirty(s, owner);
        return;
    }
    if (s.card_of(at) != first) {
        dirty(s, at);
        return;
    }
    AtomicRef<uint8_t> c(*first);
    uint8_t cur = c.load(std::memory_order_relaxed);
    while (cur == kCardClean &&
           !c.compare_exchange_weak(cur, kCardDirtyRange, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    mark_region(s, at);
}

// ---- shading during a cycle, with claims other threads may race ----

void push_shaded(Worker& w, uintptr_t a) { w.shaded.push_back(a); }

// shade(s, a) for an old object met through an old object's slot.
void shade_old(Worker& w, uintptr_t a) {
    ParMinor& pm = w.pm;
    HeapState& s = pm.s;
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= pm.block_bound) return;  // promoted by this collection, and shaded then
        BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        const uint64_t bit = uint64_t{1} << (granule & 63);
        if ((AtomicRef<uint64_t>(meta.starts[granule >> 6]).load(std::memory_order_relaxed) & bit) == 0) return;
        AtomicRef<uint64_t> marks(meta.marks[granule >> 6]);
        if (marks.load(std::memory_order_relaxed) & bit) return;
        if (marks.fetch_or(bit, std::memory_order_relaxed) & bit) return;
        push_shaded(w, a);
    } else if (s.in_large(a)) {
        const size_t page = (a - s.large_lo) / kPageBytes;
        if (page >= pm.page_bound) return;  // promoted by this collection
        const uint32_t head_plus_one = s.large_page_head[page];
        if (head_plus_one == 0 || s.large_object_at(head_plus_one - 1) != a) return;
        AtomicRef<uint8_t> bits(header_of(a)->gc_bits);
        if (bits.load(std::memory_order_relaxed) & kGcLargeMarked) return;
        if (bits.fetch_or(kGcLargeMarked, std::memory_order_relaxed) & kGcLargeMarked) return;
        push_shaded(w, a);
    }
}

// is_marked(s, a), for an old key.
bool marked(const ParMinor& pm, uintptr_t a) noexcept {
    const HeapState& s = pm.s;
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= pm.block_bound) return true;  // promoted, so marked
        BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        const uint64_t word = AtomicRef<uint64_t>(meta.marks[granule >> 6]).load(std::memory_order_relaxed);
        return ((word >> (granule & 63)) & 1) != 0;
    }
    if (s.in_large(a)) {
        if ((a - s.large_lo) / kPageBytes >= pm.page_bound) return true;
        return (AtomicRef<uint8_t>(header_of(a)->gc_bits).load(std::memory_order_relaxed) & kGcLargeMarked) != 0;
    }
    return true;
}

// ---- allocation ----

// A survivor-space copy of `total` bytes, or 0 when the space has no room.
uintptr_t survivor_allocate(Worker& w, size_t total) {
    if (w.buffer_limit - w.buffer_cursor >= total) {
        const uintptr_t at = w.buffer_cursor;
        w.buffer_cursor += total;
        return at;
    }
    ParMinor& pm = w.pm;
    const size_t need = (total + kBufferAlign - 1) & ~(kBufferAlign - 1);
    // A large object takes a piece of its own; the buffer keeps its room.
    const bool own = need > pm.buffer_bytes / 4;
    const size_t want = own ? need : pm.buffer_bytes;
    uintptr_t top = pm.survivor_top.load(std::memory_order_relaxed);
    for (;;) {
        const size_t take = std::min<size_t>(want, pm.survivor_end - top);
        if (take < need) return 0;
        if (pm.survivor_top.compare_exchange_weak(top, top + take, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            if (own) {
                w.slack += take - total;
                return top;
            }
            w.slack += w.buffer_limit - w.buffer_cursor;
            w.buffer_cursor = top + total;
            w.buffer_limit = top + take;
            return top;
        }
    }
}

[[noreturn]] void outgrew_reservation() {
    gc_fatal("a parallel minor collection's promotion outgrew the block or page table room it reserved");
}

// A block for `region` from the shared lists or the frontier, under the lock.
bool next_block(Worker& w, BumpRegion& region, bool whole_block) {
    ParMinor& pm = w.pm;
    HeapState& s = pm.s;
    std::lock_guard<std::mutex> lock(pm.alloc_mutex);
    if (!whole_block && !s.recyclable_blocks.empty()) {
        region.block = s.recyclable_blocks.back();
        s.recyclable_blocks.pop_back();
        region.next_line = 0;
        region.cursor = region.limit = 0;
        return true;
    }
    // The table must not grow past what was reserved: another thread may be
    // reading it.
    if (s.free_blocks.empty() && s.blocks.size() == s.blocks.capacity()) outgrew_reservation();
    const uint32_t index = s.acquire_free_block();
    if (index == UINT32_MAX) {
        region = BumpRegion{};
        return false;
    }
    region.block = index;
    if (whole_block) {
        region.cursor = s.block_base(index);
        region.limit = region.cursor + kBlockBytes;
        region.next_line = static_cast<uint32_t>(kLinesPerBlock);
    } else {
        region.next_line = 0;
        region.cursor = region.limit = 0;
    }
    return true;
}

// HeapState::refill_region for a thread's own region: the next hole of the
// block it holds needs no lock (the block is its alone), another block does.
bool refill(Worker& w, BumpRegion& region, bool whole_block) {
    HeapState& s = w.pm.s;
    if (whole_block) return next_block(w, region, true);
    for (;;) {
        if (region.block != UINT32_MAX) {
            const BlockMeta& meta = *s.blocks[region.block];
            uint32_t line = region.next_line;
            while (line < kLinesPerBlock && meta.line_used[line]) ++line;
            if (line < kLinesPerBlock) {
                uint32_t end = line + 1;
                while (end < kLinesPerBlock && !meta.line_used[end]) ++end;
                region.cursor = s.block_base(region.block) + line * kLineBytes;
                region.limit = s.block_base(region.block) + end * kLineBytes;
                region.next_line = end;
                return true;
            }
        }
        if (!next_block(w, region, false)) return false;
    }
}

// HeapState::mature_allocate in the thread's own regions.
uintptr_t mature_allocate(Worker& w, size_t total) {
    HeapState& s = w.pm.s;
    BumpRegion* region = &w.small;
    if (total > kLineBytes && (w.small.limit - w.small.cursor) < total) region = &w.medium;
    while (region->limit - region->cursor < total) {
        if (!refill(w, *region, region == &w.medium)) return 0;
    }
    const uintptr_t header = region->cursor;
    region->cursor += total;
    BlockMeta& meta = *s.blocks[region->block];
    const uintptr_t lo = s.block_base(region->block);
    const size_t granule = (header + kHeaderBytes - lo) / kGranuleBytes;
    // One writer (the block is this region's), other threads' atomic reads.
    AtomicRef<uint64_t> starts(meta.starts[granule >> 6]);
    starts.store(starts.load(std::memory_order_relaxed) | (uint64_t{1} << (granule & 63)), std::memory_order_relaxed);
    s.mark_lines(meta, lo, header, total, false);
    return header;
}

uintptr_t old_allocate(Worker& w, size_t total) {
    if (total <= kMaxMediumObjectBytes) return mature_allocate(w, total);
    ParMinor& pm = w.pm;
    HeapState& s = pm.s;
    std::lock_guard<std::mutex> lock(pm.alloc_mutex);
    const size_t pages = (total + kPageBytes - 1) / kPageBytes;
    if (s.large_frontier + pages > s.large_page_head.capacity() ||
        s.large_frontier + pages > s.large_page_committed.capacity()) {
        outgrew_reservation();
    }
    return s.large_allocate(total);
}

// ---- copying ----

// evacuate_young's copy, for the thread that claimed the object (`bits`: its
// gc_bits before the claim).
uintptr_t copy(Worker& w, uintptr_t object, uint8_t bits) {
    ParMinor& pm = w.pm;
    GcState& g = pm.g;
    HeapState& s = pm.s;
    ObjectHeader* h = header_of(object);
    const size_t total = kHeaderBytes + h->size;
    const unsigned age = std::min<unsigned>(((bits & kGcAgeMask) >> kGcAgeShift) + 1u, 7u);
    const bool young_enough = age < g.tenure_age;
    uintptr_t dest = young_enough ? survivor_allocate(w, total) : 0;
    const bool promoted = dest == 0;
    if (promoted) {
        if (young_enough) w.overflow = true;
        dest = old_allocate(w, total);
        if (dest == 0) gc_fatal("the old generation's reservation is exhausted while promoting a young object");
    }
    std::memcpy(reinterpret_cast<void*>(dest), h, total);
    auto* nh = reinterpret_cast<ObjectHeader*>(dest);
    uint8_t nbits = static_cast<uint8_t>(bits & ~(kGcForwarded | kGcBusy | kGcAgeMask));
    if (!promoted) nbits = static_cast<uint8_t>(nbits | (age << kGcAgeShift));
    nh->gc_bits = nbits;
    const uintptr_t moved = dest + kHeaderBytes;
    if (promoted) {
        // Ahead of a pending sweep's cursor: marked (note_large_allocated).
        s.note_large_allocated(dest);
        if (g.shade) {
            // Grey, as shade marks it; no other thread can name it yet.
            if (s.in_mature(moved)) {
                const uint32_t index = s.block_index(moved);
                BlockMeta& meta = *s.blocks[index];
                const size_t granule = (moved - s.block_base(index)) / kGranuleBytes;
                const uint64_t bit = uint64_t{1} << (granule & 63);
                if ((AtomicRef<uint64_t>(meta.marks[granule >> 6]).fetch_or(bit, std::memory_order_relaxed) & bit) == 0) {
                    push_shaded(w, moved);
                }
            } else if ((nh->gc_bits & kGcLargeMarked) == 0) {
                nh->gc_bits = static_cast<uint8_t>(nh->gc_bits | kGcLargeMarked);
                push_shaded(w, moved);
            }
        }
        w.promoted_bytes += total;
        ++w.promoted_objects;
        if (!s.in_eden(object)) w.tenured_bytes += total;
    } else {
        s.set_young_start(moved);  // a word of this thread's buffer
    }
    w.copied_bytes += total;
    ++w.copied_objects;
    *reinterpret_cast<uintptr_t*>(object) = moved;
    AtomicRef<uint8_t>(h->gc_bits)
        .store(static_cast<uint8_t>((bits & ~kGcBusy) | kGcForwarded), std::memory_order_release);
    w.stack.push_back(moved);
    return moved;
}

// Where from-space object `object` is now, copying it if no thread has.
uintptr_t forward(Worker& w, uintptr_t object) {
    AtomicRef<uint8_t> bits(header_of(object)->gc_bits);
    uint8_t b = bits.load(std::memory_order_acquire);
    unsigned spins = 0;
    for (;;) {
        if (b & kGcForwarded) return *reinterpret_cast<const uintptr_t*>(object);
        if (b & kGcBusy) {
            if (++spins < 1024) cpu_relax();
            else std::this_thread::yield();
            b = bits.load(std::memory_order_acquire);
            continue;
        }
        if (bits.compare_exchange_weak(b, static_cast<uint8_t>(b | kGcBusy), std::memory_order_acquire,
                                       std::memory_order_acquire)) {
            return copy(w, object, b);
        }
    }
}

// live_address for a Minor state, with another thread perhaps copying `a`.
uintptr_t live_now(const GcState& g, uintptr_t a) noexcept {
    if (!g.heap.is_young(a) || !g.in_from(a)) return a;
    const uint8_t b = AtomicRef<uint8_t>(header_of(a)->gc_bits).load(std::memory_order_acquire);
    return (b & kGcForwarded) ? *reinterpret_cast<const uintptr_t*>(a) : 0;
}

// ---- the tracer's slow paths: minor_visit, weak_visit, ephemeron_visit ----

void par_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    Worker& w = worker_of(t);
    GcState& g = w.pm.g;
    if (!g.heap.is_reference_tag(word)) return;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    uintptr_t now = a;
    if (g.in_from(a)) {
        now = forward(w, a);
        *slot = (word & ~kAddressMask) | now;
    } else if (g.shade && !g.heap.is_young(a)) {
        // minor_visit: an old object an old one names is shaded.
        if (t.owner_old()) shade_old(w, a);
        return;
    }
    if (t.owner_old() && g.heap.is_young(now)) remember(g.s, t.owner(), slot);
}

void par_weak(Tracer& t, uint64_t* slot, uint64_t cleared) {
    Worker& w = worker_of(t);
    GcState& g = w.pm.g;
    if (!g.heap.is_reference_tag(*slot)) return;
    if (g.shade && !g.heap.is_young(static_cast<uintptr_t>(*slot & kAddressMask))) {
        if (t.owner_old()) w.cycle_weak.push_back(WeakEntry{slot, cleared, t.owner(), true});
        return;
    }
    w.weak.push_back(WeakEntry{slot, cleared, t.owner(), t.owner_old()});
}

void par_ephemeron(Tracer& t, uint64_t* key, uint64_t* value, uint64_t cleared_key, uint64_t cleared_value) {
    Worker& w = worker_of(t);
    GcState& g = w.pm.g;
    const uint64_t word = *key;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    if (a == 0 || !g.heap.contains(a) || !g.heap.is_reference_tag(word)) {
        t.visit(value);
        return;
    }
    if (g.shade && !g.heap.is_young(a)) {
        if (!marked(w.pm, a) && t.owner_old()) {
            w.cycle_ephemerons.push_back(EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), true});
            if (g.heap.holds_young(*value)) t.visit(value);
            return;
        }
        t.visit(value);
        return;
    }
    // A key another thread is copying right now reads as not yet alive: the
    // fixpoint looks again once every thread is done.
    const uintptr_t live = live_now(g, a);
    if (!live) {
        w.ephemerons.push_back(EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), t.owner_old()});
        return;
    }
    *key = (word & ~kAddressMask) | live;
    if (t.owner_old() && g.heap.is_young(live)) dirty(g.s, t.owner());
    t.visit(value);
}

Worker::Worker(ParMinor& p, Heap& heap, uintptr_t lo, uintptr_t hi)
    : pm(p), tracer(heap, Tracer::Purpose::Minor, lo, hi, &par_visit, &par_weak, &par_ephemeron, this) {}

// ---- sharing the work and ending it (heap_mark.cpp's protocol) ----

void share(Worker& w) {
    ParMinor& pm = w.pm;
    if (pm.idle.load(std::memory_order_relaxed) == 0 || w.stack.size() < 2) return;
    std::vector<uintptr_t> chunk;
    if (w.stack.size() >= 2 * kChunk) {
        chunk.assign(w.stack.end() - static_cast<std::ptrdiff_t>(kChunk), w.stack.end());
        w.stack.resize(w.stack.size() - kChunk);
    } else {
        if (pm.pool_size.load(std::memory_order_relaxed) != 0) return;
        const auto half = static_cast<std::ptrdiff_t>(w.stack.size() / 2);
        chunk.assign(w.stack.begin(), w.stack.begin() + half);
        w.stack.erase(w.stack.begin(), w.stack.begin() + half);
    }
    std::lock_guard<std::mutex> lock(pm.pool_mutex);
    pm.pool.push_back(std::move(chunk));
    pm.pool_size.fetch_add(1);
}

bool take(Worker& w) {
    ParMinor& pm = w.pm;
    std::lock_guard<std::mutex> lock(pm.pool_mutex);
    if (pm.pool.empty()) return false;
    std::vector<uintptr_t>& chunk = pm.pool.back();
    w.stack.insert(w.stack.end(), chunk.begin(), chunk.end());
    pm.pool.pop_back();
    pm.pool_size.fetch_sub(1);
    return true;
}

// More grey work for a thread whose stack is empty and that found the cards
// all taken, or false once every thread is idle and the pool is empty. No
// card work appears after the cards are taken, so idle threads and an empty
// pool mean nothing is left to copy.
bool acquire(Worker& w) {
    ParMinor& pm = w.pm;
    if (take(w)) return true;
    pm.idle.fetch_add(1);
    unsigned spins = 0;
    for (;;) {
        if (pm.pool_size.load() > 0) {
            pm.idle.fetch_sub(1);
            if (take(w)) return true;
            pm.idle.fetch_add(1);
            continue;
        }
        if (pm.idle.load() == pm.threads) return false;
        if (++spins < 64) cpu_relax();
        else std::this_thread::yield();
    }
}

bool take_cards(Worker& w) {
    ParMinor& pm = w.pm;
    const size_t count = pm.cards.size();
    if (pm.card_next.load(std::memory_order_relaxed) >= count) return false;
    // Fewer at once when there are few: a minor collection whose young
    // survivors hang from 16 cards would otherwise give them all to one
    // thread, the rest waiting on what it shares.
    const size_t step = std::clamp<size_t>(count / (size_t{8} * pm.threads), 1, kCardsAtOnce);
    const size_t at = pm.card_next.fetch_add(step, std::memory_order_relaxed);
    if (at >= count) return false;
    const size_t end = std::min(count, at + step);
    for (size_t i = at; i < end; ++i) scan_card_work(w.tracer, pm.cards[i]);
    return true;
}

void work(Worker& w) {
    Heap& heap = w.pm.g.heap;
    unsigned since_share = 0;
    for (;;) {
        // The copies first: depth first keeps the stack short, and what a
        // card's objects name is scanned while it is in cache.
        while (!w.stack.empty()) {
            const uintptr_t object = w.stack.back();
            w.stack.pop_back();
            scan_object(w.tracer, object, heap.is_old(object));
            if (++since_share >= kShareEvery) {
                since_share = 0;
                share(w);
            }
        }
        if (take_cards(w)) continue;
        if (w.cards_done_ns == 0 && w.pm.s.log) w.cards_done_ns = gc_now_ns();
        if (!acquire(w)) return;
    }
}

void helper_work(void* context, unsigned helper) {
    ParMinor& pm = *static_cast<ParMinor*>(context);
    work(*pm.workers[helper + 1]);
}

// Every thread drains until all of the work is done.
void drain_all(ParMinor& pm, MarkThreads& helpers, unsigned helper_count) {
    pm.idle.store(0);
    helpers.run(helper_count, &helper_work, &pm);
    work(*pm.workers[0]);
    helpers.wait();
    if (pm.pool_size.load() != 0) gc_fatal("a parallel minor collection ended with grey work in the pool");
    for (unsigned i = 0; i < pm.threads; ++i) {
        if (!pm.workers[i]->stack.empty()) {
            gc_fatal("a parallel minor collection ended with grey work on a stack");
        }
    }
}

// trace_to_fixpoint's ephemeron rounds: on the calling thread, the helpers
// parked; a value it traces is the first thread's grey work, drained by all.
void ephemeron_fixpoint(ParMinor& pm, MarkThreads& helpers, unsigned helper_count) {
    GcState& g = pm.g;
    Worker& main = *pm.workers[0];
    for (;;) {
        std::vector<EphemeronEntry> pending;
        for (auto& w : pm.workers) {
            pending.insert(pending.end(), w->ephemerons.begin(), w->ephemerons.end());
            w->ephemerons.clear();
        }
        bool progress = false;
        for (const EphemeronEntry& e : pending) {
            const uint64_t word = *e.key;
            const uintptr_t live = live_now(g, static_cast<uintptr_t>(word & kAddressMask));
            if (!live) {
                main.ephemerons.push_back(e);
                continue;
            }
            *e.key = (word & ~kAddressMask) | live;
            main.tracer.set_owner(e.owner, e.owner_old);
            if (e.owner_old && g.heap.is_young(live)) dirty(g.s, e.owner);
            main.tracer.visit(e.value);
            progress = true;
        }
        if (!progress) return;
        drain_all(pm, helpers, helper_count);
    }
}

// Reserves the block table and the large pages' table so that what this
// collection can promote never moves them: at most every young byte, in
// blocks at least a quarter full (a medium object wastes at most 8 KB of
// one) and in pages one each at most beyond its bytes, plus each thread's
// two regions. False when the frontiers lack that room.
bool reserve_room(HeapState& s, uint64_t young, unsigned threads) {
    const size_t blocks = 2 * young / kBlockBytes + 2 * threads + 4;
    const size_t blocks_left = s.mature_reserve / kBlockBytes - s.blocks.size();
    const size_t pages = 2 * young / kPageBytes + 4;
    const size_t pages_left = s.large_reserve / kPageBytes - s.large_frontier;
    if (blocks_left < blocks || pages_left < pages) return false;
    s.blocks.reserve(s.blocks.size() + blocks);
    s.large_page_head.reserve(s.large_frontier + pages);
    s.large_page_committed.reserve(s.large_frontier + pages);
    return true;
}

// Whether a 32 KB region of the old generation has a dirty card, counted.
size_t dirty_regions(const HeapState& s) {
    const size_t mature = s.blocks.size();
    const size_t large = (static_cast<size_t>(s.large_frontier) * kPageBytes + (size_t{1} << kCardRegionShift) - 1) >>
                         kCardRegionShift;
    const uint8_t* large_regions = s.regions + ((s.large_lo - s.mature_lo) >> kCardRegionShift);
    size_t n = 0;
    for (size_t i = 0; i < mature; ++i) n += s.regions[i] != 0;
    for (size_t i = 0; i < large; ++i) n += large_regions[i] != 0;
    return n;
}

// A dirty region's card work, counted as this much copying: scanning the
// objects on a dirty card costs about what copying a few KB does.
constexpr uint64_t kDirtyRegionBytes = 4 * 1024;

} // namespace

unsigned parallel_minor_threads(GcState& g) {
    HeapState& s = g.s;
    const HeapConfig& c = s.config;
    if (c.parallel_minor == ParallelMinor::Off) return 1;
    const unsigned most = configured_threads(c.minor_threads);
    if (most <= 1) return 1;
    const uint64_t young = (g.eden_top - s.eden_lo) + (g.from_top - s.survivor_lo[g.from]);
    if (c.parallel_minor == ParallelMinor::Auto) {
        // Parallel only when one thread would pass the pause goal. Shorter
        // collections gain less than parallel copying costs the program
        // after it (measured, the cause not pinned down: probably the
        // helpers' writes, forwarding words in eden and the copies, leave
        // the lines the program touches next in other cores' caches; and
        // their wake-up). bro's gc_pause probe ran 7-12% slower with its
        // 0.4-0.8 ms minors halved on 8 threads. The estimate: the copying
        // expected (the last minor collection's survival rate times what
        // this one collects) and the card work, at what serial collections
        // have cost per byte of that (note_serial_evacuation).
        const uint64_t expected = static_cast<uint64_t>(s.minor_survival * static_cast<double>(young));
        const uint64_t work = expected + kDirtyRegionBytes * dirty_regions(s);
        const double ms = static_cast<double>(work) * s.minor_ns_per_byte / 1e6;
        if (ms < c.parallel_minor_ms) {
            g.serial_card_bytes = work - expected;
            return 1;
        }
    }
    // All of them: a thread of the parallel path is ~25% slower than the
    // serial one (the claims, the gathering, the buffers) and misses more
    // besides, so two threads measured barely faster than one (synth 200 p50
    // 0.54 against 0.55 ms; four 0.37, eight 0.24).
    if (!reserve_room(s, young, most)) return 1;
    return most;
}

void note_serial_evacuation(GcState& g, uint64_t ns) {
    // What it did, not what was expected: a misjudged survival rate is not
    // a cost. Small ones are mostly the roots, whatever their bytes.
    const uint64_t work = g.copied_bytes + g.serial_card_bytes;
    if (!g.learns_cost || work < (uint64_t{256} << 10)) return;
    const double sample = static_cast<double>(ns) / static_cast<double>(work);
    g.s.minor_ns_per_byte = 0.75 * g.s.minor_ns_per_byte + 0.25 * sample;
}

void evacuate_parallel(GcState& g, PhaseClock& clock, unsigned threads) {
    HeapState& s = g.s;
    ParMinor pm(g);
    pm.block_bound = static_cast<uint32_t>(s.blocks.size());
    pm.page_bound = s.large_frontier;
    const uintptr_t to_lo = s.survivor_lo[g.to];
    if (s.survivor_top[g.to] != to_lo || ((to_lo - s.eden_lo) & (kBufferAlign - 1)) != 0) {
        gc_fatal("a parallel minor collection's survivor space is not empty and aligned");
    }
    pm.survivor_top.store(to_lo);
    pm.survivor_end = to_lo + s.survivor_bytes;
    pm.buffer_bytes = std::clamp<size_t>((s.survivor_bytes / 64) & ~(kBufferAlign - 1), kBufferAlign, kBufferBytes);

    MarkThreads& helpers = heap_helpers(s);
    const unsigned helper_count = std::min(helpers.ensure(threads - 1), threads - 1);
    pm.threads = helper_count + 1;
    const uintptr_t lo = g.shade ? g.heap.reservation_base() : g.heap.young_base();
    const size_t span = g.shade ? g.heap.reservation_bytes() : g.heap.young_span();
    for (unsigned i = 0; i < pm.threads; ++i) pm.workers.push_back(std::make_unique<Worker>(pm, g.heap, lo, lo + span));

    gather_dirty_cards(g, pm.cards);
    clock.lap(kPhaseCards);
    s.parallel_promotion = true;
    visit_roots(s, pm.workers[0]->tracer);
    clock.lap(kPhaseRoots);
    const uint64_t parallel_from = s.log ? gc_now_ns() : 0;
    drain_all(pm, helpers, helper_count);
    ephemeron_fixpoint(pm, helpers, helper_count);
    clock.lap(kPhaseTrace);
    s.parallel_promotion = false;
    // The log's cards phase is the gathering and the calling thread's share
    // of the card work, as far as when it found the cards all taken; its
    // trace phase the rest.
    if (s.log && pm.workers[0]->cards_done_ns > parallel_from) {
        const uint64_t cards = std::min(g.phase_ns[kPhaseTrace], pm.workers[0]->cards_done_ns - parallel_from);
        g.phase_ns[kPhaseCards] += cards;
        g.phase_ns[kPhaseTrace] -= cards;
    }

    // The threads' buffers and regions back to the heap: the survivor
    // space's top past every buffer, the blocks with room left recyclable
    // again (the next refill finds their holes from the start).
    s.survivor_top[g.to] = pm.survivor_top.load();
    uint64_t slack = 0;
    for (auto& wp : pm.workers) {
        Worker& w = *wp;
        slack += w.slack + (w.buffer_limit - w.buffer_cursor);
        for (const BumpRegion* r : {&w.small, &w.medium}) {
            if (r->block == UINT32_MAX) continue;
            const BlockMeta& meta = *s.blocks[r->block];
            if (std::find(std::begin(meta.line_used), std::end(meta.line_used), uint8_t{0}) != std::end(meta.line_used)) {
                s.recyclable_blocks.push_back(r->block);
            }
        }
        g.weak.insert(g.weak.end(), w.weak.begin(), w.weak.end());
        g.ephemerons.insert(g.ephemerons.end(), w.ephemerons.begin(), w.ephemerons.end());
        for (const WeakEntry& e : w.cycle_weak) cycle_note_weak(s, e);
        for (const EphemeronEntry& e : w.cycle_ephemerons) cycle_note_ephemeron(s, e);
        cycle_add_grey(s, w.shaded);
        g.copied_bytes += w.copied_bytes;
        g.copied_objects += w.copied_objects;
        g.promoted_bytes += w.promoted_bytes;
        g.promoted_objects += w.promoted_objects;
        g.tenured_bytes += w.tenured_bytes;
        s.old_allocated_since_full += w.promoted_bytes;
        g.survivor_overflow = g.survivor_overflow || w.overflow;
    }
    s.survivor_slack[g.to] = slack;
    s.relocation_epoch += g.copied_objects;
    g.minor_threads = pm.threads;
    settle_weakness(g);
    clock.lap(kPhaseWeak);
}

} // namespace brass::gc::detail
