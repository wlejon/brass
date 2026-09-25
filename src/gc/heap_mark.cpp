// A full collection's marking of the old generation.
//
// By the time it runs the young generation is empty (heap_collect.cpp), so
// marking moves nothing, allocates nothing and writes no slot: it sets mark
// bits (a mature object's in its block's side bitmap, a large object's in its
// header) and, as each marked object is scanned, the line marks the sweep
// reclaims by. That makes it safe to spread over several threads: a mark bit
// is claimed with an atomic or, so exactly one thread scans each object.
//
// Each thread keeps its own mark stack. A thread with plenty of work hands a
// chunk to a shared pool whenever another is idle; a thread that runs dry
// takes one; the marking ends when every thread is idle and the pool is
// empty. Objects are scanned through a small ring of addresses whose headers
// were prefetched when they entered it, which hides most of the cache misses
// a mark loop otherwise waits on.
//
// Roots are visited on the collecting thread before the others start, and the
// ephemeron fixpoint runs there between rounds of parallel marking. Weak
// slots and dead ephemerons are gathered per thread and settled afterwards
// (heap_weak.cpp).

#include "heap_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__) && !defined(_M_ARM64)
#include <xmmintrin.h>
#endif

namespace brass::gc::detail {

namespace {

constexpr size_t kChunk = 256;   // mark-stack entries handed between threads at once
constexpr size_t kRing = 16;     // objects prefetched ahead of the one being scanned
constexpr unsigned kShareEvery = 64;

inline void prefetch(const void* p) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_ARM64)
    __prefetch(p);
#else
    _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#endif
#else
    __builtin_prefetch(p);
#endif
}

struct MarkShared;

struct MarkWorker {
    MarkShared& sh;
    Tracer tracer;
    std::vector<uintptr_t> stack;
    std::vector<WeakEntry> weak;
    std::vector<EphemeronEntry> ephemerons;
    uint64_t marked_bytes = 0;
    uint64_t marked_objects = 0;

    MarkWorker(MarkShared& shared, Heap& heap, uintptr_t lo, uintptr_t hi, Tracer::VisitFn visit,
               Tracer::WeakFn weak_fn, Tracer::EphemeronFn ephemeron_fn)
        : sh(shared), tracer(heap, Tracer::Purpose::Full, lo, hi, visit, weak_fn, ephemeron_fn, this) {}
};

struct MarkShared {
    HeapState& s;
    Heap& heap;
    unsigned threads = 1;
    std::mutex mutex;
    std::vector<std::vector<uintptr_t>> pool;
    std::atomic<size_t> pool_size{0};
    std::atomic<unsigned> idle{0};

    MarkShared(HeapState& state, Heap& h) : s(state), heap(h) {}
};

MarkWorker& worker_of(Tracer& t) noexcept { return *static_cast<MarkWorker*>(t.state()); }

[[noreturn]] void bad_reference(uintptr_t a) {
    char msg[160];
    std::snprintf(msg, sizeof msg, "a reference to %p names no block in use of the mature space",
                  reinterpret_cast<void*>(a));
    gc_fatal(msg);
}

// Sets `bit` in `word`; true when this call set it.
template <bool Parallel>
bool claim(uint64_t& word, uint64_t bit) noexcept {
    if constexpr (Parallel) {
        std::atomic_ref<uint64_t> w(word);
        if (w.load(std::memory_order_relaxed) & bit) return false;
        return (w.fetch_or(bit, std::memory_order_relaxed) & bit) == 0;
    } else {
        if (word & bit) return false;
        word |= bit;
        return true;
    }
}

template <bool Parallel>
bool claim_large(uint8_t& bits) noexcept {
    if constexpr (Parallel) {
        std::atomic_ref<uint8_t> b(bits);
        if (b.load(std::memory_order_relaxed) & kGcLargeMarked) return false;
        return (b.fetch_or(kGcLargeMarked, std::memory_order_relaxed) & kGcLargeMarked) == 0;
    } else {
        if (bits & kGcLargeMarked) return false;
        bits = static_cast<uint8_t>(bits | kGcLargeMarked);
        return true;
    }
}

bool is_marked(const HeapState& s, uintptr_t a) noexcept {
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size()) return true;
        BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        const uint64_t word = std::atomic_ref<uint64_t>(meta.marks[granule >> 6]).load(std::memory_order_relaxed);
        return ((word >> (granule & 63)) & 1) != 0;
    }
    if (s.in_large(a)) {
        const uint8_t bits = std::atomic_ref<uint8_t>(header_of(a)->gc_bits).load(std::memory_order_relaxed);
        return (bits & kGcLargeMarked) != 0;
    }
    return true;
}

template <bool Parallel>
void mark_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    (void)slot;
    MarkWorker& w = worker_of(t);
    if (!w.sh.heap.is_reference_tag(word)) return;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    HeapState& s = w.sh.s;
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size() || !s.blocks[index]->in_use) bad_reference(a);
        BlockMeta& meta = *s.blocks[index];
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        if (claim<Parallel>(meta.marks[granule >> 6], uint64_t{1} << (granule & 63))) w.stack.push_back(a);
    } else if (s.in_large(a)) {
        if (claim_large<Parallel>(header_of(a)->gc_bits)) w.stack.push_back(a);
    }
}

void mark_weak(Tracer& t, uint64_t* slot, uint64_t cleared) {
    MarkWorker& w = worker_of(t);
    if (!w.sh.heap.is_reference_tag(*slot)) return;
    w.weak.push_back(WeakEntry{slot, cleared, t.owner(), t.owner_old()});
}

void mark_ephemeron(Tracer& t, uint64_t* key, uint64_t* value, uint64_t cleared_key, uint64_t cleared_value) {
    MarkWorker& w = worker_of(t);
    const uint64_t word = *key;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    const Heap& heap = w.sh.heap;
    // A key that is no reference, or names no old object (the young
    // generation is empty by now), cannot die here.
    if (a == 0 || !heap.contains(a) || !heap.is_reference_tag(word) || heap.is_young(a) ||
        is_marked(w.sh.s, a)) {
        t.visit(value);
        return;
    }
    w.ephemerons.push_back(EphemeronEntry{key, value, cleared_key, cleared_value, t.owner(), t.owner_old()});
}

// A scanned object's lines and bytes: recorded when it is scanned rather than
// when it is marked, since its header is in cache only then.
template <bool Parallel>
void account(MarkWorker& w, uintptr_t object) noexcept {
    HeapState& s = w.sh.s;
    const size_t total = kHeaderBytes + header_of(object)->size;
    w.marked_bytes += total;
    ++w.marked_objects;
    if (!s.in_mature(object)) return;
    const uint32_t index = s.block_index(object);
    BlockMeta& meta = *s.blocks[index];
    const uintptr_t lo = s.block_base(index);
    const size_t first = (object - kHeaderBytes - lo) / kLineBytes;
    const size_t last = (object - kHeaderBytes + total - 1 - lo) / kLineBytes;
    for (size_t l = first; l <= last; ++l) {
        if constexpr (Parallel) std::atomic_ref<uint8_t>(meta.line_mark[l]).store(1, std::memory_order_relaxed);
        else meta.line_mark[l] = 1;
    }
}

// Hands part of this thread's stack to the pool when another thread idles. A
// deep stack gives a chunk off its top. A shallow one (a depth-first walk of
// a tree or a list keeps only a handful of entries) gives the bottom half:
// the entries pushed first, which head the largest unexplored subgraphs. It
// does so only while the pool is empty, so shallow stacks do not grind the
// pool's chunks into ever smaller ones.
void share(MarkWorker& w) {
    MarkShared& sh = w.sh;
    if (sh.idle.load(std::memory_order_relaxed) == 0 || w.stack.size() < 2) return;
    std::vector<uintptr_t> chunk;
    if (w.stack.size() >= 2 * kChunk) {
        chunk.assign(w.stack.end() - static_cast<std::ptrdiff_t>(kChunk), w.stack.end());
        w.stack.resize(w.stack.size() - kChunk);
    } else {
        if (sh.pool_size.load(std::memory_order_relaxed) != 0) return;
        const auto half = static_cast<std::ptrdiff_t>(w.stack.size() / 2);
        chunk.assign(w.stack.begin(), w.stack.begin() + half);
        w.stack.erase(w.stack.begin(), w.stack.begin() + half);
    }
    std::lock_guard<std::mutex> lock(sh.mutex);
    sh.pool.push_back(std::move(chunk));
    sh.pool_size.fetch_add(1);
}

bool take(MarkWorker& w) {
    MarkShared& sh = w.sh;
    std::lock_guard<std::mutex> lock(sh.mutex);
    if (sh.pool.empty()) return false;
    std::vector<uintptr_t>& chunk = sh.pool.back();
    w.stack.insert(w.stack.end(), chunk.begin(), chunk.end());
    sh.pool.pop_back();
    sh.pool_size.fetch_sub(1);
    return true;
}

// More work for an idle thread, or false once every thread is idle and the
// pool is empty (nothing can then produce more).
bool acquire(MarkWorker& w) {
    MarkShared& sh = w.sh;
    if (take(w)) return true;
    sh.idle.fetch_add(1);
    unsigned spins = 0;
    for (;;) {
        if (sh.pool_size.load() > 0) {
            sh.idle.fetch_sub(1);
            if (take(w)) return true;
            sh.idle.fetch_add(1);
            continue;
        }
        if (sh.idle.load() == sh.threads) return false;
        if (++spins < 64) continue;
        std::this_thread::yield();
    }
}

// A large object is scanned in slices when marking in parallel, so its
// references are claimed by several threads rather than one: the slices past
// the first go on the mark stack as entries of their own, the object's
// address with the slice number in the top 16 bits (a plain entry's are
// zero), where sharing spreads them like any other work. Only a layout that
// can scan a byte range is sliced; a Custom layout's range function visits,
// in the slice holding offset 0, what a full trace visits outside the
// payload (object.hpp, TraceRangeFn).
constexpr size_t kSliceBytes = 16 * 1024;
constexpr unsigned kSliceShift = 48;

size_t slice_bytes_for(size_t size) noexcept {
    // At most 65535 slices, however large the object.
    const size_t floor = (size + 0xFFFE) / 0xFFFF;
    return std::max(kSliceBytes, (floor + kGranuleBytes - 1) & ~(kGranuleBytes - 1));
}

bool sliceable(const LayoutDescriptor& d) noexcept {
    return d.kind == LayoutKind::Words || d.kind == LayoutKind::Mask ||
           (d.kind == LayoutKind::Custom && d.trace_range != nullptr);
}

void scan_parallel(MarkWorker& w, uintptr_t entry) {
    const uintptr_t o = entry & kAddressMask;
    const size_t size = header_of(o)->size;
    const size_t slice = entry >> kSliceShift;
    if (slice != 0) {
        const size_t bytes = slice_bytes_for(size);
        scan_object_range(w.tracer, o, slice * bytes, std::min(size, (slice + 1) * bytes));
        return;
    }
    account<true>(w, o);
    if (size <= kSliceBytes || !sliceable(layout_descriptor(header_of(o)->layout))) {
        scan_object(w.tracer, o, true);
        return;
    }
    const size_t bytes = slice_bytes_for(size);
    const size_t slices = (size + bytes - 1) / bytes;
    for (size_t i = slices - 1; i >= 1; --i) w.stack.push_back(o | (static_cast<uintptr_t>(i) << kSliceShift));
    scan_object_range(w.tracer, o, 0, bytes);
}

template <bool Parallel>
void drain(MarkWorker& w) {
    uintptr_t ring[kRing];
    size_t head = 0;
    size_t count = 0;
    unsigned since_share = 0;
    for (;;) {
        while (count < kRing && !w.stack.empty()) {
            const uintptr_t o = w.stack.back();
            w.stack.pop_back();
            prefetch(reinterpret_cast<const void*>((o & kAddressMask) - kHeaderBytes));
            ring[(head + count) % kRing] = o;
            ++count;
        }
        if (count == 0) {
            if constexpr (Parallel) {
                if (acquire(w)) continue;
            }
            return;
        }
        const uintptr_t o = ring[head];
        head = (head + 1) % kRing;
        --count;
        if constexpr (Parallel) {
            scan_parallel(w, o);
        } else {
            account<Parallel>(w, o);
            scan_object(w.tracer, o, true);
        }
        if constexpr (Parallel) {
            if (++since_share >= kShareEvery) {
                since_share = 0;
                share(w);
            }
        }
    }
}

unsigned mark_thread_count(const HeapState& s) {
    if (s.heap.old_used_bytes() < s.config.parallel_mark_bytes) return 1;
    unsigned n = s.config.mark_threads;
    if (n == 0) n = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
    return std::max(n, 1u);
}

// Runs every worker's stack (and whatever it spreads to) to completion.
void mark_closure(MarkShared& sh, std::vector<std::unique_ptr<MarkWorker>>& workers) {
    if (workers.size() == 1) {
        drain<false>(*workers[0]);
        return;
    }
    // Each worker thread starts at `go`, once `threads` counts exactly the
    // threads that were created: the termination count depends on it.
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(workers.size() - 1);
    for (size_t i = 1; i < workers.size(); ++i) {
        try {
            threads.emplace_back([&go, &w = *workers[i]] {
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                drain<true>(w);
            });
        } catch (...) {
            break;  // fewer threads: the ones that exist share the work
        }
    }
    sh.threads = static_cast<unsigned>(threads.size() + 1);
    sh.idle.store(0);
    go.store(true, std::memory_order_release);
    drain<true>(*workers[0]);
    for (std::thread& t : threads) t.join();
    // A worker whose thread could not be created never ran; its stack is
    // empty (only worker 0 received roots), so nothing is lost.
}

} // namespace

void mark_old_generation(GcState& g, PhaseClock& clock) {
    HeapState& s = g.s;
    MarkShared sh(s, g.heap);
    const unsigned threads = mark_thread_count(s);
    const bool parallel = threads > 1;
    const uintptr_t lo = s.mature_lo;
    const uintptr_t hi = s.large_lo + s.large_reserve;
    std::vector<std::unique_ptr<MarkWorker>> workers;
    for (unsigned i = 0; i < threads; ++i) {
        workers.push_back(std::make_unique<MarkWorker>(sh, g.heap, lo, hi,
                                                       parallel ? &mark_visit<true> : &mark_visit<false>,
                                                       &mark_weak, &mark_ephemeron));
    }
    MarkWorker& main = *workers[0];
    main.stack.reserve(4096);

    visit_roots(s, main.tracer);
    clock.lap(kPhaseRoots);

    for (;;) {
        mark_closure(sh, workers);
        std::vector<EphemeronEntry> pending;
        for (auto& w : workers) {
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
        if (!progress) break;
    }
    clock.lap(kPhaseTrace);

    for (auto& w : workers) {
        g.weak.insert(g.weak.end(), w->weak.begin(), w->weak.end());
        g.ephemerons.insert(g.ephemerons.end(), w->ephemerons.begin(), w->ephemerons.end());
        g.marked_bytes += w->marked_bytes;
        g.marked_objects += w->marked_objects;
    }
}

} // namespace brass::gc::detail
