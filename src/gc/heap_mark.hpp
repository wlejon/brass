#pragma once

// The old generation's marker, shared by heap_mark.cpp (the parallel
// machinery: per-thread mark stacks, the shared pool, large-object slices)
// and heap_mark_incremental.cpp (the marking cycle: start, slices, finish).
//
// The workers, their stacks and the pool outlive a slice: a slice that runs
// out of time leaves its grey work where it was (each worker's stack, the
// pool's chunks) and the next slice resumes from there.

#include "heap_internal.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace brass::gc::detail {

// A slice with no deadline: it runs until the grey work is gone.
inline constexpr uint64_t kNoDeadline = UINT64_MAX;

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
    unsigned threads = 1;  // threads draining in the slice under way
    std::mutex mutex;
    std::vector<std::vector<uintptr_t>> pool;
    std::atomic<size_t> pool_size{0};
    std::atomic<unsigned> idle{0};
    // The slice's deadline (gc_now_ns), and whether a worker has seen it
    // pass: every worker then stops at its next look at the clock.
    uint64_t deadline = kNoDeadline;
    std::atomic<bool> stop{false};

    MarkShared(HeapState& state, Heap& h) : s(state), heap(h) {}
};

using MarkWorkers = std::vector<std::unique_ptr<MarkWorker>>;

// The threads a cycle marks with: 1 below HeapConfig::parallel_mark_bytes.
unsigned mark_thread_count(const HeapState& s);
// `threads` workers over the old generation; claims are atomic when more
// than one. Roots go on the first one's stack.
MarkWorkers make_mark_workers(MarkShared& sh, unsigned threads);
// Drains the grey work (every worker's stack and the pool) on `run` threads
// (1, or all the workers) until it is gone or sh.deadline passes. True when
// it is gone; the ephemerons met are left in the workers.
bool mark_drain(MarkShared& sh, MarkWorkers& workers, unsigned run);
// Whether old object `a` is marked (anything else counts as marked).
bool is_marked(const HeapState& s, uintptr_t a) noexcept;

} // namespace brass::gc::detail
