#pragma once

// Fork-join over a fixed set of independent work items, for the parts of an
// ahead-of-time build that are per function (the optimizer's function-local
// pass runs, instruction selection through emission).
//
// The contract that keeps a parallel build's output byte-identical to a serial
// one is the caller's: an item reads nothing another item writes, and writes
// only state it owns (its function, its own result slot). Under that contract
// the order and the threads items run on cannot show in the result, so this
// helper is free to schedule them as it likes, and a thread count of 1 runs
// every item on the calling thread, in the same order, through the same code.
//
// Why not runtime::CompilePool: that is the JIT's background queue (priorities,
// heat, cancellation per owner, a worker held back for latency-bound jobs,
// workers placed away from the mutator's cache), built for jobs that arrive
// one at a time while a program runs. A build's fan-out is all known at once
// and the caller waits for all of it. The convention it shares with the pool's
// workers: anything a thread allocates from its ScratchPool is freed on that
// thread before the item ends.
#include <cstddef>
#include <functional>
#include <vector>

namespace brass {

// The number of threads `requested` means: itself, or for 0 the machine's
// hardware threads (at least 1).
size_t resolve_thread_count(size_t requested) noexcept;

// How many threads a fan-out over `total_weight` units of work (a module's
// blocks, say) uses when `requested` are allowed: 1 below `min_weight_per_thread`
// units, so a small module never starts a thread, and never more than
// `items`. Only the speed of a build depends on it, never its output.
size_t threads_for_work(size_t requested, size_t items, size_t total_weight,
                        size_t min_weight_per_thread) noexcept;

// Runs body(order[0]), body(order[1]), ... each exactly once, on up to
// `threads` threads, the calling thread one of them; returns when all have
// run. Items start in `order` (put the largest first: the last item to start
// bounds the wall time). With `threads` <= 1 they run on the caller, in order.
//
// An item that throws does not stop the others. Once every item has run, the
// exception of the item with the smallest value (not position in `order`) is
// rethrown, so which error a failed build reports does not depend on timing.
void parallel_for_each(const std::vector<size_t>& order, size_t threads,
                       const std::function<void(size_t)>& body);

// `order` for parallel_for_each: the indices 0..weights.size()-1, heaviest
// first, ties by index.
std::vector<size_t> heaviest_first(const std::vector<size_t>& weights);

} // namespace brass
