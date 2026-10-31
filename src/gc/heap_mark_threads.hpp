#pragma once

// The helper threads a parallel mark drain runs on (heap_mark.cpp,
// mark_drain), parked between drains rather than created and joined by each.
// A marking cycle's slice is a fraction of a millisecond: creating seven
// threads and joining them costs about 0.1 ms of it, and the helpers start
// marking some 60 us in, so a 0.4 ms slice on eight threads marked little
// more than one on a single thread. A parked helper wakes in microseconds.
// HeapConfig::mark_thread_pool (BRASS_GC_MARK_POOL=0 turns it off) is the
// A/B switch. The parallel minor collection shares the pool.
//
// Creating them is ~0.1 ms a thread, so the collection that first wants them
// pays most of a millisecond for seven: bro's first parallel minor took 3.9
// ms on eight threads against 3.0 on one, 0.7 ms of it their creation.
// Heap::start_helper_threads creates them ahead of time, where the host says
// that costs nothing.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace brass::gc {
struct HeapConfig;
}

namespace brass::gc::detail {

struct HeapState;
class MarkThreads;

// The threads (the collecting one among them) a HeapConfig field asking for
// `configured` means: itself, or for 0 half the hardware threads, at most 8.
unsigned configured_threads(unsigned configured);
// The threads a large parallel minor collection copies with
// (HeapConfig::large_minor_threads): itself, or for 0 half the hardware
// threads, uncapped; 1 when large_minor_threads_on is false, or when it is 0
// and minor_threads is set (an explicit count is the host's).
unsigned large_minor_thread_count(const HeapConfig& config);
// The heap's helpers, the pool made at first use (by a parallel drain or
// minor collection, or Heap::start_helper_threads ahead of them).
MarkThreads& heap_helpers(HeapState& s);

class MarkThreads {
public:
    using Job = void (*)(void* context, unsigned helper);

    MarkThreads() = default;
    MarkThreads(const MarkThreads&) = delete;
    MarkThreads& operator=(const MarkThreads&) = delete;
    ~MarkThreads();

    // Creates helpers until there are `n`; returns how many there are (fewer
    // when a thread could not be created). None runs anything yet.
    unsigned ensure(unsigned n);
    // job(context, i) on helpers 0..n-1 (n at most what ensure returned).
    // The caller then works alongside them and calls wait().
    void run(unsigned n, Job job, void* context);
    // Until every helper of the last run() has returned from its job.
    void wait();

private:
    void loop(unsigned helper, uint64_t seen);

    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<std::thread> threads_;
    uint64_t generation_ = 0;  // one per run(): a helper runs each at most once
    unsigned running_ = 0;     // helpers the last run() asked for
    Job job_ = nullptr;
    void* context_ = nullptr;
    bool quit_ = false;
    std::atomic<unsigned> done_{0};
};

} // namespace brass::gc::detail
