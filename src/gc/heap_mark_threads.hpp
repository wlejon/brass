#pragma once

// The helper threads a parallel mark drain runs on (heap_mark.cpp,
// mark_drain), parked between drains rather than created and joined by each.
// A marking cycle's slice is a fraction of a millisecond: creating seven
// threads and joining them costs about 0.1 ms of it, and the helpers start
// marking some 60 us in, so a 0.4 ms slice on eight threads marked little
// more than one on a single thread. A parked helper wakes in microseconds.
// HeapConfig::mark_thread_pool (BRASS_GC_MARK_POOL=0 turns it off) is the
// A/B switch.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace brass::gc::detail {

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
