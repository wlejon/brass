#include "heap_mark_threads.hpp"

#include "heap_internal.hpp"

#include <algorithm>

namespace brass::gc::detail {

void MarkThreadsDeleter::operator()(MarkThreads* threads) const noexcept { delete threads; }

unsigned configured_threads(unsigned configured) {
    if (configured != 0) return configured;
    // Read once: glibc's hardware_concurrency reads /sys each call, ~20 us
    // that every quiet minor collection would pay (the planet's measured).
    static const unsigned hardware = std::thread::hardware_concurrency();
    return std::clamp(hardware / 2, 1u, 8u);
}

unsigned large_minor_thread_count(const HeapConfig& config) {
    if (!config.large_minor_threads_on) return 1;
    if (config.large_minor_threads != 0) return config.large_minor_threads;
    // A host that set the minor collection's count has said how many cores
    // it gives the collector; only the default widens.
    if (config.minor_threads != 0) return 1;
    static const unsigned hardware = std::thread::hardware_concurrency();
    return std::max(hardware / 2, 1u);
}

MarkThreads& heap_helpers(HeapState& s) {
    if (!s.mark_threads) s.mark_threads.reset(new MarkThreads);
    return *s.mark_threads;
}

MarkThreads::~MarkThreads() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : threads_) t.join();
}

unsigned MarkThreads::ensure(unsigned n) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (threads_.size() < n) {
        const auto helper = static_cast<unsigned>(threads_.size());
        try {
            // Created between runs: it waits for the next generation, not
            // one that has already gone by.
            threads_.emplace_back([this, helper, seen = generation_] { loop(helper, seen); });
        } catch (...) {
            break;  // fewer helpers: the ones that exist share the work
        }
    }
    return static_cast<unsigned>(threads_.size());
}

void MarkThreads::run(unsigned n, Job job, void* context) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
        context_ = context;
        running_ = n;
        done_.store(0, std::memory_order_relaxed);
        ++generation_;
    }
    wake_.notify_all();
}

void MarkThreads::wait() {
    // The caller's own drain ends when the grey work is gone or the deadline
    // has passed, and the helpers' then end within microseconds: a spin, as
    // joining them was.
    while (done_.load(std::memory_order_acquire) < running_) std::this_thread::yield();
}

void MarkThreads::loop(unsigned helper, uint64_t seen) {
    for (;;) {
        Job job = nullptr;
        void* context = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return quit_ || generation_ != seen; });
            if (quit_) return;
            seen = generation_;
            if (helper >= running_) continue;
            job = job_;
            context = context_;
        }
        job(context, helper);
        done_.fetch_add(1, std::memory_order_release);
    }
}

} // namespace brass::gc::detail

namespace brass::gc {

unsigned Heap::start_helper_threads() {
    detail::HeapState& s = *s_;
    if (s.collecting) {
        detail::gc_fatal("start_helper_threads during a collection (a trace function, root "
                         "source or hook called it)");
    }
    // As many as the larger of the two parallel paths that use the pool
    // would ask for; a path that never runs on it asks for none.
    unsigned threads = 1;
    if (s.config.mark_thread_pool) {
        threads = std::max(threads, detail::configured_threads(s.config.mark_threads));
    }
    if (s.config.parallel_minor != ParallelMinor::Off) {
        threads = std::max(threads, detail::configured_threads(s.config.minor_threads));
        threads = std::max(threads, detail::large_minor_thread_count(s.config));
    }
    if (threads <= 1) return 0;
    return detail::heap_helpers(s).ensure(threads - 1);
}

} // namespace brass::gc
