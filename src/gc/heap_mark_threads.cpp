#include "heap_mark_threads.hpp"

#include "heap_internal.hpp"

namespace brass::gc::detail {

void MarkThreadsDeleter::operator()(MarkThreads* threads) const noexcept { delete threads; }

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
