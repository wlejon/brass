#include <brass/runtime/compile_pool.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

namespace brass::runtime {

namespace {

// At exit the shared pool is drained and joined before the process's other
// statics go: a job still compiling would be running brass while they are
// torn down. Hosts that exit through their own shutdown call shutdown()
// themselves, earlier; this is for those that just return from main.
void shutdown_shared_at_exit() {
    CompilePool::shared().shutdown();
}

// BRASS_JIT_THREADS, or 0 when unset or not a positive count.
long env_thread_count() {
#if defined(_MSC_VER)
    // MSVC deprecates getenv (C4996, an error under /WX).
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, "BRASS_JIT_THREADS") != 0 || !owned) return 0;
    const long n = std::strtol(owned, nullptr, 10);
    std::free(owned);
#else
    const char* env = std::getenv("BRASS_JIT_THREADS");
    const long n = env ? std::strtol(env, nullptr, 10) : 0;
#endif
    return n > 0 ? n : 0;
}

} // namespace

size_t CompilePool::default_thread_count() {
    if (const long n = env_thread_count()) return static_cast<size_t>(std::min<long>(n, 64));
    const size_t hw = std::thread::hardware_concurrency();
    return std::clamp<size_t>(hw / 4, 1, 4);
}

CompilePool& CompilePool::shared() {
    // Leaked: workers may still be parked on it when static destructors run,
    // and the at-exit shutdown above is what stops them.
    static CompilePool* pool = new CompilePool(SharedTag{}, default_thread_count());
    return *pool;
}

CompilePool::CompilePool(SharedTag, size_t threads) : is_shared_(true), configured_threads_(threads) {}

CompilePool::CompilePool(size_t threads) : is_shared_(false), configured_threads_(threads) {
    if (threads > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        start_locked(threads);
    }
}

CompilePool::~CompilePool() {
    shutdown();
}

void CompilePool::start_locked(size_t threads) {
    if (!workers_.empty() || threads == 0) return;
    stopping_ = false;
    closed_ = false;
    configured_threads_ = threads;
    workers_.reserve(threads);
    for (size_t i = 0; i < threads; ++i) workers_.emplace_back(&CompilePool::worker_loop, this);
}

void CompilePool::start(size_t threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_locked(threads);
}

bool CompilePool::submit(Owner owner, uint8_t priority, std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || stopping_) return false;
        if (workers_.empty() && is_shared_) {
            static const bool registered = [] { return std::atexit(&shutdown_shared_at_exit) == 0; }();
            (void)registered;
            start_locked(configured_threads_);
        }
        queue_.push_back(Job{owner, priority, next_seq_++, std::move(job)});
    }
    cv_work_.notify_one();
    return true;
}

size_t CompilePool::cancel(Owner owner) {
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = std::remove_if(queue_.begin(), queue_.end(), [owner](const Job& j) { return j.owner == owner; });
        dropped = static_cast<size_t>(queue_.end() - it);
        queue_.erase(it, queue_.end());
    }
    cv_done_.notify_all();
    return dropped;
}

bool CompilePool::owner_idle_locked(Owner owner) const {
    if (auto it = running_.find(owner); it != running_.end() && it->second != 0) return false;
    return std::none_of(queue_.begin(), queue_.end(), [owner](const Job& j) { return j.owner == owner; });
}

void CompilePool::wait_owner(Owner owner) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] {
        // With no workers, queued jobs never run: only the running ones count.
        if (workers_.empty()) return running_.count(owner) == 0 || running_.at(owner) == 0;
        return owner_idle_locked(owner);
    });
}

void CompilePool::wait_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return busy_ == 0 && (queue_.empty() || workers_.empty()); });
}

size_t CompilePool::queued(Owner owner) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(
        std::count_if(queue_.begin(), queue_.end(), [owner](const Job& j) { return j.owner == owner; }));
}

size_t CompilePool::running(Owner owner) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = running_.find(owner);
    return it == running_.end() ? 0 : it->second;
}

void CompilePool::shutdown() {
    std::vector<std::thread> workers;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        if (is_shared_) closed_ = true;
        if (workers_.empty()) return;
        stopping_ = true;
        workers.swap(workers_);
    }
    cv_work_.notify_all();
    cv_done_.notify_all();
    for (std::thread& t : workers) {
        if (t.joinable()) t.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = false;
}

bool CompilePool::is_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !workers_.empty() && !stopping_;
}

size_t CompilePool::thread_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return workers_.size();
}

void CompilePool::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_) return;
            // Highest priority, then oldest.
            auto best = std::min_element(queue_.begin(), queue_.end(), [](const Job& a, const Job& b) {
                return a.priority != b.priority ? a.priority > b.priority : a.seq < b.seq;
            });
            job = std::move(*best);
            queue_.erase(best);
            ++running_[job.owner];
            ++busy_;
        }
        try {
            job.run();
        } catch (...) {
            // A job reports its own failures; one that throws anyway is
            // dropped rather than taking the process down.
        }
        job.run = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--running_[job.owner] == 0) running_.erase(job.owner);
            --busy_;
        }
        cv_done_.notify_all();
    }
}

} // namespace brass::runtime
