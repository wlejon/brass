#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/tier_timeline.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace brass::runtime {

bool process_exiting() noexcept {
#if defined(_WIN32)
    using ShutdownInProgress = BOOLEAN(NTAPI*)();
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return false;
    const auto fn = reinterpret_cast<ShutdownInProgress>(GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
    return fn && fn();
#else
    return false;
#endif
}

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
    return std::clamp<size_t>(hw / 4, 1, 8);
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
    if (process_exiting()) return 0;
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
    if (process_exiting()) return;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] {
        // With no workers, queued jobs never run: only the running ones count.
        if (workers_.empty()) return running_.count(owner) == 0 || running_.at(owner) == 0;
        return owner_idle_locked(owner);
    });
}

void CompilePool::wait_idle() {
    if (process_exiting()) return;
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
    if (process_exiting()) return;
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

// Jobs below this priority (tier-2 compiles, which can run for a long time
// on a large function) may fill all workers but one: the last is kept for
// the short, latency-bound ones (Tier 1, whose callers run in Tier 0 through
// a bridge until it lands, and OSR entries).
constexpr uint8_t kReservedPriority = 128;

std::vector<CompilePool::Job>::iterator CompilePool::pick_locked() {
    const size_t bulk_limit = workers_.size() > 1 ? workers_.size() - 1 : workers_.size();
    const bool bulk_ok = bulk_running_ < bulk_limit;
    // Highest priority, then oldest, among the jobs that may start now.
    auto best = queue_.end();
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->priority < kReservedPriority && !bulk_ok) continue;
        if (best == queue_.end() || it->priority > best->priority ||
            (it->priority == best->priority && it->seq < best->seq)) {
            best = it;
        }
    }
    return best;
}

void CompilePool::worker_loop() {
    detail::t_on_compile_worker = true;
    for (;;) {
        Job job;
        bool bulk = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            auto best = queue_.end();
            cv_work_.wait(lock, [this, &best] {
                if (stopping_) return true;
                best = pick_locked();
                return best != queue_.end();
            });
            if (stopping_) return;
            job = std::move(*best);
            queue_.erase(best);
            ++running_[job.owner];
            ++busy_;
            bulk = job.priority < kReservedPriority;
            if (bulk) ++bulk_running_;
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
            if (bulk) --bulk_running_;
        }
        cv_done_.notify_all();
        // A bulk slot freed: a worker parked on a queue of only bulk jobs may
        // take one now.
        if (bulk) cv_work_.notify_one();
    }
}

} // namespace brass::runtime
