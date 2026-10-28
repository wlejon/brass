#include <brass/runtime/compile_pool.hpp>
#include <brass/core/env.hpp>
#include <brass/runtime/tier_timeline.hpp>

#include <algorithm>
#include <chrono>
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
    const long n = std::strtol(env_value("BRASS_JIT_THREADS").c_str(), nullptr, 10);
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
    live_ += threads;
    for (size_t i = 0; i < threads; ++i) workers_.emplace_back(&CompilePool::worker_loop, this);
}

void CompilePool::spawn_locked() {
    if (queue_.size() <= idle_ || live_ >= configured_threads_) return;
    // Join the workers that ended idle: each recorded itself and let go of
    // the lock on its way out, so joining under it cannot wait on it.
    if (!exited_.empty()) {
        for (auto it = workers_.begin(); it != workers_.end();) {
            if (std::find(exited_.begin(), exited_.end(), it->get_id()) == exited_.end()) {
                ++it;
                continue;
            }
            it->join();
            it = workers_.erase(it);
        }
        exited_.clear();
    }
    ++live_;
    workers_.emplace_back(&CompilePool::worker_loop, this);
}

void CompilePool::start(size_t threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_locked(threads);
}

bool CompilePool::submit(Owner owner, uint8_t priority, std::function<void()> job) {
    return submit(owner, priority, std::move(job), nullptr);
}

bool CompilePool::submit(Owner owner, uint8_t priority, std::function<void()> job,
                         std::function<uint64_t()> heat) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || stopping_) return false;
        const uint64_t at_submit = heat ? heat() : 0;
        queue_.push_back(Job{owner, priority, next_seq_++, std::move(job), std::move(heat), at_submit});
        if (is_shared_) {
            static const bool registered = [] { return std::atexit(&shutdown_shared_at_exit) == 0; }();
            (void)registered;
            spawn_locked();
        }
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
        if (live_ == 0) return running_.count(owner) == 0 || running_.at(owner) == 0;
        return owner_idle_locked(owner);
    });
}

void CompilePool::wait_idle() {
    if (process_exiting()) return;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return busy_ == 0 && (queue_.empty() || live_ == 0); });
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
    exited_.clear();
    stopping_ = false;
}

bool CompilePool::is_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_ > 0 && !stopping_;
}

size_t CompilePool::thread_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_;
}

// Jobs below this priority (tier-2 compiles, which can run for a long time
// on a large function) may fill all workers but one: the last is kept for
// the short, latency-bound ones (Tier 1, whose callers run in Tier 0 through
// a bridge until it lands, and OSR entries).
constexpr uint8_t kReservedPriority = 128;

std::vector<CompilePool::Job>::iterator CompilePool::pick_locked() {
    // Against the pool's size, not the threads alive now: the shared pool
    // starts another worker for the latency-bound job that cannot start.
    const size_t size = is_shared_ ? configured_threads_ : workers_.size();
    const size_t bulk_limit = size > 1 ? size - 1 : size;
    const bool bulk_ok = bulk_running_ < bulk_limit;
    // Highest priority, then most heat gained while queued, then oldest,
    // among the jobs that may start now. Arrival order alone left a hot
    // function's tier-up behind a burst of lukewarm ones requested just
    // before it (mesh_churn_2k: 8 to 10 ms queued at its start).
    auto best = queue_.end();
    uint64_t best_gain = 0;
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->priority < kReservedPriority && !bulk_ok) continue;
        const uint64_t now = it->heat ? it->heat() : 0;
        const uint64_t gain = now > it->heat_at_submit ? now - it->heat_at_submit : 0;
        if (best == queue_.end() || it->priority > best->priority ||
            (it->priority == best->priority &&
             (gain > best_gain || (gain == best_gain && it->seq < best->seq)))) {
            best = it;
            best_gain = gain;
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
            auto ready = [this, &best] {
                if (stopping_) return true;
                best = pick_locked();
                return best != queue_.end();
            };
            ++idle_;
            bool woke = true;
            if (is_shared_) {
                // A shared worker with nothing to do for this long ends; the
                // next submit starts another.
                woke = cv_work_.wait_for(lock, std::chrono::seconds(30), ready);
            } else {
                cv_work_.wait(lock, ready);
            }
            --idle_;
            if (stopping_ || !woke) {
                --live_;
                if (!stopping_) exited_.push_back(std::this_thread::get_id());
                return;
            }
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
