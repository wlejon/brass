#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace brass::runtime {

// Worker threads that run compile jobs (tier-2 tier-ups, OSR entries) off
// the mutator. shared() is the process's one pool: every program's
// background compiler submits to it, so a process running many programs
// (every script, eval and worker of a host) compiles on a fixed number of
// threads, sized for the machine, however many programs it holds. A
// standalone BackgroundCompiler owns a private pool instead.
//
// Jobs run highest priority first, then in submission order. Each job is
// submitted on behalf of an owner (a program's background compiler, its
// OSR coordinator), which can drop its queued jobs and wait out its running
// ones without touching anyone else's.
class CompilePool {
public:
    using Owner = const void*;

    // The process's pool, sized default_thread_count(). Its threads start
    // with the first job submitted to it, so a process that never tiers
    // anything up never starts one. BRASS_JIT_THREADS in the environment
    // overrides the size.
    static CompilePool& shared();
    // A quarter of the hardware threads, at least one and at most four: a
    // compile job is long, and the threads it would take are the mutator's
    // and the host's own.
    static size_t default_thread_count();

    // A private pool of `threads` workers, started now (none: jobs queue
    // until start()).
    explicit CompilePool(size_t threads);
    ~CompilePool();

    CompilePool(const CompilePool&) = delete;
    CompilePool& operator=(const CompilePool&) = delete;

    // Queues `job` for `owner`. False, and the job dropped, once the pool
    // is shut down. A job must not throw; one that does is dropped with
    // its exception.
    bool submit(Owner owner, uint8_t priority, std::function<void()> job);
    // Drops `owner`'s queued jobs; returns how many.
    size_t cancel(Owner owner);
    // Blocks until `owner` has no job queued or running.
    void wait_owner(Owner owner);
    // Blocks until nothing is queued or running.
    void wait_idle();
    size_t queued(Owner owner) const;
    size_t running(Owner owner) const;

    // Starts `threads` workers when none run (a running pool keeps its
    // threads). A shut-down pool starts again.
    void start(size_t threads);
    // Drops every queued job, waits out the running ones and joins the
    // workers. The pool starts again on start(), or, for shared(), never:
    // shutting the shared pool down is for a process about to exit, and a
    // later submit is refused.
    void shutdown();
    bool is_running() const;
    size_t thread_count() const;

private:
    struct Job {
        Owner owner;
        uint8_t priority;
        uint64_t seq;
        std::function<void()> run;
    };

    void worker_loop();
    void start_locked(size_t threads);
    bool owner_idle_locked(Owner owner) const;

    const bool is_shared_;
    size_t configured_threads_;
    mutable std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::vector<std::thread> workers_;
    std::vector<Job> queue_;
    std::unordered_map<Owner, size_t> running_;
    size_t busy_ = 0;
    uint64_t next_seq_ = 1;
    bool stopping_ = false;
    bool closed_ = false;

    struct SharedTag {};
    CompilePool(SharedTag, size_t threads);
};

} // namespace brass::runtime
