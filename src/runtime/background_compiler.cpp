#include <brass/runtime/background_compiler.hpp>
#include <brass/mir/loop_opt.hpp>
#include <iostream>
#include <iomanip>

namespace brass::runtime {

std::string_view to_string(CompilePriority priority) noexcept {
    switch (priority) {
        case CompilePriority::Normal: return "Normal";
        case CompilePriority::High:   return "High";
    }
    return "Unknown";
}

std::string_view to_string(CompileStatus status) noexcept {
    switch (status) {
        case CompileStatus::Pending:   return "Pending";
        case CompileStatus::Compiling: return "Compiling";
        case CompileStatus::Completed: return "Completed";
        case CompileStatus::Failed:    return "Failed";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, CompilePriority priority) {
    return os << to_string(priority);
}

std::ostream& operator<<(std::ostream& os, CompileStatus status) {
    return os << to_string(status);
}

BackgroundCompiler& BackgroundCompiler::instance() {
    static BackgroundCompiler s_instance(BackgroundCompilerConfig{0, Target::host(), nullptr, &CompilePool::shared()});
    return s_instance;
}

BackgroundCompiler::BackgroundCompiler(size_t num_threads)
    : BackgroundCompiler(BackgroundCompilerConfig{num_threads, Target::host()}) {}

BackgroundCompiler::BackgroundCompiler(const BackgroundCompilerConfig& config)
    : config_(config),
      installer_(config.table ? *config.table : FunctionDispatchTable::instance(), config.target),
      own_pool_(config.pool ? nullptr : std::make_unique<CompilePool>(config.num_threads)),
      pool_(config.pool ? config.pool : own_pool_.get()) {}

BackgroundCompiler::~BackgroundCompiler() {
    stop();
}

void BackgroundCompiler::start(size_t num_threads) {
    if (own_pool_) own_pool_->start(num_threads);
}

void BackgroundCompiler::stop() {
    if (process_exiting()) return;
    cancel_pending();
    pool_->wait_owner(this);
    if (own_pool_) own_pool_->shutdown();
}

size_t BackgroundCompiler::cancel_pending() {
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dropped = pool_->cancel(this);
        // A worker may already hold one of these: it finds its name gone
        // and drops it (run_task).
        for (const std::string& name : queued_names_) {
            active_names_.erase(name);
            statuses_.erase(name);
        }
        queued_names_.clear();
    }
    cv_fn_done_.notify_all();
    return dropped;
}

bool BackgroundCompiler::is_running() const noexcept {
    return own_pool_ ? own_pool_->is_running() : true;
}

void BackgroundCompiler::wait_idle() {
    pool_->wait_owner(this);
}

bool BackgroundCompiler::wait_for_function(std::string_view fn_name, std::chrono::milliseconds timeout) {
    std::string key(fn_name);
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_fn_done_.wait_for(lock, timeout, [&]() {
        auto it = statuses_.find(key);
        if (it == statuses_.end()) {
            // Function was not even queued or already cleaned up
            return !active_names_.contains(key);
        }
        return it->second == CompileStatus::Completed || it->second == CompileStatus::Failed;
    });
}

bool BackgroundCompiler::enqueue(
    std::string_view fn_name,
    const Module& module,
    FunctionHandle* handle,
    CompilePriority priority,
    TierLevel target_tier
) {
    // The cheap refusals first: the invocation hook calls this on every
    // call past the threshold until the code is replaced.
    if (handle && !tier2_candidate(*handle)) return false;
    if (is_queued_or_compiling(fn_name)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.tasks_deduplicated++;
        return false;
    }
    // The bindings are captured before the clone is taken: a handle rebound
    // after this point is not published to, one rebound before it is not
    // compiled for the old module.
    if (!handle) handle = installer_.dispatch_table().get_or_create(fn_name);
    auto mod_copy = clone_for_tier2(installer_.dispatch_table(), module, fn_name);
    if (!mod_copy) return false;
    const Tier2Bindings bindings = installer_.capture_tier2_bindings(*handle, *mod_copy, fn_name, &module);
    // Bound to another module's Function: this module's code is not its.
    if (bindings.target_foreign) return false;
    return enqueue_copy(fn_name, std::move(mod_copy), handle, priority, target_tier, &bindings);
}

bool BackgroundCompiler::tier2_candidate(const FunctionHandle& handle) {
    // Tier 0 without native code, or Tier 1 (whose baseline entry tier 2
    // replaces). Tier-2 code, or native code installed some other way, is
    // not recompiled, nor is a Function tier 2 already rejected.
    const TierLevel t = handle.tier();
    if (t == TierLevel::Tier2_Optimized) return false;
    if (handle.has_native_entry() && t != TierLevel::Tier1_Baseline) return false;
    return !handle.tier2_rejected();
}

bool BackgroundCompiler::enqueue(
    std::string_view fn_name,
    std::unique_ptr<Module> module_copy,
    FunctionHandle* handle,
    CompilePriority priority,
    TierLevel target_tier
) {
    return enqueue_copy(fn_name, std::move(module_copy), handle, priority, target_tier, nullptr);
}

bool BackgroundCompiler::enqueue_copy(std::string_view fn_name, std::unique_ptr<Module> module_copy,
                                      FunctionHandle* handle, CompilePriority priority, TierLevel target_tier,
                                      const Tier2Bindings* bindings) {
    if (!module_copy) return false;
    std::string key(fn_name);

    std::lock_guard<std::mutex> lock(mutex_);

    // A private pool stopped since its last task starts again.
    if (own_pool_ && !own_pool_->is_running() && config_.num_threads > 0) own_pool_->start(config_.num_threads);

    // Request deduplication: if function is already queued or compiling, reject duplicate
    if (active_names_.contains(key)) {
        stats_.tasks_deduplicated++;
        return false;
    }

    if (!handle) {
        handle = installer_.dispatch_table().get_or_create(fn_name);
    }

    if (!tier2_candidate(*handle)) return false;
    Tier2Bindings captured =
        bindings ? *bindings : installer_.capture_tier2_bindings(*handle, *module_copy, key, nullptr);
    if (captured.target_foreign) return false;

    auto task = std::make_shared<CompileTask>();
    task->id = next_task_id_++;
    task->function_name = key;
    task->module_copy = std::move(module_copy);
    task->target_tier = target_tier;
    task->priority = priority;
    task->status = CompileStatus::Pending;
    task->handle = handle;
    task->bindings = std::move(captured);
    task->enqueue_time = std::chrono::high_resolution_clock::now();

    // The pool calls back under its own lock only to queue; this lock is
    // never taken by the pool, so holding it here is safe.
    if (!pool_->submit(this, static_cast<uint8_t>(priority), [this, task] { run_task(*task); })) return false;

    active_names_.insert(key);
    queued_names_.insert(key);
    statuses_[key] = CompileStatus::Pending;
    stats_.tasks_enqueued++;
    return true;
}

bool BackgroundCompiler::is_queued_or_compiling(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_names_.contains(std::string(fn_name));
}

CompileStatus BackgroundCompiler::get_task_status(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = statuses_.find(std::string(fn_name));
    if (it != statuses_.end()) {
        return it->second;
    }
    return CompileStatus::Pending;
}

size_t BackgroundCompiler::queue_size() const {
    return pool_->queued(this);
}

size_t BackgroundCompiler::active_workers() const {
    return pool_->running(this);
}

BackgroundCompilerStats BackgroundCompiler::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void BackgroundCompiler::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = BackgroundCompilerStats{};
}

void BackgroundCompiler::dump_stats(std::ostream& os) const {
    BackgroundCompilerStats s = stats();
    double compile_ms = static_cast<double>(s.total_compile_time_us) / 1000.0;
    os << "=== Background JIT Compiler Statistics ===\n"
       << "  Worker Threads:     " << thread_count() << "\n"
       << "  Tasks Enqueued:     " << s.tasks_enqueued << "\n"
       << "  Tasks Completed:    " << s.tasks_completed << "\n"
       << "  Tasks Failed:       " << s.tasks_failed << "\n"
       << "  Tasks Deduplicated: " << s.tasks_deduplicated << "\n"
       << "  Total Compile Time: " << std::fixed << std::setprecision(2) << compile_ms << " ms\n"
       << "==========================================\n";
}

void BackgroundCompiler::run_task(CompileTask& task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Cancelled after a worker took it from the pool's queue.
        if (queued_names_.erase(task.function_name) == 0) return;
        task.status = CompileStatus::Compiling;
        statuses_[task.function_name] = CompileStatus::Compiling;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    // install_tier2 reports compile errors as results; anything still
    // thrown (bad_alloc, a bug) fails this task and leaves the function
    // on its lower tier rather than terminating the process.
    CodeInstallResult res;
    try {
        res = installer_.install_tier2(*task.handle, std::move(task.module_copy), task.function_name, task.bindings);
    } catch (const std::exception& e) {
        res = {false, nullptr, std::string("Tier-2 compilation threw: ") + e.what(), 0};
        task.handle->mark_tier2_rejected(task.bindings.target);
    } catch (...) {
        res = {false, nullptr, "Tier-2 compilation threw a non-standard exception", 0};
        task.handle->mark_tier2_rejected(task.bindings.target);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto dur_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_names_.erase(task.function_name);
        if (res.success) {
            task.status = CompileStatus::Completed;
            statuses_[task.function_name] = CompileStatus::Completed;
            stats_.tasks_completed++;
        } else {
            task.status = CompileStatus::Failed;
            task.error_message = res.error_message;
            statuses_[task.function_name] = CompileStatus::Failed;
            stats_.tasks_failed++;
        }
        stats_.total_compile_time_us += dur_us;
    }
    cv_fn_done_.notify_all();
}

} // namespace brass::runtime
