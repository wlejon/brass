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
    static BackgroundCompiler s_instance(2);
    return s_instance;
}

BackgroundCompiler::BackgroundCompiler(size_t num_threads)
    : BackgroundCompiler(BackgroundCompilerConfig{num_threads, Target::host()}) {}

BackgroundCompiler::BackgroundCompiler(const BackgroundCompilerConfig& config)
    : config_(config), installer_(config.target) {
    if (config_.num_threads > 0) {
        start(config_.num_threads);
    }
}

BackgroundCompiler::~BackgroundCompiler() {
    stop();
}

void BackgroundCompiler::start(size_t num_threads) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (running_) {
        if (workers_.size() == num_threads) {
            return;
        }
        lock.unlock();
        stop();
        lock.lock();
    }

    stopping_ = false;
    running_ = true;
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&BackgroundCompiler::worker_loop, this, i);
    }
}

void BackgroundCompiler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        stopping_ = true;
    }
    cv_work_.notify_all();

    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    workers_.clear();
    running_ = false;
    stopping_ = false;
    queue_ = std::priority_queue<CompileTask>();
    active_names_.clear();
    statuses_.clear();
    busy_workers_ = 0;
}

bool BackgroundCompiler::is_running() const noexcept {
    return running_.load(std::memory_order_relaxed);
}

void BackgroundCompiler::wait_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_idle_.wait(lock, [this]() {
        return queue_.empty() && busy_workers_ == 0;
    });
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
    auto mod_copy = clone_module(module);
    if (!mod_copy) return false;
    return enqueue(fn_name, std::move(mod_copy), handle, priority, target_tier);
}

bool BackgroundCompiler::enqueue(
    std::string_view fn_name,
    std::unique_ptr<Module> module_copy,
    FunctionHandle* handle,
    CompilePriority priority,
    TierLevel target_tier
) {
    if (!module_copy) return false;
    std::string key(fn_name);

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return false;

    // 1. Ensure worker threads are started
    if (!running_ && config_.num_threads > 0) {
        stopping_ = false;
        running_ = true;
        workers_.reserve(config_.num_threads);
        for (size_t i = 0; i < config_.num_threads; ++i) {
            workers_.emplace_back(&BackgroundCompiler::worker_loop, this, i);
        }
    }

    // 2. Request deduplication: if function is already queued or compiling, reject duplicate
    if (active_names_.contains(key)) {
        stats_.tasks_deduplicated++;
        return false;
    }

    if (!handle) {
        handle = FunctionDispatchTable::instance().get_or_create(fn_name);
    }

    // If native entry is already installed, no need to compile
    if (handle->has_native_entry()) {
        return false;
    }

    active_names_.insert(key);
    statuses_[key] = CompileStatus::Pending;

    CompileTask task;
    task.id = next_task_id_++;
    task.function_name = key;
    task.module_copy = std::move(module_copy);
    task.target_tier = target_tier;
    task.priority = priority;
    task.status = CompileStatus::Pending;
    task.handle = handle;
    task.enqueue_time = std::chrono::high_resolution_clock::now();

    queue_.push(std::move(task));
    stats_.tasks_enqueued++;

    cv_work_.notify_one();
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
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

size_t BackgroundCompiler::active_workers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return busy_workers_;
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

void BackgroundCompiler::worker_loop(size_t /*worker_id*/) {
    while (true) {
        CompileTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [this]() {
                return stopping_ || !queue_.empty();
            });

            if (stopping_ && queue_.empty()) {
                break;
            }

            task = std::move(const_cast<CompileTask&>(queue_.top()));
            queue_.pop();
            task.status = CompileStatus::Compiling;
            statuses_[task.function_name] = CompileStatus::Compiling;
            busy_workers_++;
        }

        // Compilation executes outside the mutex
        auto t0 = std::chrono::high_resolution_clock::now();
        CodeInstallResult res = installer_.install_tier2(
            *task.handle,
            std::move(task.module_copy),
            task.function_name
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t dur_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );

        {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_workers_--;
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

            cv_fn_done_.notify_all();
            if (queue_.empty() && busy_workers_ == 0) {
                cv_idle_.notify_all();
            }
        }
    }
}

} // namespace brass::runtime
