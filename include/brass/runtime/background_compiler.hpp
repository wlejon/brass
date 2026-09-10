#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/code_installer.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iosfwd>
#include <cstdint>

namespace brass::runtime {

enum class CompilePriority : uint8_t {
    Normal = 0, // Invocation tier-ups
    High = 1    // Hot loops / OSR
};

enum class CompileStatus : uint8_t {
    Pending = 0,
    Compiling = 1,
    Completed = 2,
    Failed = 3
};

std::string_view to_string(CompilePriority priority) noexcept;
std::string_view to_string(CompileStatus status) noexcept;
std::ostream& operator<<(std::ostream& os, CompilePriority priority);
std::ostream& operator<<(std::ostream& os, CompileStatus status);

struct CompileTask {
    uint64_t id = 0;
    std::string function_name;
    std::unique_ptr<Module> module_copy;
    TierLevel target_tier = TierLevel::Tier2_Optimized;
    CompilePriority priority = CompilePriority::Normal;
    CompileStatus status = CompileStatus::Pending;
    FunctionHandle* handle = nullptr;
    std::string error_message;
    std::chrono::high_resolution_clock::time_point enqueue_time;
    std::chrono::high_resolution_clock::time_point finish_time;

    // Strict priority ordering: High priority first, then FIFO by ID
    bool operator<(const CompileTask& other) const noexcept {
        if (priority != other.priority) {
            return static_cast<uint8_t>(priority) < static_cast<uint8_t>(other.priority);
        }
        return id > other.id;
    }
};

struct BackgroundCompilerConfig {
    size_t num_threads = 2;
    Target target = Target::host();
};

struct BackgroundCompilerStats {
    uint64_t tasks_enqueued = 0;
    uint64_t tasks_completed = 0;
    uint64_t tasks_failed = 0;
    uint64_t tasks_deduplicated = 0;
    uint64_t total_compile_time_us = 0;
};

class BackgroundCompiler {
public:
    explicit BackgroundCompiler(size_t num_threads = 2);
    explicit BackgroundCompiler(const BackgroundCompilerConfig& config);
    ~BackgroundCompiler();

    BackgroundCompiler(const BackgroundCompiler&) = delete;
    BackgroundCompiler& operator=(const BackgroundCompiler&) = delete;

    static BackgroundCompiler& instance();

    // Lifecycle
    void start(size_t num_threads);
    void stop();
    bool is_running() const noexcept;
    void wait_idle();
    bool wait_for_function(std::string_view fn_name, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    // Enqueue compilation task
    bool enqueue(
        std::string_view fn_name,
        const Module& module,
        FunctionHandle* handle = nullptr,
        CompilePriority priority = CompilePriority::Normal,
        TierLevel target_tier = TierLevel::Tier2_Optimized
    );

    bool enqueue(
        std::string_view fn_name,
        std::unique_ptr<Module> module_copy,
        FunctionHandle* handle = nullptr,
        CompilePriority priority = CompilePriority::Normal,
        TierLevel target_tier = TierLevel::Tier2_Optimized
    );

    // Queries
    bool is_queued_or_compiling(std::string_view fn_name) const;
    CompileStatus get_task_status(std::string_view fn_name) const;
    size_t queue_size() const;
    size_t active_workers() const;
    size_t thread_count() const noexcept { return workers_.size(); }

    // Statistics
    BackgroundCompilerStats stats() const;
    void reset_stats();
    void dump_stats(std::ostream& os) const;

    CodeInstaller& installer() noexcept { return installer_; }
    const CodeInstaller& installer() const noexcept { return installer_; }

private:
    void worker_loop(size_t worker_id);

    BackgroundCompilerConfig config_;
    CodeInstaller installer_;

    mutable std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_idle_;
    std::condition_variable cv_fn_done_;

    std::vector<std::thread> workers_;
    std::priority_queue<CompileTask> queue_;
    std::unordered_set<std::string> active_names_;
    std::unordered_map<std::string, CompileStatus> statuses_;

    std::atomic<bool> stopping_{false};
    std::atomic<bool> running_{false};
    size_t busy_workers_ = 0;
    uint64_t next_task_id_ = 1;

    BackgroundCompilerStats stats_;
};

} // namespace brass::runtime
