#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <algorithm>
#include <limits>

namespace brass::runtime {

using ParallelKernelFn = void (*)(uint64_t start_idx, uint64_t end_idx, void* context);

enum class ReductionKind : uint32_t {
    None = 0,
    SumI64 = 1,
    SumF64 = 2,
    ProdI64 = 3,
    ProdF64 = 4,
    MinI64 = 5,
    MinF64 = 6,
    MaxI64 = 7,
    MaxF64 = 8,

    // Aliases
    Sum = SumI64,
    Product = ProdI64,
    Min = MinI64,
    Max = MaxI64,
};

inline bool is_reduction(ReductionKind k) noexcept {
    return k != ReductionKind::None;
}

inline bool is_fp_reduction(ReductionKind k) noexcept {
    return k == ReductionKind::SumF64 || k == ReductionKind::ProdF64 ||
           k == ReductionKind::MinF64 || k == ReductionKind::MaxF64;
}

inline int64_t combine_reduction_i64(ReductionKind kind, int64_t acc, int64_t val) noexcept {
    switch (kind) {
        case ReductionKind::SumI64: return acc + val;
        case ReductionKind::ProdI64: return acc * val;
        case ReductionKind::MinI64: return std::min(acc, val);
        case ReductionKind::MaxI64: return std::max(acc, val);
        default: return acc;
    }
}

inline double combine_reduction_f64(ReductionKind kind, double acc, double val) noexcept {
    switch (kind) {
        case ReductionKind::SumF64: return acc + val;
        case ReductionKind::ProdF64: return acc * val;
        case ReductionKind::MinF64: return std::min(acc, val);
        case ReductionKind::MaxF64: return std::max(acc, val);
        default: return acc;
    }
}

struct ParallelChunk {
    uint64_t start = 0;
    uint64_t end = 0;
};

class WorkStealingQueue {
public:
    WorkStealingQueue() = default;

    void push_back(ParallelChunk chunk) {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.push_back(chunk);
    }

    bool pop_front(ParallelChunk& chunk) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (deque_.empty()) return false;
        chunk = deque_.front();
        deque_.pop_front();
        return true;
    }

    bool steal_back(ParallelChunk& chunk) {
        std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
        if (!lock.owns_lock() || deque_.empty()) return false;
        chunk = deque_.back();
        deque_.pop_back();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.clear();
    }

private:
    std::deque<ParallelChunk> deque_;
    mutable std::mutex mtx_;
};

struct ParallelWorker {
    uint32_t id = 0;
    WorkStealingQueue queue;
    int64_t red_i64 = 0;
    double red_f64 = 0.0;
};

class ParallelRuntime {
public:
    static ParallelRuntime& instance();

    explicit ParallelRuntime(uint32_t num_workers = 0);
    ~ParallelRuntime();

    ParallelRuntime(const ParallelRuntime&) = delete;
    ParallelRuntime& operator=(const ParallelRuntime&) = delete;

    void set_num_workers(uint32_t n);
    uint32_t get_num_workers() const;

    void parallel_for(
        uint64_t trip_count,
        uint64_t grain_size,
        ParallelKernelFn kernel,
        void* context,
        ReductionKind reduction = ReductionKind::None,
        void* red_target = nullptr
    );

    void record_reduction_i64(uint32_t worker_id, int64_t val);
    void record_reduction_f64(uint32_t worker_id, double val);

    static void* alloc_context(size_t bytes);
    static void free_context(void* ptr);

private:
    void start_workers(uint32_t count);
    void stop_workers();
    void worker_loop(uint32_t worker_id);
    void execute_work(uint32_t worker_id);

    uint32_t num_workers_ = 0;
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<ParallelWorker>> worker_data_;

    std::mutex cv_mtx_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    std::atomic<bool> stop_flag_{false};

    ParallelKernelFn current_kernel_ = nullptr;
    void* current_context_ = nullptr;
    ReductionKind current_reduction_ = ReductionKind::None;
    std::atomic<uint32_t> active_tasks_{0};
    std::atomic<uint32_t> task_generation_{0};
};

} // namespace brass::runtime

extern "C" {

void brass_parallel_for(
    uint64_t trip_count,
    uint64_t grain_size,
    brass::runtime::ParallelKernelFn kernel,
    void* context,
    brass::runtime::ReductionKind reduction,
    void* red_target
);

void brass_set_parallel_workers(uint32_t workers);
uint32_t brass_get_parallel_workers();

void brass_parallel_reduce_i64(int64_t val);
void brass_parallel_reduce_f64(double val);

void* brass_parallel_alloc_context(uint64_t bytes);
void brass_parallel_free_context(void* ptr);

}