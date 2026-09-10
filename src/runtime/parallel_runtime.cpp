#include <brass/runtime/parallel_runtime.hpp>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace brass::runtime {

namespace {

thread_local uint32_t t_current_worker_id = 0;
thread_local bool t_is_worker_thread = false;

int64_t get_initial_reduction_i64(ReductionKind kind) noexcept {
    switch (kind) {
        case ReductionKind::SumI64: return 0;
        case ReductionKind::ProdI64: return 1;
        case ReductionKind::MinI64: return std::numeric_limits<int64_t>::max();
        case ReductionKind::MaxI64: return std::numeric_limits<int64_t>::min();
        default: return 0;
    }
}

double get_initial_reduction_f64(ReductionKind kind) noexcept {
    switch (kind) {
        case ReductionKind::SumF64: return 0.0;
        case ReductionKind::ProdF64: return 1.0;
        case ReductionKind::MinF64: return std::numeric_limits<double>::infinity();
        case ReductionKind::MaxF64: return -std::numeric_limits<double>::infinity();
        default: return 0.0;
    }
}

} // namespace

ParallelRuntime& ParallelRuntime::instance() {
    static ParallelRuntime s_instance;
    return s_instance;
}

ParallelRuntime::ParallelRuntime(uint32_t num_workers) {
    if (num_workers == 0) {
        uint32_t hw = std::thread::hardware_concurrency();
        num_workers = std::clamp(hw == 0 ? 4u : hw, 2u, 8u);
    }
    set_num_workers(num_workers);
}

ParallelRuntime::~ParallelRuntime() {
    stop_workers();
}

void ParallelRuntime::set_num_workers(uint32_t n) {
    if (n == 0) n = 1;
    if (n == num_workers_ && !worker_data_.empty()) return;

    stop_workers();

    num_workers_ = n;
    worker_data_.clear();
    worker_data_.reserve(num_workers_);
    for (uint32_t i = 0; i < num_workers_; ++i) {
        auto w = std::make_unique<ParallelWorker>();
        w->id = i;
        worker_data_.push_back(std::move(w));
    }

    start_workers(num_workers_);
}

uint32_t ParallelRuntime::get_num_workers() const {
    return num_workers_;
}

void ParallelRuntime::start_workers(uint32_t count) {
    stop_flag_.store(false, std::memory_order_relaxed);
    workers_.clear();
    // Worker 0 is always the master thread. Background threads are 1 .. count-1.
    for (uint32_t i = 1; i < count; ++i) {
        workers_.emplace_back(&ParallelRuntime::worker_loop, this, i);
    }
}

void ParallelRuntime::stop_workers() {
    stop_flag_.store(true, std::memory_order_release);
    task_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
}

void ParallelRuntime::worker_loop(uint32_t worker_id) {
    t_current_worker_id = worker_id;
    t_is_worker_thread = true;
    uint32_t last_gen = 0;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(cv_mtx_);
            task_cv_.wait(lock, [this, &last_gen]() {
                return stop_flag_.load(std::memory_order_relaxed) ||
                       task_generation_.load(std::memory_order_acquire) != last_gen;
            });

            if (stop_flag_.load(std::memory_order_relaxed)) break;
            last_gen = task_generation_.load(std::memory_order_acquire);
        }

        execute_work(worker_id);

        if (active_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(cv_mtx_);
            done_cv_.notify_one();
        }
    }
}

void ParallelRuntime::execute_work(uint32_t worker_id) {
    t_current_worker_id = worker_id;
    ParallelChunk chunk;

    while (true) {
        // 1. Try local queue
        if (worker_data_[worker_id]->queue.pop_front(chunk)) {
            current_kernel_(chunk.start, chunk.end, current_context_);
            continue;
        }

        // 2. Local queue empty, attempt work stealing from peers
        bool stolen = false;
        for (uint32_t offset = 1; offset < num_workers_; ++offset) {
            uint32_t victim = (worker_id + offset) % num_workers_;
            if (worker_data_[victim]->queue.steal_back(chunk)) {
                stolen = true;
                current_kernel_(chunk.start, chunk.end, current_context_);
                break;
            }
        }

        if (stolen) continue;

        // 3. Check if all queues are truly empty
        bool any_work_left = false;
        for (uint32_t i = 0; i < num_workers_; ++i) {
            if (!worker_data_[i]->queue.empty()) {
                any_work_left = true;
                break;
            }
        }

        if (!any_work_left) {
            break;
        }

        std::this_thread::yield();
    }
}

void ParallelRuntime::record_reduction_i64(uint32_t worker_id, int64_t val) {
    if (worker_id < worker_data_.size()) {
        worker_data_[worker_id]->red_i64 = combine_reduction_i64(
            current_reduction_, worker_data_[worker_id]->red_i64, val
        );
    }
}

void ParallelRuntime::record_reduction_f64(uint32_t worker_id, double val) {
    if (worker_id < worker_data_.size()) {
        worker_data_[worker_id]->red_f64 = combine_reduction_f64(
            current_reduction_, worker_data_[worker_id]->red_f64, val
        );
    }
}

void ParallelRuntime::parallel_for(
    uint64_t trip_count,
    uint64_t grain_size,
    ParallelKernelFn kernel,
    void* context,
    ReductionKind reduction,
    void* red_target
) {
    if (trip_count == 0 || !kernel) return;

    // Single-worker or trivial workload fallback
    if (num_workers_ <= 1 || trip_count <= grain_size) {
        current_reduction_ = reduction;
        if (is_reduction(reduction)) {
            worker_data_[0]->red_i64 = get_initial_reduction_i64(reduction);
            worker_data_[0]->red_f64 = get_initial_reduction_f64(reduction);
        }

        t_current_worker_id = 0;
        kernel(0, trip_count, context);

        if (is_reduction(reduction) && red_target != nullptr) {
            if (is_fp_reduction(reduction)) {
                double* target = reinterpret_cast<double*>(red_target);
                *target = combine_reduction_f64(reduction, *target, worker_data_[0]->red_f64);
            } else {
                int64_t* target = reinterpret_cast<int64_t*>(red_target);
                *target = combine_reduction_i64(reduction, *target, worker_data_[0]->red_i64);
            }
        }
        return;
    }

    if (grain_size == 0) {
        grain_size = std::max<uint64_t>(1, trip_count / (num_workers_ * 4));
    }
    grain_size = std::min<uint64_t>(grain_size, trip_count);

    uint64_t num_chunks = (trip_count + grain_size - 1) / grain_size;

    for (uint32_t i = 0; i < num_workers_; ++i) {
        worker_data_[i]->queue.clear();
        if (is_reduction(reduction)) {
            worker_data_[i]->red_i64 = get_initial_reduction_i64(reduction);
            worker_data_[i]->red_f64 = get_initial_reduction_f64(reduction);
        }
    }

    for (uint64_t c = 0; c < num_chunks; ++c) {
        uint64_t start = c * grain_size;
        uint64_t end = std::min(start + grain_size, trip_count);
        uint32_t target_worker = static_cast<uint32_t>(c % num_workers_);
        worker_data_[target_worker]->queue.push_back({start, end});
    }

    current_kernel_ = kernel;
    current_context_ = context;
    current_reduction_ = reduction;
    active_tasks_.store(num_workers_, std::memory_order_release);
    task_generation_.fetch_add(1, std::memory_order_acq_rel);

    // Wake up background workers
    task_cv_.notify_all();

    // Master executes worker 0 partition
    execute_work(0);

    // Master decrements task counter and waits for background workers
    if (active_tasks_.fetch_sub(1, std::memory_order_acq_rel) > 1) {
        std::unique_lock<std::mutex> lock(cv_mtx_);
        done_cv_.wait(lock, [this]() {
            return active_tasks_.load(std::memory_order_acquire) == 0;
        });
    }

    // Combine reduction aggregation
    if (is_reduction(reduction) && red_target != nullptr) {
        if (is_fp_reduction(reduction)) {
            double* target = reinterpret_cast<double*>(red_target);
            double total = *target;
            for (uint32_t i = 0; i < num_workers_; ++i) {
                total = combine_reduction_f64(reduction, total, worker_data_[i]->red_f64);
            }
            *target = total;
        } else {
            int64_t* target = reinterpret_cast<int64_t*>(red_target);
            int64_t total = *target;
            for (uint32_t i = 0; i < num_workers_; ++i) {
                total = combine_reduction_i64(reduction, total, worker_data_[i]->red_i64);
            }
            *target = total;
        }
    }

    current_kernel_ = nullptr;
    current_context_ = nullptr;
}

void* ParallelRuntime::alloc_context(size_t bytes) {
    return std::calloc(1, bytes > 0 ? bytes : 8);
}

void ParallelRuntime::free_context(void* ptr) {
    std::free(ptr);
}

} // namespace brass::runtime

extern "C" {

void brass_parallel_for(
    uint64_t trip_count,
    uint64_t grain_size,
    brass::runtime::ParallelKernelFn kernel,
    void* context,
    brass::runtime::ReductionKind reduction,
    void* red_target
) {
    brass::runtime::ParallelRuntime::instance().parallel_for(
        trip_count, grain_size, kernel, context, reduction, red_target
    );
}

void brass_set_parallel_workers(uint32_t workers) {
    brass::runtime::ParallelRuntime::instance().set_num_workers(workers);
}

uint32_t brass_get_parallel_workers() {
    return brass::runtime::ParallelRuntime::instance().get_num_workers();
}

void brass_parallel_reduce_i64(int64_t val) {
    brass::runtime::ParallelRuntime::instance().record_reduction_i64(
        brass::runtime::t_current_worker_id, val
    );
}

void brass_parallel_reduce_f64(double val) {
    brass::runtime::ParallelRuntime::instance().record_reduction_f64(
        brass::runtime::t_current_worker_id, val
    );
}

void* brass_parallel_alloc_context(uint64_t bytes) {
    return brass::runtime::ParallelRuntime::alloc_context(static_cast<size_t>(bytes));
}

void brass_parallel_free_context(void* ptr) {
    brass::runtime::ParallelRuntime::free_context(ptr);
}

}