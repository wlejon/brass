#include <brass/core/parallel_for.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace brass {

namespace {

#if !defined(_WIN32)
// The workers run the same passes the calling thread would, on the module's
// largest functions too, and some walks recurse with the size of a function
// (its dominator tree, its loop nest). A process's main thread typically has
// 8 MB; a std::thread gets what the platform gives secondary threads (512 KB
// on macOS). Reserved, not committed: untouched pages cost nothing.
constexpr size_t kWorkerStackBytes = size_t{64} << 20;

class Worker {
public:
    explicit Worker(std::function<void()> fn) : fn_(std::make_unique<std::function<void()>>(std::move(fn))) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, kWorkerStackBytes);
        started_ = pthread_create(&thread_, &attr, &Worker::entry, fn_.get()) == 0;
        pthread_attr_destroy(&attr);
        // Without a thread of its own (resources exhausted), the work runs
        // here: the result is the same, only slower.
        if (!started_) (*fn_)();
    }
    void join() {
        if (started_) pthread_join(thread_, nullptr);
        started_ = false;
    }

private:
    static void* entry(void* arg) {
        (*static_cast<std::function<void()>*>(arg))();
        return nullptr;
    }
    std::unique_ptr<std::function<void()>> fn_;
    pthread_t thread_{};
    bool started_ = false;
};
#else
class Worker {
public:
    explicit Worker(std::function<void()> fn) : thread_(std::move(fn)) {}
    void join() { thread_.join(); }

private:
    std::thread thread_;
};
#endif

} // namespace

size_t resolve_thread_count(size_t requested) noexcept {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1;
}

size_t threads_for_work(size_t requested, size_t items, size_t total_weight,
                        size_t min_weight_per_thread) noexcept {
    size_t n = resolve_thread_count(requested);
    if (min_weight_per_thread > 0) n = std::min(n, std::max<size_t>(1, total_weight / min_weight_per_thread));
    n = std::min(n, std::max<size_t>(1, items));
    return n;
}

std::vector<size_t> heaviest_first(const std::vector<size_t>& weights) {
    std::vector<size_t> order(weights.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    return order;
}

void parallel_for_each(const std::vector<size_t>& order, size_t threads,
                       const std::function<void(size_t)>& body) {
    std::mutex error_mutex;
    std::exception_ptr error;
    size_t error_item = 0;
    auto run_one = [&](size_t item) {
        try {
            body(item);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error || item < error_item) {
                error = std::current_exception();
                error_item = item;
            }
        }
    };

    const size_t workers = std::min(threads, order.size());
    if (workers <= 1) {
        for (size_t item : order) run_one(item);
    } else {
        std::atomic<size_t> next{0};
        auto loop = [&] {
            for (;;) {
                const size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= order.size()) return;
                run_one(order[k]);
            }
        };
        std::vector<std::unique_ptr<Worker>> pool;
        pool.reserve(workers - 1);
        for (size_t t = 1; t < workers; ++t) pool.push_back(std::make_unique<Worker>(loop));
        loop();
        for (auto& w : pool) w->join();
    }
    if (error) std::rethrow_exception(error);
}

} // namespace brass
