#include "test_framework.hpp"
#include <brass/runtime/parallel_runtime.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace brass::runtime;

TEST_CASE("WorkStealingQueue Basic Operations") {
    WorkStealingQueue queue;
    CHECK(queue.empty());
    CHECK_EQ(queue.size(), 0u);

    ParallelChunk chunk;
    CHECK(!queue.pop_front(chunk));
    CHECK(!queue.steal_back(chunk));

    queue.push_back({0, 100});
    queue.push_back({100, 200});
    queue.push_back({200, 300});
    CHECK(!queue.empty());
    CHECK_EQ(queue.size(), 3u);

    // pop_front gets first element
    CHECK(queue.pop_front(chunk));
    CHECK_EQ(chunk.start, 0u);
    CHECK_EQ(chunk.end, 100u);

    // steal_back gets back element
    CHECK(queue.steal_back(chunk));
    CHECK_EQ(chunk.start, 200u);
    CHECK_EQ(chunk.end, 300u);

    // remaining element
    CHECK(queue.pop_front(chunk));
    CHECK_EQ(chunk.start, 100u);
    CHECK_EQ(chunk.end, 200u);

    CHECK(queue.empty());
}

TEST_CASE("Parallel Workers Configuration") {
    uint32_t orig = brass_get_parallel_workers();
    CHECK(orig > 0u);

    brass_set_parallel_workers(4);
    CHECK_EQ(brass_get_parallel_workers(), 4u);

    brass_set_parallel_workers(2);
    CHECK_EQ(brass_get_parallel_workers(), 2u);

    // Restore
    brass_set_parallel_workers(orig);
    CHECK_EQ(brass_get_parallel_workers(), orig);
}

struct ParallelContext {
    int64_t* input;
    int64_t* output;
};

static void square_kernel(uint64_t start, uint64_t end, void* context) {
    auto* ctx = static_cast<ParallelContext*>(context);
    for (uint64_t i = start; i < end; ++i) {
        ctx->output[i] = ctx->input[i] * ctx->input[i];
    }
}

TEST_CASE("brass_parallel_for DOALL Execution") {
    const size_t N = 10000;
    std::vector<int64_t> input(N);
    std::vector<int64_t> output(N, 0);
    for (size_t i = 0; i < N; ++i) {
        input[i] = static_cast<int64_t>(i);
    }

    ParallelContext ctx{input.data(), output.data()};
    brass_parallel_for(
        N,
        64,
        square_kernel,
        &ctx,
        ReductionKind::None,
        nullptr
    );

    for (size_t i = 0; i < N; ++i) {
        int64_t expected = static_cast<int64_t>(i) * static_cast<int64_t>(i);
        CHECK_EQ(output[i], expected);
    }
}

struct SumReduceContext {
    int64_t* arr;
};

static void sum_kernel(uint64_t start, uint64_t end, void* context) {
    auto* ctx = static_cast<SumReduceContext*>(context);
    int64_t local_sum = 0;
    for (uint64_t i = start; i < end; ++i) {
        local_sum += ctx->arr[i];
    }
    brass_parallel_reduce_i64(local_sum);
}

TEST_CASE("brass_parallel_for Sum Reduction") {
    const size_t N = 20000;
    std::vector<int64_t> arr(N, 1);
    SumReduceContext ctx{arr.data()};

    int64_t red_target = 0;
    brass_parallel_for(
        N,
        128,
        sum_kernel,
        &ctx,
        ReductionKind::SumI64,
        &red_target
    );

    CHECK_EQ(red_target, static_cast<int64_t>(N));
}

struct MinMaxContext {
    int64_t* arr;
};

static void min_kernel(uint64_t start, uint64_t end, void* context) {
    auto* ctx = static_cast<MinMaxContext*>(context);
    int64_t local_min = std::numeric_limits<int64_t>::max();
    for (uint64_t i = start; i < end; ++i) {
        local_min = std::min(local_min, ctx->arr[i]);
    }
    brass_parallel_reduce_i64(local_min);
}

static void max_kernel(uint64_t start, uint64_t end, void* context) {
    auto* ctx = static_cast<MinMaxContext*>(context);
    int64_t local_max = std::numeric_limits<int64_t>::min();
    for (uint64_t i = start; i < end; ++i) {
        local_max = std::max(local_max, ctx->arr[i]);
    }
    brass_parallel_reduce_i64(local_max);
}

TEST_CASE("brass_parallel_for Min and Max Reductions") {
    const size_t N = 5000;
    std::vector<int64_t> arr(N);
    for (size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<int64_t>(i * 3 + 7);
    }
    arr[1234] = -999;
    arr[4321] = 999999;

    MinMaxContext ctx{arr.data()};

    int64_t min_result = 0;
    brass_parallel_for(
        N,
        64,
        min_kernel,
        &ctx,
        ReductionKind::MinI64,
        &min_result
    );
    CHECK_EQ(min_result, -999);

    int64_t max_result = 0;
    brass_parallel_for(
        N,
        64,
        max_kernel,
        &ctx,
        ReductionKind::MaxI64,
        &max_result
    );
    CHECK_EQ(max_result, 999999);
}

TEST_CASE("Parallel Context Allocation and Free") {
    void* ptr = brass_parallel_alloc_context(128);
    CHECK(ptr != nullptr);
    brass_parallel_free_context(ptr);
}
