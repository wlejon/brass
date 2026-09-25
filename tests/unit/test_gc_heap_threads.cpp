// gc::Heap threading: a heap per thread, each collecting on its own while the
// others run, sharing only the process-wide layout registry.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <atomic>
#include <thread>
#include <vector>

using namespace brass::gc;

namespace {

HeapConfig thread_config(StressMode stress) {
    HeapConfig c;
    c.eden_bytes = 128 * 1024;
    c.survivor_bytes = 64 * 1024;
    c.mature_reserve_bytes = size_t{128} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{1} << 20;
    c.read_environment = false;
    c.stress = stress;
    return c;
}

void trace_weak_first(uintptr_t payload, size_t, Tracer& t) {
    auto* w = reinterpret_cast<uint64_t*>(payload);
    t.visit_weak(&w[0], 0);
    t.visit(&w[1]);
}

// A binary tree of `depth` levels; each node [left, right, value].
uintptr_t build_tree(Heap& h, LayoutId layout, int depth, uint64_t& counter) {
    uint64_t left = 0;
    uint64_t right = 0;
    h.add_root(&left);
    h.add_root(&right);
    if (depth > 0) {
        left = build_tree(h, layout, depth - 1, counter);
        right = build_tree(h, layout, depth - 1, counter);
    }
    uintptr_t node = h.allocate(24, layout);
    h.store(node, 0, left);
    h.store(node, 1, right);
    reinterpret_cast<uint64_t*>(node)[2] = ++counter;
    h.remove_root(&right);
    h.remove_root(&left);
    return node;
}

uint64_t tree_sum(uintptr_t node) {
    if (!node) return 0;
    return Heap::load(node, 2) + tree_sum(static_cast<uintptr_t>(Heap::load(node, 0))) +
           tree_sum(static_cast<uintptr_t>(Heap::load(node, 1)));
}

} // namespace

TEST_CASE("gc::Heap threads - eight heaps allocate and collect concurrently") {
    constexpr int kThreads = 8;
    std::atomic<int> failures{0};
    std::atomic<int> started{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &failures, &started] {
            // Layout registration races with the other threads' allocations.
            const LayoutId tree = mask_layout(0b011, static_cast<uint32_t>(1000 + t));
            LayoutDescriptor d;
            d.kind = LayoutKind::Custom;
            d.trace = &trace_weak_first;
            const LayoutId weak_pair = register_layout(d);
            const StressMode stress = (t % 4 == 0) ? StressMode::Alternate : StressMode::None;
            Heap h(thread_config(stress));
            HeapScope scope(h);
            started.fetch_add(1);
            while (started.load() < kThreads) std::this_thread::yield();

            uint64_t keep = 0;
            uint64_t pair = 0;
            h.add_root(&keep);
            h.add_root(&pair);
            for (int round = 0; round < 6; ++round) {
                uint64_t counter = 0;
                keep = build_tree(h, tree, 9, counter);  // 1023 nodes; the previous tree is garbage
                const uint64_t expected = counter * (counter + 1) / 2;
                pair = h.allocate(16, weak_pair);
                uint64_t dropped = build_tree(h, tree, 3, counter);
                h.store(static_cast<uintptr_t>(pair), 0, dropped);
                h.store(static_cast<uintptr_t>(pair), 1, keep);
                h.collect(round % 2 ? CollectionKind::Full : CollectionKind::Minor);
                if (Heap::load(static_cast<uintptr_t>(pair), 0) != 0) failures.fetch_add(1);
                if (Heap::load(static_cast<uintptr_t>(pair), 1) != keep) failures.fetch_add(1);
                if (tree_sum(static_cast<uintptr_t>(keep)) != expected) failures.fetch_add(1);
            }
            h.verify();
            h.remove_root(&pair);
            h.remove_root(&keep);
            if (Heap::current() != &h) failures.fetch_add(1);
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(failures.load(), 0);
    CHECK(Heap::current() == nullptr);
}

TEST_CASE("gc::Heap threads - heaps created and destroyed on many threads release their memory") {
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    for (int t = 0; t < 16; ++t) {
        threads.emplace_back([&failures] {
            for (int i = 0; i < 4; ++i) {
                Heap h(thread_config(StressMode::None));
                uint64_t root = 0;
                h.add_root(&root);
                for (int k = 0; k < 2000; ++k) {
                    uintptr_t n = h.allocate(32, mask_layout(0b1, 7));
                    h.store(n, 0, root);
                    root = n;
                }
                h.collect(CollectionKind::Full);
                size_t len = 0;
                for (uintptr_t n = static_cast<uintptr_t>(root); n; n = static_cast<uintptr_t>(Heap::load(n, 0))) ++len;
                if (len != 2000) failures.fetch_add(1);
                h.remove_root(&root);
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(failures.load(), 0);
}
