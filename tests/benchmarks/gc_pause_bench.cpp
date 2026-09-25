// Pause times of gc::Heap against the size of the live old generation.
//
// For each live-heap size N: builds N old objects (binary trees of 32-byte
// nodes with two references), then runs a mutator that allocates young
// garbage, keeps a small rotating young working set, and stores young
// references into random old objects (dirtying cards). Reports the minor
// pauses the mutator saw and the pause of explicit full collections.
//
//   brass_gc_pause_bench [N ...]     (default: 100k 300k 1M 1.5M 3M)

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace brass::gc;

namespace {

const LayoutId kNode = mask_layout(0b0111, 1);  // [left, right, extra, value]

struct MinorStats {
    uint64_t count = 0;
    uint64_t total_ns = 0;
    uint64_t max_ns = 0;
    double avg_us() const { return count ? static_cast<double>(total_ns) / static_cast<double>(count) / 1e3 : 0.0; }
    double max_us() const { return static_cast<double>(max_ns) / 1e3; }
};

struct Result {
    size_t objects;
    double old_mb;
    MinorStats quiet;
    MinorStats writing;
    double full_avg_ms;
    double full_max_ms;
};

// A complete binary tree of `count` nodes rooted in `*root`.
void build_old_graph(Heap& heap, uint64_t* root, size_t count) {
    // Nodes are allocated old directly: this is the long-lived heap. No
    // collection runs while they are built (the full threshold is far off).
    std::vector<uint64_t> nodes(count);
    for (size_t i = 0; i < count; ++i) {
        nodes[i] = heap.allocate(32, kNode, kAllocOld);
        reinterpret_cast<uint64_t*>(nodes[i])[3] = i;
    }
    for (size_t i = 0; i < count; ++i) {
        const size_t l = 2 * i + 1, r = 2 * i + 2;
        if (l < count) heap.store(static_cast<uintptr_t>(nodes[i]), 0, nodes[l]);
        if (r < count) heap.store(static_cast<uintptr_t>(nodes[i]), 1, nodes[r]);
    }
    *root = nodes.empty() ? 0 : nodes[0];
}

Result run(size_t objects) {
    HeapConfig config;
    config.read_environment = false;
    config.eden_bytes = size_t{8} << 20;
    config.survivor_bytes = size_t{1} << 20;
    // Full collections only when asked: the minor pauses measured below
    // must not include one.
    config.min_full_threshold_bytes = size_t{8} << 30;
    Heap heap(config);
    HeapScope scope(heap);

    uint64_t root = 0;
    heap.add_root(&root);
    std::vector<uint64_t> old_nodes;
    build_old_graph(heap, &root, objects);
    // Collect addresses of old nodes to mutate (they never move).
    old_nodes.reserve(objects);
    heap.for_each_object([&](uintptr_t o) { old_nodes.push_back(o); });

    constexpr size_t kWorkingSet = 4096;
    std::vector<uint64_t> working(kWorkingSet, 0);
    for (auto& slot : working) heap.add_root(&slot);

    std::mt19937_64 rng(42);
    // About 40 minor collections' worth of allocation per phase: first a
    // mutator that only allocates, then one that also stores young
    // references into random old objects (every 256th allocation).
    const size_t allocations = config.eden_bytes;
    auto phase = [&](bool write_old, MinorStats& out) {
        const uint64_t minors_before = heap.stats().minor_collections;
        const uint64_t ns_before = heap.stats().minor_pause_ns_total;
        uint64_t last = minors_before;
        for (size_t i = 0; i < allocations; ++i) {
            const uintptr_t young = heap.allocate(32, kNode);
            if ((i & 15) == 0) working[rng() % kWorkingSet] = young;  // a few survive a while
            if (write_old && (i & 255) == 0) {
                // An old object now names a young one: remembered by its card.
                heap.store(static_cast<uintptr_t>(old_nodes[rng() % old_nodes.size()]), 2, young);
            }
            const uint64_t minors = heap.stats().minor_collections;
            if (minors != last) {
                out.max_ns = std::max(out.max_ns, heap.stats().last_pause_ns);
                last = minors;
            }
        }
        out.count = heap.stats().minor_collections - minors_before;
        out.total_ns = heap.stats().minor_pause_ns_total - ns_before;
    };
    MinorStats quiet, writing;
    phase(false, quiet);
    phase(true, writing);

    constexpr int kFulls = 3;
    uint64_t full_ns = 0, full_max_ns = 0;
    for (int i = 0; i < kFulls; ++i) {
        heap.collect(CollectionKind::Full);
        full_ns += heap.stats().last_pause_ns;
        full_max_ns = std::max(full_max_ns, heap.stats().last_pause_ns);
    }

    Result r;
    r.objects = objects;
    r.old_mb = static_cast<double>(heap.old_used_bytes()) / (1 << 20);
    r.quiet = quiet;
    r.writing = writing;
    r.full_avg_ms = static_cast<double>(full_ns) / kFulls / 1e6;
    r.full_max_ms = static_cast<double>(full_max_ns) / 1e6;
    for (auto& slot : working) heap.remove_root(&slot);
    heap.remove_root(&root);
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<size_t>(std::strtoull(argv[i], nullptr, 10)));
    if (sizes.empty()) sizes = {100000, 300000, 1000000, 1500000, 3000000};
    std::printf("minor: young collections while only allocating; +writes: while also storing young\n"
                "references into random old objects. full: explicit full collections.\n\n");
    std::printf("%11s %8s | %7s %8s %8s | %7s %8s %8s | %8s %8s\n", "old objects", "old MB", "minors",
                "avg us", "max us", "+writes", "avg us", "max us", "full ms", "max ms");
    for (size_t n : sizes) {
        const Result r = run(n);
        std::printf("%11zu %8.1f | %7llu %8.1f %8.1f | %7llu %8.1f %8.1f | %8.2f %8.2f\n", r.objects, r.old_mb,
                    static_cast<unsigned long long>(r.quiet.count), r.quiet.avg_us(), r.quiet.max_us(),
                    static_cast<unsigned long long>(r.writing.count), r.writing.avg_us(), r.writing.max_us(),
                    r.full_avg_ms, r.full_max_ms);
    }
    return 0;
}
