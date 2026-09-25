// Pause times of gc::Heap against the size of the live old generation.
//
// Two workloads, each at several live-heap sizes N:
//
//   graph  builds N old objects (binary trees of 32-byte nodes with two
//          references), then runs a mutator that allocates young garbage,
//          keeps a small rotating young working set, and stores young
//          references into random old objects (dirtying cards). Reports the
//          minor pauses the mutator saw and the pause of explicit full
//          collections.
//   churn  a JavaScript-shaped program: N long-lived records, each with a
//          small array and a string, pushed into one growable list, then 200
//          batches that each push 20,000 short-lived records into a fresh
//          list. The lists grow by reallocating their backing store, so the
//          old generation sees large backing stores die while young objects
//          they named are still in the nursery. Reports the batch times and
//          the collections that ran inside the batches.
//
//   brass_gc_pause_bench [N ...]   (default: 100k 300k 1M 1.5M 3M)
//   brass_gc_pause_bench --check   one size of each, against generous bounds;
//                                  exits non-zero past any of them

#include <brass/gc/heap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace brass::gc;

namespace {

using Clock = std::chrono::steady_clock;

const LayoutId kNode = mask_layout(0b0111, 1);  // [left, right, extra, value]

LayoutId words_layout() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Words;
        d.type_tag = 2;
        d.name = "bench.words";
        return register_layout(d);
    }();
    return id;
}

double ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

// The default configuration, with only BRASS_GC_LOG and BRASS_GC_MARK_THREADS
// taken from the environment (stress or verification would measure nothing).
std::string env_value(const char* name) {
#if defined(_MSC_VER)
    char* owned = nullptr;  // getenv is C4996 under MSVC
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return {};
    std::string v(owned);
    std::free(owned);
    return v;
#else
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
}

HeapConfig bench_config() {
    HeapConfig config;
    config.read_environment = false;
    config.log = !env_value("BRASS_GC_LOG").empty();
    const std::string threads = env_value("BRASS_GC_MARK_THREADS");
    if (!threads.empty()) config.mark_threads = static_cast<unsigned>(std::strtoul(threads.c_str(), nullptr, 10));
    return config;
}

// Watches a heap's collections from the mutator: after each allocation,
// `poll` notices a collection that ran and records its pause by kind.
struct PauseWatch {
    explicit PauseWatch(Heap& h) : heap(h) { sync(); }
    void sync() {
        minors = heap.stats().minor_collections;
        fulls = heap.stats().full_collections;
    }
    void poll() {
        const HeapStats& s = heap.stats();
        if (s.minor_collections == minors && s.full_collections == fulls) return;
        const bool full = s.full_collections != fulls;
        (full ? full_count : minor_count) += 1;
        uint64_t& max = full ? full_max_ns : minor_max_ns;
        max = std::max(max, s.last_pause_ns);
        (full ? full_total_ns : minor_total_ns) += s.last_pause_ns;
        sync();
    }
    Heap& heap;
    uint64_t minors = 0, fulls = 0;
    uint64_t minor_count = 0, minor_total_ns = 0, minor_max_ns = 0;
    uint64_t full_count = 0, full_total_ns = 0, full_max_ns = 0;
};

// ---- graph ------------------------------------------------------------------

struct GraphResult {
    size_t objects;
    double old_mb;
    uint64_t quiet_minors, quiet_max_ns, quiet_total_ns;
    uint64_t writing_minors, writing_max_ns, writing_total_ns;
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

GraphResult run_graph(size_t objects) {
    HeapConfig config = bench_config();
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
    // About 20 minor collections' worth of allocation per phase: first a
    // mutator that only allocates, then one that also stores young
    // references into random old objects (every 256th allocation).
    const size_t allocations = config.eden_bytes;
    auto phase = [&](bool write_old, uint64_t& count, uint64_t& max_ns, uint64_t& total_ns) {
        PauseWatch watch(heap);
        for (size_t i = 0; i < allocations; ++i) {
            const uintptr_t young = heap.allocate(32, kNode);
            if ((i & 15) == 0) working[rng() % kWorkingSet] = young;  // a few survive a while
            if (write_old && (i & 255) == 0) {
                // An old object now names a young one: remembered by its card.
                heap.store(static_cast<uintptr_t>(old_nodes[rng() % old_nodes.size()]), 2, young);
            }
            watch.poll();
        }
        count = watch.minor_count;
        max_ns = watch.minor_max_ns;
        total_ns = watch.minor_total_ns;
    };
    GraphResult r{};
    phase(false, r.quiet_minors, r.quiet_max_ns, r.quiet_total_ns);
    phase(true, r.writing_minors, r.writing_max_ns, r.writing_total_ns);

    constexpr int kFulls = 3;
    uint64_t full_ns = 0, full_max_ns = 0;
    for (int i = 0; i < kFulls; ++i) {
        heap.collect(CollectionKind::Full);
        full_ns += heap.stats().last_pause_ns;
        full_max_ns = std::max(full_max_ns, heap.stats().last_pause_ns);
    }

    r.objects = objects;
    r.old_mb = static_cast<double>(heap.old_used_bytes()) / (1 << 20);
    r.full_avg_ms = ms(full_ns) / kFulls;
    r.full_max_ms = ms(full_max_ns);
    for (auto& slot : working) heap.remove_root(&slot);
    heap.remove_root(&root);
    return r;
}

// ---- churn ------------------------------------------------------------------

// A growable list, as a JavaScript array: a header [length, backing] and a
// backing store of `capacity` words that is reallocated at twice the size
// when full. `*slot` is a root naming the header.
class List {
public:
    List(Heap& heap, uint64_t* slot) : heap_(heap), slot_(slot) {
        *slot_ = heap_.allocate(16, words_layout());
    }
    // `*value` is a root: allocating a larger backing store can move it.
    void push(uint64_t* value) {
        uintptr_t header = static_cast<uintptr_t>(*slot_);
        const uint64_t length = Heap::load(header, 0);
        uintptr_t backing = static_cast<uintptr_t>(Heap::load(header, 1));
        const uint64_t capacity = backing ? header_of(backing)->size / 8 : 0;
        if (length == capacity) {
            const uint64_t grown = std::max<uint64_t>(8, capacity * 2);
            const uintptr_t fresh = heap_.allocate(grown * 8, words_layout());
            header = static_cast<uintptr_t>(*slot_);
            backing = static_cast<uintptr_t>(Heap::load(header, 1));
            if (length) {
                std::memcpy(reinterpret_cast<void*>(fresh), reinterpret_cast<void*>(backing), length * 8);
                heap_.remember_range(fresh, fresh, fresh + length * 8);
            }
            heap_.store(header, 1, fresh);
            backing = fresh;
        }
        auto* slot = reinterpret_cast<uint64_t*>(backing) + length;
        *slot = *value;
        heap_.write_barrier_interior(reinterpret_cast<uintptr_t>(slot), *value);
        reinterpret_cast<uint64_t*>(header)[0] = length + 1;
    }

private:
    Heap& heap_;
    uint64_t* slot_;
};

struct ChurnResult {
    size_t objects;
    double worst_ms, median_ms, avg_ms;
    uint64_t batch_minors, batch_fulls;
    double batch_minor_max_ms, batch_full_max_ms;
    uint64_t build_minors, build_fulls;
    double build_minor_max_ms, build_full_max_ms;
};

// A record [a, b, c] with `b` a two-element list and `c` a 16-byte string
// (the long-lived records), or [x, y] with `y` a one-element list.
void make_record(Heap& heap, uint64_t* out, uint64_t* tmp, uint64_t n, bool long_lived) {
    tmp[0] = heap.allocate(16, words_layout());                       // the small list's header
    const uintptr_t items = heap.allocate(long_lived ? 16 : 8, words_layout());
    reinterpret_cast<uint64_t*>(items)[0] = n;
    if (long_lived) reinterpret_cast<uint64_t*>(items)[1] = n + 1;
    // The header may have been promoted by the allocation of `items`.
    const uintptr_t list = static_cast<uintptr_t>(tmp[0]);
    reinterpret_cast<uint64_t*>(list)[0] = long_lived ? 2 : 1;
    heap.store(list, 1, items);
    tmp[1] = long_lived ? heap.allocate(16, kLeafLayout) : 0;         // the string
    const uintptr_t record = heap.allocate(long_lived ? 24 : 16, words_layout());
    auto* w = reinterpret_cast<uint64_t*>(record);
    w[0] = n;
    w[1] = tmp[0];
    if (long_lived) w[2] = tmp[1];
    *out = record;
    tmp[0] = tmp[1] = 0;
}

ChurnResult run_churn(size_t objects) {
    Heap heap(bench_config());
    HeapScope scope(heap);

    uint64_t live = 0, batch = 0, record = 0, tmp[2] = {0, 0};
    for (uint64_t* r : {&live, &batch, &record, &tmp[0], &tmp[1]}) heap.add_root(r);

    ChurnResult r{};
    r.objects = objects;
    PauseWatch build(heap);
    List live_list(heap, &live);
    for (size_t i = 0; i < objects; ++i) {
        make_record(heap, &record, tmp, i, true);
        live_list.push(&record);
        build.poll();
    }
    r.build_minors = build.minor_count;
    r.build_fulls = build.full_count;
    r.build_minor_max_ms = ms(build.minor_max_ns);
    r.build_full_max_ms = ms(build.full_max_ns);

    constexpr int kBatches = 200;
    constexpr int kBatchRecords = 20000;
    std::vector<double> times;
    PauseWatch during(heap);
    for (int b = 0; b < kBatches; ++b) {
        const auto start = Clock::now();
        List list(heap, &batch);
        for (int j = 0; j < kBatchRecords; ++j) {
            make_record(heap, &record, tmp, static_cast<uint64_t>(j), false);
            list.push(&record);
            during.poll();
        }
        times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    r.batch_minors = during.minor_count;
    r.batch_fulls = during.full_count;
    r.batch_minor_max_ms = ms(during.minor_max_ns);
    r.batch_full_max_ms = ms(during.full_max_ns);
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    r.worst_ms = sorted.back();
    r.median_ms = sorted[sorted.size() / 2];
    double sum = 0;
    for (double t : times) sum += t;
    r.avg_ms = sum / static_cast<double>(times.size());
    for (uint64_t* s : {&live, &batch, &record, &tmp[0], &tmp[1]}) heap.remove_root(s);
    return r;
}

// ---- report -----------------------------------------------------------------

void print_graph_header() {
    std::printf("graph: minors while only allocating; +writes: while also storing young references\n"
                "into random old objects. full: explicit full collections.\n");
    std::printf("%11s %8s | %7s %8s %8s | %7s %8s %8s | %8s %8s\n", "old objects", "old MB", "minors",
                "avg us", "max us", "+writes", "avg us", "max us", "full ms", "max ms");
}

void print_graph(const GraphResult& r) {
    auto avg_us = [](uint64_t total, uint64_t n) { return n ? static_cast<double>(total) / 1e3 / n : 0.0; };
    std::printf("%11zu %8.1f | %7llu %8.1f %8.1f | %7llu %8.1f %8.1f | %8.2f %8.2f\n", r.objects, r.old_mb,
                static_cast<unsigned long long>(r.quiet_minors), avg_us(r.quiet_total_ns, r.quiet_minors),
                static_cast<double>(r.quiet_max_ns) / 1e3, static_cast<unsigned long long>(r.writing_minors),
                avg_us(r.writing_total_ns, r.writing_minors), static_cast<double>(r.writing_max_ns) / 1e3,
                r.full_avg_ms, r.full_max_ms);
}

void print_churn_header() {
    std::printf("\nchurn: 200 batches of 20,000 short-lived records over N long-lived ones.\n"
                "batch ms; collections inside the batches; collections while building.\n");
    std::printf("%9s | %7s %7s %7s | %6s %8s %6s %8s | %6s %8s %6s %8s\n", "records", "worst", "median",
                "avg", "minors", "max ms", "fulls", "max ms", "minors", "max ms", "fulls", "max ms");
}

void print_churn(const ChurnResult& r) {
    std::printf("%9zu | %7.2f %7.2f %7.2f | %6llu %8.2f %6llu %8.2f | %6llu %8.2f %6llu %8.2f\n", r.objects,
                r.worst_ms, r.median_ms, r.avg_ms, static_cast<unsigned long long>(r.batch_minors),
                r.batch_minor_max_ms, static_cast<unsigned long long>(r.batch_fulls), r.batch_full_max_ms,
                static_cast<unsigned long long>(r.build_minors), r.build_minor_max_ms,
                static_cast<unsigned long long>(r.build_fulls), r.build_full_max_ms);
}

// Bounds loose enough for a loaded or slow machine: each one sits several
// times above what the collector does, so crossing one is a regression in
// kind (a minor that rescans the old generation, a full per batch, a full
// mark that stopped scaling), not noise.
int check() {
    int failures = 0;
    auto bound = [&](const char* what, double value, double limit) {
        const bool ok = value <= limit;
        std::printf("  %-44s %9.2f  (limit %.2f)%s\n", what, value, limit, ok ? "" : "  FAILED");
        if (!ok) ++failures;
    };
    print_graph_header();
    const GraphResult g = run_graph(1000000);
    print_graph(g);
    print_churn_header();
    const ChurnResult c = run_churn(500000);
    print_churn(c);
    std::printf("\n");
    bound("graph: minor pause, max ms", ms(g.quiet_max_ns), 10.0);
    bound("graph: minor pause with old writes, max ms", ms(g.writing_max_ns), 15.0);
    bound("graph: full pause at 1M old objects, max ms", g.full_max_ms, 150.0);
    bound("churn: worst batch ms", c.worst_ms, 60.0);
    bound("churn: minor pause inside batches, max ms", c.batch_minor_max_ms, 15.0);
    bound("churn: full collections inside batches", static_cast<double>(c.batch_fulls), 4.0);
    bound("churn: minor pause while building, max ms", c.build_minor_max_ms, 60.0);
    bound("churn: full pause while building, max ms", c.build_full_max_ms, 250.0);
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--check") == 0) return check();
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<size_t>(std::strtoull(argv[i], nullptr, 10)));
    if (sizes.empty()) sizes = {100000, 300000, 1000000, 1500000, 3000000};
    print_graph_header();
    for (size_t n : sizes) print_graph(run_graph(n));
    print_churn_header();
    for (size_t n : sizes) print_churn(run_churn(n));
    return 0;
}
