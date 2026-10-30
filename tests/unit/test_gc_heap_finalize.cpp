// gc::Heap finalizers in bulk: the two-context registration, the lists kept
// in place across many collections (young deaths, promotion, old deaths), a
// finalizer whose allocation collects (what that collection finds dead runs
// in the same run, after the batch, never nested), one that throws, and the
// minor collection's weak processing leaving old targets alone.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <new>
#include <stdexcept>
#include <vector>

using namespace brass::gc;

namespace {

constexpr uint64_t kCleared = 0x7FFC000000000000ULL;

HeapConfig fin_config() {
    HeapConfig c;
    c.eden_bytes = 256 * 1024;
    c.survivor_bytes = 128 * 1024;
    c.mature_reserve_bytes = size_t{64} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{16} << 20;
    c.read_environment = false;
    c.verify = true;
    return c;
}

const LayoutId kLeafNode = mask_layout(0, 41);
const LayoutId kRefNode = mask_layout(0b1, 42);

// Per-object finalizer bookkeeping: how often each id's finalizer ran.
std::vector<int> g_runs;

void count_pair(void* a, void* b) {
    const auto id = reinterpret_cast<uintptr_t>(a);
    const auto tag = reinterpret_cast<uintptr_t>(b);
    if (tag != 0xC0FFEE) throw std::logic_error("finalizer got the wrong second context word");
    g_runs.at(id) += 1;
}

void count_one(void* a) { g_runs.at(reinterpret_cast<uintptr_t>(a)) += 1; }

} // namespace

TEST_CASE("gc::Heap finalize - the two-context form passes both words, once") {
    Heap h(fin_config());
    g_runs.assign(2, 0);
    uint64_t keep = h.allocate(8, kLeafNode);
    h.add_root(&keep);
    h.add_finalizer(static_cast<uintptr_t>(keep), &count_pair, reinterpret_cast<void*>(uintptr_t{0}),
                    reinterpret_cast<void*>(uintptr_t{0xC0FFEE}));
    {
        uintptr_t dead = h.allocate(8, kLeafNode);
        h.add_finalizer(dead, &count_pair, reinterpret_cast<void*>(uintptr_t{1}),
                        reinterpret_cast<void*>(uintptr_t{0xC0FFEE}));
    }
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_runs[0], 0);
    CHECK_EQ(g_runs[1], 1);
    CHECK_EQ(h.stats().last_finalizers, uint64_t{1});
    keep = 0;
    h.collect(CollectionKind::Full);
    CHECK_EQ(g_runs[0], 1);
    h.collect(CollectionKind::Full);
    CHECK_EQ(g_runs[0], 1);
    CHECK_EQ(g_runs[1], 1);
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap finalize - thousands across young deaths, promotion and old deaths, each once") {
    Heap h(fin_config());
    constexpr size_t kCount = 6000;
    g_runs.assign(kCount, 0);
    // A rooted array of the survivors: every third object, the rest die young.
    uint64_t holder = h.allocate(8 * kCount, mask_layout(~uint64_t{0}, 43), kAllocOld);
    h.add_root(&holder);
    for (size_t i = 0; i < kCount; ++i) {
        const uintptr_t obj = h.allocate(16, kLeafNode);
        if (i % 2 == 0) h.add_finalizer(obj, &count_one, reinterpret_cast<void*>(i));
        else h.add_finalizer(obj, &count_pair, reinterpret_cast<void*>(i), reinterpret_cast<void*>(uintptr_t{0xC0FFEE}));
        if (i % 3 == 0) h.store(static_cast<uintptr_t>(holder), i, obj);
    }
    h.collect(CollectionKind::Minor);
    h.collect(CollectionKind::Minor);
    h.collect(CollectionKind::Minor);  // the survivors are old now
    for (size_t i = 0; i < kCount; ++i) CHECK_EQ(g_runs[i], i % 3 == 0 ? 0 : 1);
    // Half the survivors let go: only a full collection decides them.
    for (size_t i = 0; i < kCount; i += 6) h.store(static_cast<uintptr_t>(holder), i, 0);
    h.collect(CollectionKind::Minor);
    for (size_t i = 0; i < kCount; ++i) CHECK_EQ(g_runs[i], i % 3 == 0 ? 0 : 1);
    h.collect(CollectionKind::Full);
    for (size_t i = 0; i < kCount; ++i) CHECK_EQ(g_runs[i], (i % 3 == 0 && i % 6 != 0) ? 0 : 1);
    holder = 0;
    h.collect(CollectionKind::Full);
    h.collect(CollectionKind::Full);
    for (size_t i = 0; i < kCount; ++i) CHECK_EQ(g_runs[i], 1);
    h.remove_root(&holder);
}

namespace {
Heap* g_heap = nullptr;
int g_nested_ran = 0;
int g_outer_ran = 0;
bool g_in_outer = false;

void nested_dead(void*) {
    // Never inside the finalizer whose allocation collected.
    if (g_in_outer) throw std::logic_error("a nested collection ran its finalizers inside a finalizer");
    ++g_nested_ran;
}

void outer_allocates(void*) {
    g_in_outer = true;
    // Objects that die at once, each with a finalizer, then enough
    // allocation to fill eden: the collection this causes finds them dead.
    for (int i = 0; i < 16; ++i) g_heap->add_finalizer(g_heap->allocate(16, kLeafNode), &nested_dead, nullptr);
    for (int i = 0; i < 4000; ++i) (void)g_heap->allocate(128, kLeafNode);
    g_in_outer = false;
    ++g_outer_ran;
}
} // namespace

TEST_CASE("gc::Heap finalize - a finalizer's own collection queues its dead for the same run") {
    Heap h(fin_config());
    g_heap = &h;
    g_nested_ran = g_outer_ran = 0;
    const uint64_t before = h.collection_count();
    for (int i = 0; i < 3; ++i) h.add_finalizer(h.allocate(16, kLeafNode), &outer_allocates, nullptr);
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_outer_ran, 3);
    CHECK(h.collection_count() > before + 1);  // the finalizers' allocations collected
    CHECK_EQ(g_nested_ran, 48);
    CHECK_EQ(h.stats().last_finalizers, uint64_t{51});
    g_heap = nullptr;
}

namespace {
int g_after_throw = 0;
void throws(void*) { throw std::bad_alloc(); }
void counts(void*) { ++g_after_throw; }
} // namespace

TEST_CASE("gc::Heap finalize - a finalizer that throws does not stop later collections' finalizers") {
    Heap h(fin_config());
    g_after_throw = 0;
    h.add_finalizer(h.allocate(16, kLeafNode), &throws, nullptr);
    bool threw = false;
    try {
        h.collect(CollectionKind::Minor);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    CHECK(threw);
    h.add_finalizer(h.allocate(16, kLeafNode), &counts, nullptr);
    h.collect(CollectionKind::Minor);
    CHECK_EQ(g_after_throw, 1);
}

namespace {
struct WeakRoots {
    std::vector<uint64_t> slots;
};
void visit_weak_roots(Tracer& t, void* context) {
    for (uint64_t& s : static_cast<WeakRoots*>(context)->slots) t.visit_weak(&s, kCleared);
}
} // namespace

TEST_CASE("gc::Heap finalize - a minor collection decides weak slots naming young objects only") {
    Heap h(fin_config());
    // Old targets, some dead: a minor collection leaves every weak slot that
    // names one exactly as it was (the tracer's young range filters them
    // before the collector sees them), and a full collection decides them.
    WeakRoots roots;
    std::vector<uint64_t> strong(64, 0);
    for (size_t i = 0; i < 64; ++i) {
        const uintptr_t old_obj = h.allocate(16, kRefNode, kAllocOld);
        roots.slots.push_back(old_obj);
        if (i % 2 == 0) strong[i] = old_obj;
    }
    for (uint64_t& s : strong) h.add_root(&s);
    // And young targets, half of them dead.
    std::vector<uint64_t> young_strong(64, 0);
    for (size_t i = 0; i < 64; ++i) {
        const uintptr_t y = h.allocate(16, kRefNode);
        roots.slots.push_back(y);
        if (i % 2 == 0) young_strong[i] = y;
    }
    for (uint64_t& s : young_strong) h.add_root(&s);
    const std::vector<uint64_t> old_before(roots.slots.begin(), roots.slots.begin() + 64);
    const auto id = h.add_root_source(&visit_weak_roots, &roots);
    h.collect(CollectionKind::Minor);
    for (size_t i = 0; i < 64; ++i) CHECK_EQ(roots.slots[i], old_before[i]);  // old: undecided, untouched
    for (size_t i = 0; i < 64; ++i) {
        if (i % 2 == 0) CHECK_EQ(roots.slots[64 + i], young_strong[i]);       // moved with its target
        else CHECK_EQ(roots.slots[64 + i], kCleared);                         // died young
    }
    h.collect(CollectionKind::Full);
    for (size_t i = 0; i < 64; ++i) CHECK_EQ(roots.slots[i], i % 2 == 0 ? strong[i] : kCleared);
    h.remove_root_source(id);
    for (uint64_t& s : strong) h.remove_root(&s);
    for (uint64_t& s : young_strong) h.remove_root(&s);
}
