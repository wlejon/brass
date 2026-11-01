// gc::Heap keeps in place every young object a word of the thread's stacks
// names (heap_pin.cpp): a raw reference held in a C++ local, untagged, tagged
// or pointing inside its object, still names the object, intact, after minor,
// full and incremental collections, and after parallel ones; what the pinned
// object names is updated as a root's would be; an old object only the stack
// names survives a full collection, and dies once nothing names it. Eden allocates around pinned objects (its
// fragments), an object no fragment fits is allocated old with its card
// dirty, survivor spaces copy around the residents an earlier pin left there,
// and a resident no longer named is copied like any other survivor. A
// suspended fiber's stack pins too.

#include "test_framework.hpp"

#include <brass/gc/fiber.hpp>
#include <brass/gc/heap.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#define PIN_TEST_NOINLINE __declspec(noinline)
#else
#define PIN_TEST_NOINLINE __attribute__((noinline))
#endif

using namespace brass::gc;

namespace {

constexpr uint16_t kPointerTag = 0xFFFC;  // a NaN-box pointer tag

HeapConfig pin_config() {
    HeapConfig c;
    c.read_environment = false;
    c.verify = true;
    return c;
}

uint64_t pattern(uintptr_t salt, size_t i) { return (salt * 0x9E3779B97F4A7C15ULL) ^ (i * 0xBF58476D1CE4E5B9ULL); }

// A leaf object of `words` payload words filled with pattern(salt, i).
PIN_TEST_NOINLINE uintptr_t make_leaf(Heap& h, size_t words, uintptr_t salt, uint32_t flags = 0) {
    const uintptr_t o = h.allocate(words * 8, kLeafLayout, flags);
    for (size_t i = 0; i < words; ++i) reinterpret_cast<uint64_t*>(o)[i] = pattern(salt, i);
    return o;
}

bool intact(uintptr_t o, size_t words, uintptr_t salt) {
    for (size_t i = 0; i < words; ++i) {
        if (reinterpret_cast<const uint64_t*>(o)[i] != pattern(salt, i)) return false;
    }
    return true;
}

// Garbage that fills eden and makes collections run.
PIN_TEST_NOINLINE void churn(Heap& h, size_t objects) {
    for (size_t i = 0; i < objects; ++i) (void)h.allocate(48, kLeafLayout);
}

const LayoutId kPair = mask_layout(0b11, 77);  // two reference words

// Stores a fresh leaf (words, salt) into slot `index` of `holder` with no
// other reference left behind: the leaf's address never stays in this frame.
PIN_TEST_NOINLINE void hang_leaf(Heap& h, uintptr_t holder, size_t index, size_t words, uintptr_t salt) {
    const uintptr_t leaf = make_leaf(h, words, salt);
    h.store(holder, index, leaf);
}

constexpr size_t kLargeWords = (size_t{2} << 20) / 8;  // past large_object_bytes

// Overwrites the stack below the caller's frame, where dead frames' words
// would otherwise still name the objects they made.
PIN_TEST_NOINLINE void scrub_stack() {
    volatile uint64_t words[2048];
    for (size_t i = 0; i < 2048; ++i) words[i] = 0;
    (void)words[0];
}

} // namespace

TEST_CASE("gc::Heap pinning - an untagged young reference on the stack survives minor and full collections in place") {
    Heap h(pin_config());
    HeapScope bind(h);
    volatile uintptr_t held = make_leaf(h, 6, 1);
    const uintptr_t at = held;
    for (int i = 0; i < 3; ++i) {
        h.collect(CollectionKind::Minor);
        CHECK_EQ(static_cast<uintptr_t>(held), at);
        CHECK(h.is_valid_object(held));
        CHECK(h.is_young(held));
        CHECK(intact(held, 6, 1));
        CHECK(h.stats().last_pinned >= 1);
    }
    h.collect(CollectionKind::Full);
    CHECK_EQ(static_cast<uintptr_t>(held), at);
    CHECK(h.is_valid_object(held));
    CHECK(intact(held, 6, 1));
    churn(h, 200000);
    CHECK(h.is_valid_object(held));
    CHECK(intact(held, 6, 1));
}

TEST_CASE("gc::Heap pinning - a tagged reference and an interior pointer pin their objects") {
    HeapConfig c = pin_config();
    c.reference_tags = {kPointerTag};
    Heap h(c);
    HeapScope bind(h);
    const uintptr_t a = make_leaf(h, 4, 2);
    volatile uint64_t boxed = (uint64_t{kPointerTag} << 48) | a;
    const uintptr_t b = make_leaf(h, 16, 3);
    volatile uintptr_t inside = b + 72;  // payload word 9
    for (int i = 0; i < 4; ++i) {
        if (i == 2) h.collect(CollectionKind::Full);
        else h.collect(CollectionKind::Minor);
        churn(h, 1000);
        CHECK_EQ(static_cast<uintptr_t>(boxed & 0xFFFFFFFFFFFFULL), a);
        CHECK(h.is_valid_object(a));
        CHECK(intact(a, 4, 2));
        CHECK_EQ(h.find_object(inside), b);
        CHECK(intact(b, 16, 3));
    }
}

// The holder's two leaves are objects and intact. Read in a frame of its own,
// so no word the caller keeps names (and pins) a leaf.
PIN_TEST_NOINLINE void check_leaves(Heap& h, uintptr_t holder) {
    const uintptr_t first = static_cast<uintptr_t>(Heap::load(holder, 0));
    const uintptr_t second = static_cast<uintptr_t>(Heap::load(holder, 1));
    CHECK(h.is_valid_object(first));
    CHECK(h.is_valid_object(second));
    CHECK(intact(first, 8, 4));
    CHECK(intact(second, 3, 5));
}

TEST_CASE("gc::Heap pinning - a pinned object's slots are roots and are updated") {
    Heap h(pin_config());
    HeapScope bind(h);
    volatile uintptr_t holder = h.allocate(16, kPair);
    hang_leaf(h, holder, 0, 8, 4);
    hang_leaf(h, holder, 1, 3, 5);
    // Aged, then promoted (tenure 2), while the holder stays where it is.
    for (int i = 0; i < 4; ++i) {
        churn(h, 5000);
        scrub_stack();
        h.collect(CollectionKind::Minor);
        CHECK(h.is_young(holder));
        check_leaves(h, holder);
    }
    CHECK_FALSE(h.is_young(static_cast<uintptr_t>(Heap::load(holder, 0))));
    // A full collection marks the promoted leaves through the pinned holder.
    h.collect(CollectionKind::Full);
    while (!h.sweep_slice(UINT64_MAX)) {
    }
    CHECK(h.is_valid_object(static_cast<uintptr_t>(Heap::load(holder, 0))));
    CHECK(intact(static_cast<uintptr_t>(Heap::load(holder, 0)), 8, 4));
    CHECK(intact(static_cast<uintptr_t>(Heap::load(holder, 1)), 3, 5));
}

// The old objects made here, their addresses kept only masked (no word of the
// stack names them once this returns).
constexpr uintptr_t kAddressMaskOut = 0x5A5A5A5A5A5A5A5AULL;
PIN_TEST_NOINLINE void make_old_unnamed(Heap& h, uintptr_t* mature, uintptr_t* large) {
    *mature = make_leaf(h, 32, 6, kAllocOld) ^ kAddressMaskOut;
    *large = make_leaf(h, kLargeWords, 7) ^ kAddressMaskOut;
}

TEST_CASE("gc::Heap pinning - the stack keeps old objects alive, and only while it names them") {
    Heap h(pin_config());
    HeapScope bind(h);
    {
        volatile uintptr_t mature = make_leaf(h, 32, 6, kAllocOld);
        volatile uintptr_t large = make_leaf(h, kLargeWords, 7);  // large-object space
        volatile uintptr_t inside = mature + 8 * 5;               // an interior word
        CHECK_FALSE(h.is_young(mature));
        CHECK_FALSE(h.is_young(large));
        h.collect(CollectionKind::Full);
        while (!h.sweep_slice(UINT64_MAX)) {
        }
        CHECK(h.is_valid_object(mature));
        CHECK(h.is_valid_object(large));
        CHECK(intact(mature, 32, 6));
        CHECK(intact(large, 4, 7));
        mature = 0;
        h.collect(CollectionKind::Full);
        while (!h.sweep_slice(UINT64_MAX)) {
        }
        const uintptr_t kept = inside - 8 * 5;
        CHECK(h.is_valid_object(kept));
        CHECK(intact(kept, 32, 6));
        inside = 0;
        large = 0;
    }
    uintptr_t mature = 0;
    uintptr_t large = 0;
    make_old_unnamed(h, &mature, &large);
    scrub_stack();
    h.collect(CollectionKind::Full);
    while (!h.sweep_slice(UINT64_MAX)) {
    }
    CHECK_FALSE(h.is_valid_object(mature ^ kAddressMaskOut));
    CHECK_FALSE(h.is_valid_object(large ^ kAddressMaskOut));
}

TEST_CASE("gc::Heap pinning - an incremental cycle keeps young stack objects alive in place") {
    {
        Heap h(pin_config());
        HeapScope bind(h);
        volatile uintptr_t holder = h.allocate(16, kPair);
        hang_leaf(h, holder, 0, 5, 8);
        const uintptr_t at = holder;
        REQUIRE(h.start_marking());
        for (int i = 0; i < 6; ++i) {
            churn(h, 20000);
            if (i % 2 == 0) h.collect(CollectionKind::Minor);
            (void)h.mark_slice(0);
        }
        h.finish_marking();
        while (!h.sweep_slice(UINT64_MAX)) {
        }
        CHECK_EQ(static_cast<uintptr_t>(holder), at);
        CHECK(h.is_valid_object(holder));
        CHECK(intact(static_cast<uintptr_t>(Heap::load(holder, 0)), 5, 8));
    }
}

TEST_CASE("gc::Heap pinning - eden allocates around pinned objects, and an object no fragment fits goes old") {
    HeapConfig c = pin_config();
    c.eden_bytes = 64 * 1024;
    c.survivor_bytes = 64 * 1024;  // an eden under 1 MB the adaptive limit never shrinks
    Heap h(c);
    HeapScope bind(h);
    // Pins spread through eden every 8 KB.
    constexpr int kPins = 7;
    volatile uintptr_t pins[kPins] = {};
    for (int i = 0; i < kPins; ++i) {
        pins[i] = make_leaf(h, 2, 100 + i);
        (void)h.allocate(8 * 1024 - 24, kLeafLayout);
    }
    h.collect(CollectionKind::Minor);
    for (int i = 0; i < kPins; ++i) {
        CHECK(h.is_young(pins[i]));
        CHECK(intact(pins[i], 2, 100 + i));
    }
    CHECK(h.stats().last_pinned >= kPins);
    // Allocation fills the fragments without overrunning the pins.
    const uint64_t minors = h.stats().minor_collections;
    std::vector<uintptr_t> got;
    for (int i = 0; i < 300; ++i) got.push_back(make_leaf(h, 5, 1000 + i));
    for (int i = 0; i < kPins; ++i) CHECK(intact(pins[i], 2, 100 + i));
    if (h.stats().minor_collections == minors) {
        for (size_t i = 0; i < got.size(); ++i) CHECK(intact(got[i], 5, 1000 + i));
    }
    h.verify();
    // 20 KB fits no fragment between the pins (each under 8 KB), even after
    // a collection that pins them again: it is allocated old, its card
    // dirty, so a young referent stored with no barrier is still found.
    h.collect(CollectionKind::Minor);
    volatile uintptr_t big = h.allocate(20 * 1024, kPair);
    CHECK_FALSE(h.is_young(big));
    hang_leaf(h, big, 0, 4, 9);
    {
        // The leaf's store, unbarriered as for an object taken to be young.
        const uintptr_t leaf = make_leaf(h, 4, 10);
        reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(big))[1] = leaf;
    }
    h.collect(CollectionKind::Minor);
    h.collect(CollectionKind::Minor);
    CHECK(intact(static_cast<uintptr_t>(Heap::load(big, 0)), 4, 9));
    CHECK(intact(static_cast<uintptr_t>(Heap::load(big, 1)), 4, 10));
    for (int i = 0; i < kPins; ++i) CHECK(intact(pins[i], 2, 100 + i));
}

TEST_CASE("gc::Heap pinning - a survivor-space pin stays through the next collection, then is copied") {
    for (const bool parallel : {false, true}) {
        HeapConfig c = pin_config();
        c.parallel_minor = parallel ? ParallelMinor::Always : ParallelMinor::Auto;
        c.minor_threads = parallel ? 4 : 1;
        c.tenure_age = 4;
        Heap h(c);
        HeapScope bind(h);
        // A root list keeps a working set of survivors, copied around the
        // pinned object each time.
        auto list = std::make_unique<uint64_t>(0);
        h.add_root(list.get());
        auto keep = std::make_unique<uint64_t>(0);
        h.add_root(keep.get());
        // The object survives one minor collection unpinned (rooted in
        // *keep only), so it sits in a survivor space.
        {
            const uintptr_t o = make_leaf(h, 3, 11);
            *keep = o;
        }
        h.collect(CollectionKind::Minor);
        volatile uintptr_t held = static_cast<uintptr_t>(*keep);
        // Off the stack, so that only `held` names the object there.
        auto at_box = std::make_unique<uintptr_t>(held);
        const uintptr_t& at = *at_box;
        CHECK(h.is_young(at));
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 2000; ++k) {
                const uintptr_t n = h.allocate(16, kPair);
                h.store(n, 0, *list);
                *list = n;
                if (k % 64 == 0) *list = 0;  // most of them die
            }
            h.collect(CollectionKind::Minor);
            CHECK_EQ(static_cast<uintptr_t>(*keep), at);
            CHECK(intact(at, 3, 11));
        }
        // Not named by the stack any more: the next collections copy it and
        // the root follows.
        held = 0;
        scrub_stack();
        for (int i = 0; i < 2; ++i) {
            churn(h, 1000);
            h.collect(CollectionKind::Minor);
        }
        CHECK(intact(static_cast<uintptr_t>(*keep), 3, 11));
        CHECK(h.is_valid_object(static_cast<uintptr_t>(*keep)));
        CHECK(static_cast<uintptr_t>(*keep) != at);
        h.verify();
        h.remove_root(keep.get());
        h.remove_root(list.get());
    }
}

TEST_CASE("gc::Heap pinning - parallel minor collections pin as the serial ones do") {
    HeapConfig c = pin_config();
    c.parallel_minor = ParallelMinor::Always;
    c.minor_threads = 8;
    Heap h(c);
    HeapScope bind(h);
    auto list = std::make_unique<uint64_t>(0);
    h.add_root(list.get());
    volatile uintptr_t held[8] = {};
    for (int round = 0; round < 12; ++round) {
        held[round % 8] = make_leaf(h, 2 + round % 5, 200 + round);
        for (int k = 0; k < 4000; ++k) {
            const uintptr_t n = h.allocate(16, kPair);
            h.store(n, 0, *list);
            *list = n;
            if (k % 512 == 0) *list = 0;
        }
        h.collect(CollectionKind::Minor);
        for (int i = 0; i <= round && i < 8; ++i) {
            // held[i] was last made in the latest round r with r % 8 == i.
            const int r = round - (round - i) % 8;
            CHECK(h.is_valid_object(held[i]));
            CHECK(intact(held[i], 2 + static_cast<size_t>(r % 5), 200 + static_cast<uintptr_t>(r)));
        }
    }
    CHECK(h.stats().parallel_evacuations > 0);
    h.remove_root(list.get());
}

namespace {

struct FiberHold {
    Heap* heap = nullptr;
    uintptr_t at = 0;
    bool ok = false;
};

PIN_TEST_NOINLINE void fiber_hold(void* arg) {
    auto* f = static_cast<FiberHold*>(arg);
    volatile uintptr_t held = make_leaf(*f->heap, 7, 12);
    f->at = held;
    Fiber::suspend();
    f->ok = held == f->at && f->heap->is_valid_object(held) && intact(held, 7, 12);
}

} // namespace

TEST_CASE("gc::Heap pinning - a suspended fiber's stack pins what it names") {
    Heap h(pin_config());
    HeapScope bind(h);
    FiberHold f;
    f.heap = &h;
    auto fiber = Fiber::create(&fiber_hold, &f, 64 * 1024);
    REQUIRE(fiber != nullptr);
    fiber->resume();
    REQUIRE(f.at != 0);
    for (int i = 0; i < 3; ++i) {
        churn(h, 10000);
        h.collect(i == 1 ? CollectionKind::Full : CollectionKind::Minor);
    }
    fiber->resume();
    CHECK(fiber->finished());
    CHECK(f.ok);
}

#if (defined(__x86_64__) || defined(__aarch64__)) && !defined(_MSC_VER)
TEST_CASE("gc::Heap pinning - a reference held only in a callee-saved register is seen") {
    Heap h(pin_config());
    HeapScope bind(h);
#if defined(__x86_64__)
    register uintptr_t held asm("r12") = make_leaf(h, 5, 13);
#else
    register uintptr_t held asm("x21") = make_leaf(h, 5, 13);
#endif
    __asm__ __volatile__("" : "+r"(held));
    const uintptr_t before = held;
    h.collect(CollectionKind::Minor);
    __asm__ __volatile__("" : "+r"(held));
    CHECK_EQ(held, before);
    CHECK(intact(held, 5, 13));
}
#endif

TEST_CASE("gc::Heap pinning - stress modes keep stack references valid") {
    for (const StressMode mode : {StressMode::Minor, StressMode::Full, StressMode::Alternate, StressMode::Incremental}) {
        HeapConfig c = pin_config();
        c.verify = false;
        Heap h(c);
        HeapScope bind(h);
        h.set_stress(mode);
        volatile uintptr_t a = make_leaf(h, 3, 14);
        volatile uintptr_t holder = h.allocate(16, kPair);
        for (int i = 0; i < 200; ++i) {
            hang_leaf(h, holder, static_cast<size_t>(i & 1), 2, 15 + static_cast<uintptr_t>(i));
            CHECK(intact(a, 3, 14));
            CHECK(intact(static_cast<uintptr_t>(Heap::load(holder, static_cast<size_t>(i & 1))), 2,
                         15 + static_cast<uintptr_t>(i)));
        }
        h.set_stress(StressMode::None);
        if (h.marking()) h.finish_marking();
        h.verify();
    }
}

TEST_CASE("gc::Heap pinning - an object only a heap root names still moves") {
    Heap h(pin_config());
    HeapScope bind(h);
    auto root = std::make_unique<uint64_t>(0);
    h.add_root(root.get());
    struct Make {
        PIN_TEST_NOINLINE static void into(Heap& heap, uint64_t* slot) { *slot = make_leaf(heap, 4, 16); }
    };
    Make::into(h, root.get());
    // Off the stack, so that nothing there names the object but by chance.
    auto before = std::make_unique<uint64_t>(*root);
    scrub_stack();
    h.collect(CollectionKind::Minor);
    CHECK(intact(static_cast<uintptr_t>(*root), 4, 16));
    CHECK(*root != *before);
    h.remove_root(root.get());
}
