// gc::Heap: allocation, the young generation's copying, promotion, the
// mature and large-object spaces, cards, full collections, tagged slots,
// custom layouts, roots, verification and the stress modes.

#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace brass::gc;

namespace {

HeapConfig small_config() {
    HeapConfig c;
    c.eden_bytes = 64 * 1024;
    c.survivor_bytes = 64 * 1024;
    c.mature_reserve_bytes = size_t{64} << 20;
    c.large_reserve_bytes = size_t{64} << 20;
    c.min_full_threshold_bytes = size_t{4} << 20;
    c.read_environment = false;
    return c;
}

// A two-field node: [0] next (reference), [1] value (number).
const LayoutId kNode = mask_layout(0b01, 17);
// Every word a reference.
const LayoutId kArray = mask_layout(~uint64_t{0}, 18);

uintptr_t make_node(Heap& h, uint64_t value) {
    uintptr_t n = h.allocate(16, kNode);
    reinterpret_cast<uint64_t*>(n)[1] = value;
    return n;
}

// A rooted chain of `count` nodes, head in *root, values count-1 .. 0.
void build_chain(Heap& h, uint64_t* root, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uintptr_t n = make_node(h, i);
        h.store(n, 0, *root);
        *root = n;
    }
}

uint64_t chain_sum(uint64_t head, size_t* length = nullptr) {
    uint64_t sum = 0;
    size_t len = 0;
    for (uintptr_t n = static_cast<uintptr_t>(head); n != 0; n = static_cast<uintptr_t>(Heap::load(n, 0))) {
        sum += Heap::load(n, 1);
        ++len;
    }
    if (length) *length = len;
    return sum;
}

} // namespace

TEST_CASE("gc::Heap - allocation writes the header and zeroes the payload") {
    Heap h(small_config());
    uintptr_t a = h.allocate(20, kNode, 0, 0x5A);
    REQUIRE(a != 0);
    CHECK_EQ(a % 8, uintptr_t{0});
    CHECK(h.is_young(a));
    CHECK(h.is_valid_object(a));
    const ObjectHeader* hdr = header_of(a);
    CHECK_EQ(hdr->size, 24u);
    CHECK_EQ(hdr->layout, kNode);
    CHECK_EQ(hdr->host_bits, 0x5A);
    CHECK_EQ(h.type_tag_of(a), 17u);
    for (size_t i = 0; i < 3; ++i) CHECK_EQ(Heap::load(a, i), uint64_t{0});

    uintptr_t z = h.allocate(0, kLeafLayout);
    CHECK_EQ(Heap::object_size(z), size_t{8});

    // Interior addresses resolve to their object; the header belongs to it too.
    CHECK_EQ(h.find_object(a + 16), a);
    CHECK_EQ(h.find_object(a - 8), a);
    CHECK_EQ(h.find_object(z), z);
    CHECK(!h.is_valid_object(a + 8));
}

TEST_CASE("gc::Heap - checked access rejects out-of-bounds and non-object addresses") {
    Heap h(small_config());
    uintptr_t a = h.allocate(16, kLeafLayout);
    h.check_access(a, 8, 8, "test");
    h.check_access(a + 8, -8, 8, "test");
    bool threw = false;
    try {
        h.check_access(a, 12, 8, "test");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        h.check_access(h.young_base() + 60000, 0, 8, "test");  // past the allocation top
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    int local = 0;
    h.check_access(reinterpret_cast<uintptr_t>(&local), 0, 4, "test");  // not heap memory: unchecked
}

TEST_CASE("gc::Heap - a minor collection moves reachable young objects and drops the rest") {
    HeapConfig c = small_config();
    c.tenure_age = 3;
    Heap h(c);
    uint64_t root = 0;
    h.add_root(&root);
    root = make_node(h, 42);
    uintptr_t dead = make_node(h, 7);
    const uintptr_t before = static_cast<uintptr_t>(root);
    const uint64_t epoch = h.relocation_epoch();

    h.collect(CollectionKind::Minor);
    CHECK_EQ(h.stats().minor_collections, uint64_t{1});
    CHECK_NE(static_cast<uintptr_t>(root), before);
    CHECK(h.is_young(static_cast<uintptr_t>(root)));
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(root), 1), uint64_t{42});
    CHECK(!h.is_valid_object(dead));
    CHECK_EQ(h.relocation_epoch(), epoch + 1);
    CHECK_EQ(header_of(static_cast<uintptr_t>(root))->age(), 1);

    h.collect(CollectionKind::Minor);
    CHECK(h.is_young(static_cast<uintptr_t>(root)));
    CHECK_EQ(header_of(static_cast<uintptr_t>(root))->age(), 2);
    h.collect(CollectionKind::Minor);  // age 3: promoted
    CHECK(h.is_old(static_cast<uintptr_t>(root)));
    CHECK(!h.is_movable(static_cast<uintptr_t>(root)));
    const uintptr_t promoted = static_cast<uintptr_t>(root);
    h.collect(CollectionKind::Minor);
    h.collect(CollectionKind::Full);
    CHECK_EQ(static_cast<uintptr_t>(root), promoted);  // old objects stay put
    CHECK_EQ(Heap::load(promoted, 1), uint64_t{42});
    h.remove_root(&root);
}

TEST_CASE("gc::Heap - chains survive many minor collections under verification") {
    HeapConfig c = small_config();
    c.verify = true;
    Heap h(c);
    uint64_t root = 0;
    h.add_root(&root);
    build_chain(h, &root, 5000);  // far more than eden holds: many minors, promotions
    CHECK(h.stats().minor_collections > 0);
    size_t len = 0;
    CHECK_EQ(chain_sum(root, &len), uint64_t{5000} * 4999 / 2);
    CHECK_EQ(len, size_t{5000});
    h.collect(CollectionKind::Full);
    CHECK_EQ(chain_sum(root, &len), uint64_t{5000} * 4999 / 2);
    h.verify();
    h.remove_root(&root);
}

TEST_CASE("gc::Heap - an old object's store of a young reference keeps the target alive") {
    Heap h(small_config());
    uint64_t holder = h.allocate(16, kNode, kAllocPinned);
    h.add_root(&holder);
    CHECK(h.is_old(static_cast<uintptr_t>(holder)));
    CHECK(header_of(static_cast<uintptr_t>(holder))->gc_bits & kGcPinned);
    {
        uintptr_t young = make_node(h, 99);
        h.store(static_cast<uintptr_t>(holder), 0, young);  // dirties the holder's card
    }
    for (int i = 0; i < 6; ++i) {
        h.collect(CollectionKind::Minor);
        h.verify();  // the card stays dirty while the target is young
    }
    const uintptr_t target = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(holder), 0));
    CHECK(h.is_valid_object(target));
    CHECK_EQ(Heap::load(target, 1), uint64_t{99});
    h.remove_root(&holder);
}

TEST_CASE("gc::Heap - a full collection reclaims old garbage and reuses its lines") {
    HeapConfig c = small_config();
    c.tenure_age = 1;
    Heap h(c);
    uint64_t keep = 0;
    h.add_root(&keep);
    std::vector<uint64_t> garbage_roots(4000, 0);
    for (auto& slot : garbage_roots) h.add_root(&slot);
    for (size_t i = 0; i < garbage_roots.size(); ++i) garbage_roots[i] = make_node(h, i);
    build_chain(h, &keep, 1000);
    h.collect(CollectionKind::Full);  // everything is old now
    for (uint64_t slot : garbage_roots) CHECK(h.is_old(static_cast<uintptr_t>(slot)));
    const size_t used_with_garbage = h.old_used_bytes();
    for (auto& slot : garbage_roots) slot = 0;
    h.collect(CollectionKind::Full);
    const size_t used_after = h.old_used_bytes();
    CHECK(used_after < used_with_garbage);
    CHECK_EQ(chain_sum(keep), uint64_t{1000} * 999 / 2);
    // Refill: the freed lines are reused before the heap grows.
    const size_t committed = h.committed_bytes();
    for (size_t i = 0; i < garbage_roots.size(); ++i) garbage_roots[i] = h.allocate(16, kNode, kAllocOld);
    CHECK(h.committed_bytes() <= committed);
    h.verify();
    for (auto& slot : garbage_roots) h.remove_root(&slot);
    h.remove_root(&keep);
}

TEST_CASE("gc::Heap - large objects: direct allocation, cards, reclamation, interior lookup") {
    Heap h(small_config());
    uint64_t big = h.allocate(200 * 1024, kArray);  // over the large threshold
    h.add_root(&big);
    const uintptr_t b = static_cast<uintptr_t>(big);
    CHECK(h.is_old(b));
    CHECK(h.is_valid_object(b));
    CHECK_EQ(h.find_object(b + 150 * 1024), b);
    CHECK_EQ(Heap::object_size(b), size_t{200 * 1024});
    for (size_t i = 0; i < 64; ++i) h.store(b, i * 100, make_node(h, i));
    for (int i = 0; i < 4; ++i) h.collect(CollectionKind::Minor);
    h.verify();
    uint64_t sum = 0;
    for (size_t i = 0; i < 64; ++i) sum += Heap::load(static_cast<uintptr_t>(Heap::load(b, i * 100)), 1);
    CHECK_EQ(sum, uint64_t{64 * 63 / 2});
    CHECK_EQ(static_cast<uintptr_t>(big), b);  // never moves

    const size_t committed = h.committed_bytes();
    big = 0;
    h.collect(CollectionKind::Full);
    CHECK(h.committed_bytes() < committed);
    CHECK(!h.is_valid_object(b));
    h.remove_root(&big);
}

TEST_CASE("gc::Heap - medium and large young objects are promoted to the right space") {
    HeapConfig c = small_config();
    c.eden_bytes = 256 * 1024;
    c.tenure_age = 1;
    Heap h(c);
    uint64_t medium = h.allocate(4000, kArray);
    uint64_t large = h.allocate(20000, kArray);  // young (under the large threshold), promoted to the LOS
    h.add_root(&medium);
    h.add_root(&large);
    CHECK(h.is_young(static_cast<uintptr_t>(medium)));
    CHECK(h.is_young(static_cast<uintptr_t>(large)));
    h.store(static_cast<uintptr_t>(medium), 3, make_node(h, 5));
    h.store(static_cast<uintptr_t>(large), 2000, make_node(h, 6));
    h.collect(CollectionKind::Minor);
    CHECK(h.is_old(static_cast<uintptr_t>(medium)));
    CHECK(h.is_old(static_cast<uintptr_t>(large)));
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(medium), 3)), 1), uint64_t{5});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(large), 2000)), 1), uint64_t{6});
    h.verify();
    h.remove_root(&medium);
    h.remove_root(&large);
}

TEST_CASE("gc::Heap - word slots keep their tag, and reference tags filter what is a reference") {
    constexpr uint64_t kTagGcref = 0x7FFD000000000000ULL;
    constexpr uint64_t kTagDouble = 0x4000000000000000ULL;
    HeapConfig c = small_config();
    c.reference_tags = {0x7FFD};
    Heap h(c);
    uint64_t tagged = 0;
    uint64_t raw = 0;
    uint64_t fake = 0;
    h.add_root(&tagged);
    h.add_root(&raw);
    h.add_root(&fake);
    uintptr_t target = make_node(h, 11);
    tagged = kTagGcref | target;
    raw = target;
    fake = kTagDouble | target;  // a number whose low bits look like a reference
    h.collect(CollectionKind::Minor);
    const uintptr_t moved = static_cast<uintptr_t>(tagged & kAddressMask);
    CHECK_NE(moved, target);
    CHECK_EQ(tagged & ~kAddressMask, kTagGcref);
    CHECK_EQ(static_cast<uintptr_t>(raw), moved);
    CHECK_EQ(fake, kTagDouble | target);  // untouched
    CHECK_EQ(Heap::load(moved, 1), uint64_t{11});
    h.remove_root(&tagged);
    h.remove_root(&raw);
    h.remove_root(&fake);
}

namespace {

// A custom layout: word 0 counts the references that follow; everything
// after them is raw data the collector must not touch.
void trace_counted(uintptr_t payload, size_t, Tracer& t) {
    auto* words = reinterpret_cast<uint64_t*>(payload);
    for (uint64_t i = 0; i < words[0]; ++i) t.visit(words + 1 + i);
}

} // namespace

TEST_CASE("gc::Heap - a custom layout's trace function decides which words are references") {
    LayoutDescriptor d;
    d.kind = LayoutKind::Custom;
    d.trace = &trace_counted;
    d.name = "counted";
    const LayoutId counted = register_layout(d);
    Heap h(small_config());
    uint64_t obj = h.allocate(8 * 4, counted);
    h.add_root(&obj);
    const uintptr_t target = make_node(h, 3);
    auto* words = reinterpret_cast<uint64_t*>(obj);
    words[0] = 1;
    words[1] = target;
    words[2] = target;  // raw data that happens to equal the address
    h.collect(CollectionKind::Minor);
    words = reinterpret_cast<uint64_t*>(obj);
    CHECK_NE(static_cast<uintptr_t>(words[1]), target);
    CHECK_EQ(words[2], static_cast<uint64_t>(target));
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(words[1]), 1), uint64_t{3});
    h.remove_root(&obj);
}

TEST_CASE("gc::Heap - root sources and object iteration") {
    Heap h(small_config());
    std::vector<uint64_t> table(50, 0);
    const auto id = h.add_root_source([&table](Tracer& t) {
        for (auto& w : table) t.visit(&w);
    });
    for (size_t i = 0; i < table.size(); ++i) table[i] = make_node(h, i);
    h.collect(CollectionKind::Minor);
    uint64_t sum = 0;
    for (uint64_t w : table) sum += Heap::load(static_cast<uintptr_t>(w), 1);
    CHECK_EQ(sum, uint64_t{50 * 49 / 2});
    size_t objects = 0;
    h.for_each_object([&](uintptr_t) { ++objects; });
    CHECK_EQ(objects, size_t{50});
    h.remove_root_source(id);
    h.collect(CollectionKind::Full);
    objects = 0;
    h.for_each_object([&](uintptr_t) { ++objects; });
    CHECK_EQ(objects, size_t{0});
}

TEST_CASE("gc::Heap - poison mode overwrites evacuated memory") {
    HeapConfig c = small_config();
    c.poison = true;
    Heap h(c);
    uint64_t root = make_node(h, 77);
    h.add_root(&root);
    const uintptr_t old_copy = static_cast<uintptr_t>(root);
    h.collect(CollectionKind::Minor);
    CHECK_EQ(Heap::load(old_copy, 1), uint64_t{0xDBDBDBDBDBDBDBDBULL});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(root), 1), uint64_t{77});
    h.remove_root(&root);
}

TEST_CASE("gc::Heap - every stress mode keeps a mutating graph intact") {
    for (StressMode mode : {StressMode::Minor, StressMode::Full, StressMode::Alternate}) {
        HeapConfig c = small_config();
        c.stress = mode;
        c.verify = mode != StressMode::Full;  // full-stress verification is quadratic; covered below
        Heap h(c);
        uint64_t root = 0;
        h.add_root(&root);
        build_chain(h, &root, 300);
        // Rewire: drop every other node, append fresh ones.
        for (uintptr_t n = static_cast<uintptr_t>(root); n != 0; n = static_cast<uintptr_t>(Heap::load(n, 0))) {
            const uintptr_t next = static_cast<uintptr_t>(Heap::load(n, 0));
            if (next) h.store(n, 0, Heap::load(next, 0));
        }
        size_t len = 0;
        chain_sum(root, &len);
        CHECK_EQ(len, size_t{150});
        build_chain(h, &root, 100);
        chain_sum(root, &len);
        CHECK_EQ(len, size_t{250});
        CHECK(h.collection_count() >= 400);
        h.verify();
        h.remove_root(&root);
    }
}

TEST_CASE("gc::Heap - safepoints collect only when asked or stressed") {
    Heap h(small_config());
    h.safepoint_at(0, 0);
    CHECK_EQ(h.collection_count(), uint64_t{0});
    h.request_collection(CollectionKind::Minor);
    h.safepoint_at(0, 0);
    CHECK_EQ(h.stats().minor_collections, uint64_t{1});
    h.safepoint_at(0, 0);
    CHECK_EQ(h.collection_count(), uint64_t{1});
    h.set_stress(StressMode::Full);
    h.safepoint_at(0, 0);
    CHECK_EQ(h.stats().full_collections, uint64_t{1});
}

TEST_CASE("gc::Heap - the thread's current heap and HeapScope") {
    CHECK(Heap::current() == nullptr);
    Heap a(small_config());
    {
        HeapScope scope(a);
        CHECK(Heap::current() == &a);
        {
            Heap b(small_config());
            HeapScope inner(b);
            CHECK(Heap::current() == &b);
        }
        CHECK(Heap::current() == &a);
    }
    CHECK(Heap::current() == nullptr);
}
