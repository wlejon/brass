// gc::Heap: allocation, the young generation's copying, promotion, the
// mature and large-object spaces, cards, full collections, tagged slots,
// custom layouts, roots, verification and the stress modes.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace brass::gc;
using brass::test::hide_address;
using brass::test::HeapRoot;
using brass::test::in_own_frame;
using brass::test::scrub_stack;
using brass::test::unhide_address;

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

TEST_CASE("gc::Heap - an uninitialized leaf, filled by its caller, lives like any other") {
    Heap h(small_config());
    // Young (moved by the minor collection) and pinned old.
    for (const uint32_t where : {0u, kAllocPinned}) {
        uintptr_t a = h.allocate(64, kLeafLayout, where | kAllocUninitialized);
        REQUIRE(a != 0);
        CHECK(h.is_valid_object(a));
        CHECK_EQ(Heap::object_size(a), size_t{64});
        for (size_t i = 0; i < 8; ++i) reinterpret_cast<uint64_t*>(a)[i] = 0xA5A5A5A5A5A5A5A5ull + i;
        uint64_t root = a;
        h.add_root(&root);
        h.collect(CollectionKind::Minor);
        h.collect(CollectionKind::Full);
        h.verify();
        for (size_t i = 0; i < 8; ++i) CHECK_EQ(Heap::load(static_cast<uintptr_t>(root), i), 0xA5A5A5A5A5A5A5A5ull + i);
        h.remove_root(&root);
    }
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
    HeapRoot<Heap> root(h);
    uint64_t dead = 0;
    uint64_t before = 0;
    in_own_frame([&] {
        *root.slot = make_node(h, 42);
        dead = hide_address(make_node(h, 7));
        before = root.hidden();
    });
    const uint64_t epoch = h.relocation_epoch();

    scrub_stack();
    h.collect(CollectionKind::Minor);
    CHECK_EQ(h.stats().minor_collections, uint64_t{1});
    CHECK_EQ(h.relocation_epoch(), epoch + 1);
    in_own_frame([&] {
        const uintptr_t r = static_cast<uintptr_t>(*root.slot);
        CHECK_NE(root.hidden(), before);
        CHECK(h.is_young(r));
        CHECK_EQ(Heap::load(r, 1), uint64_t{42});
        CHECK(!h.is_valid_object(static_cast<uintptr_t>(unhide_address(dead))));
        CHECK_EQ(header_of(r)->age(), 1);
    });

    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        CHECK(h.is_young(static_cast<uintptr_t>(*root.slot)));
        CHECK_EQ(header_of(static_cast<uintptr_t>(*root.slot))->age(), 2);
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);  // age 3: promoted
    in_own_frame([&] {
        CHECK(h.is_old(static_cast<uintptr_t>(*root.slot)));
        CHECK(!h.is_movable(static_cast<uintptr_t>(*root.slot)));
    });
    const uint64_t promoted = root.hidden();
    h.collect(CollectionKind::Minor);
    h.collect(CollectionKind::Full);
    CHECK_EQ(root.hidden(), promoted);  // old objects stay put
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(*root.slot), 1), uint64_t{42});
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

TEST_CASE("gc::Heap - releasing the young generation's unused pages keeps its objects") {
    HeapConfig c = small_config();
    c.verify = true;
    Heap h(c);
    uint64_t root = 0;
    h.add_root(&root);
    build_chain(h, &root, 5000);  // more than eden holds: minors, survivors
    CHECK(h.stats().minor_collections > 0);
    CHECK(h.release_unused_young() > 0);
    size_t len = 0;
    CHECK_EQ(chain_sum(root, &len), uint64_t{5000} * 4999 / 2);
    // After a minor collection eden is empty: all of it goes, and allocation
    // over the released pages starts from zeroed memory.
    h.collect(CollectionKind::Minor);
    CHECK(h.release_unused_young() >= c.eden_bytes);
    const uintptr_t fresh = h.allocate(48, kArray);
    for (size_t i = 0; i < 6; ++i) CHECK_EQ(reinterpret_cast<uint64_t*>(fresh)[i], uint64_t{0});
    build_chain(h, &root, 5000);
    CHECK_EQ(chain_sum(root, &len), uint64_t{5000} * 4999 / 2 * 2);
    CHECK_EQ(len, size_t{10000});
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
    HeapRoot<Heap> big(h);
    uint64_t b = 0;  // hidden
    in_own_frame([&] {
        *big.slot = h.allocate(200 * 1024, kArray);  // over the large threshold
        const uintptr_t a = static_cast<uintptr_t>(*big.slot);
        b = hide_address(a);
        CHECK(h.is_old(a));
        CHECK(h.is_valid_object(a));
        CHECK_EQ(h.find_object(a + 150 * 1024), a);
        CHECK_EQ(Heap::object_size(a), size_t{200 * 1024});
        for (size_t i = 0; i < 64; ++i) h.store(a, i * 100, make_node(h, i));
    });
    for (int i = 0; i < 4; ++i) h.collect(CollectionKind::Minor);
    h.verify();
    in_own_frame([&] {
        const uintptr_t a = static_cast<uintptr_t>(*big.slot);
        uint64_t sum = 0;
        for (size_t i = 0; i < 64; ++i) sum += Heap::load(static_cast<uintptr_t>(Heap::load(a, i * 100)), 1);
        CHECK_EQ(sum, uint64_t{64 * 63 / 2});
    });
    CHECK_EQ(big.hidden(), b);  // never moves

    // A dead large object's run is freed and its pages kept committed for the
    // next one (heap_spaces.cpp, kRetainedLargeBytes): the same size again
    // takes the same run, zeroed, and commits nothing more.
    const size_t committed = h.committed_bytes();
    *big.slot = 0;
    scrub_stack();
    h.collect(CollectionKind::Full);
    CHECK(!h.is_valid_object(static_cast<uintptr_t>(unhide_address(b))));
    CHECK_EQ(h.committed_bytes(), committed);
    in_own_frame([&] { *big.slot = h.allocate(200 * 1024, kArray); });
    CHECK_EQ(big.hidden(), b);
    CHECK_EQ(h.committed_bytes(), committed);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(unhide_address(b)), 100), uint64_t{0});
}

TEST_CASE("gc::Heap - medium and large young objects are promoted to the right space") {
    HeapConfig c = small_config();
    c.eden_bytes = 256 * 1024;
    c.tenure_age = 1;
    Heap h(c);
    HeapRoot<Heap> medium(h);
    HeapRoot<Heap> large(h);
    in_own_frame([&] {
        *medium.slot = h.allocate(4000, kArray);
        *large.slot = h.allocate(20000, kArray);  // young (under the large threshold), promoted to the LOS
        CHECK(h.is_young(static_cast<uintptr_t>(*medium.slot)));
        CHECK(h.is_young(static_cast<uintptr_t>(*large.slot)));
        h.store(static_cast<uintptr_t>(*medium.slot), 3, make_node(h, 5));
        h.store(static_cast<uintptr_t>(*large.slot), 2000, make_node(h, 6));
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        const auto m = static_cast<uintptr_t>(*medium.slot);
        const auto l = static_cast<uintptr_t>(*large.slot);
        CHECK(h.is_old(m));
        CHECK(h.is_old(l));
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(m, 3)), 1), uint64_t{5});
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(Heap::load(l, 2000)), 1), uint64_t{6});
    });
    h.verify();
}

TEST_CASE("gc::Heap - word slots keep their tag, and reference tags filter what is a reference") {
    constexpr uint64_t kTagGcref = 0x7FFD000000000000ULL;
    constexpr uint64_t kTagDouble = 0x4000000000000000ULL;
    HeapConfig c = small_config();
    c.reference_tags = {0, 0x7FFD};
    Heap h(c);
    HeapRoot<Heap> tagged(h);
    HeapRoot<Heap> raw(h);
    HeapRoot<Heap> fake(h);
    uint64_t target = 0;  // hidden
    in_own_frame([&] {
        const uintptr_t t = make_node(h, 11);
        target = hide_address(t);
        *tagged.slot = kTagGcref | t;
        *raw.slot = t;
        *fake.slot = kTagDouble | t;  // a number whose low bits look like a reference
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        const uintptr_t t = static_cast<uintptr_t>(unhide_address(target));
        const uintptr_t moved = static_cast<uintptr_t>(*tagged.slot & kAddressMask);
        CHECK_NE(moved, t);
        CHECK_EQ(*tagged.slot & ~kAddressMask, kTagGcref);
        CHECK_EQ(static_cast<uintptr_t>(*raw.slot), moved);
        CHECK_EQ(*fake.slot, kTagDouble | t);  // untouched
        CHECK_EQ(Heap::load(moved, 1), uint64_t{11});
    });
}

TEST_CASE("gc::Heap - a heap whose reference tags leave out 0 takes no raw word for a reference") {
    constexpr uint64_t kTagObject = 0xFFF1000000000000ULL;
    HeapConfig c = small_config();
    c.reference_tags = {0xFFF1};
    Heap h(c);
    // Held in heap memory: a raw word on the stack is read as a reference by
    // the stack scan whatever the reference tags (gc_contract.md 2.5).
    HeapRoot<Heap> tagged(h);
    auto words = std::make_unique<uint64_t[]>(2);
    uint64_t* raw = &words[0];
    uint64_t* derived = &words[1];
    uint64_t target = 0;  // hidden
    in_own_frame([&] {
        const uintptr_t t = make_node(h, 12);
        target = hide_address(t);
        *tagged.slot = kTagObject | t;
        *raw = t;  // a raw pointer the host holds, not a reference
        *derived = t + 8;
        CHECK(!h.is_reference_tag(t));
        CHECK(h.is_reference_tag(*tagged.slot));
    });
    h.add_root(raw);
    h.add_root_source([raw, derived](Tracer& t) {
        t.visit_conservative(raw);  // an untyped register holding it
        t.visit_derived(derived);
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        const uintptr_t t = static_cast<uintptr_t>(unhide_address(target));
        const uintptr_t moved = static_cast<uintptr_t>(*tagged.slot & kAddressMask);
        CHECK_NE(moved, t);
        CHECK_EQ(*tagged.slot & ~kAddressMask, kTagObject);
        CHECK_EQ(static_cast<uintptr_t>(*raw), t);          // untouched
        CHECK_EQ(static_cast<uintptr_t>(*derived), t + 8);  // untouched
        CHECK_EQ(Heap::load(moved, 1), uint64_t{12});
    });
    // The barrier ignores a raw word too: storing one into an old object
    // leaves its card clean.
    uint64_t old = h.allocate(16, kNode, kAllocOld);
    h.add_root(&old);
    h.collect(CollectionKind::Full);  // cleans every card
    const uintptr_t young = make_node(h, 13);
    const size_t card = (static_cast<uintptr_t>(old) - h.old_base()) >> kCardShift;
    h.store(static_cast<uintptr_t>(old), 0, young);
    CHECK_EQ(h.card_table_base()[card], kCardClean);
    h.store(static_cast<uintptr_t>(old), 0, 0);
    h.remove_root(raw);
    h.remove_root(&old);
}

TEST_CASE("gc::Heap - remember() covers a bulk copy into an old object") {
    Heap h(small_config());
    uint64_t young[3] = {0, 0, 0};
    for (auto& y : young) h.add_root(&y);
    for (int i = 0; i < 3; ++i) young[i] = make_node(h, 40 + static_cast<uint64_t>(i));
    // A large array, old from birth, filled by memcpy with no per-word barrier.
    uint64_t big = h.allocate(8 * 20000, kArray);
    h.add_root(&big);
    REQUIRE(h.is_old(static_cast<uintptr_t>(big)));
    h.collect(CollectionKind::Full);  // clean cards; the nodes are old now too
    for (int i = 0; i < 3; ++i) young[i] = make_node(h, 50 + static_cast<uint64_t>(i));
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(big) + 8 * 19000), young, sizeof young);
    h.remember(static_cast<uintptr_t>(big));
    for (auto& y : young) h.remove_root(&y);
    h.collect(CollectionKind::Minor);  // the nodes survive only through the big array
    for (int i = 0; i < 3; ++i) {
        const uintptr_t node = static_cast<uintptr_t>(Heap::load(static_cast<uintptr_t>(big), 19000 + static_cast<size_t>(i)));
        CHECK(h.is_valid_object(node));
        CHECK_EQ(Heap::load(node, 1), uint64_t{50} + static_cast<uint64_t>(i));
    }
    h.verify();
    h.remove_root(&big);
}

TEST_CASE("gc::Heap - the bump region can live in a host's buffer") {
    Heap h(small_config());
    HeapRoot<Heap> kept(h);
    HeapRoot<Heap> root(h);
    in_own_frame([&] { *kept.slot = make_node(h, 21); });
    const Heap::AllocationBuffer before = *h.allocation_buffer();
    // On the C++ heap: the buffer's top is an eden address, which a stack
    // word would read as naming the object below it.
    auto host_buffer = std::make_unique<Heap::AllocationBuffer>();
    Heap::AllocationBuffer& host = *host_buffer;
    h.bind_allocation_buffer(&host);
    CHECK(h.allocation_buffer() == &host);
    CHECK(host.top == before.top);
    CHECK(host.end == before.end);
    // An object bumped by hand in the host's buffer, as inline generated
    // code does, is a heap object like any other.
    uint64_t inline_obj = 0;  // hidden
    in_own_frame([&] {
        const size_t total = kHeaderBytes + 16;
        REQUIRE(host.end - host.top >= total);
        auto* header = reinterpret_cast<ObjectHeader*>(host.top);
        header->size = 16;
        header->layout = kNode;
        header->gc_bits = 0;
        header->host_bits = 0;
        const uintptr_t obj = host.top + kHeaderBytes;
        host.top += total;
        reinterpret_cast<uint64_t*>(obj)[0] = *kept.slot;
        reinterpret_cast<uint64_t*>(obj)[1] = 22;
        *root.slot = obj;
        inline_obj = hide_address(obj);
        const uintptr_t via_heap = h.allocate(16, kNode);
        CHECK_EQ(via_heap, obj + 16 + kHeaderBytes);  // the heap bumps the same region
        CHECK(h.is_valid_object(obj));
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        const auto r = static_cast<uintptr_t>(*root.slot);
        CHECK_NE(root.hidden(), inline_obj);
        CHECK_EQ(Heap::load(r, 1), uint64_t{22});
        CHECK_EQ(Heap::load(r, 0), *kept.slot);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(*kept.slot), 1), uint64_t{21});
    });
    CHECK(host.top <= host.end);  // the collection reset the host's region
    h.set_stress(StressMode::Minor);
    CHECK(host.top == host.end);
    h.set_stress(StressMode::None);
    h.bind_allocation_buffer(nullptr);
    CHECK(h.allocation_buffer() != &host);
    CHECK(h.allocation_buffer()->top == host.top);
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
    HeapRoot<Heap> obj(h);
    uint64_t target = 0;  // hidden
    in_own_frame([&] {
        *obj.slot = h.allocate(8 * 4, counted);
        const uintptr_t t = make_node(h, 3);
        target = hide_address(t);
        auto* words = reinterpret_cast<uint64_t*>(*obj.slot);
        words[0] = 1;
        words[1] = t;
        words[2] = t;  // raw data that happens to equal the address
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        const uint64_t t = unhide_address(target);
        auto* words = reinterpret_cast<uint64_t*>(*obj.slot);
        CHECK_NE(words[1], t);
        CHECK_EQ(words[2], t);
        CHECK_EQ(Heap::load(static_cast<uintptr_t>(words[1]), 1), uint64_t{3});
    });
}

TEST_CASE("gc::Heap - root sources and object iteration") {
    Heap h(small_config());
    std::vector<uint64_t> table(50, 0);
    const auto id = h.add_root_source([&table](Tracer& t) {
        for (auto& w : table) t.visit(&w);
    });
    in_own_frame([&] {
        for (size_t i = 0; i < table.size(); ++i) table[i] = make_node(h, i);
    });
    scrub_stack();
    h.collect(CollectionKind::Minor);
    in_own_frame([&] {
        uint64_t sum = 0;
        for (uint64_t w : table) sum += Heap::load(static_cast<uintptr_t>(w), 1);
        CHECK_EQ(sum, uint64_t{50 * 49 / 2});
    });
    size_t objects = 0;
    h.for_each_object([&](uintptr_t) { ++objects; });
    CHECK_EQ(objects, size_t{50});
    h.remove_root_source(id);
    scrub_stack();
    h.collect(CollectionKind::Full);
    objects = 0;
    h.for_each_object([&](uintptr_t) { ++objects; });
    CHECK_EQ(objects, size_t{0});
}

TEST_CASE("gc::Heap - poison mode overwrites evacuated memory") {
    HeapConfig c = small_config();
    c.poison = true;
    Heap h(c);
    HeapRoot<Heap> root(h);
    in_own_frame([&] { *root.slot = make_node(h, 77); });
    const uint64_t old_copy = root.hidden();
    scrub_stack();
    h.collect(CollectionKind::Minor);
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(unhide_address(old_copy)), 1), uint64_t{0xDBDBDBDBDBDBDBDBULL});
    CHECK_EQ(Heap::load(static_cast<uintptr_t>(*root.slot), 1), uint64_t{77});
}

TEST_CASE("gc::Heap - every stress mode keeps a mutating graph intact") {
    for (StressMode mode : {StressMode::Minor, StressMode::Full, StressMode::Alternate, StressMode::Incremental}) {
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
        // Incremental: a collection every fourth step, a cycle every eleventh
        // or so (its start, seven slices, minors among them, its finish).
        CHECK(h.collection_count() >= (mode == StressMode::Incremental ? 100u : 400u));
        if (mode == StressMode::Incremental) CHECK(h.stats().full_collections >= 20);
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
