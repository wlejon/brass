// gc::Heap pretenuring (docs/gc_contract.md 2.14): try_allocate_pretenured
// makes young objects before the first minor collection, then zeroed old ones
// whose card is dirty, and the next minor collection promotes what they were
// filled with (even by a store no barrier saw) without a stay in the survivor
// space; a site's probes refuse it once its objects die young (from the start
// or after a long-lived phase), and a site whose objects live keeps
// allocating old; an allocation memento behind a young object follows it into
// the survivor space and counts its promotion.
//
// Objects are touched only in frames of their own and named across a
// collection by heap roots, so no stale stack word pins one in place.

#include "stack_scrub.hpp"
#include "test_framework.hpp"

#include <brass/gc/heap.hpp>

using namespace brass::gc;
using brass::test::collect_scrubbed;
using brass::test::from_own_frame;
using brass::test::HeapRoot;
using brass::test::in_own_frame;
using brass::test::scrub_stack;

namespace {

HeapConfig plain_config() {
    HeapConfig c;
    c.read_environment = false;  // stress would replace the collections these wait for
    return c;
}

const LayoutId kNode = mask_layout(0b01, 47);  // word 0 a reference, word 1 a number

LayoutId memento_layout_id() {
    static const LayoutId id = [] {
        LayoutDescriptor d;
        d.kind = LayoutKind::Leaf;
        d.type_tag = 0x7E57;
        d.name = "test.memento";
        return register_layout(d);
    }();
    return id;
}

// `n` objects of `site`, each naming the one before it from its word 0 and
// the newest named by `head`'s slot (0 there drops them all as they are
// made): how many of them were made old.
size_t make_chain(Heap& h, PretenureSite& site, size_t n, uint64_t* head) {
    return from_own_frame([&] {
        size_t old = 0;
        for (size_t i = 0; i < n; ++i) {
            const uintptr_t o = h.try_allocate_pretenured(16, kNode, site);
            if (o == 0) return ~size_t{0};
            old += h.is_old(o) ? 1 : 0;
            if (head != nullptr) {
                reinterpret_cast<uint64_t*>(o)[0] = *head;
                *head = o;
            }
        }
        return old;
    });
}

constexpr size_t kEvery = PretenureSite::kProbeEvery;

} // namespace

TEST_CASE("gc::Heap pretenuring - young before the first minor, then old, and what it holds is promoted") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    PretenureSite site;
    // The site's first object is a probe; the second is young, as everything
    // before the first minor collection.
    CHECK(from_own_frame([&] { return h.is_young(h.try_allocate_pretenured(16, kNode, site)); }));
    CHECK(from_own_frame([&] { return h.is_young(h.try_allocate_pretenured(16, kNode, site)); }));
    CHECK_EQ(site.made, 1u);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(site.dead, 1u);
    CHECK_EQ(site.refused, 0u);  // one probe is no verdict

    HeapRoot<Heap> holder(h);
    in_own_frame([&] {
        // A judged site's next object is a probe again; the one after is old.
        CHECK(h.is_young(h.try_allocate_pretenured(16, kNode, site)));
        const uintptr_t old = h.try_allocate_pretenured(16, kNode, site);
        REQUIRE(old != 0);
        CHECK(h.is_old(old));
        CHECK_EQ(Heap::load(old, 0), uint64_t{0});
        CHECK_EQ(Heap::load(old, 1), uint64_t{0});
        const uintptr_t child = h.allocate(16, kNode);
        REQUIRE(h.is_young(child));
        // A store no barrier sees, as a pretenured literal's own fields are
        // filled: the dirty card and the pass find it.
        reinterpret_cast<uint64_t*>(old)[0] = child;
        *holder.slot = old;
    });
    collect_scrubbed(h, CollectionKind::Minor);
    // Promoted at its first collection, where one copied into the survivor
    // space waits for the second.
    CHECK(from_own_frame([&] { return h.is_old(static_cast<uintptr_t>(Heap::load(*holder.slot, 0))); }));
}

TEST_CASE("gc::Heap pretenuring - a site whose objects die young is refused before any is made old") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    PretenureSite site;
    CHECK_EQ(make_chain(h, site, 64 * kEvery, nullptr), size_t{0});
    CHECK_EQ(site.made, 64u);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK(site.dead > 32u);
    CHECK(site.refused != 0u);
    // From now on all young, probes or not.
    CHECK_EQ(make_chain(h, site, 64 * kEvery, nullptr), size_t{0});
    CHECK_EQ(site.made, 64u);
}

TEST_CASE("gc::Heap pretenuring - a site whose objects live is not refused and allocates old") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    PretenureSite site;
    HeapRoot<Heap> head(h);
    CHECK_EQ(make_chain(h, site, 64 * kEvery, head.slot), size_t{0});
    collect_scrubbed(h, CollectionKind::Minor);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(site.dead, 0u);
    CHECK_EQ(site.refused, 0u);
    // Old but for the probes.
    CHECK_EQ(make_chain(h, site, 64 * kEvery, head.slot), 64 * kEvery - 64);
    collect_scrubbed(h, CollectionKind::Minor);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(site.dead, 0u);
    CHECK_EQ(site.refused, 0u);
}

TEST_CASE("gc::Heap pretenuring - a site that turns short-lived is refused within a few minor collections") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    PretenureSite site;
    HeapRoot<Heap> head(h);
    CHECK_EQ(make_chain(h, site, 64 * kEvery, head.slot), size_t{0});
    collect_scrubbed(h, CollectionKind::Minor);
    // A long-lived phase, more probes than a verdict weighs (256), each eden
    // old but for its first 64 probes.
    for (int round = 0; round < 4; ++round) {
        CHECK_EQ(make_chain(h, site, 256 * kEvery, head.slot), 256 * kEvery - 64);
        collect_scrubbed(h, CollectionKind::Minor);
    }
    CHECK_EQ(site.refused, 0u);
    // Then dropped as soon as made: refused by the fourth minor collection.
    *head.slot = 0;
    int rounds = 0;
    while (site.refused == 0 && rounds < 8) {
        CHECK_EQ(make_chain(h, site, 256 * kEvery, nullptr), 256 * kEvery - 64);
        collect_scrubbed(h, CollectionKind::Minor);
        ++rounds;
    }
    CHECK(site.refused != 0u);
    CHECK(rounds <= 4);
    CHECK_EQ(make_chain(h, site, 64 * kEvery, nullptr), size_t{0});
}

TEST_CASE("gc::Heap pretenuring - minor collections in a marking cycle judge probes, its finish drops them") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    PretenureSite site;
    HeapRoot<Heap> head(h);
    collect_scrubbed(h, CollectionKind::Minor);
    REQUIRE(h.start_marking());
    // Old (and grey) but for the probes, kept.
    CHECK_EQ(make_chain(h, site, 64 * kEvery, head.slot), 64 * kEvery - 64);
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(site.made, 64u);
    CHECK_EQ(site.dead, 0u);
    // Probes dropped at once, which only the finish sees.
    *head.slot = 0;
    CHECK_EQ(make_chain(h, site, 64 * kEvery, nullptr), 64 * kEvery - 64);
    scrub_stack();
    h.finish_marking();
    collect_scrubbed(h, CollectionKind::Minor);
    CHECK_EQ(site.made, 128u);
    CHECK_EQ(site.dead, 0u);
    CHECK_EQ(site.refused, 0u);
}

TEST_CASE("gc::Heap pretenuring - a memento follows its object and counts its promotion") {
    Heap h(plain_config());
    REQUIRE(h.stress() == StressMode::None);
    set_memento_layout(memento_layout_id());
    uint64_t promoted = 0;
    HeapRoot<Heap> kept(h);
    const bool adjacent = from_own_frame([&] {
        const uintptr_t object = h.allocate(16, kNode);
        const uintptr_t memento = h.allocate(8, memento_layout_id());
        reinterpret_cast<uint64_t*>(memento)[0] = reinterpret_cast<uint64_t>(&promoted);
        *kept.slot = object;
        return memento == object + 16 + kHeaderBytes;
    });
    REQUIRE(adjacent);
    collect_scrubbed(h, CollectionKind::Minor);  // into the survivor space, the memento behind it
    CHECK(from_own_frame([&] { return h.is_young(static_cast<uintptr_t>(*kept.slot)); }));
    CHECK_EQ(promoted, uint64_t{0});
    collect_scrubbed(h, CollectionKind::Minor);  // promoted: counted
    CHECK(from_own_frame([&] { return h.is_old(static_cast<uintptr_t>(*kept.slot)); }));
    CHECK_EQ(promoted, uint64_t{1});
    set_memento_layout(0xFFFF);
}
