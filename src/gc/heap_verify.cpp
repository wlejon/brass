// Questions about objects (is this an object, which object contains this
// address, where did this object go) and heap verification: a walk over
// every object and root that checks each reference names a real object and
// each old-to-young reference has a dirty card.

#include "heap_internal.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace brass::gc {

using detail::GcState;
using detail::HeapState;

namespace {

// The highest set bit at an index in [min_bit, from_bit], or SIZE_MAX.
size_t find_prev_bit(const uint64_t* words, size_t from_bit, size_t min_bit) noexcept {
    size_t w = from_bit >> 6;
    uint64_t word = words[w] & (~uint64_t{0} >> (63 - (from_bit & 63)));
    for (;;) {
        if (word) {
            const size_t bit = (w << 6) + detail::clz_index64(word);
            return bit >= min_bit ? bit : SIZE_MAX;
        }
        if (w == 0 || (w << 6) <= min_bit) return SIZE_MAX;
        word = words[--w];
    }
}

} // namespace

bool Heap::is_valid_object(uintptr_t a) const noexcept {
    HeapState& s = *s_;
    if ((a & (kGranuleBytes - 1)) != 0 || !contains(a)) return false;
    if (is_young(a)) {
        if (s.in_eden(a)) {
            if (a < s.eden_lo + kHeaderBytes || a >= alloc_.top) return false;
            if (a >= s.eden_synced) s.sync_eden_starts();
        } else {
            const int w = s.in_survivor(0, a) ? 0 : 1;
            if (a < s.survivor_lo[w] + kHeaderBytes || a >= s.survivor_top[w]) return false;
        }
        return s.young_start(a) && !header_of(a)->forwarded();
    }
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size() || !s.blocks[index]->in_use) return false;
        const size_t granule = (a - s.block_base(index)) / kGranuleBytes;
        return ((s.blocks[index]->starts[granule >> 6] >> (granule & 63)) & 1) != 0;
    }
    if (s.in_large(a)) {
        const uint32_t head = s.large_head_of(a);
        return head != UINT32_MAX && a == s.large_lo + static_cast<uintptr_t>(head) * kPageBytes + kHeaderBytes;
    }
    return false;
}

uintptr_t Heap::find_object(uintptr_t a) const noexcept {
    return detail::containing_object(*s_, a, false);
}

namespace detail {

uintptr_t containing_object(HeapState& s, uintptr_t a, bool accept_forwarded) noexcept {
    const Heap& heap = s.heap;
    const Heap::AllocationBuffer& alloc_ = *const_cast<Heap&>(heap).allocation_buffer();
    if (!heap.contains(a)) return 0;
    auto is_young = [&heap](uintptr_t x) { return heap.is_young(x); };
    auto inside = [a](uintptr_t payload) -> uintptr_t {
        return (a + kHeaderBytes >= payload && a < payload + header_of(payload)->size) ? payload : 0;
    };
    if (is_young(a)) {
        uintptr_t lo, top;
        if (s.in_eden(a)) {
            lo = s.eden_lo;
            top = alloc_.top;
            if (a < top && a + kHeaderBytes > s.eden_synced) s.sync_eden_starts();
        } else {
            const int w = s.in_survivor(0, a) ? 0 : 1;
            lo = s.survivor_lo[w];
            top = s.survivor_top[w];
        }
        if (a >= top) return 0;
        const uintptr_t probe = std::min(a + kHeaderBytes, top - kGranuleBytes);
        const size_t bit = find_prev_bit(s.young_starts.data(), s.young_bit(probe), s.young_bit(lo + kHeaderBytes));
        if (bit == SIZE_MAX) return 0;
        const uintptr_t payload = s.eden_lo + bit * kGranuleBytes;
        if (header_of(payload)->forwarded() && !accept_forwarded) return 0;
        return inside(payload);
    }
    if (s.in_mature(a)) {
        const uint32_t index = s.block_index(a);
        if (index >= s.blocks.size() || !s.blocks[index]->in_use) return 0;
        const uintptr_t lo = s.block_base(index);
        const uintptr_t probe = std::min(a + kHeaderBytes, lo + kBlockBytes - kGranuleBytes);
        const size_t bit = find_prev_bit(s.blocks[index]->starts, (probe - lo) / kGranuleBytes, 1);
        if (bit == SIZE_MAX) return 0;
        return inside(lo + bit * kGranuleBytes);
    }
    if (s.in_large(a)) {
        const uint32_t head = s.large_head_of(a);
        if (head == UINT32_MAX) return 0;
        return inside(s.large_lo + static_cast<uintptr_t>(head) * kPageBytes + kHeaderBytes);
    }
    return 0;
}

} // namespace detail

void Heap::remember_interior(uintptr_t address) noexcept {
    const uintptr_t object = find_object(address);
    if (object != 0) cards_[(object - old_lo_) >> kCardShift] = kCardDirty;
}

void Heap::check_access(uintptr_t base, int64_t offset, size_t size, const char* what) const {
    if (!contains(base)) return;
    const uintptr_t object = find_object(base);
    if (object == 0 || base < object) {
        throw std::runtime_error(std::string("Memory Error: Invalid GC object access in ") + what);
    }
    const int64_t rel = static_cast<int64_t>(base - object) + offset;
    if (rel < 0 || static_cast<uint64_t>(rel) + size > header_of(object)->size) {
        throw std::runtime_error("Memory Error: GC object access out of bounds");
    }
}

bool Heap::is_movable(uintptr_t object) const noexcept { return is_young(object); }

uintptr_t Heap::survivor_of(uintptr_t object) const noexcept {
    HeapState& s = *s_;
    if (!s.collecting) return object;
    GcState g(s, const_cast<Heap&>(*this), s.collecting_kind);
    g.from = s.from_survivor;
    g.to = 1 - g.from;
    g.eden_top = s.gc_eden_top;
    g.from_top = s.gc_from_top;
    return detail::live_address(g, object);
}

void Heap::for_each_object(const std::function<void(uintptr_t)>& fn) const {
    HeapState& s = *s_;
    s.sync_eden_starts();
    auto walk_bits = [&](const uint64_t* words, size_t first_bit, size_t end_bit, uintptr_t base) {
        for (size_t w = first_bit >> 6; w < (end_bit + 63) >> 6; ++w) {
            uint64_t word = words[w];
            while (word) {
                const size_t bit = (w << 6) + detail::ctz64(word);
                word &= word - 1;
                if (bit < first_bit || bit >= end_bit) continue;
                fn(base + bit * kGranuleBytes);
            }
        }
    };
    walk_bits(s.young_starts.data(), s.young_bit(s.eden_lo), s.young_bit(alloc_.top), s.eden_lo);
    for (int w = 0; w < 2; ++w) {
        walk_bits(s.young_starts.data(), s.young_bit(s.survivor_lo[w]), s.young_bit(s.survivor_top[w]), s.eden_lo);
    }
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        const detail::BlockMeta& meta = *s.blocks[i];
        if (!meta.in_use) continue;
        walk_bits(meta.starts, 0, kGranulesPerBlock, s.block_base(static_cast<uint32_t>(i)));
    }
    std::vector<uint32_t> heads;
    heads.reserve(s.large_objects.size());
    for (const auto& entry : s.large_objects) heads.push_back(entry.first);
    for (uint32_t head : heads) fn(s.large_lo + static_cast<uintptr_t>(head) * kPageBytes + kHeaderBytes);
}

void Heap::verify() { detail::verify_heap(*this, "on request"); }

namespace detail {

namespace {

struct VerifyState {
    const char* when;
};

[[noreturn]] void verify_fail(const char* when, const std::string& what) {
    const std::string msg = std::string("heap verification ") + when + ": " + what;
    gc_fatal(msg.c_str());
}

std::string describe(uintptr_t owner) {
    char buf[160];
    if (owner == 0) {
        std::snprintf(buf, sizeof buf, "a root");
    } else {
        const ObjectHeader* h = header_of(owner);
        std::snprintf(buf, sizeof buf, "object %p (layout %u '%s', %u bytes)", reinterpret_cast<void*>(owner),
                      static_cast<unsigned>(h->layout), layout_descriptor(h->layout).name,
                      static_cast<unsigned>(h->size));
    }
    return buf;
}

void verify_visit(Tracer& t, uint64_t* slot, uint64_t word) {
    const auto& v = *static_cast<VerifyState*>(t.state());
    Heap& heap = t.heap();
    if (!heap.is_reference_tag(word)) return;
    const uintptr_t a = static_cast<uintptr_t>(word & kAddressMask);
    if (!heap.is_valid_object(a)) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "slot %p holds 0x%016llx, inside the heap but not an object",
                      static_cast<void*>(slot), static_cast<unsigned long long>(word));
        verify_fail(v.when, describe(t.owner()) + ": " + buf);
    }
    if (t.owner_old() && heap.is_young(a)) {
        const uint8_t card = heap.card_table_base()[(t.owner() - heap.old_base()) >> kCardShift];
        if (card != kCardDirty) {
            char buf[200];
            std::snprintf(buf, sizeof buf,
                          "slot %p names young object %p but the owner's card is clean (a store without "
                          "a write barrier)",
                          static_cast<void*>(slot), reinterpret_cast<void*>(a));
            verify_fail(v.when, describe(t.owner()) + ": " + buf);
        }
    }
}

void check_header(const char* when, uintptr_t object, size_t max_total, bool young) {
    const ObjectHeader* h = header_of(object);
    char buf[200];
    if (h->size == 0 || (h->size % kGranuleBytes) != 0 || kHeaderBytes + h->size > max_total) {
        std::snprintf(buf, sizeof buf, "object %p has a bad size %u", reinterpret_cast<void*>(object),
                      static_cast<unsigned>(h->size));
        verify_fail(when, buf);
    }
    if (h->layout >= layout_count()) {
        std::snprintf(buf, sizeof buf, "object %p has an unregistered layout %u", reinterpret_cast<void*>(object),
                      static_cast<unsigned>(h->layout));
        verify_fail(when, buf);
    }
    if (h->forwarded()) {
        std::snprintf(buf, sizeof buf, "object %p is a forwarded copy still in a live space",
                      reinterpret_cast<void*>(object));
        verify_fail(when, buf);
    }
    if (young && (h->gc_bits & (kGcPinned | kGcLargeMarked))) {
        std::snprintf(buf, sizeof buf, "young object %p carries old-generation bits 0x%02x",
                      reinterpret_cast<void*>(object), static_cast<unsigned>(h->gc_bits));
        verify_fail(when, buf);
    }
}

} // namespace

void verify_heap(Heap& heap, const char* when) {
    HeapState& s = Collector::state(heap);
    VerifyState v{when};
    Tracer t(heap, Tracer::Purpose::Verify, s.base, s.base + s.reserve_bytes, &verify_visit, &weak_visit,
             &ephemeron_visit, &v);
    const size_t max_young = Collector::max_young_total(heap);
    heap.for_each_object([&](uintptr_t object) {
        if (heap.is_young(object)) {
            check_header(when, object, max_young, true);
            scan_object(t, object, false);
        } else if (s.in_mature(object)) {
            check_header(when, object, kMaxMediumObjectBytes, false);
            const uintptr_t block_end = s.block_base(s.block_index(object)) + kBlockBytes;
            if (object + header_of(object)->size > block_end) {
                verify_fail(when, describe(object) + " runs past the end of its block");
            }
            scan_object(t, object, true);
        } else {
            check_header(when, object, SIZE_MAX, false);
            scan_object(t, object, true);
        }
    });
    visit_roots(s, t);
}

} // namespace detail

} // namespace brass::gc
