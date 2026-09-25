#pragma once

// The object model of brass's heap (gc::Heap, heap.hpp): the header every
// object carries, the address arithmetic every part of the collector shares,
// and the process-wide registry of object layouts that says how an object's
// references are found. docs/gc_contract.md is the full contract.

#include <cstddef>
#include <cstdint>

namespace brass::gc {

class Tracer;

// Objects are aligned to, and sized in, 8-byte granules.
inline constexpr size_t kGranuleBytes = 8;

// A reference is the address of an object's PAYLOAD. The 8-byte header sits
// immediately before it.
inline constexpr size_t kHeaderBytes = 8;

// A word slot holds a reference in its low 48 bits; the high 16 bits are a
// tag the heap preserves when it updates the slot (zero for a raw gcref).
inline constexpr uint64_t kAddressMask = 0x0000FFFFFFFFFFFFULL;
inline constexpr unsigned kTagShift = 48;

// Mature space: 32 KiB blocks of 256-byte lines (mark-region).
inline constexpr size_t kBlockBytes = 32 * 1024;
inline constexpr size_t kLineBytes = 256;
inline constexpr size_t kLinesPerBlock = kBlockBytes / kLineBytes;
inline constexpr size_t kGranulesPerBlock = kBlockBytes / kGranuleBytes;
// The largest object (header included) a mature block holds; anything larger
// that becomes old lives in the large-object space.
inline constexpr size_t kMaxMediumObjectBytes = 8 * 1024;

// Large-object space granularity.
inline constexpr size_t kPageBytes = 4096;

// Cards: one byte per 512 bytes of the old generation (mature blocks and
// large objects). A store into a mature object dirties the card holding the
// object's payload address. A large object's first card dirty means the
// whole object; a store whose slot address is known dirties the slot's own
// card instead (kCardDirtyRange when that is the first card), so only the
// 512 bytes around it are rescanned.
inline constexpr unsigned kCardShift = 9;
inline constexpr size_t kCardBytes = size_t{1} << kCardShift;
inline constexpr uint8_t kCardDirty = 0x00;
inline constexpr uint8_t kCardClean = 0x01;
inline constexpr uint8_t kCardDirtyRange = 0x02;

// Header bits the collector owns (ObjectHeader::gc_bits).
inline constexpr uint8_t kGcForwarded = 0x01;  // young copy left behind; payload word 0 = new address
inline constexpr uint8_t kGcPinned = 0x02;     // never moves (allocated old)
inline constexpr uint8_t kGcLargeMarked = 0x04;// large object reached by the running full collection
inline constexpr unsigned kGcAgeShift = 4;     // survived minor collections (0..7)
inline constexpr uint8_t kGcAgeMask = 0x70;

using LayoutId = uint16_t;

struct ObjectHeader {
    uint32_t size;       // payload bytes, a multiple of 8, at least 8
    LayoutId layout;     // registered layout (layout_descriptor)
    uint8_t gc_bits;     // collector-owned (kGc*)
    uint8_t host_bits;   // free for the host; the collector copies and never reads them

    [[nodiscard]] uint8_t age() const noexcept { return static_cast<uint8_t>((gc_bits & kGcAgeMask) >> kGcAgeShift); }
    [[nodiscard]] bool forwarded() const noexcept { return (gc_bits & kGcForwarded) != 0; }
    [[nodiscard]] uintptr_t payload() const noexcept {
        return reinterpret_cast<uintptr_t>(this) + kHeaderBytes;
    }
};
static_assert(sizeof(ObjectHeader) == kHeaderBytes, "the object header is one word");

[[nodiscard]] inline ObjectHeader* header_of(uintptr_t payload) noexcept {
    return reinterpret_cast<ObjectHeader*>(payload - kHeaderBytes);
}

[[nodiscard]] constexpr size_t align_granule(size_t bytes) noexcept {
    return (bytes + (kGranuleBytes - 1)) & ~(kGranuleBytes - 1);
}

// Payload bytes an allocation of `requested` bytes gets (never zero: a
// forwarded young object keeps its new address in payload word 0).
[[nodiscard]] constexpr size_t payload_bytes_for(size_t requested) noexcept {
    return requested == 0 ? kGranuleBytes : align_granule(requested);
}

// How the collector finds an object's references.
enum class LayoutKind : uint8_t {
    Leaf,    // none (strings, raw bytes)
    Mask,    // payload word i is a word slot iff bit i of `mask` (bit 63: words 63 and up)
    Words,   // every payload word is a word slot
    Custom,  // the host's `trace` visits them
};

// Traces one object of a Custom layout: visits every reference it holds
// through `tracer` (tracer.hpp). Called during a collection; it must not
// allocate, must not read another object's payload except through the
// values the tracer writes back, and must visit the same slots each time for
// the same object state.
using TraceFn = void (*)(uintptr_t payload, size_t payload_bytes, Tracer& tracer);

// Visits only the references stored in payload bytes [begin, end) of one
// object of a Custom layout (a slot counts when its first byte lies in the
// range), under the same rules as a TraceFn. A minor collection uses it to
// rescan just the dirty cards of a large old object, and a parallel full
// collection to split a large object's scan between threads. Visiting more
// than the range is allowed, only slower. In a full collection
// (Tracer::Purpose::Full) the call whose range begins at offset 0 must also
// visit whatever the full trace visits outside the payload, since the
// object's ranges then stand in for its trace.
using TraceRangeFn = void (*)(uintptr_t payload, size_t payload_bytes, size_t begin, size_t end,
                              Tracer& tracer);

struct LayoutDescriptor {
    LayoutKind kind = LayoutKind::Leaf;
    uint64_t mask = 0;        // Mask
    uint32_t type_tag = 0;    // the host's label for objects of this layout
    TraceFn trace = nullptr;  // Custom
    // Custom, optional: without it a dirty card of a large object rescans
    // the whole object.
    TraceRangeFn trace_range = nullptr;
    const char* name = "";    // diagnostics (heap verification names it)
};

// The layout registry is process-wide and append-only: layouts are shared by
// every heap on every thread, and a LayoutId never changes meaning.
// register_layout throws std::length_error once 65535 layouts exist.
LayoutId register_layout(const LayoutDescriptor& descriptor);

// The Mask layout for (mask, type_tag), registered on first use and shared
// thereafter: what brass_gc_alloc(size, mask, tag) allocates with.
LayoutId mask_layout(uint64_t mask, uint32_t type_tag);

// The layout `id` names. `id` must have been returned by register_layout or
// mask_layout (or be kLeafLayout).
const LayoutDescriptor& layout_descriptor(LayoutId id) noexcept;

// How many layouts are registered (ids are 0 .. count-1).
size_t layout_count() noexcept;

// Layout 0: no references, type tag 0.
inline constexpr LayoutId kLeafLayout = 0;

} // namespace brass::gc
