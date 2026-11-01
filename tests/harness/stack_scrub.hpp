#pragma once

// For tests that assert an object dies, moves, ages or is promoted at a given
// collection. Every collection reads the collecting thread's stack
// conservatively and keeps (young objects: in place) whatever a word of it
// names, stale words of dead frames included (docs/gc_contract.md 2.5). Such
// a test makes the object in a function of its own that is never inlined
// (BRASS_TEST_NOINLINE), keeps it only where no stack word names it (a heap
// root, a slot of another object, or its address masked), reads it back
// through such helpers, and calls scrub_stack() before the collection, so the
// dead frames' words are gone.

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#define BRASS_TEST_NOINLINE __declspec(noinline)
#define BRASS_TEST_FORCEINLINE __forceinline
#else
#define BRASS_TEST_NOINLINE __attribute__((noinline))
#define BRASS_TEST_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace brass::test {

// Overwrites the stack below the caller's frame, where the frames it called
// left their words.
BRASS_TEST_NOINLINE inline void scrub_stack() {
    volatile uint64_t words[16384];
    for (size_t i = 0; i < 16384; ++i) words[i] = 0;
    (void)words[0];
}

// An address kept in a local without naming its object: every bit of it is
// flipped against a pattern, so its low 48 bits name no address of the heap
// (a heap whose reference tags are all tags reads any high bits).
inline constexpr uint64_t kAddressMaskOut = 0x5A5A5A5A5A5A5A5AULL;
inline constexpr uint64_t hide_address(uint64_t a) noexcept { return a ^ kAddressMaskOut; }
// Not inline: the compiler would be free to compute the unhidden address
// early and keep it in a callee-saved register across a collection.
BRASS_TEST_NOINLINE inline uint64_t unhide_address(uint64_t h) noexcept { return h ^ kAddressMaskOut; }

// Stores a hidden address into `slot` unhidden, in a frame of its own (the
// unhidden word never lives in the caller's).
BRASS_TEST_NOINLINE inline void set_hidden(uint64_t* slot, uint64_t hidden) { *slot = unhide_address(hidden); }

// A root slot no stack word holds (a std::unique_ptr's word, on the C++
// heap), registered with the heap for the slot's life.
template <class HeapT>
struct HeapRoot {
    HeapT& heap;
    uint64_t* slot;
    explicit HeapRoot(HeapT& h, uint64_t value = 0) : heap(h), slot(new uint64_t(value)) { heap.add_root(slot); }
    ~HeapRoot() {
        heap.remove_root(slot);
        delete slot;
    }
    HeapRoot(const HeapRoot&) = delete;
    HeapRoot& operator=(const HeapRoot&) = delete;
    // The slot's word hidden (hide_address), for comparing addresses across
    // a collection without naming the object.
    BRASS_TEST_NOINLINE uint64_t hidden() const { return hide_address(*slot); }
};

// Runs `f` in a frame below the caller's, never inlined into it: whatever
// `f` reads of the heap stays in that frame, which scrub_stack() then
// overwrites. A test that wants an object to move or die touches it only
// inside such frames, and calls scrub_stack() right before collecting. A
// frame the caller calls next sits where this one was, and the slots it never
// writes keep this one's words: scrub_stack() between the two when the next
// one collects.
template <class F>
BRASS_TEST_NOINLINE void in_own_frame(F&& f) {
    f();
}

// The same, returning what `f` returns (a count, a hidden address: nothing
// that names an object).
template <class F>
BRASS_TEST_NOINLINE auto from_own_frame(F&& f) {
    return f();
}

// scrub_stack() and then the collection, so the frames below the caller's
// name nothing. Inlined into the caller: a frame of its own would sit where
// earlier frames were, and the slots it never writes would keep their words.
// A helper that scrubs and then collects is BRASS_TEST_FORCEINLINE for the
// same reason.
template <class HeapT, class Kind>
BRASS_TEST_FORCEINLINE void collect_scrubbed(HeapT& heap, Kind kind) {
    scrub_stack();
    heap.collect(kind);
}

}  // namespace brass::test
