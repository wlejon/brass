// What the thread's stacks name stays where it is.
//
// Host code (an embedder's C++, a runtime's natives) holds references to heap
// objects in plain locals: on its stack, or in registers the compiler keeps
// them in. No stack map describes those frames, so a collection could not
// update such a reference when it moved the object, and the reference would
// name the dead copy afterwards. So every collection that moves young objects
// first reads every word of the thread's stacks, from its own frame (the
// callee-saved registers spilled where the scan sees them) to each stack's
// base, and of each switched-out fiber's stack from where it switched away.
// A word that, untagged or under one of the heap's reference tags, falls in
// a young object's payload (its start or anywhere inside; not its header,
// where the bounds of the spaces and of the object before it fall) names that
// object, and the object is pinned: the evacuation leaves it where it is,
// visits its slots as a root's (so what it names survives, and is updated),
// and keeps it young; it is a resident from then on (kGcResident), copied
// like any other young object by the first evacuation that does not find it
// named again (heap_collect.cpp, reclaim_young; GcState::in_from). A marking
// cycle's finish that leaves the young generation in place marks it, as a
// root (HeapState::stack_objects through visit_roots). Old objects never
// move, so a word naming one is passed over.
//
// The scan is conservative: a word that merely looks like such a reference
// (a number, a stale slot of a dead frame) keeps its object for that
// collection too. It never makes anything unsound, only something live a
// little longer.

#include "heap_internal.hpp"

#include <brass/core/env.hpp>

#include <algorithm>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define BRASS_PIN_NOINLINE __declspec(noinline)
#else
#define BRASS_PIN_NOINLINE __attribute__((noinline))
#endif

namespace brass::gc::detail {

bool stack_pinning() noexcept {
    static const bool on = [] {
        const std::string v = env_value("BRASS_GC_PIN_STACK");
        return !v.empty() && v != "0";
    }();
    return on;
}

namespace {

// More than any stack brass or a host makes: a span past it is a bound gone
// wrong, not a stack, and is not read.
constexpr uintptr_t kMaxStackSpan = uintptr_t{1} << 30;

struct StackScan {
    const Heap* heap;
    std::vector<uintptr_t>* candidates;
};

// Every word of [lo, hi) that names an address in the young generation,
// untagged or under a reference tag. Old objects never move, so the stacks'
// names for them need nothing, and keep nothing alive: a host roots what it
// keeps.
void scan_words(void* context, uintptr_t lo, uintptr_t hi) {
    auto& scan = *static_cast<StackScan*>(context);
    lo = (lo + sizeof(uint64_t) - 1) & ~(uintptr_t{sizeof(uint64_t)} - 1);
    if (hi <= lo || hi - lo > kMaxStackSpan) return;
    const Heap& heap = *scan.heap;
    for (uintptr_t p = lo; p + sizeof(uint64_t) <= hi; p += sizeof(uint64_t)) {
        const uint64_t word = *reinterpret_cast<const uint64_t*>(p);
        if ((word >> kTagShift) != 0 && !heap.is_reference_tag(word)) continue;
        const auto a = static_cast<uintptr_t>(word & kAddressMask);
        if (heap.is_young(a)) scan.candidates->push_back(a);
    }
}

// The stacks from this frame up: below its caller's, whose spilled registers
// it therefore reads.
BRASS_PIN_NOINLINE void scan_from_here(StackScan& scan) {
    volatile uintptr_t marker = 0;
    for_each_thread_stack(reinterpret_cast<uintptr_t>(&marker), &scan_words, &scan);
    marker = 1;
}

// Puts the callee-saved registers in this frame, where scan_from_here reads
// them: a reference a caller keeps only in a register is seen too.
BRASS_PIN_NOINLINE void scan_thread_stacks(StackScan& scan) {
#if defined(_WIN32)
    CONTEXT registers;
    RtlCaptureContext(&registers);
    scan_from_here(scan);
    // The context lives until the scan has read it.
    const void* volatile keep = &registers;
    (void)keep;
#elif defined(__x86_64__)
    uintptr_t registers[6];
    __asm__ __volatile__(
        "movq %%rbx, 0(%0)\n\t"
        "movq %%rbp, 8(%0)\n\t"
        "movq %%r12, 16(%0)\n\t"
        "movq %%r13, 24(%0)\n\t"
        "movq %%r14, 32(%0)\n\t"
        "movq %%r15, 40(%0)\n\t"
        :
        : "r"(registers)
        : "memory");
    scan_from_here(scan);
    __asm__ __volatile__("" : : "r"(registers) : "memory");
#elif defined(__aarch64__)
    uintptr_t registers[12];
    __asm__ __volatile__(
        "stp x19, x20, [%0, #0]\n\t"
        "stp x21, x22, [%0, #16]\n\t"
        "stp x23, x24, [%0, #32]\n\t"
        "stp x25, x26, [%0, #48]\n\t"
        "stp x27, x28, [%0, #64]\n\t"
        "stp x29, x30, [%0, #80]\n\t"
        :
        : "r"(registers)
        : "memory");
    scan_from_here(scan);
    __asm__ __volatile__("" : : "r"(registers) : "memory");
#else
#error "brass gc: no register capture for the stack scan on this architecture"
#endif
}

} // namespace

void gather_stack_objects(HeapState& s) {
    s.stack_objects.clear();
    if (!stack_pinning()) return;
    std::vector<uintptr_t> candidates;
    candidates.reserve(256);
    StackScan scan{&s.heap, &candidates};
    scan_thread_stacks(scan);
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    // Each candidate to the object whose payload holds it
    // (young_object_containing). An address in an object's header is no
    // reference to it (a reference is the payload's address, or inside it)
    // but the bound of a space or of the object before it, which the
    // collector's own frames hold: eden's base would pin eden's first object.
    std::vector<uintptr_t>& objects = s.stack_objects;
    for (const uintptr_t a : candidates) {
        const uintptr_t object = young_object_containing(s, a);
        if (object == 0) continue;
        if (objects.empty() || objects.back() != object) objects.push_back(object);
    }
    // Interior addresses of one object come in order, so adjacent duplicates
    // are all there are; sorted for the pinned list's sake.
    std::sort(objects.begin(), objects.end());
    objects.erase(std::unique(objects.begin(), objects.end()), objects.end());
}

void pin_stack_objects(GcState& g) {
    HeapState& s = g.s;
    if (!s.pinned.empty()) gc_fatal("an evacuation began with objects still pinned by the last");
    for (const uintptr_t object : s.stack_objects) {
        if (!g.heap.is_young(object) || !g.in_from(object)) continue;
        ObjectHeader* h = header_of(object);
        if (h->gc_bits & (kGcPinned | kGcForwarded)) continue;
        h->gc_bits = static_cast<uint8_t>(h->gc_bits | kGcPinned);
        s.pinned.push_back(object);
    }
    g.pinned_objects = s.pinned.size();
}

void scan_pinned(HeapState& s, Tracer& tracer) {
    for (const uintptr_t object : s.pinned) scan_object(tracer, object, false);
}

} // namespace brass::gc::detail
