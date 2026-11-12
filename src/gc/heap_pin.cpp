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
// move, but a collection that marks the old generation (a full collection, a
// marking cycle's start and its finish's rescan) reads the stacks for them
// too, and a word naming an old object's payload marks it as a root: a raw
// reference in a local keeps whatever it names alive, old or young. A minor
// collection decides nothing old, so it passes over them.
//
// The scan is conservative: a word that merely looks like such a reference
// (a number, a stale slot of a dead frame) keeps its object for that
// collection too. It never makes anything unsound, only something live a
// little longer. The collector keeps its own words out of what it reads: it
// clears the stack below the collection's frame before the scan, and of the
// registers it reads only the callee-saved integer ones, where a caller's
// references can be; the rest are the collector's scratch.

#include "heap_internal.hpp"

#include <algorithm>
#include <cstring>

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

namespace {

// More than any stack brass or a host makes: a span past it is a bound gone
// wrong, not a stack, and is not read.
constexpr uintptr_t kMaxStackSpan = uintptr_t{1} << 30;

struct StackScan {
    const Heap* heap;
    std::vector<uintptr_t>* candidates;
    bool old;  // old addresses too (StackObjects::WithOld)
};

// Every word of [lo, hi) that names an address in the young generation (and,
// when the scan asks for them, the old), untagged or under a reference tag.
void scan_words(void* context, uintptr_t lo, uintptr_t hi) {
    auto& scan = *static_cast<StackScan*>(context);
    lo = (lo + sizeof(uint64_t) - 1) & ~(uintptr_t{sizeof(uint64_t)} - 1);
    if (hi <= lo || hi - lo > kMaxStackSpan) return;
    const Heap& heap = *scan.heap;
    const bool old = scan.old;
    for (uintptr_t p = lo; p + sizeof(uint64_t) <= hi; p += sizeof(uint64_t)) {
        const uint64_t word = *reinterpret_cast<const uint64_t*>(p);
        if ((word >> kTagShift) != 0 && !heap.is_reference_tag(word)) continue;
        const auto a = static_cast<uintptr_t>(word & kAddressMask);
        if (old ? heap.contains(a) : heap.is_young(a)) scan.candidates->push_back(a);
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
#if defined(_WIN32) && defined(_M_X64)
    // Only the callee-saved registers (integer plus XMM6..XMM15 on Win64,
    // where MSVC may hold 64-bit/128-bit aggregates across calls): whatever
    // else the context holds is the collector's own scratch, and a stale
    // reference in it would keep its object.
    uintptr_t registers[28];
    {
        CONTEXT context;
        RtlCaptureContext(&context);
        registers[0] = context.Rbx;
        registers[1] = context.Rbp;
        registers[2] = context.Rsi;
        registers[3] = context.Rdi;
        registers[4] = context.R12;
        registers[5] = context.R13;
        registers[6] = context.R14;
        registers[7] = context.R15;
        std::memcpy(&registers[8], &context.Xmm6, 10 * sizeof(M128A));
        SecureZeroMemory(&context, sizeof(context));
    }
    scan_from_here(scan);
    // The registers live until the scan has read them.
    const void* volatile keep = registers;
    (void)keep;
#elif defined(_WIN32) && defined(_M_ARM64)
    uintptr_t registers[12];
    {
        CONTEXT context;
        RtlCaptureContext(&context);
        for (int i = 0; i < 10; ++i) registers[i] = context.X[19 + i];
        registers[10] = context.Fp;
        registers[11] = context.Lr;
        SecureZeroMemory(&context, sizeof(context));
    }
    scan_from_here(scan);
    const void* volatile keep = registers;
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

// The collection's own work before the scan (a lazy sweep finished, a
// verification) walked objects in frames as deep as the scan's, and the
// scan's frames would read what those left in their slots the scan never
// writes (the register context's unused parts, padding): the collector would
// keep what it had just looked at. Cleared first, from below this frame's
// caller.
constexpr size_t kScanFrameWords = 1024;

BRASS_PIN_NOINLINE void clear_scan_frames() {
    volatile uint64_t words[kScanFrameWords];
    for (size_t i = 0; i < kScanFrameWords; ++i) words[i] = 0;
    (void)words[0];
}

} // namespace

void gather_stack_objects(HeapState& s, StackObjects which) {
    s.stack_objects.clear();
    std::vector<uintptr_t> candidates;
    candidates.reserve(256);
    StackScan scan{&s.heap, &candidates, which == StackObjects::WithOld};
    clear_scan_frames();
    scan_thread_stacks(scan);
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    // Each candidate to the object whose payload holds it
    // (young_object_containing, containing_object for the old spaces). An
    // address in an object's header is no reference to it (a reference is
    // the payload's address, or inside it) but the bound of a space or of the
    // object before it, which the collector's own frames hold: eden's base
    // would pin eden's first object.
    std::vector<uintptr_t>& objects = s.stack_objects;
    for (const uintptr_t a : candidates) {
        uintptr_t object = 0;
        if (s.heap.is_young(a)) {
            object = young_object_containing(s, a);
        } else {
            object = containing_object(s, a, false);
            if (object > a) object = 0;
        }
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
