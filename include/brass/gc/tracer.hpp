#pragma once

// The one interface through which a collection reaches references: a host's
// Custom layout trace function (object.hpp) and a root source (heap.hpp) are
// handed a Tracer and call it once per slot. docs/gc_contract.md, "Tracing".
//
// A slot is a 64-bit word. Its low 48 bits are a reference (an object's
// payload address) when they fall inside the heap and its high 16 bits are
// one of the heap's reference tags (HeapConfig::reference_tags; by default
// every tag). The collector may rewrite the low 48 bits when it moves the
// object and never changes the high 16. A zero word, and any word whose low
// 48 bits lie outside the heap, is left alone, so a slot that holds a number
// or a pointer to memory the heap does not own may be visited freely --
// except a number whose low 48 bits could fall inside the heap under a
// reference tag: such a slot must not be visited (the host's trace function
// decides from its own value representation).

#include <brass/gc/object.hpp>
#include <cstdint>

namespace brass::gc {

class Heap;

class Tracer {
public:
    enum class Purpose : uint8_t { Minor, Full, Verify };

    // A strong reference: keeps the target alive and is updated if it moves.
    void visit(uint64_t* slot) noexcept {
        const uint64_t word = *slot;
        const uintptr_t address = static_cast<uintptr_t>(word & kAddressMask);
        if (address - range_lo_ < range_span_) visit_slow_(*this, slot, word);
    }
    void visit_ref(uintptr_t* slot) noexcept { visit(reinterpret_cast<uint64_t*>(slot)); }

    // A strong reference that may point inside its object rather than at its
    // start (a derived reference, or a slot a conservative source cannot
    // type): the containing object is kept alive and the slot keeps its
    // offset into it when it moves. A word naming no object is left alone.
    // Slower than visit; for roots only.
    void visit_derived(uint64_t* slot) noexcept;
    // A word that may or may not be a reference (an untyped interpreter
    // register): visited as a strong reference only when it names the start
    // of an object, otherwise left alone.
    void visit_conservative(uint64_t* slot) noexcept;

    // A weak reference: does not keep the target alive. Once the collection
    // has decided liveness, the slot is updated if the target survived and
    // set to `cleared` (the whole word) if it did not.
    void visit_weak(uint64_t* slot, uint64_t cleared) noexcept {
        const uintptr_t address = static_cast<uintptr_t>(*slot & kAddressMask);
        if (address - range_lo_ < range_span_) weak_(*this, slot, cleared);
    }

    // An ephemeron (a WeakMap entry): `*value_slot` is traced only once
    // `*key_slot`'s target is known to be alive by some other path; if the key
    // dies, both slots are set to their cleared words. A key that is not a
    // reference into the heap counts as alive.
    void visit_ephemeron(uint64_t* key_slot, uint64_t* value_slot,
                         uint64_t cleared_key, uint64_t cleared_value) noexcept {
        ephemeron_(*this, key_slot, value_slot, cleared_key, cleared_value);
    }

    [[nodiscard]] Heap& heap() const noexcept { return *heap_; }
    [[nodiscard]] Purpose purpose() const noexcept { return purpose_; }

    // Collector-internal from here on (heap_collect.cpp builds the tracers).
    using VisitFn = void (*)(Tracer&, uint64_t* slot, uint64_t word);
    using WeakFn = void (*)(Tracer&, uint64_t* slot, uint64_t cleared);
    using EphemeronFn = void (*)(Tracer&, uint64_t* key_slot, uint64_t* value_slot,
                                 uint64_t cleared_key, uint64_t cleared_value);

    Tracer(Heap& heap, Purpose purpose, uintptr_t lo, uintptr_t hi, VisitFn visit,
           WeakFn weak, EphemeronFn ephemeron, void* state) noexcept
        : range_lo_(lo), range_span_(hi - lo), visit_slow_(visit), weak_(weak),
          ephemeron_(ephemeron), heap_(&heap), state_(state), purpose_(purpose) {}

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;

    // The object whose slots are being visited (0 for a root), and whether it
    // lives in the old generation: a minor collection re-dirties its card
    // when a slot still names a young object afterwards.
    void set_owner(uintptr_t owner, bool owner_old) noexcept {
        owner_ = owner;
        owner_old_ = owner_old;
    }
    [[nodiscard]] uintptr_t owner() const noexcept { return owner_; }
    [[nodiscard]] bool owner_old() const noexcept { return owner_old_; }
    [[nodiscard]] void* state() const noexcept { return state_; }

private:
    uintptr_t range_lo_;
    uintptr_t range_span_;
    VisitFn visit_slow_;
    WeakFn weak_;
    EphemeronFn ephemeron_;
    Heap* heap_;
    void* state_;
    uintptr_t owner_ = 0;
    bool owner_old_ = false;
    Purpose purpose_;
};

} // namespace brass::gc
