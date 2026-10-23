#pragma once

#include <brass/mir/opcodes.hpp>
#include <cstdint>

namespace brass {

// Proof certificate explaining why a GC write_barrier was safely elided on a store.
enum class BarrierCertKind : uint8_t {
    None = 0,
    YoungGenBumpAlloc,    // Target object was allocated in nursery in the same block with no intervening GC point
    StackOrNonGC,         // Target address is a stack alloca or non-GC memory
    NonRefStore,          // Stored value is proven not to carry a heap reference
    RedundantSameEpoch,   // Identical write_barrier(obj, val) already executed in this block since last GC point
    ConstantOrNullRef,    // Stored value is a constant 0 / null reference

    // Descriptive aliases used in verification specs
    FreshAllocSinceSafepoint = YoungGenBumpAlloc,
    NonHeapValue = NonRefStore
};

inline constexpr const char* barrier_cert_kind_name(BarrierCertKind kind) noexcept {
    switch (kind) {
        case BarrierCertKind::None:               return "None";
        case BarrierCertKind::YoungGenBumpAlloc:  return "YoungGenBumpAlloc";
        case BarrierCertKind::StackOrNonGC:       return "StackOrNonGC";
        case BarrierCertKind::NonRefStore:        return "NonRefStore";
        case BarrierCertKind::RedundantSameEpoch: return "RedundantSameEpoch";
        case BarrierCertKind::ConstantOrNullRef:  return "ConstantOrNullRef";
    }
    return "Unknown";
}

// Proof certificate recorded when an optimization pass folds or eliminates a guard.
struct GuardElimCert {
    enum class Reason : uint8_t {
        DominatedSameCond = 0, // Guard condition is implied by a dominating guard or branch edge
        ImpliedRelational,     // Guard condition is implied by relational/scaled dominating facts
        RangeProven,           // Value range analysis proved the guard condition is always non-zero
        NonNegativeMask,       // Bitwise AND mask guarantees non-negative value within upper bound
        ConstantFolded,        // Guard condition is a literal non-zero integer constant
        SmtProven,             // Symbolic bitvector solver proved guard condition is unreachable when 0

        // Descriptive aliases
        DominatedByGuard = DominatedSameCond,
        ConstantRange = RangeProven
    };

    uint32_t eliminated_resume_id = 0;
    uint32_t block_id = 0;
    Reason reason = Reason::DominatedSameCond;
    uint32_t witness_inst_or_val_id = 0;
    Opcode check_op = Opcode::unreachable;
    int64_t bound_min = 0;
    int64_t bound_max = 0;
};

} // namespace brass
