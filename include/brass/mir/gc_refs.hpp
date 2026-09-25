#pragma once

#include <brass/mir/instruction.hpp>

// Rules for references into the moving GC heap, shared by the verifier and
// by every pass that moves or merges instructions.
//
// A *derived* gcref is a gcref computed by pointer arithmetic (`add`/`sub`
// of a gcref and an integer), or a `select` that may yield one. It points
// into the middle of an object, and nothing records which object: stack
// maps list gcref values as object roots, so a derived gcref that is live
// when the GC runs is treated as an object header and corrupts the heap.
// Hence a derived gcref is a short-lived address computation:
//   - it is used only in the block that defines it;
//   - no GC point lies between its definition and any of its uses;
//   - it is never a branch argument, deopt state value, return value,
//     stored value, or an operand of a GC point.
// Passes rematerialize the computation next to its use instead of hoisting,
// sinking, merging or forwarding it. See docs/gc_contract.md.
//
// `bitcast_i64_tagged` reads the bits of a tagged value, which for a
// reference name where its object is now. No pass hoists it, merges it with
// another, or carries it across a GC point in place of a fresh one: it is
// absent from every pure-instruction list, so LICM, CSE, GVN and PRE leave
// it where the frontend put it.
namespace brass {

// True for a value whose computation matches the definition above.
bool is_derived_gcref(const Value* val) noexcept;

// True for an instruction during which the GC may run: calls (other than
// runtime helpers known not to allocate), safepoints and coroutine ops.
bool may_trigger_gc(const Instruction& inst) noexcept;

} // namespace brass
