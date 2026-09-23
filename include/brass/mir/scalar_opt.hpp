#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

// Small function-level scalar cleanups: the passes other transforms lean on
// to tidy up after themselves (constant folding, dominator-scoped CSE, dead
// code elimination) and the scalar loop passes (LICM and induction-variable
// strength reduction). Each is a plain function over one function, so the
// pass manager can schedule it like any other pass.
namespace brass {

// Folds integer instructions with constant operands and algebraic identities
// (x + 0, x * 1, x - x, ...), and selects on a constant condition.
bool fold_constants(Function& fn);

// Replaces a pure instruction by an identical one that dominates it.
bool dominator_cse(Function& fn);

// Removes side-effect-free instructions and non-entry block parameters that
// nothing reads, until nothing more goes.
bool eliminate_dead_code(Function& fn);

// Removes block-parameter cycles (an induction variable and its add/sub
// chain) that feed nothing but themselves.
bool eliminate_dead_induction_cycles(Function& fn);

// fold_constants, dominator_cse, then eliminate_dead_code: the cleanup most
// transforms want after they change a function.
bool scalar_cleanup(Function& fn);

// Hoists loop-invariant pure instructions into each loop's preheader
// (creating the preheader when a loop lacks one), innermost loops first.
bool hoist_loop_invariants(Function& fn);

// Induction-variable strength reduction: an indexed access whose index is a
// basic i64 induction variable `i` (or `i + inv`) with scale s > 1 is
// rewritten to a byte-offset induction variable `s*i` with scale 1, and
// `i * s` / `i << log2(s)` become that variable. The loop's exit compare
// moves to the scaled variable only when constant bounds prove the scaled
// values cannot overflow. Accesses whose base is a GC reference are left
// alone: the rebased address would be a derived gcref live across the loop.
bool strength_reduce_induction_variables(Function& fn);

struct LoopCleanupOptions {
    bool licm = true;          // hoist_loop_invariants
    bool fold_and_dce = true;  // fold, CSE, DCE and dead induction cycles
    size_t max_iterations = 8;
};

// Iterates LICM and the scalar cleanups to a fixed point (or the iteration
// limit): the scalar part of the loop pipeline.
bool loop_cleanup(Function& fn, const LoopCleanupOptions& options = {});

} // namespace brass
