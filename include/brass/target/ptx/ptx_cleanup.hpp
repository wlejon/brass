#pragma once

// PtxCleanup: register-level and branch-level clean-up of a ptx::Function
// between PtxISel and the verifier (PtxTarget runs lower -> cleanup -> verify
// -> print). The ISel output is correct but noisy: block-argument copies,
// lane extract/insert moves, constants materialized in registers that only
// immediates read, `bra` to the very next block. ptxas would remove all of
// it; these passes remove it in the compiler so the printed PTX reads like
// the hand-written kernels.
//
// The IR is "SSA-ish": most registers have exactly one def, block parameters
// and parallel-copy scratch registers have several. Every pass computes
// def/use counts first and only rewrites registers whose def count it can
// reason about. None of the passes changes program semantics, and every pass
// is a no-op on input that is already clean (cleanup() applied twice prints
// the same text). See docs/ptx_backend_design.md ("Stage 6a notes").

#include <brass/target/ptx/ptx_ir.hpp>

#include <cstddef>

namespace brass::ptx {

// 1. Copy propagation / coalescing of plain register moves (same class,
//    unguarded, no modifiers):
//      - `mov d, s` with d and s both single-def: every use of d becomes s
//        and the mov is deleted;
//      - `mov d, s` where s is single-def, single-use, defined unguarded by
//        a scalar instruction earlier in the same block with no def or read
//        of d in between: that instruction writes d directly and the mov is
//        deleted (coalesces `add %t, ...; mov %param, %t` on back-edges).
//    Guarded movs (taken-edge copies of br_if) are never touched.
//    Returns the number of movs removed.
size_t propagate_copies(Function& fn);

// 2. Dead instruction elimination: deletes instructions whose destination
//    registers are never read and that have no side effects. Side effects:
//    st, atom, bar, call, ret, bra, trap, exit, shfl (warp-collective) and a
//    mov from a non-invariant special register (%clock, %warpid, ...).
//    Loads (any state space) with a dead result are deleted. Runs to a
//    fixed point; returns the number of instructions removed.
size_t eliminate_dead_instructions(Function& fn);

// 3. Branch simplification: removes blocks that are unreachable from the
//    first block (label references and fall-through edges both count),
//    drops a `bra L` that ends a block when L is the next block, turns the
//    ending pair `@p bra A; bra B` into `@p bra A` when B is next and into
//    `@!p bra B` when A is next. Fall-through is a valid block end for the
//    verifier and for ptxas. Returns the number of branches and blocks
//    removed.
size_t simplify_branches(Function& fn);

// 4. Register renumbering: renames every register of every class densely in
//    ascending order of the old index and sets reg_counts to the number of
//    registers actually mentioned, so the .reg declarations are tight.
void renumber_registers(Function& fn);

struct CleanupStats {
    size_t copies_removed = 0;
    size_t dead_removed = 0;
    size_t branches_removed = 0;   // bra instructions and unreachable blocks
    size_t insts_before = 0;
    size_t insts_after = 0;
};

// The whole pipeline: (1) and (2) alternate until neither changes anything,
// then (3), then (4).
CleanupStats cleanup(Function& fn);

// Total instruction count over all blocks (labels are not instructions).
size_t instruction_count(const Function& fn);

} // namespace brass::ptx
