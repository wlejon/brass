#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// Moves a pure computation out of a block whose branch decides whether it is
// needed: an instruction every use of which lies down one outgoing edge is
// moved to the head of that edge's block, so the other edges stop paying for
// it. Uses down several edges each get a copy at theirs when that spares an
// edge or lets a copy move on (one edge is taken, so one copy runs where one
// ran, and each is weighed again at its block's own branch). A front end that leaves a fast path through many exits (a
// guard's miss takes the values the slow path wants, boxed) computes each
// exit's values before its test; after this pass they are computed on the
// exit.
//
// Moved: arithmetic, bitwise, comparison, conversion and select instructions
// of non-GC types (gc_refs.hpp: a derived gcref and `bitcast_i64_tagged` stay
// where the front end put them), integer division excepted (it traps where it
// stands), and `bitcast_tagged_i64` when no GC point follows it in its block
// (past one, the bits it would box are no longer what the collector kept
// current). Loads stay: nothing here asks what is stored in between. Any
// other value is the same function of its operands wherever it is computed,
// so moving it past a GC point changes only which of the two is live across
// it.
//
// An edge's block must be entered from nowhere else, which it is once
// critical edges are split; the pass splits them first. Returns how many
// instructions it placed.
size_t sink_code(Function& fn);

} // namespace brass
