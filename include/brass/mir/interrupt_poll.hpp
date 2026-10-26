#pragma once

// Interrupt polls for compiled code (Module::interrupt_poll).
//
// Tier 2, OSR and AOT code polls on every back edge of its CFG: the edge's
// source compares the stack pointer with the word at [pinned TLS +
// offset] and, when it is below, goes through a cold block that calls the
// module's handler before the loop header runs again. The polls go in
// after the optimizer, into the copy of the function that is lowered, so no
// loop transform sees a call in the loop; register allocation splits the
// loop's values around the cold call rather than keep them out of the
// caller-saved registers on the hot path.
//
// Tier 0 and Tier 1 poll in their own backedge code (fast_interpreter.cpp,
// the baseline emitters) from the same attribute.

#include <brass/mir/function.hpp>

#include <cstddef>

namespace brass {

// True when `fn`'s module has an interrupt poll and `fn` has a back edge.
bool needs_interrupt_polls(const Function& fn);

// Inserts the polls into `fn` (whose module must have an interrupt poll):
// one cold handler block per loop header. Returns the number of back edges
// polled. Predecessor lists are rebuilt.
size_t insert_interrupt_polls(Function& fn);

} // namespace brass
