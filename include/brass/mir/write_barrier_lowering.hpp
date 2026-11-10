#pragma once

// The write barrier's inline young-target filter (Module::young_range).
//
// brass's barrier (gc_contract.md 2.4) does nothing for a target object in
// the young generation, and those are most of the barriers compiled code
// runs: a fresh object's fields, a record written every iteration. When the
// module names where the pinned TLS block holds the thread's young range,
// each `write_barrier %obj, %val` of a function's lowered copy becomes
//
//     %a = and %obj, 0x0000FFFFFFFFFFFF          ; the address under any tag
//     br_if (%a - [tls + base]) <u [tls + span], cont, cold
//   cold:                                          ; set_cold
//     call @brass_gc_write_barrier(%obj, %val)
//     br cont
//
// The range is the heap's young reservation, fixed for its lifetime, so the
// test is sound at any point, inside a marking cycle too: it never skips a
// store the barrier would act on. (The value half of the barrier's filter,
// barrier_value_base/span, moves when a marking cycle opens and stays out of
// line.)
//
// Like the interrupt polls this runs after the optimizer, on the copy of the
// function that is lowered (function_codegen), so every pass before it sees
// one `write_barrier` instruction: write_barrier_elim, alias analysis (which
// takes it for no memory access) and the rest. The baseline emitters test
// the same range inline from the same attribute; the interpreters call the
// barrier directly.

#include <brass/mir/function.hpp>

#include <cstddef>

namespace brass {

// True when `fn`'s module has a young range and a pinned TLS register and
// `fn` holds a write_barrier whose target is an i64 (a raw or tagged
// address). A gcref or pointer target keeps the plain call.
bool needs_write_barrier_expansion(const Function& fn);

// Expands those barriers as above. Returns how many it expanded.
// Predecessor lists are rebuilt.
size_t expand_write_barriers(Function& fn);

} // namespace brass
