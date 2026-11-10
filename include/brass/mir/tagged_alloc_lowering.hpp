#pragma once

// The tagged allocator's inline window (Module::tagged_allocator).
//
// When the module names where the pinned TLS block holds the thread's
// allocation window, each call of its tagged allocator in a function's
// lowered copy becomes
//
//     %tls  = pinned_tls_read
//     %top  = load.i64 %tls, <top>
//     %next = add (add %top, %bytes), 8                ; the collector header
//     br_if (ule %next, load.i64 %tls, <end>), fast, cold
//   fast:
//     store.i64 %tls, <top>, %next
//     store.i64 %top, (or %header, %bytes)             ; the collector header
//     br cont(bitcast.tagged.i64 (or (add %top, 8), <tag>))
//   cold:                                              ; set_cold
//     %slow = call.tagged @alloc(%bytes, %header)
//     br cont(%slow)
//   cont(%obj): ...
//
// Allocations of constant size in one block with no GC point between them
// form a group: the first takes the window for all of them, and each later
// one, when the first took the fast path, writes its collector header into
// the space reserved for it (its own window test only after a cold first).
// Up to the continuation's first GC point, uses of the object's bits
// (bitcast.i64.tagged) and of its address (and with the 48-bit mask) take
// the values computed here as parameters.
//
// The fast path is no GC point; only the cold call is. The window's end is
// eden's adaptive limit, which a collection moves, so it is read at every
// allocation (gc_contract.md 3). Like the barrier's filter this runs after
// the optimizer, on the copy of the function that is lowered
// (function_codegen), so every pass before it sees one call: escape
// analysis, scalar replacement, write_barrier_elim. The baseline tier and
// the interpreters call the allocator.

#include <brass/mir/function.hpp>

#include <cstddef>

namespace brass {

// True when `fn`'s module has a tagged allocator with an inline window and a
// pinned TLS register, and `fn` calls the allocator.
bool needs_tagged_allocation_expansion(const Function& fn);

// Expands those calls as above. Returns how many it expanded. Predecessor
// lists are rebuilt.
size_t expand_tagged_allocations(Function& fn);

} // namespace brass
