#pragma once

#include <brass/core/env.hpp>
#include <brass/mir/module.hpp>

namespace brass::codegen {

// Whether the allocator must zero the registers and spill slots dead at each
// call of a function of `mod` (LirFunction::clear_dead_roots_at_calls): so
// the collector's conservative stack scan finds no dead object there. Not for
// a module that has chosen to leave them (Module::precise_stack_roots).
// BRASS_NO_PRECISE_ROOTS=1 (read once a process) zeroes them for every
// module, for A/B.
inline bool clears_dead_roots_at_calls(const Module* mod) {
    static const bool seam_off = !brass::env_value("BRASS_NO_PRECISE_ROOTS").empty();
    return seam_off || mod == nullptr || !mod->precise_stack_roots();
}

}  // namespace brass::codegen
