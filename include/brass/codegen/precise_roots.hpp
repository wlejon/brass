#pragma once

#include <brass/mir/module.hpp>
#include <cstdlib>

namespace brass::codegen {

// Whether the allocator must zero the registers and spill slots dead at each
// call of a function of `mod` (LirFunction::clear_dead_roots_at_calls). Not
// for a module whose host reads its frames only through their stack maps
// (Module::precise_stack_roots): a map names what is live at its call, and a
// dead location is written before anything reads it again, so the zeroing is
// there only for a conservative scan. BRASS_NO_PRECISE_ROOTS=1 (read once a
// process) zeroes them for every module, as before, for A/B.
inline bool clears_dead_roots_at_calls(const Module* mod) {
    static const bool seam_off = std::getenv("BRASS_NO_PRECISE_ROOTS") != nullptr;
    return seam_off || mod == nullptr || !mod->precise_stack_roots();
}

}  // namespace brass::codegen
