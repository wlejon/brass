#pragma once

#include <brass/mir/function.hpp>

namespace brass {

// Versions each inner counted loop whose range checks (`ult idx, len`, idx
// affine in the loop's variable, len invariant) the preheader can decide from
// the loop's first and last iteration: a copy with those checks taken as
// passed runs when every one passes at both ends, the loop as it is
// otherwise. Returns whether any loop was versioned.
bool version_loops_on_range_checks(Function& fn);

} // namespace brass
