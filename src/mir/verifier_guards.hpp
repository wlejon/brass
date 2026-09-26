#pragma once

#include <brass/mir/function.hpp>
#include <functional>
#include <string>

namespace brass {

// A guard that exits to a resume target (no exit stub) hands the lower tier
// that continues after a deopt only its state values: the resume block's
// parameters take them by position and each state value is restored under
// its own name. Every other value live into the resume block would be read
// unset there, so each value live into it must be one of the guard's state
// values (or a parameter of the block).
void verify_guard_resume_state(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
