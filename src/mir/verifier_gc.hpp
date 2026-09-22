#pragma once

#include <brass/mir/function.hpp>
#include <functional>
#include <string>

namespace brass {

// Checks the derived-gcref rules of <brass/mir/gc_refs.hpp> over `fn`.
void verify_derived_gcrefs(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
