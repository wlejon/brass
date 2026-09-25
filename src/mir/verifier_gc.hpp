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

// Checks bitcast_i64_tagged / bitcast_tagged_i64; false for any other opcode.
bool verify_tagged_bitcast(
    const Instruction& inst,
    const std::string& prefix,
    const std::function<void(const std::string&)>& report_error
);

// Checks keep_alive: one operand, no result; false for any other opcode.
bool verify_keep_alive(
    const Instruction& inst,
    const std::string& prefix,
    const std::function<void(const std::string&)>& report_error
);

// Checks the buffer of an `alloca.tagged`: whole, aligned tagged words.
void verify_tagged_alloca(
    const Instruction& inst,
    const std::string& prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
