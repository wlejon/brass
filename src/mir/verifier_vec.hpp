#pragma once

#include <brass/mir/instruction.hpp>
#include <string>
#include <functional>

namespace brass {

bool verify_vector_instruction(
    const Instruction* inst,
    const std::string& inst_prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
