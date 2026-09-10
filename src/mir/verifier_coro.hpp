#pragma once

#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <string>
#include <functional>

namespace brass {

bool verify_coro_instruction(
    const Instruction* inst,
    const BasicBlock* bb,
    const std::string& inst_prefix,
    const std::function<void(const std::string&)>& report_error
);

} // namespace brass
