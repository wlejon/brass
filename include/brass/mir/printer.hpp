#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <ostream>
#include <string>

namespace brass {

void print_module(const Module& mod, std::ostream& os);
std::string to_string(const Module& mod);

void print_function(const Function& fn, std::ostream& os);
std::string to_string(const Function& fn);

void print_block(const BasicBlock& bb, std::ostream& os);
std::string to_string(const BasicBlock& bb);

void print_instruction(const Instruction& inst, std::ostream& os);
std::string to_string(const Instruction& inst);

} // namespace brass
