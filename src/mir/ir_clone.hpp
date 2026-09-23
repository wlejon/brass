#pragma once

#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/uses.hpp>
#include <string_view>
#include <unordered_map>

// Instruction-level cloning and use walking shared by the passes that
// duplicate code (jump threading, loop unswitching). One implementation
// keeps every field of an instruction — immediates, symbols, deopt state,
// every kind of edge — in the copy, so a pass cannot silently drop one.
namespace brass::ir {

using ValueMap = std::unordered_map<const Value*, Value*>;
using BlockMap = std::unordered_map<const BasicBlock*, BasicBlock*>;

// Creates a block in `fn` named `name`, appended at the end.
BasicBlock* new_block(Function& fn, std::string_view name);

// Creates a fresh block parameter value of type `type` on `bb`.
Value* new_block_param(Function& fn, BasicBlock* bb, Type type);

// Copies `src` without its operands, state or edges: opcode, type,
// immediates, symbols, memory type, debug location, and a fresh result value
// (recorded in `values`).
Instruction* clone_shell(Function& fn, const Instruction& src, ValueMap& values);

// Fills the operands, deopt state and edges of `dst` from `src`, remapping
// values through `values` and target blocks through `blocks`; anything not
// in a map is kept as is.
void clone_uses(const Instruction& src, Instruction& dst, const ValueMap& values, const BlockMap& blocks);

// clone_shell followed by clone_uses, for code whose operands are all
// already mapped.
Instruction* clone_instruction(Function& fn, const Instruction& src, ValueMap& values, const BlockMap& blocks);

} // namespace brass::ir
