#pragma once

#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <string_view>
#include <unordered_map>

// Instruction-level cloning and use walking shared by the passes that
// duplicate code (jump threading, loop unswitching). One implementation
// keeps every field of an instruction — immediates, symbols, deopt state,
// every kind of edge — in the copy, so a pass cannot silently drop one.
namespace brass::ir {

using ValueMap = std::unordered_map<const Value*, Value*>;
using BlockMap = std::unordered_map<const BasicBlock*, BasicBlock*>;

// Calls `f(Value*)` for every value `inst` reads: operands, deopt state and
// the arguments of every outgoing edge.
template <typename F>
void for_each_use(const Instruction& inst, F&& f) {
    for (Value* v : inst.operands()) if (v) f(v);
    for (Value* v : inst.state_map()) if (v) f(v);
    for (Value* v : inst.branch_target().args) if (v) f(v);
    for (Value* v : inst.true_target().args) if (v) f(v);
    for (Value* v : inst.false_target().args) if (v) f(v);
    for (const SwitchCase& sc : inst.switch_cases()) {
        for (Value* v : sc.target.args) if (v) f(v);
    }
}

// Calls `f(BranchTarget&)` for every outgoing edge slot of `inst` that names
// a block (br, br_if, switch cases and default, invoke normal and unwind).
template <typename F>
void for_each_target(Instruction& inst, F&& f) {
    if (inst.branch_target().block) f(inst.branch_target());
    if (inst.true_target().block) f(inst.true_target());
    if (inst.false_target().block) f(inst.false_target());
    for (SwitchCase& sc : inst.switch_cases()) {
        if (sc.target.block) f(sc.target);
    }
}

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
