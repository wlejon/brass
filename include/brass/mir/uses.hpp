#pragma once

#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

// Use walking and use replacement shared by every MIR pass. An instruction
// reads values through three kinds of slot: its operands, its deopt state map
// and the arguments of each outgoing edge (br, br_if, switch cases and
// default, invoke normal and unwind). Walking them in one place keeps a pass
// from silently missing a kind of use.
namespace brass {

// Calls `f(Value*&)` for every value slot of `inst`, null slots included, so
// `f` may rewrite them.
template <typename F>
void for_each_use_slot(Instruction& inst, F&& f) {
    for (Value*& v : inst.operands()) f(v);
    for (Value*& v : inst.state_map()) f(v);
    for (Value*& v : inst.branch_target().args) f(v);
    for (Value*& v : inst.true_target().args) f(v);
    for (Value*& v : inst.false_target().args) f(v);
    for (SwitchCase& sc : inst.switch_cases()) {
        for (Value*& v : sc.target.args) f(v);
    }
}

// Calls `f(Value*)` for every non-null value `inst` reads.
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

// Calls `f(BranchTarget&)` for every outgoing edge of `inst` that names a
// block. br uses the branch target, br_if and invoke the true/false (normal/
// unwind) targets, switch the default target and its cases.
template <typename F>
void for_each_edge(Instruction& inst, F&& f) {
    if (inst.branch_target().block) f(inst.branch_target());
    if (inst.true_target().block) f(inst.true_target());
    if (inst.false_target().block) f(inst.false_target());
    for (SwitchCase& sc : inst.switch_cases()) {
        if (sc.target.block) f(sc.target);
    }
}

template <typename F>
void for_each_edge(const Instruction& inst, F&& f) {
    if (inst.branch_target().block) f(inst.branch_target());
    if (inst.true_target().block) f(inst.true_target());
    if (inst.false_target().block) f(inst.false_target());
    for (const SwitchCase& sc : inst.switch_cases()) {
        if (sc.target.block) f(sc.target);
    }
}

// True when `inst` reads `val` in any slot.
bool uses_value(const Instruction& inst, const Value* val) noexcept;

// Replaces every use of `old_val` in `inst` by `new_val`; returns how many
// slots changed.
size_t replace_uses_in(Instruction& inst, const Value* old_val, Value* new_val);

// Replaces every use of `old_val` in `bb`'s instructions.
size_t replace_uses_in(BasicBlock& bb, const Value* old_val, Value* new_val);

// Replaces every use of `old_val` in `fn` by `new_val` (nothing when either
// is null or they are equal); returns how many slots changed.
size_t replace_all_uses(Function& fn, const Value* old_val, Value* new_val);

// Replaces the uses of `old_val` in the instructions `where` accepts.
size_t replace_uses_if(Function& fn, const Value* old_val, Value* new_val,
                       const std::function<bool(const Instruction&)>& where);

// Number of slots in `fn` that read `val`.
size_t count_uses(const Function& fn, const Value* val);

// True when some instruction in `fn` reads `val`.
bool has_uses(const Function& fn, const Value* val);

// Use counts of every value `fn` reads.
std::unordered_map<const Value*, uint32_t> compute_use_counts(const Function& fn);

// Removes parameter `index` of `bb` and the matching argument on every edge
// into `bb` from any block of its function, whatever kind of terminator the
// edge comes from. Scans every terminator, not the cached predecessor list,
// which an earlier edit may have left stale. The parameter must be unused.
void remove_block_param(BasicBlock& bb, size_t index);

} // namespace brass
