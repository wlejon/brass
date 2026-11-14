#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>
#include <unordered_map>
#include <vector>

// Values a moving collection makes stale (GcSensitive: what LICM may hoist,
// scalar_opt.hpp hoist_loop_invariants) and computing them again after a
// point that may collect (interrupt_poll.hpp: an address held across a back
// edge's handler call).
namespace brass {

// The pure instructions of scalar_opt.cpp: LICM hoists them, CSE merges them.
bool is_pure_instruction(const Instruction* inst);

// An instruction a copy of which, placed where its operands reach, computes
// its value again: a pure instruction, a read of a tagged value's bits, or a
// plain load. A load is only copied below the original or where what it
// reads was read before, so its address is one that was read.
bool is_rematerializable(const Instruction& inst);

// Values computed from where an object is now: a read of a tagged value's
// bits (`bitcast_i64_tagged`), a pure instruction or plain load with such a
// value among its operands, or a block parameter with one among its incoming
// values; only a 64-bit word or pointer is one, as nothing narrower holds an
// address. Past a GC point such a value may be stale, the object having moved
// (gc_refs.hpp). One is *clonable* when no block parameter lies on its way
// to the tagged values: copies of the instructions on that way compute it
// again from them.
class GcSensitive {
public:
    bool operator()(const Value* v) { return state(v) != 0; }
    bool clonable(const Value* v) { return state(v) == 1; }

private:
    // 0 none, 1 clonable, 2 sensitive through a block parameter.
    int state(const Value* v);
    bool param_sensitive(const Value* param);

    std::unordered_map<const Value*, int> memo_;
    int depth_ = 0;
};

// After `after` (no terminator), the vars of these indices computed again,
// in order.
struct RematPoint {
    Instruction* after = nullptr;
    std::vector<size_t> vars;
};

// Computes the vars (instructions) again at their points and rewrites every
// use of a var to the version that reaches it: the original below its
// definition, a copy below a point, and where versions meet a new block
// parameter (pruned SSA construction at the iterated dominance frontier of
// the definitions, where the var is live). A copy is the var with each
// operand that is a var taken as its version there; past a GC point
// (gc_refs.hpp may_trigger_gc) an operand GcSensitive calls clonable is
// copied too, from the tagged values up; every other operand is shared, so
// it must reach the point. Each var's definition dominates each of its
// points. Returns the number of instructions copied.
size_t rematerialize_after(Function& fn, const std::vector<Instruction*>& vars, const std::vector<RematPoint>& points);

} // namespace brass
