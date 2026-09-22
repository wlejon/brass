#pragma once

// Affine address forms and the per-instruction safety classes shared by the
// loop transforms that reorder or distribute iterations (auto-parallelization
// and tiling). Both need the same two facts about a loop body: which
// instructions may run a different number of times or in a different order
// without changing the answer, and whether two memory accesses made by
// different iterations can touch the same bytes.

#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace brass::affine {

// Checked signed 64-bit arithmetic (portable; no compiler builtins): true,
// with the result in `out`, when the exact result fits.
bool checked_add(int64_t a, int64_t b, int64_t& out) noexcept;
bool checked_sub(int64_t a, int64_t b, int64_t& out) noexcept;
bool checked_mul(int64_t a, int64_t b, int64_t& out) noexcept;

// An address as a linear combination:
//   sum(invariants[v] * v) + sum(terms[(iv, sym)] * iv * sym) + constant
// where every `v` and `sym` is fixed for the whole loop nest and every `iv`
// is one of the induction variables the form was built over (`sym` is null
// for a plain constant coefficient). All arithmetic feeding the address is
// 64-bit, so the combination is exact for any address that is in bounds.
struct Form {
    bool ok = false;
    std::map<const Value*, int64_t> invariants;
    std::map<std::pair<const Value*, const Value*>, int64_t> terms;
    int64_t constant = 0;
    uint32_t size = 0;           // bytes accessed
    const Value* pointer = nullptr;  // the address operand, for alias queries
    bool is_store = false;
};

using InvariantFn = std::function<bool(const Value*)>;

bool is_plain_memory_access(Opcode op) noexcept;
bool is_plain_store(Opcode op) noexcept;

// The address of a load/store/load_indexed/store_indexed as a Form over
// `ivs`. `invariant` says whether a value is fixed for the whole nest.
Form address_form(const Instruction& mem, const std::vector<const Value*>& ivs, const InvariantFn& invariant);

// Two forms with the same invariant part, terms, constant and size name the
// same address at the same induction-variable values.
bool same_form(const Form& a, const Form& b) noexcept;

// True when the two accesses provably lie in different objects.
bool distinct_objects(const AliasAnalysis& aa, const Value* p1, const Value* p2);

// An instruction that computes a value from its operands only: no memory,
// no calls, no traps. Running it more often, or not at all, changes nothing.
bool is_pure_nontrapping(const Instruction& inst) noexcept;

// A value defined outside `loop` (or a constant, which can be re-created).
bool defined_outside(const LoopInfo& loop, const Value* v) noexcept;
bool is_int_constant(const Value* v, int64_t& out) noexcept;
// iconst/fconst: a value that can be re-created anywhere by copying it.
bool is_plain_constant(const Value* v) noexcept;

// Visits every value slot an instruction reads: operands, deopt state and
// the arguments of every outgoing edge, so the slot can be rewritten.
void for_each_use_slot(Instruction& inst, const std::function<void(Value*&)>& f);

// True when some instruction in a block outside `loop` reads `v`.
bool used_outside(const Function& fn, const LoopInfo& loop, const Value* v);

} // namespace brass::affine
