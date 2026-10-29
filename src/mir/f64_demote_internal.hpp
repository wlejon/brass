#pragma once

#include <brass/mir/f64_demote.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <unordered_set>
#include <vector>

namespace brass {

bool is_safe_integer_f64(double v) noexcept;
bool get_const_int_or_f64_int(const Value* val, int64_t& out) noexcept;
bool is_unprofitable_loop(const LoopInfo* loop, const Function& fn);

// A loop limit canonicalize_loop_bounds replaced: the header's `cmp` now
// reads, at operand `idx`, ceil(limit) as an i32 clamped to `max` (so the
// bounds can bound the IV), where it read `original`. That changes the trip
// count when ceil(limit) > max, so it may stand only where the entry check
// proves it does not (version_on_entry_params); restore() puts the original
// back.
struct ClampedLimit {
    Instruction* cmp = nullptr;
    size_t idx = 0;
    Value* original = nullptr;
    Value* limit = nullptr;
    double max = 0.0;
    BasicBlock* preheader = nullptr;
    BasicBlock* header = nullptr;
    Value* fits = nullptr;  // ceil(limit) <= max, in the preheader
    void restore() const { cmp->set_operand(idx, original); }
};

// For a clamp the entry cannot test: the preheader branches on `fits` to the
// loop or to a copy of everything reachable from its header, every clamped
// limit in the copy restored, so a limit over the clamp runs the loop as
// written. Returns false, changing nothing, when the copy would be too big.
bool version_loop_on_clamp(Function& fn, const ClampedLimit& site, const std::vector<ClampedLimit>& all);

// Versions each loop whose counter is stepped by a loop-invariant value the
// bounds cannot show positive on 1 <= step <= 2^31 (f64_demote_step.cpp).
// Returns whether any loop was versioned.
bool version_loops_on_step(Function& fn);

bool canonicalize_loop_bounds(Function& fn, const DominatorTree& dom, const LoopAnalysis& loop_analysis,
                              std::vector<ClampedLimit>* clamped);

// Whether `limit` is a value the entry block can compute again from the
// function's parameters alone (constants, f64 parameters, and their sums,
// differences and products), which is what lets the entry check test it.
bool limit_computable_at_entry(const Function& fn, const Value* limit);

// The f64 entry parameters a loop's limit is made from (a `ceil` of it, or of
// a sum or difference with it), which the bounds take to be an integer in
// [0, sized_limit_max]: a width or a count. Nothing proves that, so a
// function with any of them runs its demoted code only when an entry check
// finds it true (version_on_entry_params).
std::unordered_set<const Value*> sized_entry_params(const Function& fn);
// The largest limit a loop's bound is canonicalized to and a sized parameter
// may hold: 2^26 - 1 where the function multiplies two loop values (a 2D
// index), so their product stays exact, else 2^31 - 1.
double sized_limit_max(const Function& fn);

// The f64 entry parameters the demoted values' exactness rests on: the
// dataflow takes every entry parameter to hold an integer, and the rewrite
// converts one to i64 without checking, so each one reached backward from
// `demote_set` through exact values must be checked before the demoted code
// runs; so must every one in `sized`.
std::vector<Value*> assumed_entry_params(const Function& fn, const std::unordered_set<Value*>& demote_set,
                                         const std::unordered_set<const Value*>& exact_ints,
                                         const std::unordered_set<const Value*>& sized);

// Puts a copy of the function as it is now (the fallback, its `clamped`
// limits restored) beside its body and a new entry block in front of both:
// when every parameter in `assumed` is an integer, not -0, inside its range
// ([0, sized_limit_max] for the sized ones, +-2^53 otherwise), and every
// clamped limit's ceiling is within its clamp, the original body runs, which
// the caller then demotes; otherwise the copy runs. Returns the original
// entry.
BasicBlock* version_on_entry_params(Function& fn, const std::vector<Value*>& assumed,
                                    const std::unordered_set<const Value*>& sized,
                                    const std::vector<ClampedLimit>& clamped);

// The name of the entry block version_on_entry_params makes: a function that
// has one was demoted already, and its fallback must stay as it is.
inline constexpr const char* kF64VersionEntryName = "f64_demote.entry";

// Removes from `exact_ints` every value not provably within +-2^53 and never
// -0.0, the conditions under which its i64 image computes the same thing.
// Returns whether anything was removed.
bool drop_unprovable_exact_values(const Function& fn, const DominatorTree& dom, const LoopAnalysis& loops,
                                  std::unordered_set<const Value*>& exact_ints,
                                  std::unordered_set<const Value*>* i32_bounded = nullptr);

void record_loop_stats(
    Function& fn,
    const LoopAnalysis& loops,
    const std::unordered_set<Value*>& demote_set,
    DemoteStats* stats
);

} // namespace brass
