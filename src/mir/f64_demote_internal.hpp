#pragma once

#include <brass/mir/f64_demote.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <unordered_set>

namespace brass {

bool is_safe_integer_f64(double v) noexcept;
bool get_const_int_or_f64_int(const Value* val, int64_t& out) noexcept;
bool is_unprofitable_loop(const LoopInfo* loop, const Function& fn);

// Removes from `exact_ints` every value not provably within +-2^53 and never
// -0.0, the conditions under which its i64 image computes the same thing.
// Returns whether anything was removed.
bool drop_unprovable_exact_values(const Function& fn, const DominatorTree& dom, const LoopAnalysis& loops,
                                  std::unordered_set<const Value*>& exact_ints);

void record_loop_stats(
    Function& fn,
    const LoopAnalysis& loops,
    const std::unordered_set<Value*>& demote_set,
    DemoteStats* stats
);

} // namespace brass
