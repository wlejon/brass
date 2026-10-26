#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

// What the optimizer knows of an i64 that holds a double's bits, the way a
// NaN-boxing front end (bronze, brasslua) keeps its numbers: a double is
// boxed as its bits with a NaN made the one canonical NaN
// (`select(ne.f64(f, f), NAN, bitcast(f))`), and a value is a number when its
// bits are at most those of -inf, 0xFFF0000000000000, as an unsigned
// integer. Every double but a NaN with its sign set is: the tags live above.
// The number check of a value just boxed, or carried round a loop as boxes,
// is then known to pass.
namespace brass::number_bits {

inline constexpr uint64_t kMaxOrderedBits = 0xFFF0000000000000ULL;

class Facts {
public:
    explicit Facts(const Function& fn);

    // True when every value `v` can take is, unsigned, at most
    // kMaxOrderedBits.
    bool at_most_ordered_max(const Value* v);

private:
    // `ordered`: a double known not to be a NaN where `v` is evaluated.
    bool holds(const Value* v, const Value* ordered, unsigned depth);
    bool param_holds(const Value* param, unsigned depth);

    const Function& fn_;
    std::unordered_set<const BasicBlock*> resume_targets_;
    std::unordered_map<const Value*, bool> known_;
    // Block parameters being decided: assumed to hold, which is sound for
    // the cycle through them (each value a parameter takes is an incoming
    // argument evaluated before it).
    std::unordered_set<const Value*> assumed_;
};

} // namespace brass::number_bits
