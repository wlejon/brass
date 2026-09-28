#pragma once

#include <brass/mir/function.hpp>
#include <unordered_map>
#include <vector>

namespace brass {

// The values live into each block of `fn`: the usual backward dataflow over
// the CFG, with an edge's arguments read by its source block and a block's
// parameters defined by it (so never live into it). Constants are values
// like any other. A block no edge reaches (a guard's resume block) still
// gets its set, from its own uses and its successors'. Each block's values
// are listed once, in no particular order.
//
// Computed per value over SSA (a use's block and its predecessors back to
// the defining block), so the cost is the size of the answer rather than
// blocks x values per round of a set-based fixpoint.
using BlockLiveIns = std::unordered_map<const BasicBlock*, std::vector<const Value*>>;
BlockLiveIns block_live_ins(const Function& fn);

// The same sets for `targets` only (every other block is absent from the
// map). What a caller that asks about a few blocks — guard resume targets,
// an OSR entry — wants: the walk is the same, the answer is not built for
// the rest.
BlockLiveIns block_live_ins(const Function& fn, const std::vector<const BasicBlock*>& targets);

} // namespace brass
