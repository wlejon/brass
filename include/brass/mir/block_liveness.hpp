#pragma once

#include <brass/mir/function.hpp>
#include <unordered_map>
#include <unordered_set>

namespace brass {

// The values live into each block of `fn`: the usual backward dataflow over
// the CFG, with an edge's arguments read by its source block and a block's
// parameters defined by it (so never live into it). Constants are values
// like any other. A block no edge reaches (a guard's resume block) still
// gets its set, from its own uses and its successors'.
std::unordered_map<const BasicBlock*, std::unordered_set<const Value*>> block_live_ins(const Function& fn);

} // namespace brass
