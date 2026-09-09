#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <cstddef>

namespace brass {

struct LoopUnswitchOptions {
    size_t max_loop_instructions = 100;
    size_t max_unswitch_depth = 3;
};

struct LoopUnswitchStats {
    size_t loops_unswitched = 0;
};

// Unswitch a loop containing a loop-invariant conditional branch.
bool unswitch_loop(Function& fn, LoopInfo& loop, const DominatorTree& dom, const LoopUnswitchOptions& opts);
bool unswitch_loop(Function& fn, LoopInfo& loop, const DominatorTree& dom);

// Unswitch all candidate loops in a function.
bool unswitch_loops_in_function(Function& fn, const LoopUnswitchOptions& opts, LoopUnswitchStats* stats = nullptr);
bool unswitch_loops_in_function(Function& fn);

// Unswitch all candidate loops across all functions in a module.
bool unswitch_loops_in_module(Module& mod, const LoopUnswitchOptions& opts, LoopUnswitchStats* stats = nullptr);
bool unswitch_loops_in_module(Module& mod);

} // namespace brass
