#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <vector>

namespace brass {

class DebugContext;

struct InlineResult {
    bool success = false;
    BasicBlock* split_head = nullptr;
    BasicBlock* split_tail = nullptr;
    std::vector<BasicBlock*> inlined_blocks;
    Value* return_value = nullptr; // BlockParam on split_tail (nullptr if void)
};

// Performs SSA block splitting at call_inst and inlines callee into caller.
// - Splits caller block into split_head and split_tail
// - Clones callee basic blocks and instructions with fresh block/value IDs
// - Maps callee parameters to caller call arguments
// - Remaps callee 'ret' to branch to split_tail with return value forwarded as block param
// - Replaces all uses of call_inst->result() with split_tail's return parameter
// - Connects split_head -> callee entry
// - Rebuilds CFG predecessors
InlineResult inline_call_site(Function& caller, Instruction* call_inst, const Function& callee, DebugContext* dbg_ctx = nullptr);

} // namespace brass
