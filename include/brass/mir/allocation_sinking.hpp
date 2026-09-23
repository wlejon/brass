#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/partial_escape.hpp>

namespace brass {

struct AllocationSinkingOptions {
    // Most distinct fields an object may have and still be scalarized.
    size_t max_fields = 32;
    PartialEscapeStats* stats = nullptr;
};

// Allocation sinking with scalar replacement. An allocation (a call its
// module declares `allocator`, or one of brass's own allocators) whose object
// does not escape on some path from it is removed: while the object is
// "virtual" its field loads and stores become SSA values, and the allocation
// is re-emitted - with the current field values stored into it - only where
// the object first escapes. An object that escapes on one edge into a block
// with other predecessors is materialized on that edge, and the pointer
// reaches the block's uses through a block parameter.
class AllocationSinkingPass {
public:
    explicit AllocationSinkingPass(Function& fn, const AllocationSinkingOptions& options = {});

    bool run();

    const PartialEscapeStats& stats() const noexcept { return stats_; }

private:
    bool sink_one(Instruction* alloc);

    Function& fn_;
    AllocationSinkingOptions options_;
    PartialEscapeStats stats_;
};

bool sink_allocations(Function& fn, const AllocationSinkingOptions& options = {});
bool sink_allocations(Module& mod, const AllocationSinkingOptions& options = {});

} // namespace brass
