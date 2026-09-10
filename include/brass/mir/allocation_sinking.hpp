#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/partial_escape.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>

namespace brass {

struct AllocationSinkingOptions {
    bool enable_scalarization = true;
    bool enable_materialization = true;
    PartialEscapeStats* stats = nullptr;
};

class AllocationSinkingPass {
public:
    explicit AllocationSinkingPass(Function& fn, const AllocationSinkingOptions& options = {});

    bool run();

    const PartialEscapeStats& stats() const noexcept { return stats_; }

private:
    bool process_candidate(const Value* alloc_val,
                           const PartialEscapeAnalysis& pea,
                           DominatorTree& dom);

    Function& fn_;
    AllocationSinkingOptions options_;
    PartialEscapeStats stats_;
};

bool sink_allocations(Function& fn, const AllocationSinkingOptions& options = {});
bool sink_allocations(Module& mod, const AllocationSinkingOptions& options = {});

} // namespace brass
