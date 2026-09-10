#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/codegen/sched_dag.hpp>
#include <cstddef>
#include <cstdint>

namespace brass::codegen {

struct SchedOptions {
    bool enable_pre_ra = false;
    bool enable_post_ra = false;
    bool enable_software_pipelining = false;
    size_t max_block_instructions = 500;
    size_t gpr_pressure_threshold = 12;
    size_t xmm_pressure_threshold = 12;
    bool balance_issue_ports = true;
};

struct SchedStats {
    size_t blocks_scheduled = 0;
    size_t instructions_scheduled = 0;
    size_t stalls_hidden = 0;
    size_t loads_hoisted = 0;
    size_t pressure_throttles = 0;
};

// Schedule instructions within a single basic block
SchedStats schedule_block(LirBlock& block, const SchedOptions& opts);
SchedStats schedule_block(LirBlock& block);

// Schedule instructions across all blocks in a function
SchedStats schedule_function(LirFunction& fn, const SchedOptions& opts);
SchedStats schedule_function(LirFunction& fn);

} // namespace brass::codegen
