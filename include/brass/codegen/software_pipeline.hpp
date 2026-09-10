#pragma once

#include <brass/codegen/lir.hpp>
#include <cstddef>
#include <cstdint>

namespace brass::codegen {

struct PipelineOptions {
    bool enable_software_pipelining = true;
    size_t min_trip_count = 2;
    size_t max_body_instructions = 100;
};

struct PipelineStats {
    size_t loops_analyzed = 0;
    size_t loops_pipelined = 0;
    size_t prologues_created = 0;
    size_t epilogues_created = 0;
};

// Check if a block is a legal candidate for loop software pipelining
bool is_pipelinable_loop(const LirFunction& fn, const LirBlock& block, const PipelineOptions& opts);
bool is_pipelinable_loop(const LirFunction& fn, const LirBlock& block);

// Modulo schedule / pipeline a single loop block
bool pipeline_loop(LirFunction& fn, LirBlock* loop_body, const PipelineOptions& opts);
bool pipeline_loop(LirFunction& fn, LirBlock* loop_body);

// Pipeline all eligible inner loops in a function
PipelineStats run_software_pipelining(LirFunction& fn, const PipelineOptions& opts);
PipelineStats run_software_pipelining(LirFunction& fn);

} // namespace brass::codegen
