#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/pgo/profile_data.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::pgo {

struct ChordEdge {
    BasicBlock* src = nullptr; // nullptr represents Exit
    BasicBlock* dst = nullptr; // nullptr represents Exit
    uint32_t counter_index = 0;
    bool is_critical = false;
    BasicBlock* split_block = nullptr;
};

struct SpanningTreeEdge {
    BasicBlock* src = nullptr; // nullptr represents Exit
    BasicBlock* dst = nullptr; // nullptr represents Exit
};

struct PgoFunctionMetadata {
    std::string name;
    uint32_t counter_base_index = 0;
    uint32_t entry_counter_index = UINT32_MAX;
    uint32_t num_edge_counters = 0;
    std::vector<std::pair<uint32_t, uint32_t>> chord_block_ids;
};

struct PgoMetadata {
    std::string module_name;
    uint64_t module_hash = 0;
    std::vector<PgoFunctionMetadata> functions;
    uint32_t total_counters = 0;

    const PgoFunctionMetadata* find_function(const std::string& name) const {
        for (const auto& f : functions) {
            if (f.name == name) return &f;
        }
        return nullptr;
    }
};

struct PgoInstrumentResult {
    Function* function = nullptr;
    uint32_t counter_base_index = 0;
    uint32_t total_counters = 0;
    uint32_t entry_counter_index = UINT32_MAX;
    std::vector<ChordEdge> chords;
    std::vector<SpanningTreeEdge> spanning_tree_edges;
    PgoMetadata metadata;
};

// Knuth-Stevenson minimal spanning-tree edge instrumentation for a single function
PgoInstrumentResult instrument_function(Function& fn, uint32_t counter_base_index);

// Knuth-Stevenson minimal spanning-tree edge instrumentation for an entire module
PgoInstrumentResult instrument_module(Module& mod);

// Runtime dumper helper: writes .bprof file from counters and metadata
bool brass_pgo_dump(const char* path, const uint64_t* counters, size_t num_counters, const PgoMetadata& meta);

// In-memory runtime counter table API
void brass_pgo_init_counters(size_t num_counters);
void brass_pgo_inc_counter(uint32_t counter_index);
const uint64_t* brass_pgo_get_counters(size_t* num_counters);
void brass_pgo_reset_counters();

} // namespace brass::pgo

extern "C" {
// Runtime callback called by instrumented code
void brass_pgo_inc(uint32_t counter_index);
}
