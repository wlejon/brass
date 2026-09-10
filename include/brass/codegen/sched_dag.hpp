#pragma once

#include <brass/codegen/lir.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string_view>

namespace brass::codegen {

enum class EdgeKind : uint8_t {
    RAW,     // True data dependence
    WAR,     // Anti-dependence
    WAW,     // Output dependence
    MemRAW,  // Store -> Load memory hazard
    MemWAR,  // Load -> Store memory hazard
    MemWAW,  // Store -> Store memory hazard
    Barrier  // Scheduling barrier (calls, safepoints, volatile, terminators)
};

std::string_view to_string(EdgeKind kind) noexcept;

struct SchedEdge {
    uint32_t target_node = 0;
    EdgeKind kind = EdgeKind::RAW;
    uint32_t latency = 1;
};

struct SchedNode {
    uint32_t id = 0;
    LirInst* inst = nullptr;
    uint32_t latency = 1;
    uint32_t height = 0;
    uint32_t depth = 0;
    bool is_pre_ra = true;
    bool is_load = false;
    bool is_store = false;
    bool is_barrier = false;

    std::vector<SchedEdge> preds;
    std::vector<SchedEdge> succs;

    uint32_t unscheduled_preds = 0;
};

class SchedDAG {
public:
    explicit SchedDAG(LirBlock& block, bool is_pre_ra = true);

    void build();

    const std::vector<SchedNode>& nodes() const noexcept { return nodes_; }
    std::vector<SchedNode>& nodes() noexcept { return nodes_; }
    const SchedNode& node(size_t i) const { return nodes_[i]; }
    SchedNode& node(size_t i) { return nodes_[i]; }
    size_t size() const noexcept { return nodes_.size(); }

    void add_edge(uint32_t from, uint32_t to, EdgeKind kind, uint32_t latency);

private:
    LirBlock& block_;
    bool is_pre_ra_;
    std::vector<SchedNode> nodes_;

    void build_nodes();
    void build_register_dependencies();
    void build_memory_dependencies();
    void build_barrier_dependencies();
    void compute_metrics();
};

uint32_t get_instruction_latency(const LirInst& inst);
bool instruction_defines_flags(const LirInst& inst) noexcept;
bool instruction_uses_flags(const LirInst& inst) noexcept;
bool is_scheduling_barrier(const LirInst& inst) noexcept;

} // namespace brass::codegen
