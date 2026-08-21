#pragma once

#include <brass/codegen/lir.hpp>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <string>

namespace brass::codegen {

struct UsePosition {
    uint32_t inst_id = 0;
    bool is_def = false;
    bool requires_reg = false;
    PReg fixed_reg;
};

struct LiveRangeSegment {
    uint32_t start = 0;
    uint32_t end = 0;

    constexpr bool contains(uint32_t id) const noexcept {
        return id >= start && id <= end;
    }

    constexpr bool overlaps(const LiveRangeSegment& other) const noexcept {
        return start <= other.end && other.start <= end;
    }
};

class LiveInterval {
public:
    VReg vreg;
    uint32_t start_id = UINT32_MAX;
    uint32_t end_id = 0;
    std::vector<LiveRangeSegment> segments;
    std::vector<UsePosition> use_positions;
    float spill_weight = 0.0f;
    PReg assigned_preg;
    int32_t assigned_spill_slot = -1;
    bool spans_call = false;

    LiveInterval() = default;
    explicit LiveInterval(VReg v) : vreg(v) {}

    void add_range(uint32_t s, uint32_t e);
    void shorten_start(uint32_t from_id);
    void add_use_pos(uint32_t id, bool is_def, bool requires_reg = true, PReg fixed = PReg{});

    bool covers(uint32_t id) const noexcept;
    bool overlaps(const LiveInterval& other) const noexcept;
    uint32_t next_use_after(uint32_t id) const noexcept;
    uint32_t first_use() const noexcept;
};

std::string to_string(const LiveInterval& interval);

struct BlockLiveness {
    uint32_t start_id = 0;
    uint32_t end_id = 0;
    std::vector<VReg> defs;
    std::vector<VReg> uses;
    std::vector<VReg> live_in;
    std::vector<VReg> live_out;
};

class LivenessAnalysis {
public:
    explicit LivenessAnalysis(LirFunction& fn);

    void run();

    const std::vector<LiveInterval>& intervals() const noexcept { return intervals_; }
    std::vector<LiveInterval>& intervals() noexcept { return intervals_; }

    LiveInterval* get_interval(VReg v);
    const LiveInterval* get_interval(VReg v) const;

    const BlockLiveness& block_liveness(const LirBlock* b) const;

private:
    LirFunction& fn_;
    std::vector<LiveInterval> intervals_;
    std::unordered_map<const LirBlock*, BlockLiveness> block_liveness_;
    std::vector<uint32_t> call_inst_ids_;

    void assign_instruction_ids();
    void compute_local_liveness();
    void compute_global_liveness();
    void build_intervals();
    void compute_spill_weights();
};

} // namespace brass::codegen
