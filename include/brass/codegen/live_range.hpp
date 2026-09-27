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
        return start < other.end && other.start < end;
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
    uint32_t max_loop_depth = 0;
    PReg register_hint;
    VReg coalesce_partner;

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
    uint32_t loop_depth = 0;
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
    uint32_t get_loop_depth_at(uint32_t inst_id) const;
    // Guard exit blocks whose uses are counted at the Jcc entering them:
    // block id -> the block's GuardExit.
    const std::unordered_map<uint32_t, const LirInst*>& guard_exits() const noexcept { return guard_exits_; }

private:
    struct BlockRange {
        uint32_t start_id;
        uint32_t end_id;
        uint32_t loop_depth;
    };

    LirFunction& fn_;
    std::vector<LiveInterval> intervals_;
    std::unordered_map<const LirBlock*, BlockLiveness> block_liveness_;
    std::vector<uint32_t> call_inst_ids_;   // in id order
    std::vector<BlockRange> block_ranges_;  // non-empty blocks, by start id
    // Guard exit blocks entered only by a Jcc: block id -> its GuardExit.
    std::unordered_map<uint32_t, const LirInst*> guard_exits_;

    void find_guard_exits();
    // The GuardExit `inst` branches to when it is a Jcc to one of
    // guard_exits_, else null.
    const LirInst* guard_exit_taken_by(const LirInst& inst) const;
    void assign_instruction_ids();
    void compute_local_liveness();
    void compute_global_liveness();
    void build_intervals();
    void compute_loop_depths();
    void compute_spill_weights();
};

} // namespace brass::codegen
