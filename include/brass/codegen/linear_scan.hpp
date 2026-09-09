#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/target/calling_conv.hpp>
#include <vector>
#include <set>
#include <unordered_map>
#include <cstdint>

namespace brass::codegen {

class LinearScanAllocator {
public:
    LinearScanAllocator(
        LirFunction& fn,
        LivenessAnalysis& liveness,
        const CallingConvention& cc
    );

    void allocate();

    x64::RegMask used_callee_saved_gprs() const noexcept { return used_callee_gprs_; }
    x64::RegMask used_callee_saved_xmms() const noexcept { return used_callee_xmms_; }
    size_t num_spill_slots() const noexcept { return next_spill_slot_; }

private:
    LirFunction& fn_;
    LivenessAnalysis& liveness_;
    CallingConvention cc_;

    std::vector<PReg> available_gprs_;
    std::vector<PReg> available_xmms_;

    std::vector<LiveInterval*> active_;

    x64::RegMask used_callee_gprs_ = 0;
    x64::RegMask used_callee_xmms_ = 0;
    size_t next_spill_slot_ = 0;

    std::unordered_map<uint32_t, std::vector<VReg>> coalesce_hints_;

    void init_register_pools();
    void build_coalesce_hints();
    void expire_old_intervals(uint32_t current_start);
    bool try_allocate_free_reg(LiveInterval& interval);
    void allocate_blocked_reg(LiveInterval& interval);
    std::set<uint8_t> get_occupied_regs(const LiveInterval& interval) const;
    std::set<uint8_t> get_hard_blocked_regs(const LiveInterval& interval) const;
    int32_t allocate_spill_slot(bool is_gcref, uint8_t size);
    int32_t allocate_spill_slot(bool is_gcref) { return allocate_spill_slot(is_gcref, 8); }
    void rewrite_instructions();
};

void run_linear_scan_regalloc(
    LirFunction& fn,
    LivenessAnalysis& liveness,
    const CallingConvention& cc
);

} // namespace brass::codegen
