#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/target/calling_conv.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace brass::codegen {

struct InstConstraints {
    uint32_t clobbered_gprs = 0;
    uint32_t clobbered_xmms = 0;
    uint32_t pinned_gprs = 0;
    uint32_t pinned_xmms = 0;
    constexpr bool empty() const noexcept {
        return (clobbered_gprs | clobbered_xmms | pinned_gprs | pinned_xmms) == 0;
    }
    InstConstraints& operator|=(const InstConstraints& o) noexcept {
        clobbered_gprs |= o.clobbered_gprs;
        clobbered_xmms |= o.clobbered_xmms;
        pinned_gprs |= o.pinned_gprs;
        pinned_xmms |= o.pinned_xmms;
        return *this;
    }
    constexpr InstConstraints operator|(const InstConstraints& o) const noexcept {
        return InstConstraints{
            clobbered_gprs | o.clobbered_gprs,
            clobbered_xmms | o.clobbered_xmms,
            pinned_gprs | o.pinned_gprs,
            pinned_xmms | o.pinned_xmms
        };
    }
};

class LinearScanAllocator {
public:
    LinearScanAllocator(
        LirFunction& fn,
        LivenessAnalysis& liveness,
        const CallingConvention& cc
    );

    void allocate();

    uint32_t used_callee_saved_gprs() const noexcept { return used_callee_gprs_; }
    uint32_t used_callee_saved_xmms() const noexcept { return used_callee_xmms_; }
    size_t num_spill_slots() const noexcept { return next_spill_slot_; }

private:
    LirFunction& fn_;
    LivenessAnalysis& liveness_;
    CallingConvention cc_;

    std::vector<PReg> available_gprs_;
    std::vector<PReg> available_xmms_;

    std::vector<LiveInterval*> active_;

    uint32_t used_callee_gprs_ = 0;
    uint32_t used_callee_xmms_ = 0;
    size_t next_spill_slot_ = 0;

    std::unordered_map<uint32_t, std::vector<VReg>> coalesce_hints_;
    std::unordered_map<uint32_t, std::vector<PReg>> fixed_preg_hints_;
    std::unordered_multimap<uint64_t, PReg> vreg_at_preg_hints_;

    // The instructions that can block a register for an interval covering
    // them — any with a clobber mask, a physical-register operand, or a fixed
    // operand constraint — in id order, stored in InstConstraints structs
    // with a sparse table (constraint_or_[k][i] is the OR of [i, i + 2^k)),
    // so the union over any id range is two lookups and an interval pays per
    // segment, not per instruction it covers.
    std::vector<uint32_t> constrained_ids_;
    std::vector<std::vector<InstConstraints>> constraint_or_;
    // Distinct vregs that appear as the index register of a memory operand.
    std::vector<VReg> mem_index_vregs_;

    void init_register_pools();
    void build_coalesce_hints();
    void build_constraint_index();
    InstConstraints constraint_or(size_t lo, size_t hi) const noexcept;
    void expire_old_intervals(uint32_t current_start);
    bool try_allocate_free_reg(LiveInterval& interval);
    void allocate_blocked_reg(LiveInterval& interval);
    // Register masks over the interval's class (bit = PReg::code).
    uint32_t get_occupied_regs(const LiveInterval& interval) const;
    uint32_t get_hard_blocked_regs(const LiveInterval& interval) const;
    int32_t allocate_spill_slot(bool is_gcref, uint8_t size);
    int32_t allocate_spill_slot(bool is_gcref) { return allocate_spill_slot(is_gcref, 8); }
    void rewrite_instructions();
    // Fill live_gcrefs of every call and safepoint with the gcref vregs live
    // across it (live after the site and not defined by it).
    void record_live_gcrefs();
};

void run_linear_scan_regalloc(
    LirFunction& fn,
    LivenessAnalysis& liveness,
    const CallingConvention& cc
);

} // namespace brass::codegen
