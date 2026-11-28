#pragma once

#include <brass/target/ptx/ptx_ir.hpp>
#include <array>
#include <cstdint>

namespace brass::ptx {

enum class PtxSchedMode : uint8_t {
    None = 0,
    LatencyHide = 1,
    MinRegPressure = 2,
    MmaInterleave = 3,
    Auto = 4
};

inline constexpr const char* ptx_sched_mode_name(PtxSchedMode m) noexcept {
    switch (m) {
        case PtxSchedMode::None:           return "None";
        case PtxSchedMode::LatencyHide:    return "LatencyHide";
        case PtxSchedMode::MinRegPressure: return "MinRegPressure";
        case PtxSchedMode::MmaInterleave:  return "MmaInterleave";
        case PtxSchedMode::Auto:           return "Auto";
    }
    return "Unknown";
}

struct PtxScheduleOptions {
    PtxSchedMode mode = PtxSchedMode::None;
    bool enable_remat = false;
    uint32_t reg_pressure_limit = 64;
};

struct PtxScheduleStats {
    uint32_t regions_scheduled = 0;
    uint32_t insts_reordered = 0;
    uint32_t remats_applied = 0;
    uint32_t peak_live_before = 0;
    uint32_t peak_live_after = 0;
};

/// Estimates the peak intra-block live virtual register count across all blocks of `fn`.
uint32_t estimate_peak_live_regs(
    const Function& fn,
    std::array<uint32_t, kRegClassCount>* per_class_peak = nullptr) noexcept;

/// Runs barrier-safe constant/address rematerialization and DAG list scheduling on `fn`.
PtxScheduleStats schedule_and_remat(Function& fn, const PtxScheduleOptions& opts);

} // namespace brass::ptx
