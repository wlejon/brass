#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx/ptx_schedule.hpp>

#include <string>
#include <string_view>
#include <cstdint>

namespace brass::target {

struct PtxOptions {
    std::string sm_arch = "sm_70"; // sm_70, sm_75, sm_80, sm_89, sm_90
    uint32_t ptx_version_major = 7;
    uint32_t ptx_version_minor = 0;
    // Optional entry launch-bound metadata (0 = omitted).
    uint32_t maxntid[3] = {0, 0, 0};
    uint32_t minnctapersm = 0;
    uint32_t maxnreg = 0;
    // Run the ptx::cleanup passes (copy propagation, dead code, branch
    // simplification, register renumbering) between ISel and the verifier.
    // Off prints the raw ISel output; the result is the same program.
    bool cleanup = true;
    // Barrier-safe intra-block PTX scheduler & rematerializer (runs after cleanup).
    ptx::PtxSchedMode sched_mode = ptx::PtxSchedMode::None;
    bool enable_ptx_remat = false;
    uint32_t sched_reg_pressure_limit = 64;
};

class PtxTarget {
public:
    static ptx::Function lower_function(const Function& fn, const PtxOptions& opts = {});
    static std::string emit_function(const Function& fn, const PtxOptions& opts = {});
    static std::string emit_module(const Module& mod, const PtxOptions& opts = {});
};

} // namespace brass::target
