#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <string>
#include <string_view>
#include <cstdint>

namespace brass::target {

struct PtxOptions {
    std::string sm_arch = "sm_70"; // sm_70, sm_75, sm_80, sm_89, sm_90
    uint32_t ptx_version_major = 7;
    uint32_t ptx_version_minor = 0;
    // Run the ptx::cleanup passes (copy propagation, dead code, branch
    // simplification, register renumbering) between ISel and the verifier.
    // Off prints the raw ISel output; the result is the same program.
    bool cleanup = true;
};

class PtxTarget {
public:
    static std::string emit_function(const Function& fn, const PtxOptions& opts = {});
    static std::string emit_module(const Module& mod, const PtxOptions& opts = {});
};

} // namespace brass::target
