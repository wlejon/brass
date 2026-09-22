#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brass::fuzz {

// Which optimizer configuration the differential fuzzer's optimized tiers run.
enum class FuzzPipeline : uint8_t {
    // Every module-level optimization pass in src/mir, each enabled and run
    // as its own named step so a broken module is pinned to one pass.
    AllPasses,
    // Exactly the production pipeline (run_pass_pipeline) under the
    // configuration Bronze compiles with.
    Bronze,
    // The fuzzer's original fixed sequence: GVN-PRE, WBE, optimize_module.
    Legacy,
};

std::string_view pipeline_name(FuzzPipeline pipeline) noexcept;
bool parse_pipeline(std::string_view text, FuzzPipeline& out) noexcept;

// Runs `pipeline` on `mod`, calling the hooks around every step. Steps named
// in `skip` (e.g. "jump_threading") are left out, so a pass with a known bug
// does not mask the rest. For the Bronze pipeline a name switches off the
// matching option (the transforms inside its "loops" stage use the names
// AllPasses gives them). Returns false when after_pass stopped it.
bool run_fuzz_pipeline(Module& mod, FuzzPipeline pipeline, const PassPipelineHooks& hooks = {},
                       const std::vector<std::string>& skip = {});

} // namespace brass::fuzz
