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
    // Exactly the production pipeline (run_pass_pipeline) under
    // production_pass_pipeline_options().
    Production,
    // The fuzzer's original fixed sequence: GVN-PRE, WBE, then
    // optimize_module's function pipeline.
    Legacy,
};

std::string_view pipeline_name(FuzzPipeline pipeline) noexcept;
bool parse_pipeline(std::string_view text, FuzzPipeline& out) noexcept;

// The declared pipeline the fuzzer runs for `pipeline`.
Pipeline fuzz_pipeline(FuzzPipeline pipeline);

// The names in `skip` that match no step of `pipeline` (a step also matches
// its base name: "cfg_simplify" names "cfg_simplify 2").
std::vector<std::string> unknown_skip_names(FuzzPipeline pipeline, const std::vector<std::string>& skip);

// Runs fuzz_pipeline(pipeline) on `mod`, calling the hooks around every
// step. Steps named in `skip` (e.g. "jump_threading") are left out, so a
// pass with a known bug does not mask the rest. Returns false when
// after_pass stopped it.
bool run_fuzz_pipeline(Module& mod, FuzzPipeline pipeline, const PassPipelineHooks& hooks = {},
                       const std::vector<std::string>& skip = {});

} // namespace brass::fuzz
