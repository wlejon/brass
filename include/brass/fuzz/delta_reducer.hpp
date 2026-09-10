#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/printer.hpp>
#include <functional>
#include <string>
#include <string_view>
#include <memory>

namespace brass::fuzz {

/// An oracle predicate returns true if the discrepancy / failure is STILL PRESENT in mod.
using OraclePredicate = std::function<bool(const Module& mod, std::string_view fn_name)>;

struct DeltaReducerOptions {
    size_t max_passes = 20;
    bool prune_instructions = true;
    bool replace_with_constants = true;
    bool remove_redundant_blocks = true;
    bool verbose = false;
};

struct ReductionResult {
    bool success = false;
    size_t initial_instructions = 0;
    size_t final_instructions = 0;
    size_t initial_blocks = 0;
    size_t final_blocks = 0;
    std::unique_ptr<Module> minimized_module;
};

class DeltaReducer {
public:
    explicit DeltaReducer(const DeltaReducerOptions& options = {}) : options_(options) {}

    /// Runs delta-debugging algorithm minimizing mod until a minimal irreducible test case is reached.
    ReductionResult reduce(const Module& initial_mod, std::string_view fn_name,
                           const OraclePredicate& oracle);

    /// Export minimized module as canonical .mir text.
    static bool export_mir(const Module& mod, const std::string& path);

    /// Export minimized module as .il format.
    static bool export_il(const Module& mod, const std::string& path);

    const DeltaReducerOptions& options() const noexcept { return options_; }
    void set_options(const DeltaReducerOptions& opts) noexcept { options_ = opts; }

private:
    DeltaReducerOptions options_;

    bool prune_unused_instructions(std::unique_ptr<Module>& mod, std::string_view fn_name, const OraclePredicate& oracle);
    bool replace_with_constants(std::unique_ptr<Module>& mod, std::string_view fn_name, const OraclePredicate& oracle);
    bool simplify_control_flow(std::unique_ptr<Module>& mod, std::string_view fn_name, const OraclePredicate& oracle);
};

} // namespace brass::fuzz
