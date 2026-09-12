#pragma once

#include <brass/runtime/tiering.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/gc/stack_map.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <mutex>
#include <iosfwd>
#include <cstdint>

namespace brass::runtime {

struct MultiTierStats {
    uint64_t tier0_invocations = 0;
    uint64_t tier1_compilations = 0;
    uint64_t tier1_invocations = 0;
    uint64_t tier2_compilations = 0;
    uint64_t tier2_invocations = 0;
    uint64_t total_tier1_compile_time_us = 0;
};

class MultiTierPipeline {
public:
    static MultiTierPipeline& instance();

    void initialize(const TieringConfig& config = TieringConfig{});
    void shutdown();
    bool is_initialized() const noexcept { return initialized_; }

    const TieringConfig& config() const noexcept { return config_; }
    TieringConfig& config() noexcept { return config_; }
    void set_config(const TieringConfig& config) noexcept;

    codegen::BaselineJitCompiler& baseline_compiler() noexcept { return baseline_compiler_; }
    const codegen::BaselineJitCompiler& baseline_compiler() const noexcept { return baseline_compiler_; }

    BackgroundCompiler& background_compiler() noexcept { return BackgroundCompiler::instance(); }

    ModuleStackMap& active_stack_maps() noexcept { return active_stack_maps_; }
    const ModuleStackMap& active_stack_maps() const noexcept { return active_stack_maps_; }

    // Synchronous Tier 1 compilation (< 5 microseconds on mutator thread)
    bool compile_and_install_tier1(std::string_view fn_name, const Function* fn = nullptr);

    // Enqueue Tier 2 background optimization
    bool enqueue_tier2(
        std::string_view fn_name,
        const Module* mod = nullptr,
        FunctionHandle* handle = nullptr
    );

    // Mutator notification on function invocation
    void on_invocation(std::string_view fn_name);

    // Multi-tier execution of an entry function in a module
    RuntimeValue execute(
        Module& mod,
        std::string_view entry_fn,
        const std::vector<RuntimeValue>& args = {}
    );

    MultiTierStats stats() const;
    void reset_stats();
    void dump_stats(std::ostream& os) const;

    void register_baseline_compiled(std::shared_ptr<codegen::BaselineCompiledFunction> fn);
    std::shared_ptr<codegen::BaselineCompiledFunction> find_baseline_compiled(std::string_view name) const;
    void clear_baseline_cache();

private:
    MultiTierPipeline() = default;
    ~MultiTierPipeline() = default;

    MultiTierPipeline(const MultiTierPipeline&) = delete;
    MultiTierPipeline& operator=(const MultiTierPipeline&) = delete;

    bool initialized_ = false;
    TieringConfig config_;
    codegen::BaselineJitCompiler baseline_compiler_;
    ModuleStackMap active_stack_maps_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<codegen::BaselineCompiledFunction>> baseline_functions_;
    MultiTierStats stats_;
};

} // namespace brass::runtime
