#pragma once

#include <brass/runtime/tiering.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/deopt.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <mutex>
#include <unordered_set>
#include <atomic>
#include <iosfwd>
#include <cstdint>

namespace brass::runtime {

struct MultiTierStats {
    std::atomic<uint64_t> tier0_invocations{0};
    std::atomic<uint64_t> tier1_compilations{0};
    std::atomic<uint64_t> tier1_invocations{0};
    std::atomic<uint64_t> tier2_compilations{0};
    std::atomic<uint64_t> tier2_invocations{0};
    std::atomic<uint64_t> total_tier1_compile_time_us{0};

    MultiTierStats() = default;

    MultiTierStats(const MultiTierStats& other) noexcept
        : tier0_invocations(other.tier0_invocations.load(std::memory_order_relaxed)),
          tier1_compilations(other.tier1_compilations.load(std::memory_order_relaxed)),
          tier1_invocations(other.tier1_invocations.load(std::memory_order_relaxed)),
          tier2_compilations(other.tier2_compilations.load(std::memory_order_relaxed)),
          tier2_invocations(other.tier2_invocations.load(std::memory_order_relaxed)),
          total_tier1_compile_time_us(other.total_tier1_compile_time_us.load(std::memory_order_relaxed)) {}

    MultiTierStats& operator=(const MultiTierStats& other) noexcept {
        if (this != &other) {
            tier0_invocations.store(other.tier0_invocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tier1_compilations.store(other.tier1_compilations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tier1_invocations.store(other.tier1_invocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tier2_compilations.store(other.tier2_compilations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tier2_invocations.store(other.tier2_invocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_tier1_compile_time_us.store(other.total_tier1_compile_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
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

    Tier0Interpreter tier0_interpreter() const noexcept { return config_.tier0_interpreter; }
    void set_tier0_interpreter(Tier0Interpreter kind) noexcept;
    bool use_fast_interpreter() const noexcept { return config_.use_fast_interpreter(); }
    void set_use_fast_interpreter(bool enable) noexcept;

    codegen::BaselineJitCompiler& baseline_compiler() noexcept { return baseline_compiler_; }
    const codegen::BaselineJitCompiler& baseline_compiler() const noexcept { return baseline_compiler_; }

    BackgroundCompiler& background_compiler() noexcept { return BackgroundCompiler::instance(); }

    ModuleStackMap& active_stack_maps() noexcept { return active_stack_maps_; }
    const ModuleStackMap& active_stack_maps() const noexcept { return active_stack_maps_; }

    // Synchronous Tier 1 compilation (< 5 microseconds on mutator thread).
    // Returns false, leaving the function in the interpreter, when the
    // baseline compiler rejects it (codegen::UnsupportedOperation); a rejected
    // function is not retried until clear_baseline_cache().
    bool compile_and_install_tier1(std::string_view fn_name, const Function* fn = nullptr);
    bool is_baseline_rejected(std::string_view fn_name) const;

    // Enqueue Tier 2 background optimization
    bool enqueue_tier2(
        std::string_view fn_name,
        const Module* mod = nullptr,
        FunctionHandle* handle = nullptr
    );

    // Mutator notification on function invocation. The feedback overload
    // is the hot path (tier-1 code passes the pointer it was compiled
    // with); the name overload looks the function up first.
    void on_invocation(std::string_view fn_name);
    void on_invocation(TieringFeedback& fb);

    // Called by ~Module: drops the Tier-0 interpreter kept for `mod`.
    static void forget_module(const Module* mod) noexcept;

    // Multi-tier execution of an entry function in a module
    RuntimeValue execute(
        Module& mod,
        std::string_view entry_fn,
        const std::vector<RuntimeValue>& args = {}
    );

    // Deopt continuation of tier-2 code (the resumer CodeInstaller
    // registers): rebuilds the failed guard's state as typed values, records
    // the failure, invalidates the optimized code once the guard site is
    // judged mis-speculated (bailing out of tier 2 for good), and finishes
    // the call in a fresh Tier-0 interpreter exactly as the interpreter's own
    // guard would. Returns the call's result bits. Inconsistent deopt state
    // is a fatal error: there is no correct value to return.
    uint64_t resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame);
    uint64_t tier2_deopts() const noexcept { return tier2_deopts_.load(std::memory_order_relaxed); }
    uint64_t tier2_invalidations() const noexcept { return tier2_invalidations_.load(std::memory_order_relaxed); }

    MultiTierStats stats() const;
    void reset_stats();
    void dump_stats(std::ostream& os) const;

    void register_external_symbol(std::string_view name, void* addr);
    void register_external_function(std::string_view name, FastHostFn fn);

    void register_baseline_compiled(std::shared_ptr<codegen::BaselineCompiledFunction> fn);
    std::shared_ptr<codegen::BaselineCompiledFunction> find_baseline_compiled(std::string_view name) const;
    void clear_baseline_cache();

private:
    MultiTierPipeline() = default;
    ~MultiTierPipeline();

    MultiTierPipeline(const MultiTierPipeline&) = delete;
    MultiTierPipeline& operator=(const MultiTierPipeline&) = delete;

    void tier_invocation(TieringFeedback& fb, std::string_view fn_name, FunctionHandle* handle);
    void setup_fast_interpreter(FastInterpreter& interp, Module& mod);
    // The Tier-0 interpreter kept across execute() calls on one module,
    // rebuilt when the module or the registered symbols change.
    FastInterpreter& persistent_fast_interpreter(Module& mod);

    std::unique_ptr<FastInterpreter> fast_interp_;
    const Module* fast_interp_module_ = nullptr;
    uint64_t fast_interp_symbols_gen_ = 0;
    uint64_t symbols_gen_ = 1;
    std::atomic<bool> fast_interp_busy_{false};

    bool initialized_ = false;
    TieringConfig config_;
    codegen::BaselineJitCompiler baseline_compiler_;
    ModuleStackMap active_stack_maps_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, void*> external_symbols_;
    std::unordered_map<std::string, FastHostFn> external_functions_;
    std::unordered_map<std::string, std::shared_ptr<codegen::BaselineCompiledFunction>> baseline_functions_;
    MultiTierStats stats_;
    std::atomic<uint64_t> tier2_deopts_{0};
    std::atomic<uint64_t> tier2_invalidations_{0};

    mutable std::mutex compiling_mutex_;
    std::unordered_set<std::string> in_progress_compilations_;
    std::unordered_set<std::string> baseline_rejected_; // under compiling_mutex_
};

} // namespace brass::runtime
