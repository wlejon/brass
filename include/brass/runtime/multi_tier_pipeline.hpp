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
#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace brass::runtime {

namespace detail {
enum class Tier1Link;
struct Tier1Node;
} // namespace detail

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

// The tiering driver of one program. instance() is the default program's;
// an owned FunctionDispatchTable owns its own (FunctionDispatchTable::
// pipeline()), which counts into that table's TieringRegistry, compiles
// into its handles (with its own background compiler), and keeps its own
// baseline code, tier-2 stack maps and Tier-0 interpreter.
class MultiTierPipeline {
public:
    static MultiTierPipeline& instance();
    // The pipeline of an owned program; `table` owns it.
    explicit MultiTierPipeline(FunctionDispatchTable& table);
    ~MultiTierPipeline();

    bool is_default() const noexcept { return is_default_; }
    FunctionDispatchTable& dispatch_table() const noexcept { return *table_; }
    TieringRegistry& tiering() const noexcept;

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

    // The default program's is BackgroundCompiler::instance(); an owned
    // program's is its own, created on first use and stopped with it.
    BackgroundCompiler& background_compiler();

    // This program's code stack maps (baseline and tier-2). While the
    // pipeline is initialized they are the GC's active maps; destroying the
    // program drops them (and the GC's pointer to them).
    ModuleStackMap& active_stack_maps() noexcept { return active_stack_maps_; }
    const ModuleStackMap& active_stack_maps() const noexcept { return active_stack_maps_; }
    void add_stack_maps(const ModuleStackMap& maps);

    // Called by an owned table's destructor before it retires its handles:
    // stops and drains this program's background compiler, and drops its
    // baseline code, stack maps and Tier-0 interpreter.
    void release_program();

    // Synchronous Tier 1 compilation (< 5 microseconds on mutator thread).
    // Returns false, leaving the function in the interpreter, when the
    // baseline compiler rejects it (codegen::UnsupportedOperation); a rejected
    // function is not retried until clear_baseline_cache(). The module
    // functions it calls that have no native entry yet are compiled first:
    // when one of them is rejected, so is this function, since its code
    // could not call it.
    //
    // A call cycle among such callees is installed as a whole, once every
    // function of it compiles: when one of them is rejected, none is.
    // Returns false too while a function it needs is being compiled on
    // another thread; tiering then asks again later.
    bool compile_and_install_tier1(std::string_view fn_name, const Function* fn = nullptr);
    bool is_baseline_rejected(std::string_view fn_name) const;

    // The program's one function-pointer representation: the address every
    // tier's func_addr of program function `name` (MIR `fn`) yields, and
    // what type feedback keys call targets on. It is the function's lazy
    // stub, callable from native code in any tier: the first call through
    // it compiles the function to Tier 1, or, when the baseline tier
    // rejects it, binds a native-to-Tier-0 bridge (tier0_bridge.cpp).
    // Registered in the dispatch table, so Tier 0 maps it back.
    void* function_address(std::string_view name, const Function* fn);
    // Whether a native-to-Tier-0 bridge can call `fn`: at most
    // kTier0BridgeMaxParams parameters, each and the result an integer,
    // pointer or f64 (or a void result). `why` says why not.
    static constexpr size_t kTier0BridgeMaxParams = 8;
    static bool tier0_bridge_supported(const Function& fn, std::string* why = nullptr);
    // Called by a bridge: runs `name` in Tier 0 with the raw argument bits,
    // returning the result's bits. It re-enters the interpreter active on
    // this thread when that one runs this program (a fresh one otherwise),
    // and an exception it throws unwinds through the native callers.
    uint64_t call_tier0_from_native(std::string_view name, const uint64_t* bits, size_t count);

    // Testing only: called with each function's name as a Tier 1 compile of
    // it starts (on the compiling thread, while it counts as in progress).
    // Set before compiles start.
    void set_tier1_compile_hook(std::function<void(std::string_view)> hook) {
        tier1_compile_hook_ = std::move(hook);
    }
    // Testing only: called with each function's name as its Tier 1 code is
    // installed, after the code is registered (find_baseline_compiled) and
    // its stubs filled, before its native entry is published. Set before
    // compiles start.
    void set_tier1_install_hook(std::function<void(std::string_view)> hook) {
        tier1_install_hook_ = std::move(hook);
    }

    // Enqueue Tier 2 background optimization
    bool enqueue_tier2(
        std::string_view fn_name,
        const Module* mod = nullptr,
        FunctionHandle* handle = nullptr
    );
    // Tier 2 on the calling thread (tier-up with background compile off):
    // compiles `fn_name` from the active module and installs it over its
    // tier-1 code. False if it is not a tier-2 candidate, is being compiled
    // already, or tier 2 rejected it (it then stays on its lower tier and
    // is not tried again). Never throws.
    bool compile_tier2_now(std::string_view fn_name, FunctionHandle* handle = nullptr);

    // Mutator notification on function invocation. The feedback overload
    // is the hot path (tier-1 code passes the pointer it was compiled
    // with); the name overload looks the function up first.
    void on_invocation(std::string_view fn_name);
    void on_invocation(TieringFeedback& fb);

    // Called by ~Module: drops the Tier-0 interpreter kept for `mod` (the
    // default program's; forget() is one pipeline's part).
    static void forget_module(const Module* mod) noexcept;
    void forget(const Module* mod) noexcept;

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
    // The Tier-0 interpreter routes its calls through `table`, the program
    // the handle belongs to.
    uint64_t resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame);
    uint64_t resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame, FunctionDispatchTable& table);
    // As above, for code compiled from `compiled_from`, which the handle
    // may since have been rebound away from (FunctionHandle::
    // rebind_mir_function): the call finishes in `compiled_from`.
    uint64_t resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame, FunctionDispatchTable& table,
                                const Function& compiled_from);
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

    MultiTierPipeline(const MultiTierPipeline&) = delete;
    MultiTierPipeline& operator=(const MultiTierPipeline&) = delete;

private:
    struct DefaultTag {};
    explicit MultiTierPipeline(DefaultTag);

    void tier_invocation(TieringFeedback& fb, std::string_view fn_name, FunctionHandle* handle);
    // Tier-1 code reaches a module function with no native entry through a
    // lazy stub that resolves to the callee's native entry. Claims `name`
    // (in progress) and compiles it into a node appended to `nodes`, whose
    // callees are those functions (std::nullopt); or returns why it has
    // nothing to compile: Ready when its native entry is published,
    // Pending or Rejected.
    std::optional<detail::Tier1Link> open_tier1_node(std::string_view name, const Function* fn,
                                                     std::deque<detail::Tier1Node>& nodes);
    // A call cycle's functions are installed together once all of them
    // and everything they reach compile (install), or none is (release).
    // Their claims are released only after every entry is published.
    void install_tier1_group(const std::vector<detail::Tier1Node*>& group);
    void release_tier1_claims(const std::vector<detail::Tier1Node*>& group, bool rejected);
    // A func_addr target's lazy stub, first called with no native entry:
    // compiles and installs it (waiting out another thread's compile of
    // it), returning its entry, or null when the baseline tier rejects it.
    void* compile_tier1_on_demand(std::string_view name);
    // The native entry of `name`'s native-to-Tier-0 bridge, built on first
    // use; null (after reporting why) when its signature has none.
    void* tier0_bridge(std::string_view name);
    // Runs `fn` (or, with `resume_fn`, resumes it at `resume_id`) in a
    // fresh Tier-0 interpreter of this pipeline's kind routing through `table`.
    RuntimeValue run_fresh_tier0(FunctionDispatchTable& table, const Function* fn,
                                 const std::vector<RuntimeValue>& args, const Function* resume_fn = nullptr,
                                 uint32_t resume_id = 0);
    void setup_fast_interpreter(FastInterpreter& interp, Module& mod);
    // The Tier-0 interpreter kept across execute() calls on one module,
    // rebuilt when the module or the registered symbols change.
    FastInterpreter& persistent_fast_interpreter(Module& mod);

    FunctionDispatchTable* const table_;
    const bool is_default_;
    std::mutex bg_mutex_;
    std::unique_ptr<BackgroundCompiler> bg_; // owned programs, under bg_mutex_

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
    std::unordered_set<std::string> tier2_in_progress_; // under compiling_mutex_
    // Native-to-Tier-0 bridges by function name (tier0_bridge.cpp); kept
    // for the pipeline's lifetime, since lazy stubs point into them.
    std::mutex bridges_mutex_;
    std::unordered_map<std::string, std::shared_ptr<void>> tier0_bridges_;
    // Testing only (set_tier1_compile_hook, set_tier1_install_hook).
    std::function<void(std::string_view)> tier1_compile_hook_;
    std::function<void(std::string_view)> tier1_install_hook_;
};

} // namespace brass::runtime
