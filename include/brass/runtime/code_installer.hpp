#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/runtime/tiering.hpp>
#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace brass {
class Interpreter;
class FastInterpreter;
namespace codegen {
class JitExecutionEngine;
class BaselineCompiledFunction;
}
}

namespace brass::runtime {

class FunctionHandle {
public:
    explicit FunctionHandle(std::string_view name, const Function* mir_fn = nullptr);
    ~FunctionHandle() = default;

    FunctionHandle(const FunctionHandle&) = delete;
    FunctionHandle& operator=(const FunctionHandle&) = delete;

    std::string_view name() const noexcept { return name_; }

    const Function* mir_function() const noexcept { return mir_function_; }
    void set_mir_function(const Function* fn) noexcept;

    // Atomic acquire/release native entry point swapping
    void* native_entry() const noexcept {
        return native_entry_.load(std::memory_order_acquire);
    }

    void set_native_entry(void* ptr) noexcept {
        native_entry_.store(ptr, std::memory_order_release);
    }

    TierLevel tier() const noexcept {
        return tier_.load(std::memory_order_acquire);
    }

    void set_tier(TierLevel t) noexcept {
        tier_.store(t, std::memory_order_release);
    }

    bool has_native_entry() const noexcept {
        return native_entry() != nullptr;
    }

    const std::atomic<void*>* native_entry_ptr() const noexcept {
        return &native_entry_;
    }

    uint64_t invocation_count() const noexcept {
        return invocation_count_.load(std::memory_order_relaxed);
    }

    const std::atomic<uint64_t>* invocation_count_ptr() const noexcept {
        return &invocation_count_;
    }

    uint64_t record_call() noexcept {
        return ++invocation_count_;
    }

    void set_jit_engine(std::shared_ptr<codegen::JitExecutionEngine> engine);
    std::shared_ptr<codegen::JitExecutionEngine> jit_engine() const;

    void set_baseline_function(std::shared_ptr<codegen::BaselineCompiledFunction> compiled);
    std::shared_ptr<codegen::BaselineCompiledFunction> baseline_function() const;

    Type return_type() const noexcept { return return_type_; }
    const std::vector<Type>& param_types() const noexcept { return param_types_; }
    void set_signature(Type ret, std::vector<Type> params);

    template <typename FuncPtr>
    FuncPtr get_function_ptr() const noexcept {
        return reinterpret_cast<FuncPtr>(native_entry());
    }

    // Dynamic invocation
    RuntimeValue call_native(const std::vector<RuntimeValue>& args = {}) const;
    RuntimeValue call(Interpreter& interp, const std::vector<RuntimeValue>& args = {});
    RuntimeValue call(FastInterpreter& interp, const std::vector<RuntimeValue>& args = {});

    // Called when the dispatch table drops this handle: clears the entry
    // point and MIR link and releases the compiled code's owners. The
    // object itself stays alive for callers that cached a pointer to it.
    void retire() noexcept;

    // Drops the optimized (tier-2) code after its speculation failed too
    // often: the entry falls back to the baseline code, or to the
    // interpreter. The engine is kept alive (frames of it may still be on
    // some stack, including the caller's). No-op without tier-2 code.
    void invalidate_optimized();
    size_t retired_engine_count() const;

    // Tier-2 entry points whose deopt resumer this handle registered.
    void add_deopt_entry(void* entry);

private:
    std::string name_;
    const Function* mir_function_ = nullptr;
    std::atomic<void*> native_entry_{nullptr};
    std::atomic<TierLevel> tier_{TierLevel::Tier0_Interpreter};
    std::atomic<uint64_t> invocation_count_{0};

    mutable std::mutex engine_mutex_;
    std::shared_ptr<codegen::JitExecutionEngine> jit_engine_;
    std::shared_ptr<codegen::BaselineCompiledFunction> baseline_function_;
    std::vector<std::shared_ptr<codegen::JitExecutionEngine>> retired_engines_;
    std::vector<void*> deopt_entries_;

    Type return_type_ = Type::void_type();
    std::vector<Type> param_types_;
};

// The function handles of one program, keyed by name within it.
//
// A default-constructed table is an owned, per-program table: two programs
// may each have a function of the same name, and each routes calls to its
// own. The table owns its handles and, through them, their compiled code
// (baseline code, tier-2 engines, deopt resumers): destroying it retires
// every handle, frees the code and moves registry_generation(), so
// interpreters re-resolve instead of reusing a pointer into it. It must
// outlive the modules its handles point into being destroyed (destroying a
// module a live owned table still routes to is a fatal error), and it must
// outlive every interpreter, compiler and installer given it.
//
// A program's whole tiering state hangs off its table: tiering() holds its
// feedback (invocation, backedge and deopt counts, bailouts) and pipeline()
// its MultiTierPipeline (tier-up, the persistent Tier-0 interpreter, baseline
// code, tier-2 stack maps, and its own background compiler). Destroying an
// owned table first stops its background compiler (queued compiles are
// dropped, an in-flight one finishes into the still-live handles), then
// drops its stack maps and code, then its feedback.
//
// instance() is the default program: the process-wide table everything uses
// unless given another. Its tiering() and pipeline() are
// TieringRegistry::instance() and MultiTierPipeline::instance(). It keeps
// the old stopgap: ~Module detaches its handles from the dying module's
// functions.
class FunctionDispatchTable {
public:
    FunctionDispatchTable();
    ~FunctionDispatchTable();

    FunctionDispatchTable(const FunctionDispatchTable&) = delete;
    FunctionDispatchTable& operator=(const FunctionDispatchTable&) = delete;

    static FunctionDispatchTable& instance();
    bool is_default() const noexcept { return is_default_; }

    TieringRegistry& tiering() const noexcept;
    MultiTierPipeline& pipeline() const noexcept;

    FunctionHandle* get_or_create(std::string_view name, const Function* fn = nullptr);
    FunctionHandle* find(std::string_view name) const;
    bool has(std::string_view name) const;
    void register_handle(std::unique_ptr<FunctionHandle> handle);
    void clear();

    // Default program only. Handles are keyed by name and outlive the MIR
    // they were registered with; a later Function allocated at a dead one's
    // address would otherwise match `mir_function() == &fn` and be routed to
    // the dead function's native code. Detaches every handle from `mod`'s
    // functions (native entries stay: other code may still call them by name).
    void forget_module(const Module& mod);

    size_t size() const;
    std::vector<FunctionHandle*> all_handles() const;

private:
    struct DefaultTag {};
    explicit FunctionDispatchTable(DefaultTag);
    friend void forget_module(const Module& mod) noexcept;
    // Name of a live handle routing into one of `mod`'s functions, or empty.
    std::string handle_into(const Module& mod) const;
    void retire_locked(std::unique_ptr<FunctionHandle> handle);

    const bool is_default_ = false;
    // Owned programs only (declared in this order so the pipeline, which
    // points at the registry, is destroyed first).
    std::unique_ptr<TieringRegistry> tiering_;
    std::unique_ptr<MultiTierPipeline> pipeline_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<FunctionHandle>> handles_;
    // Handles dropped by clear() or replaced by register_handle(); callers
    // may hold pointers resolved before registry_generation() moved.
    std::vector<std::unique_ptr<FunctionHandle>> retired_;
};

using DispatchTable = FunctionDispatchTable;

// Called by ~Module: drops the default program's raw pointers into `mod`
// (dispatch-table handles, the tiering registry's active module, the
// pipeline's persistent interpreter) so none of them can dangle or alias a
// later module allocated at the same address. A live owned table that still
// routes a function to `mod` is a fatal error.
void forget_module(const Module& mod) noexcept;

// Whether every guard of `optimized` can deoptimize into `tier0` (the
// function the lower tiers run): a guard with the same resume id and state
// map size, whose exit stub names a function of tier0's module or whose
// resume id has a resume target — the exits the interpreter takes. On
// false, `why` names the first guard that cannot.
bool deopt_targets_valid(const Function& optimized, const Function* tier0, std::string& why);

struct CodeInstallResult {
    bool success = false;
    void* entry_point = nullptr;
    std::string error_message;
    size_t code_size = 0;
};

// Installs tier-2 code into handles of one dispatch table (the default
// program unless given another, which must outlive the installer): other
// functions of the compiled module are published to that table's handles,
// and the deopt continuation resumes in an interpreter routed through it.
class CodeInstaller {
public:
    explicit CodeInstaller(const Target& target = Target::host());
    explicit CodeInstaller(FunctionDispatchTable& table, const Target& target = Target::host());

    FunctionDispatchTable& dispatch_table() const noexcept { return *table_; }
    ~CodeInstaller() = default;

    CodeInstallResult install_tier2(
        FunctionHandle& handle,
        const Module& module,
        std::string_view fn_name
    );

    CodeInstallResult install_tier2(
        FunctionHandle& handle,
        std::unique_ptr<Module> module,
        std::string_view fn_name
    );

    void register_external_symbol(std::string_view name, void* address);

    const Target& target() const noexcept { return target_; }

private:
    Target target_;
    FunctionDispatchTable* table_;
    mutable std::mutex symbols_mutex_;
    std::unordered_map<std::string, void*> external_symbols_;
};

} // namespace brass::runtime
