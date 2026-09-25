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
#include <unordered_set>
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

class OsrCoordinator;

// Calls the native code at `addr` from C++ with the platform's calling
// convention: `args` passed as `ptypes` (null: each argument's own kind
// decides), the result read as `ret_type`. What an interpreter uses to call
// native code it has no function for, and FunctionHandle::call_native.
// Throws on a host without an invoke thunk.
RuntimeValue invoke_native_address(void* addr, Type ret_type, const std::vector<Type>* ptypes,
                                   const std::vector<RuntimeValue>& args);

class FunctionHandle {
public:
    explicit FunctionHandle(std::string_view name, const Function* mir_fn = nullptr);
    ~FunctionHandle() = default;

    FunctionHandle(const FunctionHandle&) = delete;
    FunctionHandle& operator=(const FunctionHandle&) = delete;

    std::string_view name() const noexcept { return name_; }

    const Function* mir_function() const noexcept { return mir_function_.load(std::memory_order_acquire); }
    void set_mir_function(const Function* fn) noexcept;
    // Detaches the handle from its function, which is being destroyed
    // (FunctionDispatchTable::forget_module). Its native code stays callable
    // but no longer implements any live Function.
    void detach_mir_function() noexcept;
    bool mir_detached() const noexcept { return mir_detached_; }
    // Binds the handle to `fn`, a different Function than the one its
    // native code was compiled from: that code does not implement `fn`, so
    // it is retired (kept alive, as invalidate_optimized does: native
    // callers may be bound to it, and frames of it may be on some stack)
    // and the handle drops back to Tier 0. Returns whether any code was
    // retired.
    bool rebind_mir_function(const Function* fn);

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

    // Installs `engine` as the handle's tier-2 code owner. An engine it
    // replaces is retired (kept alive with the handle, as
    // invalidate_optimized does): native callers may have bound its entry.
    void set_jit_engine(std::shared_ptr<codegen::JitExecutionEngine> engine);
    std::shared_ptr<codegen::JitExecutionEngine> jit_engine() const;

    // Atomically publishes tier-2 code at `entry`, owned by `engine`, that
    // was compiled while the handle was bound to `compiled_from`: engine,
    // signature, entry and tier change together under the handle's lock.
    // Refused (returns false; `engine` is retired, not freed, since its
    // deopt resumer is already registered) when the handle was rebound,
    // detached or retired since, or, with `require_no_entry`, has native
    // code. Code it replaces (tier-1 or older tier-2) is retired as
    // set_jit_engine does: its callers may be bound to it.
    bool publish_optimized(std::shared_ptr<codegen::JitExecutionEngine> engine, void* entry,
                           const Function* compiled_from, Type ret, std::vector<Type> params,
                           bool require_no_entry);
    // Tier 2 rejected the handle's current Function (UnsupportedOperation,
    // a failed pass, a guard with no deopt target): automatic tier-up does
    // not try it again until the handle is bound to another Function.
    void mark_tier2_rejected();
    // Tier 2 rejected `fn`, the Function a compile was taken from: the
    // handle stays eligible if it has since been bound to another one.
    void mark_tier2_rejected(const Function* fn);
    bool tier2_rejected() const;

    void set_baseline_function(std::shared_ptr<codegen::BaselineCompiledFunction> compiled);
    std::shared_ptr<codegen::BaselineCompiledFunction> baseline_function() const;

    // The signature of the code the handle routes to. It is an immutable
    // snapshot, replaced (never modified) under the handle's lock together
    // with the entry point it describes: a call reads both at once and
    // finishes with the types of the code it entered, however the handle
    // is rebound meanwhile.
    struct Signature {
        Type ret = Type::void_type();
        std::vector<Type> params;
    };
    std::shared_ptr<const Signature> signature() const;
    Type return_type() const { return signature()->ret; }
    std::vector<Type> param_types() const { return signature()->params; }
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

    // Tier-2 entry points whose deopt resumer this handle registered, each
    // with the Function it was compiled from (the one its frames resume
    // in). Rebinding keeps them registered: the code stays alive for frames
    // still running it.
    void add_deopt_entry(void* entry, const Function* compiled_from);
    // The Function the tier-2 code at `entry` was compiled from, or null if
    // its module has been destroyed (forget_deopt_functions) or `entry` is
    // not this handle's.
    const Function* deopt_function(void* entry) const;
    // Forgets every deopt entry's Function in `dying` (a module being
    // destroyed): a later deopt of that code is a fatal error.
    void forget_deopt_functions(const std::unordered_set<const Function*>& dying);

private:
    void set_mir_function_locked(const Function* fn);
    std::string name_;
    std::atomic<const Function*> mir_function_{nullptr};
    std::atomic<void*> native_entry_{nullptr};
    std::atomic<TierLevel> tier_{TierLevel::Tier0_Interpreter};
    std::atomic<uint64_t> invocation_count_{0};

    mutable std::mutex engine_mutex_;
    std::shared_ptr<codegen::JitExecutionEngine> jit_engine_;
    std::shared_ptr<codegen::BaselineCompiledFunction> baseline_function_;
    std::vector<std::shared_ptr<codegen::JitExecutionEngine>> retired_engines_;
    // Baseline code of functions the handle was rebound away from.
    std::vector<std::shared_ptr<codegen::BaselineCompiledFunction>> retired_baselines_;
    struct DeoptEntry {
        void* entry;
        const Function* compiled_from;
    };
    std::vector<DeoptEntry> deopt_entries_;
    bool mir_detached_ = false;
    bool retired_ = false;               // under engine_mutex_
    bool tier2_rejected_ = false;        // under engine_mutex_
    const Function* tier2_rejected_fn_ = nullptr; // under engine_mutex_

    std::shared_ptr<const Signature> sig_ = std::make_shared<const Signature>(); // under engine_mutex_
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
// feedback (invocation, backedge and deopt counts, bailouts, and through
// tiering().type_feedback() its call-target and shape feedback), osr() its
// OsrCoordinator (OSR stubs and enablement), and pipeline()
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
    // The program's on-stack replacement (OsrCoordinator::instance() for
    // the default program).
    OsrCoordinator& osr() const noexcept;

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

    // A program function pointer is a code address: canonically the
    // function's lazy stub (MultiTierPipeline::function_address), which every
    // tier's func_addr yields, or an address tier-2 code took of its own
    // copy. Each is registered here with the function's name, so Tier 0 can
    // map a pointer native code made back to the function it calls.
    void register_code_address(const void* addr, std::string_view name);
    // The function registered at `addr`, or "" if none.
    std::string function_name_at(const void* addr) const;

private:
    struct DefaultTag {};
    explicit FunctionDispatchTable(DefaultTag);
    friend void forget_module(const Module& mod) noexcept;
    // Name of a live handle routing into one of `mod`'s functions, or empty.
    std::string handle_into(const Module& mod) const;
    // Every handle forgets the Functions of `mod` its retained tier-2 code
    // would resume in (FunctionHandle::forget_deopt_functions).
    void forget_deopt_functions(const Module& mod);
    void retire_locked(std::unique_ptr<FunctionHandle> handle);

    const bool is_default_ = false;
    // Owned programs only (declared in this order so the pipeline, which
    // points at the registry, is destroyed first).
    std::unique_ptr<TieringRegistry> tiering_;
    std::unique_ptr<MultiTierPipeline> pipeline_;
    // Holds OSR stubs counting into tiering_, so it is destroyed before it.
    std::unique_ptr<OsrCoordinator> osr_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<FunctionHandle>> handles_;
    // Handles dropped by clear() or replaced by register_handle(); callers
    // may hold pointers resolved before registry_generation() moved.
    std::vector<std::unique_ptr<FunctionHandle>> retired_;
    // Keyed by address, so re-registering one (every func_addr of a stub)
    // adds nothing; one entry per stub plus one per function of each tier-2
    // engine installed, retired ones included (they stay alive, see
    // FunctionHandle::set_jit_engine). clear() empties it.
    std::unordered_map<const void*, std::string> code_addresses_; // under mutex_
};

using DispatchTable = FunctionDispatchTable;

// The program this thread is running: the default program unless a
// ProgramScope is open. MultiTierPipeline::execute and the interpreters' run
// entry points open one for their table, so runtime helpers that native code
// calls with only a name (the IL runtime's function resolver) reach the
// running program's handles.
FunctionDispatchTable& current_program() noexcept;

// Makes `table` this thread's current_program() until destroyed (scopes
// nest; the previous program is restored). `table` must outlive the scope.
class ProgramScope {
public:
    explicit ProgramScope(FunctionDispatchTable& table) noexcept;
    ~ProgramScope();
    ProgramScope(const ProgramScope&) = delete;
    ProgramScope& operator=(const ProgramScope&) = delete;

private:
    FunctionDispatchTable* prev_;
};

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

// The Functions the handles were bound to when a module was cloned for a
// tier-2 compile. The code is validated and published against these, and its
// frames deoptimize into them, never into whatever a handle is bound to when
// the compile finishes: a handle rebound in between is not published to.
struct Tier2Bindings {
    const Function* target = nullptr;
    // The handle was bound to a Function other than the compiled module's
    // one of that name: its code would run and deoptimize as another
    // function, so the install is refused before compiling (and the handle
    // is not marked rejected, its own Function was never tried).
    bool target_foreign = false;
    // Other functions of the module whose handles may receive the compiled
    // code, by name. A name absent here is never published.
    std::unordered_map<std::string, const Function*> siblings;
};

// The module a tier-2 compile of `fn_name` for `table`'s program works on: in
// a program whose pipeline runs it, the function alone
// (clone_function_module), its other functions linked through their stubs;
// otherwise a copy of the whole module.
std::unique_ptr<Module> clone_for_tier2(FunctionDispatchTable& table, const Module& module, std::string_view fn_name);

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

    // `module` is a clone the caller just took: the handles' bindings are
    // captured now, with no source module to check the bindings against.
    // A bound handle is refused only when its Function's name or module
    // name differs from the clone's: a handle bound to another module of
    // the same name is not caught, so prefer the overload taking the source.
    CodeInstallResult install_tier2(
        FunctionHandle& handle,
        std::unique_ptr<Module> module,
        std::string_view fn_name
    );

    // Compiles `module` (a clone) against the bindings captured when it was
    // taken (capture_tier2_bindings).
    CodeInstallResult install_tier2(
        FunctionHandle& handle,
        std::unique_ptr<Module> module,
        std::string_view fn_name,
        const Tier2Bindings& bindings
    );

    // Records what `handle` and the handles of `module`'s other functions
    // are bound to now. With `source` (the module being cloned), a sibling
    // is recorded only if its handle is bound to source's Function of that
    // name: code compiled from one Function is never published for another.
    // A bound `handle` whose Function is not source's of `fn_name` (without
    // source: whose name or module name is not the clone's) is recorded as
    // target_foreign, and install_tier2 then refuses it. An unbound handle
    // is never foreign.
    Tier2Bindings capture_tier2_bindings(const FunctionHandle& handle, const Module& module,
                                         std::string_view fn_name, const Module* source) const;

    void register_external_symbol(std::string_view name, void* address);

    const Target& target() const noexcept { return target_; }

private:
    Target target_;
    FunctionDispatchTable* table_;
    mutable std::mutex symbols_mutex_;
    std::unordered_map<std::string, void*> external_symbols_;
};

} // namespace brass::runtime
