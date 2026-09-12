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

private:
    std::string name_;
    const Function* mir_function_ = nullptr;
    std::atomic<void*> native_entry_{nullptr};
    std::atomic<TierLevel> tier_{TierLevel::Tier0_Interpreter};
    std::atomic<uint64_t> invocation_count_{0};

    mutable std::mutex engine_mutex_;
    std::shared_ptr<codegen::JitExecutionEngine> jit_engine_;
    std::shared_ptr<codegen::BaselineCompiledFunction> baseline_function_;

    Type return_type_ = Type::void_type();
    std::vector<Type> param_types_;
};

class FunctionDispatchTable {
public:
    static FunctionDispatchTable& instance();

    FunctionHandle* get_or_create(std::string_view name, const Function* fn = nullptr);
    FunctionHandle* find(std::string_view name) const;
    bool has(std::string_view name) const;
    void register_handle(std::unique_ptr<FunctionHandle> handle);
    void clear();

    size_t size() const;
    std::vector<FunctionHandle*> all_handles() const;

private:
    FunctionDispatchTable() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<FunctionHandle>> handles_;
};

struct CodeInstallResult {
    bool success = false;
    void* entry_point = nullptr;
    std::string error_message;
    size_t code_size = 0;
};

class CodeInstaller {
public:
    explicit CodeInstaller(const Target& target = Target::host());
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
    mutable std::mutex symbols_mutex_;
    std::unordered_map<std::string, void*> external_symbols_;
};

} // namespace brass::runtime
