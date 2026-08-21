#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/resume_table.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/embedding/host_gc.hpp>

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cstdint>
#include <cstddef>

namespace brass {

struct EngineOptions {
    Target target = Target::host();
    bool enable_optimizations = true;
    bool enable_stress_gc = false;
};

class CompiledModule {
public:
    explicit CompiledModule(std::unique_ptr<codegen::JitExecutionEngine> jit_engine);
    ~CompiledModule() = default;

    CompiledModule(const CompiledModule&) = delete;
    CompiledModule& operator=(const CompiledModule&) = delete;
    CompiledModule(CompiledModule&&) noexcept = default;
    CompiledModule& operator=(CompiledModule&&) noexcept = default;

    // Symbol and function pointer retrieval
    void* get_symbol_address(std::string_view name) const;

    template <typename FuncPtr>
    FuncPtr get_function_ptr(std::string_view name) const {
        return reinterpret_cast<FuncPtr>(get_symbol_address(name));
    }

    // Stack map query
    const ModuleStackMap& stack_maps() const noexcept;
    ModuleStackMap& stack_maps() noexcept;

    // Resume tables & interior resume points
    const runtime::ResumeTableRegistry& resume_tables() const noexcept;
    const runtime::FunctionResumeTable* get_resume_table(std::string_view fn_name) const noexcept;
    void* get_resume_target_address(std::string_view fn_name, uint32_t resume_id) const;

    // Runtime patching
    const runtime::PatchRegistry& patch_sites() const noexcept;
    runtime::PatchRegistry& patch_sites() noexcept;

    bool patch_constant(std::string_view site_name, int32_t new_val);
    bool patch_constant(std::string_view site_name, int64_t new_val);
    bool patch_call(std::string_view site_name, const void* new_target);
    bool patch_call(std::string_view site_name, std::string_view new_target_fn);
    bool patch_call(std::string_view site_name, const char* new_target_fn) {
        return patch_call(site_name, std::string_view(new_target_fn));
    }
    bool patch_call(std::string_view site_name, const std::string& new_target_fn) {
        return patch_call(site_name, std::string_view(new_target_fn));
    }

    // Stack walk helper
    size_t walk_stack(
        uintptr_t rbp,
        uintptr_t return_ip,
        brass_root_visitor_fn visitor,
        void* user_data
    ) const;

    size_t walk_stack(
        uintptr_t rbp,
        uintptr_t return_ip,
        const std::function<void(void**)>& visitor
    ) const;

    // Dynamic invocation helper
    RuntimeValue invoke(std::string_view name, const std::vector<RuntimeValue>& args = {});
    RuntimeValue resume(std::string_view name, uint32_t resume_id, const std::vector<RuntimeValue>& args = {});

    // Access underlying JIT engine
    codegen::JitExecutionEngine* jit_engine() noexcept { return jit_engine_.get(); }
    const codegen::JitExecutionEngine* jit_engine() const noexcept { return jit_engine_.get(); }

private:
    std::unique_ptr<codegen::JitExecutionEngine> jit_engine_;
};

class HostEngine {
public:
    HostEngine();
    explicit HostEngine(const Target& target);
    explicit HostEngine(const EngineOptions& options);
    ~HostEngine() = default;

    HostEngine(const HostEngine&) = delete;
    HostEngine& operator=(const HostEngine&) = delete;
    HostEngine(HostEngine&&) noexcept = default;
    HostEngine& operator=(HostEngine&&) noexcept = default;

    // External host symbols registration
    void register_external_symbol(std::string_view name, void* address);

    // Host GC attachment & automatic bridge registration
    void register_host_gc(HostGC* gc);

    // In-memory compilation of MIR Module
    std::unique_ptr<CompiledModule> compile(const Module& mod);
    std::unique_ptr<CompiledModule> compile(Module& mod);

    const Target& target() const noexcept { return options_.target; }
    const EngineOptions& options() const noexcept { return options_; }

private:
    EngineOptions options_;
    std::unordered_map<std::string, void*> registered_symbols_;
    HostGC* attached_gc_ = nullptr;
};

using EmbeddingEngine = HostEngine;

} // namespace brass
