#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/gc/stack_map.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace brass::codegen {

class JitMemoryBlock {
public:
    JitMemoryBlock() = default;
    explicit JitMemoryBlock(size_t size);
    ~JitMemoryBlock();

    JitMemoryBlock(const JitMemoryBlock&) = delete;
    JitMemoryBlock& operator=(const JitMemoryBlock&) = delete;
    JitMemoryBlock(JitMemoryBlock&& other) noexcept;
    JitMemoryBlock& operator=(JitMemoryBlock&& other) noexcept;

    uint8_t* data() noexcept { return ptr_; }
    const uint8_t* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool is_valid() const noexcept { return ptr_ != nullptr; }

    void make_executable();
    void make_read_write();
    void reset();

private:
    uint8_t* ptr_ = nullptr;
    size_t size_ = 0;
};

class JitExecutionEngine {
public:
    explicit JitExecutionEngine(const Target& target);
    JitExecutionEngine();
    ~JitExecutionEngine();

    JitExecutionEngine(const JitExecutionEngine&) = delete;
    JitExecutionEngine& operator=(const JitExecutionEngine&) = delete;
    JitExecutionEngine(JitExecutionEngine&&) noexcept;
    JitExecutionEngine& operator=(JitExecutionEngine&&) noexcept;

    // Register host external function/symbol
    void register_external_symbol(std::string_view name, void* address);

    // Compilation & loading
    bool compile_and_load(const Module& mod, size_t code_padding = 0);
    bool compile_and_load(const Module& mod, size_t code_padding, const SchedOptions& sched_opts);
    bool load_object(const object::ObjectFile& obj, size_t code_padding = 0);

    const SchedOptions& sched_options() const noexcept { return sched_opts_; }
    void set_sched_options(const SchedOptions& opts) { sched_opts_ = opts; }

    // Function/symbol lookup
    void* get_symbol_address(std::string_view name) const;

    template <typename FuncPtr>
    FuncPtr get_function_ptr(std::string_view name) const {
        return reinterpret_cast<FuncPtr>(get_symbol_address(name));
    }

    // Stack map access
    const ModuleStackMap& stack_maps() const noexcept { return stack_maps_; }
    ModuleStackMap& stack_maps() noexcept { return stack_maps_; }

    // Resume tables & interior resume points
    const runtime::ResumeTableRegistry& resume_tables() const noexcept { return resume_tables_; }
    const runtime::FunctionResumeTable* get_resume_table(std::string_view fn_name) const noexcept;
    void* get_resume_target_address(std::string_view fn_name, uint32_t resume_id) const;

    // Exception tables
    const runtime::ExceptionTableRegistry& exception_tables() const noexcept { return exception_tables_; }

    // Runtime patching API
    const runtime::PatchRegistry& patch_sites() const noexcept { return patch_sites_; }
    runtime::PatchRegistry& patch_sites() noexcept { return patch_sites_; }
    bool patch_const32(std::string_view site_name, int32_t new_val);
    bool patch_const64(std::string_view site_name, int64_t new_val);
    bool patch_call(std::string_view site_name, const void* new_target);
    bool patch_call(std::string_view site_name, std::string_view new_target_fn);
    bool patch_call(std::string_view site_name, const char* new_target_fn) {
        return patch_call(site_name, std::string_view(new_target_fn));
    }
    bool patch_call(std::string_view site_name, const std::string& new_target_fn) {
        return patch_call(site_name, std::string_view(new_target_fn));
    }

    // Dynamic invocation helper using RuntimeValue
    RuntimeValue invoke(std::string_view name, const std::vector<RuntimeValue>& args);
    RuntimeValue invoke(std::string_view name);
    RuntimeValue resume(std::string_view name, uint32_t resume_id);
    RuntimeValue resume(std::string_view name, uint32_t resume_id, const std::vector<RuntimeValue>& args);

private:
    Target target_;
    JitMemoryBlock code_mem_;

    std::unordered_map<std::string, void*> symbol_table_;
    std::unordered_map<std::string, void*> external_symbols_;
    std::unordered_map<std::string, std::pair<Type, std::vector<Type>>> function_signatures_;
    ModuleStackMap stack_maps_;
    runtime::ResumeTableRegistry resume_tables_;
    runtime::PatchRegistry patch_sites_;
    runtime::ExceptionTableRegistry exception_tables_;
    std::vector<uintptr_t> registered_exception_fns_;
    uint8_t* text_section_base_ = nullptr;
    // Windows SEH registration tracking
    void* pdata_table_ = nullptr;
    size_t pdata_count_ = 0;
    uintptr_t code_base_ = 0;
    SchedOptions sched_opts_;

    void register_seh_tables(const object::ObjectFile& obj, uint8_t* base_ptr);
    void unregister_seh_tables();
};

} // namespace brass::codegen
