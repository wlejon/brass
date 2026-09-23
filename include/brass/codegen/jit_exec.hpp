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

namespace brass::runtime {
class MultiTierPipeline;
}

namespace brass::codegen {

class JitMemoryBlock {
public:
    JitMemoryBlock() = default;
    explicit JitMemoryBlock(size_t size);
    // One mapping holding the code pages first and `data_size` bytes of data
    // pages after them, so that every image-relative 32-bit offset in the
    // loaded object (.pdata's function and unwind-info RVAs, Addr32NB
    // relocations) is a small positive distance from the code base. Two
    // separate mappings can land in either order and further apart than an
    // RVA can express. Where code must be its own MAP_JIT mapping (Apple
    // Silicon) only the code pages are allocated, and size() says so.
    JitMemoryBlock(size_t code_size, size_t data_size);
    ~JitMemoryBlock();

    JitMemoryBlock(const JitMemoryBlock&) = delete;
    JitMemoryBlock& operator=(const JitMemoryBlock&) = delete;
    JitMemoryBlock(JitMemoryBlock&& other) noexcept;
    JitMemoryBlock& operator=(JitMemoryBlock&& other) noexcept;

    uint8_t* data() noexcept { return ptr_; }
    const uint8_t* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool is_valid() const noexcept { return ptr_ != nullptr; }

    // W^X: a block is allocated read-write; once the code is in place, the
    // first `code_size` bytes (rounded up to pages; 0 = the whole block) are
    // turned read-execute and registered as JIT code. There is no call that
    // makes a page writable and executable at once. False if the OS refused.
    [[nodiscard]] bool make_executable_read_only(size_t code_size = 0);
    // Back to read-write (not executable) for rewriting the whole block.
    [[nodiscard]] bool make_read_write();
    void reset();

private:
    uint8_t* ptr_ = nullptr;
    size_t size_ = 0;
};

class DataMemoryBlock {
public:
    DataMemoryBlock() = default;
    explicit DataMemoryBlock(size_t size, void* address_hint = nullptr);
    ~DataMemoryBlock();

    DataMemoryBlock(const DataMemoryBlock&) = delete;
    DataMemoryBlock& operator=(const DataMemoryBlock&) = delete;
    DataMemoryBlock(DataMemoryBlock&& other) noexcept;
    DataMemoryBlock& operator=(DataMemoryBlock&& other) noexcept;

    uint8_t* data() noexcept { return ptr_; }
    const uint8_t* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool is_valid() const noexcept { return ptr_ != nullptr; }
    void reset();

private:
    uint8_t* ptr_ = nullptr;
    size_t size_ = 0;
};

bool is_jit_code_address(const void* addr) noexcept;
size_t jit_system_page_size() noexcept;

struct alignas(16) AArch64InvokeArgs {
    uint64_t x[8] = {0};
    alignas(16) uint8_t v[8][16] = {{0}};
    const uint64_t* stack_words = nullptr;
    uint64_t stack_word_count = 0;
    void* target_fn = nullptr;
};

struct alignas(16) AArch64InvokeResult {
    uint64_t x0 = 0;
    uint64_t x1 = 0;
    alignas(16) uint8_t q0[16] = {0};
};

void partition_aarch64_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    AArch64InvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
);

struct alignas(16) X64SysVInvokeArgs {
    uint64_t gpr[6] = {0};                 // RDI, RSI, RDX, RCX, R8, R9
    alignas(16) uint8_t xmm[8][16] = {{0}};// XMM0..XMM7
    const uint64_t* stack_words = nullptr;
    uint64_t stack_word_count = 0;
    void* target_fn = nullptr;
};

struct alignas(16) X64SysVInvokeResult {
    uint64_t rax = 0;
    uint64_t rdx = 0;
    alignas(16) uint8_t xmm0[16] = {0};
};

struct alignas(16) X64Win64InvokeArgs {
    uint64_t gpr[4] = {0};                 // RCX, RDX, R8, R9
    alignas(16) uint8_t xmm[4][16] = {{0}};// XMM0..XMM3
    const uint64_t* stack_words = nullptr;
    uint64_t stack_word_count = 0;
    void* target_fn = nullptr;
};

struct alignas(16) X64Win64InvokeResult {
    uint64_t rax = 0;
    // Explicit, so the 16-byte slot below needs no compiler padding (MSVC
    // warns C4324 on implicit alignment padding in consumers' /W4 builds).
    uint64_t reserved = 0;
    alignas(16) uint8_t xmm0[16] = {0};
};

void partition_x64_sysv_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    X64SysVInvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
);

void partition_x64_win64_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    X64Win64InvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
);

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
extern "C" void x64_sysv_invoke_thunk(
    const X64SysVInvokeArgs* args,
    X64SysVInvokeResult* result
);
#endif
extern "C" void x64_win64_invoke_thunk(
    const X64Win64InvokeArgs* args,
    X64Win64InvokeResult* result
);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__GNUC__) || defined(__clang__)
extern "C" void aarch64_invoke_thunk(
    const AArch64InvokeArgs* args,
    AArch64InvokeResult* result
);
#endif
#endif

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
    void register_function_signature(std::string_view name, Type ret_type, std::vector<Type> param_types = {});

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

    // OSR entry offsets and addresses
    size_t get_osr_entry_offset(std::string_view fn_name) const;
    void* get_osr_entry_address(std::string_view fn_name) const;

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

    // Multi-tier pipeline integration
    runtime::MultiTierPipeline* multi_tier_pipeline() noexcept;
    const runtime::MultiTierPipeline* multi_tier_pipeline() const noexcept;
    void set_multi_tier_enabled(bool enabled);
    bool is_multi_tier_enabled() const noexcept;

    // Dynamic invocation helper using RuntimeValue
    RuntimeValue invoke(std::string_view name, const std::vector<RuntimeValue>& args);
    RuntimeValue invoke(std::string_view name);
    RuntimeValue resume(std::string_view name, uint32_t resume_id);
    RuntimeValue resume(std::string_view name, uint32_t resume_id, const std::vector<RuntimeValue>& args);

private:
    Target target_;
    JitMemoryBlock code_mem_;
    DataMemoryBlock data_mem_;

    std::unordered_map<std::string, void*> symbol_table_;
    std::unordered_map<std::string, void*> external_symbols_;
    std::unordered_map<std::string, std::pair<Type, std::vector<Type>>> function_signatures_;
    ModuleStackMap stack_maps_;
    runtime::ResumeTableRegistry resume_tables_;
    runtime::PatchRegistry patch_sites_;
    std::unordered_map<std::string, size_t> osr_entry_offsets_;
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
