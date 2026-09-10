#pragma once

#include <brass/embedding/nanbox.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <exception>



namespace brass::object {
struct Section;
}

namespace brass::runtime {

// 1. Exception representation: BrassException envelope storing HostValue
class BrassException : public std::exception {
public:
    explicit BrassException(HostValue val) noexcept : value_(val) {}

    [[nodiscard]] HostValue value() const noexcept { return value_; }
    [[nodiscard]] const char* what() const noexcept override {
        return "BrassException: runtime exception";
    }

private:
    HostValue value_;
};

// 2. Exception scope entry
struct ExceptionScopeEntry {
    uint32_t begin_offset = 0;       // Start of protected call / region (relative to function)
    uint32_t end_offset = 0;         // End of protected call / region
    uint32_t landing_pad_offset = 0; // Landing pad offset within the function
};

// 3. Function exception table
class FunctionExceptionTable {
public:
    FunctionExceptionTable() = default;
    explicit FunctionExceptionTable(std::string function_name, uint32_t code_offset = 0, uint32_t code_size = 0)
        : function_name_(std::move(function_name)), code_offset_(code_offset), code_size_(code_size) {}

    void set_function_name(std::string name) { function_name_ = std::move(name); }
    [[nodiscard]] const std::string& function_name() const noexcept { return function_name_; }

    void set_code_offset(uint32_t off) noexcept { code_offset_ = off; }
    [[nodiscard]] uint32_t code_offset() const noexcept { return code_offset_; }

    void set_code_size(uint32_t sz) noexcept { code_size_ = sz; }
    [[nodiscard]] uint32_t code_size() const noexcept { return code_size_; }

    void set_frame_size(uint32_t sz) noexcept { frame_size_ = sz; }
    [[nodiscard]] uint32_t frame_size() const noexcept { return frame_size_; }

    void set_saved_callee_gprs(uint32_t mask) noexcept { saved_callee_gprs_ = mask; }
    [[nodiscard]] uint32_t saved_callee_gprs() const noexcept { return saved_callee_gprs_; }

    void add_scope(uint32_t begin_off, uint32_t end_off, uint32_t landing_pad_off) {
        scopes_.push_back(ExceptionScopeEntry{begin_off, end_off, landing_pad_off});
    }

    [[nodiscard]] const std::vector<ExceptionScopeEntry>& scopes() const noexcept { return scopes_; }
    [[nodiscard]] bool has_scopes() const noexcept { return !scopes_.empty(); }

    [[nodiscard]] const ExceptionScopeEntry* find_scope(uint32_t func_ip_offset) const noexcept;

private:
    std::string function_name_;
    uint32_t code_offset_ = 0;
    uint32_t code_size_ = 0;
    uint32_t frame_size_ = 0;
    uint32_t saved_callee_gprs_ = 0;
    std::vector<ExceptionScopeEntry> scopes_;
};

// 4. Registry for JIT and AOT exception lookup
class ExceptionTableRegistry {
public:
    ExceptionTableRegistry() = default;

    void register_table(std::string_view fn_name, FunctionExceptionTable table);
    [[nodiscard]] const FunctionExceptionTable* get_table(std::string_view fn_name) const noexcept;

    void register_function_mapping(uintptr_t fn_start, uintptr_t fn_size, const FunctionExceptionTable& table);
    void unregister_function_mapping(uintptr_t fn_start);
    void clear();

    [[nodiscard]] const FunctionExceptionTable* find_function_by_pc(uintptr_t pc) const noexcept;
    [[nodiscard]] const ExceptionScopeEntry* find_scope_by_pc(
        uintptr_t pc,
        uintptr_t* out_fn_start = nullptr,
        const FunctionExceptionTable** out_table = nullptr
    ) const noexcept;

    [[nodiscard]] const std::unordered_map<std::string, FunctionExceptionTable>& tables() const noexcept { return tables_; }

private:
    std::unordered_map<std::string, FunctionExceptionTable> tables_;
    struct RangeEntry {
        uintptr_t start = 0;
        uintptr_t end = 0;
        FunctionExceptionTable table;
    };
    std::vector<RangeEntry> ranges_;
};

ExceptionTableRegistry& get_global_exception_registry() noexcept;

// 5. Thread-local exception APIs
void brass_set_current_exception(HostValue val) noexcept;
HostValue brass_get_current_exception() noexcept;
HostValue brass_take_current_exception() noexcept;
bool brass_has_current_exception() noexcept;
void brass_clear_current_exception() noexcept;

struct SavedRegisters {
    uint64_t r15 = 0;
    uint64_t r14 = 0;
    uint64_t r13 = 0;
    uint64_t r12 = 0;
    uint64_t rdi = 0;
    uint64_t rsi = 0;
    uint64_t rbx = 0;
};

// 6. In-memory landing pad jump & dispatcher
[[noreturn]] void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& saved_regs = SavedRegisters{}
);

extern "C" {
[[noreturn]] void brass_throw(HostValue val);
[[noreturn]] void brass_rethrow();
}

// 7. Win64 SEH scope table emitter and personality routine
void emit_win64_seh_scope_table(object::Section& xdata_sec, const FunctionExceptionTable& table);

#if defined(_WIN32)
extern "C" int brass_seh_personality(
    void* ExceptionRecord,
    void* EstablisherFrame,
    void* ContextRecord,
    void* DispatcherContext
);
#endif

// 8. SysV DWARF LSDA emitter
void emit_sysv_lsda(object::Section& lsda_sec, const FunctionExceptionTable& table);

} // namespace brass::runtime
