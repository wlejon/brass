#pragma once

#include <brass/embedding/nanbox.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <map>
#include <exception>
#include <mutex>



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
    ExceptionTableRegistry(const ExceptionTableRegistry& other) {
        std::lock_guard<std::mutex> lock(other.mutex_);
        tables_ = other.tables_;
        ranges_ = other.ranges_;
    }
    ExceptionTableRegistry& operator=(const ExceptionTableRegistry& other) {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            tables_ = other.tables_;
            ranges_ = other.ranges_;
        }
        return *this;
    }
    ExceptionTableRegistry(ExceptionTableRegistry&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        tables_ = std::move(other.tables_);
        ranges_ = std::move(other.ranges_);
    }
    ExceptionTableRegistry& operator=(ExceptionTableRegistry&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            tables_ = std::move(other.tables_);
            ranges_ = std::move(other.ranges_);
        }
        return *this;
    }

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
    mutable std::mutex mutex_;
    std::unordered_map<std::string, FunctionExceptionTable> tables_;
    struct RangeEntry {
        uintptr_t start = 0;
        uintptr_t end = 0;
        FunctionExceptionTable table;
    };
    // By start address: a lookup is a search, not a scan of every function
    // the program ever compiled, and an entry's table stays where it is
    // while other functions come and go.
    std::map<uintptr_t, RangeEntry> ranges_;
    const RangeEntry* find_range(uintptr_t pc) const noexcept;
};

ExceptionTableRegistry& get_global_exception_registry() noexcept;

// 5. Thread-local exception APIs
void brass_set_current_exception(HostValue val) noexcept;
HostValue brass_get_current_exception() noexcept;
HostValue brass_take_current_exception() noexcept;
bool brass_has_current_exception() noexcept;
void brass_clear_current_exception() noexcept;

struct SavedRegisters {
    // x86_64 callee-saved
    uint64_t r15 = 0;
    uint64_t r14 = 0;
    uint64_t r13 = 0;
    uint64_t r12 = 0;
    uint64_t rdi = 0;
    uint64_t rsi = 0;
    uint64_t rbx = 0;
    // AArch64 callee-saved
    uint64_t x19 = 0, x20 = 0, x21 = 0, x22 = 0;
    uint64_t x23 = 0, x24 = 0, x25 = 0, x26 = 0;
    uint64_t x27 = 0, x28 = 0, fp = 0, lr = 0;
    uint64_t d8 = 0, d9 = 0, d10 = 0, d11 = 0;
    uint64_t d12 = 0, d13 = 0, d14 = 0, d15 = 0;
};

// 6. In-memory landing pad jump & dispatcher
[[noreturn]] void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& saved_regs = SavedRegisters{}
);

// Generated code names its throw, rethrow and personality routines by the
// canonical symbols brass_throw, brass_rethrow and brass_seh_personality /
// brass_sysv_personality. brass's definitions are these brass_default_*
// functions, bound to the canonical names by an engine's symbol table (the
// JIT) or by the one image in the process that exports them under those
// names (a host runtime, for AOT objects). Like brass_gc_write_barrier, no
// canonical name is defined here, so a process that also links brass
// statically still has exactly one definition of each.
extern "C" {
[[noreturn]] void brass_default_throw(HostValue val);
[[noreturn]] void brass_default_rethrow();
}

// 7. Win64 SEH scope table emitter and personality routine
//
// The scope table follows the handler RVA in a function's .xdata; each
// begin/end/landing-pad entry is an ADDR32NB relocation against fn_symbol.
void emit_win64_seh_scope_table(object::Section& xdata_sec, const FunctionExceptionTable& table,
                                std::string_view fn_symbol);

// SEH exception code of a brass throw raised through the OS dispatcher
// (customer bit set, "BRS"); ExceptionInformation[0] holds the value bits.
inline constexpr uint32_t BRASS_SEH_EXCEPTION_CODE = 0xE0425253u;

// Landing pad for a frame at return address control_pc, from a scope table
// (handler_data, image-relative) or, with no table, from the JIT registry.
// Returns the absolute pad address, or 0 when the frame has no scope there.
uint64_t brass_seh_find_landing_pad(uint64_t control_pc, uint64_t image_base, const void* handler_data) noexcept;

// Raises val to the first frame the OS unwinder can see that has a brass
// landing pad for it; does not return then. On Win64 that is a Win64 SEH
// exception; on other x86-64 and AArch64 hosts the DWARF unwinder finds the
// frame and its pad is entered directly, abandoning the frames in between,
// unless one of them has unwinding of its own, when it throws a C++
// BrassException for the C++ unwinder to take there instead
// (exception_raise_unwind.cpp). Callers raise from frames holding nothing to
// unwind. Returns false (and raises nothing) when no such frame exists,
// or on any other host. Only frames below the innermost generated-code entry
// (GeneratedCodeEntryScope) count, here and in brass_seh_raise_above: the
// C++ frames above it must see a C++ exception, which the caller throws.
bool brass_seh_raise(HostValue val);

// As brass_seh_raise, for a throw on behalf of a deoptimized native frame:
// only frames above the frame of the function starting at `deopted_entry`
// (the first such frame up the stack; it and everything below it are
// skipped) and below `stack_limit` count. Raises when the first frame there
// with a landing pad for its call site is found; returns false (raising
// nothing) when that range has none, when the deoptimized frame is not on
// the stack, or where brass_seh_raise cannot raise.
bool brass_seh_raise_above(HostValue val, const void* deopted_entry, uintptr_t stack_limit);

// The language handler of every Win64 function with brass landing pads. It
// lands two kinds of exception at a pad: a brass SEH exception, and a C++
// BrassException (MSVC's C++ exception ABI) thrown by compiled code the
// generated code called, whose compiled frames unwind with their destructors
// run. A host or runtime function called from generated code raises into
// its caller's pads by throwing a BrassException.
extern "C" int brass_default_seh_personality(
    void* ExceptionRecord,
    void* EstablisherFrame,
    void* ContextRecord,
    void* DispatcherContext
);

// 8. SysV DWARF LSDA emitter and personality routine
//
// The LSDA is gcc's layout: no LPStart or type table, a udata4 call-site
// table of (begin, length, pad) function offsets, one catch-all action.
void emit_sysv_lsda(object::Section& lsda_sec, const FunctionExceptionTable& table);

// The personality routine the .eh_frame CIE of generated code names off
// Windows (x86-64 and AArch64; the Itanium unwinder's signature, with its
// enums as ints). It lands a C++ BrassException thrown by a function the
// generated code called at the frame's pad for that call, found in the
// frame's LSDA or, without one, in the JIT registry; everything else passes
// through. It is brass_seh_personality's counterpart.
extern "C" int brass_default_sysv_personality(int version, int actions, uint64_t exception_class,
                                      void* exception_object, void* context);

} // namespace brass::runtime
