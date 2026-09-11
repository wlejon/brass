#include <brass/runtime/exception.hpp>
#include <mutex>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
extern "C" void brass_jump_to_landing_pad_msvc(void* ip, void* rbp, void* rsp, uint64_t val);
#endif

namespace brass::runtime {

const ExceptionScopeEntry* FunctionExceptionTable::find_scope(uint32_t func_ip_offset) const noexcept {
    if (func_ip_offset == 0) return nullptr;
    uint32_t call_ip = func_ip_offset - 1;
    for (const auto& s : scopes_) {
        if (call_ip >= s.begin_offset && call_ip < s.end_offset) {
            return &s;
        }
    }
    return nullptr;
}

void ExceptionTableRegistry::register_table(std::string_view fn_name, FunctionExceptionTable table) {
    std::lock_guard<std::mutex> lock(mutex_);
    tables_[std::string(fn_name)] = std::move(table);
}

const FunctionExceptionTable* ExceptionTableRegistry::get_table(std::string_view fn_name) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tables_.find(std::string(fn_name));
    if (it != tables_.end()) {
        return &it->second;
    }
    return nullptr;
}

void ExceptionTableRegistry::register_function_mapping(
    uintptr_t fn_start,
    uintptr_t fn_size,
    const FunctionExceptionTable& table
) {
    std::lock_guard<std::mutex> lock(mutex_);
    RangeEntry re;
    re.start = fn_start;
    re.end = fn_start + fn_size;
    re.table = table;
    ranges_.push_back(std::move(re));
}

void ExceptionTableRegistry::unregister_function_mapping(uintptr_t fn_start) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = ranges_.begin(); it != ranges_.end(); ++it) {
        if (it->start == fn_start) {
            ranges_.erase(it);
            break;
        }
    }
}

void ExceptionTableRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    tables_.clear();
    ranges_.clear();
}

const FunctionExceptionTable* ExceptionTableRegistry::find_function_by_pc(uintptr_t pc) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uintptr_t lookup_pc = (pc > 0) ? (pc - 1) : pc;
    for (const auto& r : ranges_) {
        if (lookup_pc >= r.start && lookup_pc < r.end) {
            return &r.table;
        }
    }
    return nullptr;
}

const ExceptionScopeEntry* ExceptionTableRegistry::find_scope_by_pc(
    uintptr_t pc,
    uintptr_t* out_fn_start,
    const FunctionExceptionTable** out_table
) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uintptr_t lookup_pc = (pc > 0) ? (pc - 1) : pc;
    for (const auto& r : ranges_) {
        if (lookup_pc >= r.start && lookup_pc < r.end) {
            if (out_fn_start) *out_fn_start = r.start;
            if (out_table) *out_table = &r.table;
            uint32_t off = static_cast<uint32_t>(pc - r.start);
            return r.table.find_scope(off);
        }
    }
    return nullptr;
}

static ExceptionTableRegistry s_global_exception_registry;

ExceptionTableRegistry& get_global_exception_registry() noexcept {
    return s_global_exception_registry;
}

// Thread-local exception storage
static thread_local HostValue t_current_exception = HostValue::undefined_val();
static thread_local bool t_has_current_exception = false;

void brass_set_current_exception(HostValue val) noexcept {
    t_current_exception = val;
    t_has_current_exception = true;
}

HostValue brass_get_current_exception() noexcept {
    return t_current_exception;
}

HostValue brass_take_current_exception() noexcept {
    HostValue val = t_current_exception;
    t_current_exception = HostValue::undefined_val();
    t_has_current_exception = false;
    return val;
}

bool brass_has_current_exception() noexcept {
    return t_has_current_exception;
}

void brass_clear_current_exception() noexcept {
    t_current_exception = HostValue::undefined_val();
    t_has_current_exception = false;
}

#if defined(__GNUC__) || defined(__clang__)
#if defined(_WIN32)
__attribute__((naked)) void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& regs
) {
    __asm__ volatile(
        "movq 40(%%rsp), %%r10\n\t"  // 5th arg (&regs) is at 40(%rsp) on Win64
        "movq %%r9, %%rax\n\t"       // 4th arg (val) -> %rax
        "movq %%rcx, %%r11\n\t"      // 1st arg (landing_pad_ip) -> %r11
        // Load callee-saved registers from &regs
        "movq 0(%%r10),  %%r15\n\t"
        "movq 8(%%r10),  %%r14\n\t"
        "movq 16(%%r10), %%r13\n\t"
        "movq 24(%%r10), %%r12\n\t"
        "movq 32(%%r10), %%rdi\n\t"
        "movq 40(%%r10), %%rsi\n\t"
        "movq 48(%%r10), %%rbx\n\t"
        // Restore target frame RBP and RSP
        "movq %%rdx, %%rbp\n\t"      // 2nd arg (target_rbp) -> %rbp
        "movq %%r8,  %%rsp\n\t"      // 3rd arg (target_rsp) -> %rsp
        "jmp *%%r11\n\t"
        :
        :
        : "memory"
    );
}
#else
__attribute__((naked)) void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& regs
) {
    __asm__ volatile(
        "movq %%rcx, %%rax\n\t"      // 4th arg (val) -> %rax
        "movq %%rdi, %%r11\n\t"      // 1st arg (landing_pad_ip) -> %r11
        "movq %%rsi, %%r10\n\t"      // 2nd arg (target_rbp) -> %r10
        "movq %%rdx, %%r9\n\t"       // 3rd arg (target_rsp) -> %r9
        "movq %%r8,  %%rdx\n\t"      // 5th arg (&regs) -> %rdx
        // Load callee-saved registers from &regs
        "movq 0(%%rdx),  %%r15\n\t"
        "movq 8(%%rdx),  %%r14\n\t"
        "movq 16(%%rdx), %%r13\n\t"
        "movq 24(%%rdx), %%r12\n\t"
        "movq 32(%%rdx), %%rdi\n\t"
        "movq 40(%%rdx), %%rsi\n\t"
        "movq 48(%%rdx), %%rbx\n\t"
        // Restore target frame RBP and RSP
        "movq %%r10, %%rbp\n\t"
        "movq %%r9,  %%rsp\n\t"
        "jmp *%%r11\n\t"
        :
        :
        : "memory"
    );
}
#endif
#elif defined(_MSC_VER)
[[noreturn]] void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& regs
) {
    (void)regs;
    brass_jump_to_landing_pad_msvc(landing_pad_ip, target_rbp, target_rsp, val.raw());
    __assume(0);
}
#else
[[noreturn]] void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& regs
) {
    (void)landing_pad_ip;
    (void)target_rbp;
    (void)target_rsp;
    (void)regs;
    throw BrassException(val);
}
#endif

#if defined(__clang__)
#define BRASS_NOINLINE_NOFP [[noreturn]] __attribute__((noinline))
#elif defined(__GNUC__)
#define BRASS_NOINLINE_NOFP [[noreturn]] __attribute__((noinline, optimize("no-omit-frame-pointer")))
#else
#define BRASS_NOINLINE_NOFP [[noreturn]]
#endif

BRASS_NOINLINE_NOFP void brass_throw_impl(HostValue val, const SavedRegisters* regs) {
    brass_set_current_exception(val);

    uintptr_t cur_rbp = 0;
    uintptr_t cur_ip = 0;

#if defined(__GNUC__) || defined(__clang__)
    void* frame = __builtin_frame_address(0);
    if (frame) {
        // frame is brass_throw_impl's RBP
        uintptr_t bt_rbp = *reinterpret_cast<uintptr_t*>(frame); // brass_throw's RBP
        if (bt_rbp) {
            cur_ip = *reinterpret_cast<uintptr_t*>(bt_rbp + 8); // return address in caller
            cur_rbp = *reinterpret_cast<uintptr_t*>(bt_rbp);     // caller's RBP
        }
    }
#elif defined(_MSC_VER)
    void** ret_addr_ptr = reinterpret_cast<void**>(_AddressOfReturnAddress());
    if (ret_addr_ptr) {
        cur_ip = reinterpret_cast<uintptr_t>(*ret_addr_ptr);
        cur_rbp = reinterpret_cast<uintptr_t>(*(ret_addr_ptr - 1));
    }
#endif

    auto& registry = get_global_exception_registry();

    // Walk call stack via RBP frame chaining
    while (cur_rbp != 0 && cur_ip != 0) {
        uintptr_t fn_start = 0;
        const FunctionExceptionTable* fn_table = nullptr;
        const ExceptionScopeEntry* scope = registry.find_scope_by_pc(cur_ip, &fn_start, &fn_table);

        if (scope && fn_table) {
            void* landing_pad_ip = reinterpret_cast<void*>(fn_start + scope->landing_pad_offset);
            void* target_rbp = reinterpret_cast<void*>(cur_rbp);
            uintptr_t target_rsp_val = cur_rbp - fn_table->frame_size();
            void* target_rsp = reinterpret_cast<void*>(target_rsp_val);

            brass_jump_to_landing_pad(landing_pad_ip, target_rbp, target_rsp, val, *regs);
        }

        // Check if cur_ip belongs to a registered Brass function
        const FunctionExceptionTable* cur_fn = registry.find_function_by_pc(cur_ip);
        if (!cur_fn) {
            // Reached non-Brass frame (host code). Stop walking.
            break;
        }

        auto* rbp_ptr = reinterpret_cast<uintptr_t*>(cur_rbp);
        uintptr_t next_rbp = rbp_ptr[0];
        uintptr_t next_ip = rbp_ptr[1];

        if (next_rbp <= cur_rbp) {
            break;
        }
        cur_rbp = next_rbp;
        cur_ip = next_ip;
    }

    // Fallback if unhandled by Brass landing pad
    throw BrassException(val);
}

extern "C" {

#if defined(__GNUC__) || defined(__clang__)
BRASS_NOINLINE_NOFP void brass_throw(HostValue val) {
    SavedRegisters regs;
    __asm__ volatile(
        "movq %%r15, %0\n\t"
        "movq %%r14, %1\n\t"
        "movq %%r13, %2\n\t"
        "movq %%r12, %3\n\t"
        "movq %%rdi, %4\n\t"
        "movq %%rsi, %5\n\t"
        "movq %%rbx, %6\n\t"
        : "=m"(regs.r15), "=m"(regs.r14), "=m"(regs.r13), "=m"(regs.r12),
          "=m"(regs.rdi), "=m"(regs.rsi), "=m"(regs.rbx)
        :
        : "memory"
    );
    brass_throw_impl(val, &regs);
}
#else
[[noreturn]] void brass_throw(HostValue val) {
    SavedRegisters regs{};
    brass_throw_impl(val, &regs);
}
#endif

[[noreturn]] void brass_rethrow() {
    HostValue val = brass_get_current_exception();
    brass_throw(val);
}

} // extern "C"

} // namespace brass::runtime
