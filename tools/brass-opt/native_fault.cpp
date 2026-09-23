#include "native_fault.hpp"

#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csetjmp>
#include <csignal>
#if defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <ucontext.h>
#endif
#endif

namespace brass {

#if defined(__aarch64__) || defined(_M_ARM64)
namespace {

// AArch64 has no divide fault: the backend guards every integer division
// with `brk #kBrkIntegerDivideByZero`, which arrives as a breakpoint
// (SIGTRAP, or EXCEPTION_BREAKPOINT on Windows). It is a division by zero
// exactly when the trapping instruction is that brk.
bool is_divide_by_zero_brk(uintptr_t pc) {
    constexpr uint32_t kBrk = 0xD4200000u | (static_cast<uint32_t>(aarch64::kBrkIntegerDivideByZero) << 5);
    if (pc < 4 || (pc & 3u) != 0) return false;
    // The reported PC is the brk itself; accept the next instruction's
    // address too, should a kernel report the PC past it.
    const uintptr_t candidates[2] = {pc, pc - 4};
    for (uintptr_t at : candidates) {
        uint32_t word = 0;
        std::memcpy(&word, reinterpret_cast<const void*>(at), sizeof(word));
        if (word == kBrk) return true;
    }
    return false;
}

} // namespace
#endif

#if defined(_WIN32)

namespace {

int arith_fault_filter(EXCEPTION_POINTERS* ep, DWORD* out) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_INT_DIVIDE_BY_ZERO || code == EXCEPTION_INT_OVERFLOW) {
        *out = code;
        return EXCEPTION_EXECUTE_HANDLER;
    }
#if defined(_M_ARM64)
    if (code == EXCEPTION_BREAKPOINT &&
        is_divide_by_zero_brk(reinterpret_cast<uintptr_t>(ep->ExceptionRecord->ExceptionAddress))) {
        *out = EXCEPTION_INT_DIVIDE_BY_ZERO;
        return EXCEPTION_EXECUTE_HANDLER;
    }
#endif
    return EXCEPTION_CONTINUE_SEARCH;
}

} // namespace

// No C++ objects with destructors here: __try needs a frame without them.
// Anything else (a C++ exception, an access violation) passes through.
const char* call_catching_arith_faults(NativeBody body, void* ctx) {
    DWORD code = 0;
    __try {
        body(ctx);
    } __except (arith_fault_filter(GetExceptionInformation(), &code)) {
        return code == EXCEPTION_INT_DIVIDE_BY_ZERO ? "integer division by zero"
                                                    : "integer overflow in division";
    }
    return nullptr;
}

#else

namespace {

thread_local sigjmp_buf* t_fault_jmp = nullptr;

void sigfpe_handler(int sig, siginfo_t* info, void*) {
    if (!t_fault_jmp) {
        // Not ours: fall back to the default action, which re-raises.
        signal(sig, SIG_DFL);
        return;
    }
    siglongjmp(*t_fault_jmp, info && info->si_code == FPE_INTDIV ? 1 : 2);
}

#if defined(__aarch64__)
uintptr_t trap_pc(void* uctx) {
    if (!uctx) return 0;
    auto* uc = static_cast<ucontext_t*>(uctx);
#if defined(__APPLE__)
#if defined(__darwin_arm_thread_state64_get_pc)
    return (uintptr_t)(__darwin_arm_thread_state64_get_pc(uc->uc_mcontext->__ss));
#else
    return static_cast<uintptr_t>(uc->uc_mcontext->__ss.__pc);
#endif
#elif defined(__linux__)
    return static_cast<uintptr_t>(uc->uc_mcontext.pc);
#else
    (void)uc;
    return 0;
#endif
}

void sigtrap_handler(int sig, siginfo_t*, void* uctx) {
    if (t_fault_jmp && is_divide_by_zero_brk(trap_pc(uctx))) siglongjmp(*t_fault_jmp, 1);
    // Not ours: the default action takes over when the brk re-executes.
    signal(sig, SIG_DFL);
}
#endif

} // namespace

const char* call_catching_arith_faults(NativeBody body, void* ctx) {
    struct sigaction act {};
    struct sigaction old {};
    act.sa_sigaction = &sigfpe_handler;
    act.sa_flags = SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    sigaction(SIGFPE, &act, &old);
#if defined(__aarch64__)
    struct sigaction trap_act {};
    struct sigaction trap_old {};
    trap_act.sa_sigaction = &sigtrap_handler;
    trap_act.sa_flags = SA_SIGINFO;
    sigemptyset(&trap_act.sa_mask);
    sigaction(SIGTRAP, &trap_act, &trap_old);
#endif

    sigjmp_buf jmp;
    sigjmp_buf* const prev = t_fault_jmp;
    const int fault = sigsetjmp(jmp, 1);
    auto restore = [&] {
        t_fault_jmp = prev;
        sigaction(SIGFPE, &old, nullptr);
#if defined(__aarch64__)
        sigaction(SIGTRAP, &trap_old, nullptr);
#endif
    };
    if (fault == 0) {
        t_fault_jmp = &jmp;
        try {
            body(ctx);
        } catch (...) {
            restore();
            throw;
        }
    }
    restore();
    if (fault == 0) return nullptr;
    return fault == 1 ? "integer division by zero" : "integer overflow in division";
}

#endif

} // namespace brass
