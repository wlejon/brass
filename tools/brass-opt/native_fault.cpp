#include "native_fault.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csetjmp>
#include <csignal>
#endif

namespace brass {

#if defined(_WIN32)

namespace {

int arith_fault_filter(DWORD code, DWORD* out) {
    if (code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_INT_OVERFLOW) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    *out = code;
    return EXCEPTION_EXECUTE_HANDLER;
}

} // namespace

// No C++ objects with destructors here: __try needs a frame without them.
// Anything else (a C++ exception, an access violation) passes through.
const char* call_catching_arith_faults(NativeBody body, void* ctx) {
    DWORD code = 0;
    __try {
        body(ctx);
    } __except (arith_fault_filter(GetExceptionCode(), &code)) {
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

} // namespace

const char* call_catching_arith_faults(NativeBody body, void* ctx) {
    struct sigaction act {};
    struct sigaction old {};
    act.sa_sigaction = &sigfpe_handler;
    act.sa_flags = SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    sigaction(SIGFPE, &act, &old);

    sigjmp_buf jmp;
    sigjmp_buf* const prev = t_fault_jmp;
    const int fault = sigsetjmp(jmp, 1);
    auto restore = [&] {
        t_fault_jmp = prev;
        sigaction(SIGFPE, &old, nullptr);
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
