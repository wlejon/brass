// A native stack overflow in JIT code (any tier: brass-il's programs carry
// no stack-limit probe unless the frontend pinned a TLS block for one) is
// reported as an error and a nonzero exit instead of dying with
// 0xC00000FD / SIGSEGV and no diagnostic.
#include "stack_guard.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

constexpr int kStackOverflowExit = 3;
constexpr char kMessage[] = "brass-il: fatal error: native stack overflow (recursion too deep)\n";

LONG CALLBACK stack_overflow_handler(PEXCEPTION_POINTERS info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_STACK_OVERFLOW) return EXCEPTION_CONTINUE_SEARCH;
    // Runs on the stack guarantee reserved below: no CRT, no allocation.
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), kMessage, sizeof(kMessage) - 1, &written, nullptr);
    TerminateProcess(GetCurrentProcess(), kStackOverflowExit);
    return EXCEPTION_CONTINUE_SEARCH;
}

} // namespace

void brass_il_install_stack_overflow_guard() {
    // Room for the handler once the guard page is gone.
    ULONG guarantee = 64 * 1024;
    SetThreadStackGuarantee(&guarantee);
    AddVectoredExceptionHandler(1, &stack_overflow_handler);
}

#else

void brass_il_install_stack_overflow_guard() {}

#endif
