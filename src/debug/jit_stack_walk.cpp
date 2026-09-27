// Native stack walks that name JIT frames, and the crash report built on them.
//
// Windows: the walk unwinds with RtlLookupFunctionEntry / RtlVirtualUnwind,
// which find JIT frames through the RUNTIME_FUNCTIONs brass registers with
// RtlAddFunctionTable; the crash report is a vectored exception handler (it
// runs on the first chance, so under a debugger once the debugger passes the
// exception on) that prints only for a fault whose address is JIT code, and
// always lets the search continue. POSIX: _Unwind_Backtrace through the
// .eh_frame brass registers; the report is a signal handler that prints and
// then restores the previous disposition so the fault recurs into it.

#include <brass/debug/jit_code_registry.hpp>
#include <brass/runtime/coroutine.hpp>
#include "jit_sinks.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <signal.h>
#include <unwind.h>
#endif

namespace brass::debug {

namespace {

std::atomic<bool> g_crash_report_installed{false};

void fill_module(NativeFrame& f) {
#if defined(_WIN32)
    HMODULE mod = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(f.ip), &mod) || !mod) {
        return;
    }
    char path[MAX_PATH];
    const DWORD n = GetModuleFileNameA(mod, path, MAX_PATH);
    if (n == 0) return;
    const char* base = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '\\' || *p == '/') base = p + 1;
    }
    f.module = base;
    f.module_offset = f.ip - reinterpret_cast<uintptr_t>(mod);
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(f.ip), &info) || !info.dli_fname) return;
    const char* base = std::strrchr(info.dli_fname, '/');
    f.module = base ? base + 1 : info.dli_fname;
    f.module_offset = f.ip - reinterpret_cast<uintptr_t>(info.dli_fbase);
#endif
}

// A frame for `ip`; `lookup_ip` is what names it (a return address minus
// one, so a call that ends its function is still inside it).
NativeFrame make_frame(uintptr_t ip, uintptr_t lookup_ip, bool nonblocking) {
    NativeFrame f;
    f.ip = ip;
    if (nonblocking) {
        char name[256];
        JitTier tier{};
        uintptr_t start = 0;
        if (find_jit_code_nonblocking(lookup_ip, name, sizeof name, &tier, &start)) {
            f.is_jit = true;
            f.jit.name = name;
            f.jit.tier = tier;
            f.jit.start = start;
        }
    } else {
        f.is_jit = find_jit_code(lookup_ip, &f.jit);
    }
    if (!f.is_jit) fill_module(f);
    return f;
}

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
void walk_context(CONTEXT ctx, size_t skip, size_t max_frames, bool nonblocking, std::vector<NativeFrame>& out) {
    for (size_t depth = 0; out.size() < max_frames && ctx.Rip != 0 && depth < 1024; ++depth) {
        const uintptr_t ip = ctx.Rip;
        if (depth >= skip) out.push_back(make_frame(ip, depth == 0 ? ip : ip - 1, nonblocking));
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
        if (fn) {
            void* handler_data = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
        } else {
            // A leaf: its return address is on top of the stack.
            if (ctx.Rsp == 0) break;
            ctx.Rip = *reinterpret_cast<const DWORD64*>(ctx.Rsp);
            ctx.Rsp += 8;
        }
    }
}
#endif

#if !defined(_WIN32)
struct UnwindState {
    std::vector<NativeFrame>* out;
    size_t skip;
    size_t max;
    size_t depth;
};

_Unwind_Reason_Code unwind_step(struct _Unwind_Context* uc, void* arg) {
    auto* s = static_cast<UnwindState*>(arg);
    int before = 0;
    uintptr_t ip = static_cast<uintptr_t>(_Unwind_GetIPInfo(uc, &before));
    if (ip == 0) return _URC_END_OF_STACK;
#if defined(__aarch64__)
    ip &= (uintptr_t{1} << 48) - 1;
#endif
    if (s->depth++ >= s->skip) s->out->push_back(make_frame(ip, before ? ip : ip - 1, false));
    return s->out->size() >= s->max ? _URC_END_OF_STACK : _URC_NO_REASON;
}
#endif

} // namespace

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
std::vector<NativeFrame> capture_native_stack(size_t max_frames) {
    std::vector<NativeFrame> out;
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    CONTEXT ctx{};
    RtlCaptureContext(&ctx);
    // Frame 0 is this function; start at its caller.
    walk_context(ctx, 1, max_frames, false, out);
#elif defined(_WIN32)
    void* ips[64];
    const USHORT n = RtlCaptureStackBackTrace(1, static_cast<DWORD>(max_frames < 64 ? max_frames : 64), ips, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        const auto ip = reinterpret_cast<uintptr_t>(ips[i]);
        out.push_back(make_frame(ip, ip - 1, false));
    }
#else
    UnwindState s{&out, 1, max_frames, 0};
    _Unwind_Backtrace(&unwind_step, &s);
#endif
    return out;
}

std::string format_native_stack(const std::vector<NativeFrame>& frames, bool with_async_stack) {
    std::string s;
    char line[512];
    for (size_t i = 0; i < frames.size(); ++i) {
        const NativeFrame& f = frames[i];
        if (f.is_jit) {
            std::snprintf(line, sizeof line, "  #%zu 0x%016llx %s+0x%llx\n", i, static_cast<unsigned long long>(f.ip),
                          detail::tool_name(f.jit.name, f.jit.tier).c_str(),
                          static_cast<unsigned long long>(f.ip - f.jit.start));
        } else if (!f.module.empty()) {
            std::snprintf(line, sizeof line, "  #%zu 0x%016llx %s+0x%llx\n", i, static_cast<unsigned long long>(f.ip),
                          f.module.c_str(), static_cast<unsigned long long>(f.module_offset));
        } else {
            std::snprintf(line, sizeof line, "  #%zu 0x%016llx\n", i, static_cast<unsigned long long>(f.ip));
        }
        s += line;
    }
    if (with_async_stack) {
        const auto async = runtime::current_async_stack();
        for (size_t i = 0; i < async.size(); ++i) {
            const std::string name = async[i].name.empty() ? std::string("<fixed-code body>") : std::string(async[i].name);
            std::snprintf(line, sizeof line, "  async #%zu %s (state %u, frame 0x%llx)\n", i, name.c_str(),
                          async[i].state_id, static_cast<unsigned long long>(async[i].frame));
            s += line;
        }
    }
    return s;
}

// ---- The crash report -------------------------------------------------------

namespace {

thread_local bool t_reporting = false;

void write_report(const char* what, uintptr_t fault_ip, const std::vector<NativeFrame>& frames) {
    char at[64];
    std::snprintf(at, sizeof at, "0x%llx", static_cast<unsigned long long>(fault_ip));
    std::string s = "brass: ";
    s += what;
    s += " in JIT code at ";
    s += at;
    if (!frames.empty() && frames[0].is_jit) s += " (" + detail::tool_name(frames[0].jit.name, frames[0].jit.tier) + ")";
    s += "\n";
    try {
        s += format_native_stack(frames, true);
    } catch (...) {
        s += format_native_stack(frames, false);
    }
    std::fputs(s.c_str(), stderr);
    std::fflush(stderr);
}

#if defined(_WIN32)
const char* fatal_code_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION: return "access violation";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
        case EXCEPTION_PRIV_INSTRUCTION: return "privileged instruction";
        case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer divide by zero";
        case EXCEPTION_INT_OVERFLOW: return "integer overflow";
        case EXCEPTION_IN_PAGE_ERROR: return "in-page error";
        case EXCEPTION_DATATYPE_MISALIGNMENT: return "misaligned access";
        default: return nullptr;
    }
}

LONG CALLBACK crash_handler(PEXCEPTION_POINTERS info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const char* what = fatal_code_name(code);
    if (!what || t_reporting) return EXCEPTION_CONTINUE_SEARCH;
    const auto ip = reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress);
    if (!find_jit_code_nonblocking(ip, nullptr, 0, nullptr, nullptr)) return EXCEPTION_CONTINUE_SEARCH;
    t_reporting = true;
    std::vector<NativeFrame> frames;
    try {
#if defined(_M_X64) || defined(__x86_64__)
        walk_context(*info->ContextRecord, 0, 64, true, frames);
#else
        frames.push_back(make_frame(ip, ip, true));
#endif
        write_report(what, ip, frames);
    } catch (...) {
    }
    t_reporting = false;
    return EXCEPTION_CONTINUE_SEARCH;
}
#else
struct sigaction g_previous[32];

uintptr_t signal_pc(void* uctx) {
    (void)uctx;
#if defined(__linux__) && defined(__x86_64__)
    return static_cast<uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs[REG_RIP]);
#elif defined(__linux__) && defined(__aarch64__)
    return static_cast<uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.pc);
#elif defined(__APPLE__) && defined(__x86_64__)
    return static_cast<uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext->__ss.__rip);
#elif defined(__APPLE__) && defined(__aarch64__)
    return static_cast<uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext->__ss.__pc);
#else
    return 0;
#endif
}

void crash_signal(int sig, siginfo_t* si, void* uctx) {
    const uintptr_t pc = signal_pc(uctx);
    if (!t_reporting && pc && find_jit_code_nonblocking(pc, nullptr, 0, nullptr, nullptr)) {
        t_reporting = true;
        std::vector<NativeFrame> frames;
        frames.push_back(make_frame(pc, pc, true));
        // The unwinder steps through the signal frame into the JIT frames.
        for (NativeFrame& f : capture_native_stack(64)) frames.push_back(std::move(f));
        write_report(sig == SIGSEGV ? "segmentation fault" : sig == SIGBUS ? "bus error"
                     : sig == SIGILL ? "illegal instruction" : "arithmetic fault", pc, frames);
        t_reporting = false;
    }
    (void)si;
    // Back to whoever handled it before; returning re-executes the fault.
    if (sig > 0 && sig < 32) sigaction(sig, &g_previous[sig], nullptr);
}
#endif

} // namespace

namespace detail {
void install_crash_handler() {
#if defined(_WIN32)
    // Last in line: every handler the host installed runs first.
    AddVectoredExceptionHandler(0, &crash_handler);
#else
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        struct sigaction sa{};
        sa.sa_sigaction = &crash_signal;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, &g_previous[sig]);
    }
#endif
}
} // namespace detail

void install_jit_crash_report() {
    bool expected = false;
    if (g_crash_report_installed.compare_exchange_strong(expected, true)) detail::install_crash_handler();
}

bool jit_crash_report_installed() noexcept {
    return g_crash_report_installed.load(std::memory_order_acquire);
}

} // namespace brass::debug
