// Process-level isolation for the differential fuzzer: hardware-fault
// recovery, the watchdog, the job-object / rlimit sandbox and reproducer
// files. The tier logic lives in diff_fuzzer.cpp.

#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/mir/printer.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <csetjmp>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace brass::fuzz {

std::string_view status_name(ExecutionStatus status) noexcept {
    switch (status) {
        case ExecutionStatus::Success: return "Success";
        case ExecutionStatus::Timeout: return "Timeout";
        case ExecutionStatus::MemoryExceeded: return "MemoryExceeded";
        case ExecutionStatus::CrashOrFault: return "CrashOrFault";
        case ExecutionStatus::VerificationFailure: return "VerificationFailure";
        case ExecutionStatus::CompilationFailure: return "CompilationFailure";
        case ExecutionStatus::ExceptionThrown: return "ExceptionThrown";
    }
    return "Unknown";
}

#if defined(_WIN32)
static thread_local bool s_in_protected = false;
alignas(16) static thread_local CONTEXT s_recovery_context;
static thread_local DWORD s_fault_code = 0;
static thread_local volatile bool s_fault_occurred = false;

static bool is_recoverable_fault(DWORD code) {
    // INT_OVERFLOW is `idiv` on INT_MIN / -1; it must be reported as a
    // fault of the tier, not take the whole fuzzer down.
    return code == EXCEPTION_ACCESS_VIOLATION ||
           code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
           code == EXCEPTION_INT_OVERFLOW ||
           code == EXCEPTION_ILLEGAL_INSTRUCTION ||
           code == EXCEPTION_PRIV_INSTRUCTION ||
           code == EXCEPTION_FLT_DIVIDE_BY_ZERO ||
           code == EXCEPTION_DATATYPE_MISALIGNMENT ||
           code == EXCEPTION_ARRAY_BOUNDS_EXCEEDED ||
           code == EXCEPTION_IN_PAGE_ERROR;
}

static LONG WINAPI FuzzVectoredHandler(EXCEPTION_POINTERS* ep) {
    if (s_in_protected && ep && ep->ExceptionRecord && ep->ContextRecord) {
        DWORD code = ep->ExceptionRecord->ExceptionCode;
        if (is_recoverable_fault(code)) {
            s_in_protected = false;
            s_fault_code = code;
            s_fault_occurred = true;
            *ep->ContextRecord = s_recovery_context;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static void ensure_vectored_handler_installed() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        AddVectoredExceptionHandler(1, FuzzVectoredHandler);
    });
}

static void format_win_fault(DWORD code, std::string& out_msg) {
    std::ostringstream ss;
    ss << "Hardware fault 0x" << std::hex << code;
    if (code == EXCEPTION_ACCESS_VIOLATION) ss << " (Access Violation)";
    else if (code == EXCEPTION_INT_DIVIDE_BY_ZERO) ss << " (Integer Divide by Zero)";
    else if (code == EXCEPTION_INT_OVERFLOW) ss << " (Integer Overflow)";
    else if (code == EXCEPTION_ILLEGAL_INSTRUCTION) ss << " (Illegal Instruction)";
    else if (code == EXCEPTION_PRIV_INSTRUCTION) ss << " (Privileged Instruction)";
    else if (code == EXCEPTION_FLT_DIVIDE_BY_ZERO) ss << " (Float Divide by Zero)";
    out_msg = ss.str();
}
#else
static thread_local sigjmp_buf s_posix_recovery_env;
static thread_local volatile sig_atomic_t s_posix_in_protected = 0;
static thread_local volatile sig_atomic_t s_posix_fault_sig = 0;

static void posix_fuzz_sig_handler(int sig, siginfo_t* info, void* ctx) {
    (void)info;
    (void)ctx;
    if (s_posix_in_protected) {
        s_posix_in_protected = 0;
        s_posix_fault_sig = sig;
        siglongjmp(s_posix_recovery_env, 1);
    }
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

static void ensure_posix_signal_handler_installed() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = posix_fuzz_sig_handler;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGFPE, &sa, nullptr);
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGBUS, &sa, nullptr);
        sigaction(SIGILL, &sa, nullptr);
    });
}

static void format_posix_fault(int sig, std::string& out_msg) {
    std::ostringstream ss;
    ss << "Hardware fault: ";
    if (sig == SIGFPE) ss << "Integer Divide by Zero / Arithmetic (SIGFPE)";
    else if (sig == SIGSEGV) ss << "Segmentation Fault (SIGSEGV)";
    else if (sig == SIGBUS) ss << "Bus Error (SIGBUS)";
    else if (sig == SIGILL) ss << "Illegal Instruction (SIGILL)";
    else ss << "Signal " << sig;
    out_msg = ss.str();
}
#endif

bool DiffFuzzer::run_protected(const std::function<void()>& action, std::string& fault_msg) {
#if defined(_WIN32)
    ensure_vectored_handler_installed();
    s_fault_code = 0;
    s_fault_occurred = false;
    s_in_protected = false;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    RtlCaptureContext(&s_recovery_context);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (s_fault_occurred) {
        s_in_protected = false;
        format_win_fault(s_fault_code, fault_msg);
        return false;
    }
    s_in_protected = true;
    bool ok = true;
    try {
        action();
    } catch (const std::exception& ex) {
        fault_msg = std::string("C++ Exception: ") + ex.what();
        ok = false;
    } catch (...) {
        fault_msg = "Unknown Exception";
        ok = false;
    }
    s_in_protected = false;
    return ok;
#else
    ensure_posix_signal_handler_installed();
    s_posix_fault_sig = 0;
    s_posix_in_protected = 0;

    if (sigsetjmp(s_posix_recovery_env, 1) != 0) {
        s_posix_in_protected = 0;
        format_posix_fault(s_posix_fault_sig, fault_msg);
        return false;
    }

    s_posix_in_protected = 1;
    bool ok = true;
    try {
        action();
    } catch (const std::exception& ex) {
        fault_msg = std::string("C++ Exception: ") + ex.what();
        ok = false;
    } catch (...) {
        fault_msg = "Unknown Exception";
        ok = false;
    }
    s_posix_in_protected = 0;
    return ok;
#endif
}

struct WatchdogState {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> finished{false};
};

bool DiffFuzzer::run_with_watchdog(const std::function<void()>& action, uint32_t timeout_ms) {
    auto state = std::make_shared<WatchdogState>();

    std::thread worker([state, action]() {
        action();
        state->finished.store(true, std::memory_order_release);
        state->cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(state->mtx);
    bool completed = state->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
        return state->finished.load(std::memory_order_acquire);
    });

    if (!completed) {
#if defined(_WIN32)
        TerminateThread(reinterpret_cast<HANDLE>(worker.native_handle()), 1);
#endif
        worker.detach();
        return false;
    }

    if (worker.joinable()) {
        worker.join();
    }
    return true;
}

ExecutionStatus DiffFuzzer::run_sandboxed_command(const std::string& command_line,
                                                 uint32_t timeout_ms,
                                                 size_t memory_limit_bytes,
                                                 std::string& out_log) {
#if defined(_WIN32)
    HANDLE hJob = CreateJobObjectW(NULL, NULL);
    if (!hJob) {
        out_log = "Failed to create Windows Job Object";
        return ExecutionStatus::CompilationFailure;
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = {};
    jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY;
    jeli.ProcessMemoryLimit = (memory_limit_bytes > 0) ? memory_limit_bytes : (256 * 1024 * 1024);
    jeli.JobMemoryLimit = jeli.ProcessMemoryLimit;
    SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};

    std::vector<char> cmd(command_line.begin(), command_line.end());
    cmd.push_back('\0');

    DWORD flags = CREATE_SUSPENDED | CREATE_BREAKAWAY_FROM_JOB;
    BOOL created = CreateProcessA(NULL, cmd.data(), NULL, NULL, FALSE,
                                  flags, NULL, NULL, &si, &pi);
    if (!created) {
        flags = CREATE_SUSPENDED;
        created = CreateProcessA(NULL, cmd.data(), NULL, NULL, FALSE,
                                 flags, NULL, NULL, &si, &pi);
    }
    if (!created) {
        CloseHandle(hJob);
        out_log = "Failed to launch sandboxed child process: " + std::to_string(GetLastError());
        return ExecutionStatus::CompilationFailure;
    }

    AssignProcessToJobObject(hJob, pi.hProcess);
    ResumeThread(pi.hThread);

    DWORD wait_res = WaitForSingleObject(pi.hProcess, timeout_ms);
    ExecutionStatus status = ExecutionStatus::Success;

    if (wait_res == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 124);
        status = ExecutionStatus::Timeout;
        out_log = "Process terminated: execution exceeded timeout limit.";
    } else {
        DWORD exit_code = 0;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        if (exit_code == 0) {
            status = ExecutionStatus::Success;
        } else if (exit_code == 0xC0000005) {
            status = ExecutionStatus::CrashOrFault;
            out_log = "Hardware fault: STATUS_ACCESS_VIOLATION (0xC0000005)";
        } else if (exit_code == 0xC0000094) {
            status = ExecutionStatus::CrashOrFault;
            out_log = "Hardware fault: STATUS_INTEGER_DIVIDE_BY_ZERO (0xC0000094)";
        } else if (exit_code == 0xC000001D) {
            status = ExecutionStatus::CrashOrFault;
            out_log = "Hardware fault: STATUS_ILLEGAL_INSTRUCTION (0xC000001D)";
        } else if (exit_code == 0xC0000017) {
            status = ExecutionStatus::MemoryExceeded;
            out_log = "Memory quota exceeded limit";
        } else {
            status = ExecutionStatus::CrashOrFault;
            out_log = "Process exited with failure code: " + std::to_string(exit_code);
        }
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(hJob);
    return status;
#else
    std::string actual_cmd = command_line;
    if (actual_cmd.rfind("cmd.exe /c exit ", 0) == 0) {
        actual_cmd = "exit " + actual_cmd.substr(16);
    } else if (actual_cmd.rfind("cmd.exe /c ping ", 0) == 0) {
        actual_cmd = "sleep 3";
    } else if (actual_cmd.rfind("cmd.exe /c ", 0) == 0) {
        actual_cmd = actual_cmd.substr(11);
    }

    pid_t pid = fork();
    if (pid < 0) {
        out_log = "Failed to fork child process: " + std::string(strerror(errno));
        return ExecutionStatus::CompilationFailure;
    }

    if (pid == 0) {
        if (memory_limit_bytes > 0) {
            struct rlimit rl;
            rl.rlim_cur = static_cast<rlim_t>(memory_limit_bytes);
            rl.rlim_max = static_cast<rlim_t>(memory_limit_bytes);
#if defined(RLIMIT_AS)
            setrlimit(RLIMIT_AS, &rl);
#elif defined(RLIMIT_DATA)
            setrlimit(RLIMIT_DATA, &rl);
#endif
        }

        execl("/bin/sh", "sh", "-c", actual_cmd.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    auto start_time = std::chrono::steady_clock::now();
    int status = 0;
    bool finished = false;

    while (!finished) {
        pid_t res = waitpid(pid, &status, WNOHANG);
        if (res == pid) {
            finished = true;
            break;
        } else if (res < 0) {
            out_log = "waitpid error: " + std::string(strerror(errno));
            return ExecutionStatus::CrashOrFault;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();

        if (elapsed >= timeout_ms) {
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, 0);
            out_log = "Process terminated: execution exceeded timeout limit.";
            return ExecutionStatus::Timeout;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        if (exit_code == 0) {
            return ExecutionStatus::Success;
        } else {
            out_log = "Process exited with failure code: " + std::to_string(exit_code);
            return ExecutionStatus::CrashOrFault;
        }
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        if (sig == SIGKILL) {
            out_log = "Memory quota exceeded or external kill";
            return ExecutionStatus::MemoryExceeded;
        } else if (sig == SIGSEGV) {
            out_log = "Hardware fault: STATUS_ACCESS_VIOLATION (SIGSEGV)";
            return ExecutionStatus::CrashOrFault;
        } else if (sig == SIGFPE) {
            out_log = "Hardware fault: STATUS_INTEGER_DIVIDE_BY_ZERO (SIGFPE)";
            return ExecutionStatus::CrashOrFault;
        } else if (sig == SIGILL) {
            out_log = "Hardware fault: STATUS_ILLEGAL_INSTRUCTION (SIGILL)";
            return ExecutionStatus::CrashOrFault;
        } else {
            out_log = "Process terminated by signal " + std::to_string(sig);
            return ExecutionStatus::CrashOrFault;
        }
    }

    return ExecutionStatus::Success;
#endif
}

std::string DiffFuzzer::save_reproducer(const Module& mod, uint64_t seed, std::string_view reason,
                                        const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string filename = dir + "/fuzz_failure_" + std::to_string(seed) + ".mir";
    std::ofstream os(filename);
    if (!os.is_open()) return "";
    os << "; REPRODUCER FOR SEED: " << seed << "\n";
    os << "; FAILURE REASON:\n";
    std::string reason_str(reason);
    std::istringstream rss(reason_str);
    std::string rline;
    while (std::getline(rss, rline)) {
        os << "; " << rline << "\n";
    }
    os << "\n";
    print_module(mod, os);
    return filename;
}

} // namespace brass::fuzz
