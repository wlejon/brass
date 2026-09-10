#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/pgo/instrument.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <csetjmp>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace brass::fuzz {

static void fuzz_deopt_exit() {}

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

static LONG WINAPI FuzzVectoredHandler(EXCEPTION_POINTERS* ep) {
    if (s_in_protected && ep && ep->ExceptionRecord && ep->ContextRecord) {
        DWORD code = ep->ExceptionRecord->ExceptionCode;
        if (code == EXCEPTION_ACCESS_VIOLATION ||
            code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
            code == EXCEPTION_ILLEGAL_INSTRUCTION ||
            code == EXCEPTION_FLT_DIVIDE_BY_ZERO ||
            code == EXCEPTION_DATATYPE_MISALIGNMENT) {
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
    else if (code == EXCEPTION_ILLEGAL_INSTRUCTION) ss << " (Illegal Instruction)";
    else if (code == EXCEPTION_FLT_DIVIDE_BY_ZERO) ss << " (Float Divide by Zero)";
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
    try {
        action();
        return true;
    } catch (const std::exception& ex) {
        fault_msg = std::string("C++ Exception: ") + ex.what();
        return false;
    } catch (...) {
        fault_msg = "Unknown Exception";
        return false;
    }
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
    (void)command_line;
    (void)timeout_ms;
    (void)memory_limit_bytes;
    out_log = "Subprocess sandboxing not supported on this platform.";
    return ExecutionStatus::Success;
#endif
}

std::string DiffFuzzer::save_reproducer(const Module& mod, uint64_t seed, std::string_view reason,
                                        const std::string& dir) {
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

DiffFuzzer::DiffFuzzer(const DiffFuzzerOptions& options) : options_(options) {}
DiffFuzzer::~DiffFuzzer() = default;

TierResult DiffFuzzer::run_tier0_interp(const Module& mod, std::string_view fn_name,
                                        const std::vector<RuntimeValue>& args) {
    TierResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    auto res_box = std::make_shared<TierResult>();
    std::string fn_name_str(fn_name);

    auto worker_task = [res_box, &mod, fn_name_str, args]() {
        std::string fault;
        bool prot_ok = run_protected([&]() {
            Interpreter interp;
            interp.set_max_instructions(500'000);
            interp.register_external_function("brass_pgo_inc", [](Interpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_void();
            });
            interp.register_external_function("brass_gc_alloc", [](Interpreter& in, const std::vector<RuntimeValue>& a) {
                size_t sz = a.empty() ? 32 : static_cast<size_t>(a[0].as_u64());
                if (sz > 65536) sz = (sz % 65536) + 32;
                return RuntimeValue::from_gcref(in.allocate_gc(sz, 0, 1));
            });
            res_box->value = interp.run(mod, fn_name_str, args);
            res_box->status = ExecutionStatus::Success;
        }, fault);
        if (!prot_ok) {
            if (fault.find("Maximum instruction execution count exceeded") != std::string::npos) {
                res_box->status = ExecutionStatus::Timeout;
                res_box->fault_message = "Instruction execution limit exceeded in Interpreter";
            } else {
                res_box->status = ExecutionStatus::CrashOrFault;
                res_box->fault_message = std::move(fault);
            }
        }
    };

    if (!run_with_watchdog(worker_task, options_.timeout_ms)) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in Interpreter";
    } else {
        res = std::move(*res_box);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

TierResult DiffFuzzer::run_tier1_jit_unopt(const Module& mod, std::string_view fn_name,
                                          const std::vector<RuntimeValue>& args) {
    TierResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    auto jit = std::make_shared<codegen::JitExecutionEngine>(Target::host());
    jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    jit->register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    jit->register_external_symbol("fuzz_deopt_exit", reinterpret_cast<void*>(&fuzz_deopt_exit));

    try {
        if (!jit->compile_and_load(mod)) {
            res.status = ExecutionStatus::CompilationFailure;
            res.fault_message = "JIT unoptimized compilation failed";
            return res;
        }
    } catch (const std::exception& e) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = std::string("JIT unoptimized compilation exception: ") + e.what();
        return res;
    }

    auto res_box = std::make_shared<TierResult>();
    std::string fn_name_str(fn_name);

    auto worker_task = [jit, res_box, fn_name_str, args]() {
        auto gc = std::make_unique<MiniCheneyGC>(256 * 1024);
        MiniCheneyGC* old_gc = brass_get_active_gc();
        brass_set_active_gc(gc.get());
        struct GcGuard {
            MiniCheneyGC* old;
            ~GcGuard() { brass_set_active_gc(old); }
        } guard{old_gc};

        std::string fault;
        bool prot_ok = run_protected([&]() {
            res_box->value = jit->invoke(fn_name_str, args);
            res_box->status = ExecutionStatus::Success;
        }, fault);
        if (!prot_ok) {
            res_box->status = ExecutionStatus::CrashOrFault;
            res_box->fault_message = std::move(fault);
        }
    };

    if (!run_with_watchdog(worker_task, options_.timeout_ms)) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in JIT unoptimized";
    } else {
        res = std::move(*res_box);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

TierResult DiffFuzzer::run_tier2_jit_opt(const Module& mod, std::string_view fn_name,
                                        const std::vector<RuntimeValue>& args) {
    TierResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    auto opt_mod = clone_module(mod);
    if (!opt_mod) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = "Failed to clone module for Tier 2 optimization";
        return res;
    }

    LoopOptOptions opt_opts;
    opt_opts.enable_gvn = true;
    opt_opts.enable_sccp = true;
    opt_opts.enable_cfg_simplify = true;
    opt_opts.enable_licm = true;
    opt_opts.enable_ivsr = true;
    opt_opts.enable_dce = true;
    opt_opts.enable_diamond_select = true;
    opt_opts.enable_unroll = true;
    opt_opts.enable_slp = true;
    opt_opts.enable_vectorize = true;
    opt_opts.enable_avx2 = true;
    opt_opts.enable_fma = true;
    opt_opts.enable_partial_escape = true;
    opt_opts.enable_allocation_sinking = true;
    opt_opts.enable_loop_fusion = true;
    opt_opts.enable_loop_distribution = true;
    opt_opts.enable_array_contraction = true;
    try {
        DiagnosticReporter diag_pre;
        gvn_pre_module(*opt_mod);
        if (!verify_module(*opt_mod, &diag_pre)) {
            res.status = ExecutionStatus::VerificationFailure;
            res.fault_message = "Verification failed after GVN-PRE: " + diag_pre.format_all();
            return res;
        }

        WriteBarrierElimination wbe;
        wbe.run_on_module(*opt_mod);

        optimize_module(*opt_mod, opt_opts);

        DiagnosticReporter diag;
        if (!verify_module(*opt_mod, &diag)) {
            res.status = ExecutionStatus::VerificationFailure;
            res.fault_message = "Verification failed after Tier 2 optimization: " + diag.format_all();
            return res;
        }

        auto jit = std::make_shared<codegen::JitExecutionEngine>(Target::host());
        jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
        jit->register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
        jit->register_external_symbol("fuzz_deopt_exit", reinterpret_cast<void*>(&fuzz_deopt_exit));

        if (!jit->compile_and_load(*opt_mod)) {
            res.status = ExecutionStatus::CompilationFailure;
            res.fault_message = "JIT optimized compilation failed";
            return res;
        }

        auto res_box = std::make_shared<TierResult>();
        std::string fn_name_str(fn_name);

        auto worker_task = [jit, res_box, fn_name_str, args]() {
            auto gc = std::make_unique<MiniCheneyGC>(256 * 1024);
            MiniCheneyGC* old_gc = brass_get_active_gc();
            brass_set_active_gc(gc.get());
            struct GcGuard {
                MiniCheneyGC* old;
                ~GcGuard() { brass_set_active_gc(old); }
            } guard{old_gc};

            std::string fault;
            bool prot_ok = run_protected([&]() {
                res_box->value = jit->invoke(fn_name_str, args);
                res_box->status = ExecutionStatus::Success;
            }, fault);
            if (!prot_ok) {
                res_box->status = ExecutionStatus::CrashOrFault;
                res_box->fault_message = std::move(fault);
            }
        };

        if (!run_with_watchdog(worker_task, options_.timeout_ms)) {
            res.status = ExecutionStatus::Timeout;
            res.fault_message = "Watchdog timeout exceeded in JIT optimized";
        } else {
            res = std::move(*res_box);
        }
    } catch (const std::exception& e) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = std::string("JIT optimized compilation exception: ") + e.what();
        return res;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

static bool values_match(const RuntimeValue& v1, const RuntimeValue& v2) {
    if (v1.is_f64()) {
        double d1 = v1.as_f64();
        double d2 = v2.as_f64();
        if (std::isnan(d1)) return std::isnan(d2);
        return (v1.raw_bits() == v2.raw_bits()) || (std::abs(d1 - d2) < 1e-6);
    }
    if (v1.is_f32()) {
        float f1 = v1.as_f32();
        float f2 = v2.as_f32();
        if (std::isnan(f1)) return std::isnan(f2);
        return (v1.raw_bits() == v2.raw_bits()) || (std::abs(f1 - f2) < 1e-6f);
    }
    if (v1.is_vector()) {
        return v1 == v2;
    }
    return v1.raw_bits() == v2.raw_bits();
}

DiffResult DiffFuzzer::run_test(const Module& mod, std::string_view fn_name,
                                const std::vector<RuntimeValue>& args, uint64_t seed) {
    DiffResult result;

    DiagnosticReporter ver_diag;
    if (!verify_module(mod, &ver_diag)) {
        result.passed = false;
        result.mismatch_reason = "Initial module verification failed: " + ver_diag.format_all();
        result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
        return result;
    }

    if (options_.tier0_interpreter) {
        result.tier0_interp = run_tier0_interp(mod, fn_name, args);
    }
    if (options_.tier1_jit_unopt) {
        result.tier1_jit_unopt = run_tier1_jit_unopt(mod, fn_name, args);
    }
    if (options_.tier2_jit_opt) {
        result.tier2_jit_opt = run_tier2_jit_opt(mod, fn_name, args);
    }

    // Check status consistency
    if (options_.tier0_interpreter && result.tier0_interp.status != ExecutionStatus::Success) {
        result.passed = false;
        result.mismatch_reason = "Tier 0 (Interpreter) failed: " + result.tier0_interp.fault_message;
        result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
        return result;
    }

    if (options_.tier1_jit_unopt && result.tier1_jit_unopt.status != ExecutionStatus::Success) {
        result.passed = false;
        result.mismatch_reason = "Tier 1 (JIT unoptimized) failed: " + result.tier1_jit_unopt.fault_message;
        result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
        return result;
    }

    if (options_.tier2_jit_opt && result.tier2_jit_opt.status != ExecutionStatus::Success) {
        result.passed = false;
        result.mismatch_reason = "Tier 2 (JIT optimized) failed: " + result.tier2_jit_opt.fault_message;
        result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
        return result;
    }

    // Compare results between tiers
    if (options_.tier0_interpreter && options_.tier1_jit_unopt) {
        if (!values_match(result.tier0_interp.value, result.tier1_jit_unopt.value)) {
            result.passed = false;
            std::ostringstream ss;
            ss << "Mismatch between Tier 0 (Interp=" << result.tier0_interp.value
               << ") and Tier 1 (JIT unopt=" << result.tier1_jit_unopt.value << ")";
            result.mismatch_reason = ss.str();
            result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
            return result;
        }
    }

    if (options_.tier1_jit_unopt && options_.tier2_jit_opt) {
        if (!values_match(result.tier1_jit_unopt.value, result.tier2_jit_opt.value)) {
            result.passed = false;
            std::ostringstream ss;
            ss << "Mismatch between Tier 1 (JIT unopt=" << result.tier1_jit_unopt.value
               << ") and Tier 2 (JIT opt=" << result.tier2_jit_opt.value << ")";
            result.mismatch_reason = ss.str();
            result.reproducer_path = save_reproducer(mod, seed, result.mismatch_reason, options_.reproducer_dir);
            return result;
        }
    }

    result.passed = true;
    return result;
}

} // namespace brass::fuzz
