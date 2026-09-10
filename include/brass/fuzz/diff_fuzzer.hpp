#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/loop_opt.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <memory>

namespace brass::fuzz {

enum class ExecutionStatus : uint8_t {
    Success,
    Timeout,
    MemoryExceeded,
    CrashOrFault,
    VerificationFailure,
    CompilationFailure,
    ExceptionThrown
};

std::string_view status_name(ExecutionStatus status) noexcept;

struct TierResult {
    ExecutionStatus status = ExecutionStatus::Success;
    RuntimeValue value = RuntimeValue::from_void();
    std::string fault_message;
    double duration_ms = 0.0;
};

struct DiffResult {
    bool passed = false;
    TierResult tier0_interp;
    TierResult tier1_jit_unopt;
    TierResult tier2_jit_opt;
    std::string mismatch_reason;
    std::string reproducer_path;
};

struct DiffFuzzerOptions {
    uint32_t timeout_ms = 500;
    size_t memory_limit_bytes = 256 * 1024 * 1024; // 256 MB
    bool tier0_interpreter = true;
    bool tier1_jit_unopt = true;
    bool tier2_jit_opt = true;
    bool sandbox_process = false;
    std::string reproducer_dir = ".";
};

class DiffFuzzer {
public:
    explicit DiffFuzzer(const DiffFuzzerOptions& options = {});
    ~DiffFuzzer();

    /// Runs tri-tier differential execution on a function in module with provided arguments.
    DiffResult run_test(const Module& mod, std::string_view fn_name,
                        const std::vector<RuntimeValue>& args, uint64_t seed);

    /// Tier 0: Pure C++ Reference Interpreter with Cheney GC.
    TierResult run_tier0_interp(const Module& mod, std::string_view fn_name,
                                const std::vector<RuntimeValue>& args);

    /// Tier 1: Native JIT unoptimized.
    TierResult run_tier1_jit_unopt(const Module& mod, std::string_view fn_name,
                                   const std::vector<RuntimeValue>& args);

    /// Tier 2: Native JIT fully optimized (GVN-PRE, PEA, loop fusion/contraction, AVX2, WBE).
    TierResult run_tier2_jit_opt(const Module& mod, std::string_view fn_name,
                                 const std::vector<RuntimeValue>& args);

    /// Protected execution runner that catches hardware faults / SEH exceptions.
    static bool run_protected(const std::function<void()>& action, std::string& fault_msg);

    /// Watchdog runner that executes an action with a strict time limit.
    static bool run_with_watchdog(const std::function<void()>& action, uint32_t timeout_ms);

    /// Sandboxed process execution on Windows (using Windows Job Objects + Memory Limit + Process Timeout).
    static ExecutionStatus run_sandboxed_command(const std::string& command_line,
                                                 uint32_t timeout_ms,
                                                 size_t memory_limit_bytes,
                                                 std::string& out_log);

    /// Saves a reproducer MIR file for a failure.
    static std::string save_reproducer(const Module& mod, uint64_t seed, std::string_view reason,
                                       const std::string& dir = ".");

    const DiffFuzzerOptions& options() const noexcept { return options_; }
    void set_options(const DiffFuzzerOptions& opts) noexcept { options_ = opts; }

private:
    DiffFuzzerOptions options_;
};

} // namespace brass::fuzz
