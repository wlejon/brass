#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/fuzz/fuzz_pipeline.hpp>
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
    TierResult tier0_interp;      // interpreter, original module
    TierResult tier1_jit_unopt;   // JIT, original module
    TierResult tier2_jit_opt;     // JIT, optimized module
    TierResult tier3_interp_opt;  // interpreter, optimized module
    TierResult tier4_fast;        // FastInterpreter (bytecode), original module
    TierResult tier5_fast_opt;    // FastInterpreter (bytecode), optimized module
    std::string mismatch_reason;
    // A stable grouping key for reports, e.g. "verify@jump_threading",
    // "interp-opt:mismatch@loop_fusion", "jit-unopt:fault".
    std::string failure_class;
    // The optimization step blamed for the failure, when one is known.
    std::string failing_pass;
    // The reason plus CLASS / PIPELINE / ARGS lines, as written into the
    // reproducer's header comment (and read back by `brass-fuzz --repro`).
    std::string reproducer_note;
    std::string reproducer_path;
};

struct DiffFuzzerOptions {
    uint32_t timeout_ms = 500;
    size_t memory_limit_bytes = 256 * 1024 * 1024; // 256 MB
    bool tier0_interpreter = true;
    bool tier1_jit_unopt = true;
    bool tier2_jit_opt = true;
    // The interpreter on the optimized module against the interpreter on the
    // original: platform-independent, and it sees optimizer bugs that JIT
    // codegen happens to mask.
    bool tier3_interp_opt = true;
    // The bytecode tier (BytecodeCompiler + FastInterpreter) on the original
    // and on the optimized module: every program is checked interpreter vs
    // FastInterpreter vs JIT.
    bool tier4_fast_interp = true;
    bool tier5_fast_interp_opt = true;
    FuzzPipeline pipeline = FuzzPipeline::AllPasses;
    // Pipeline steps left out (see run_fuzz_pipeline).
    std::vector<std::string> skip_passes;
    // On a wrong answer from an optimized tier, re-run the pipeline checking
    // the answer after every step to name the first step that changes it.
    bool bisect = true;
    // Budget for running the whole optimization pipeline once.
    uint32_t pipeline_timeout_ms = 20000;
    bool sandbox_process = false;
    // Write a reproducer file for every failure into reproducer_dir.
    bool save_reproducers = true;
    std::string reproducer_dir = ".";
};

// The optimized module (or why there is none).
struct OptimizeOutcome {
    std::unique_ptr<Module> module;
    ExecutionStatus status = ExecutionStatus::Success;
    std::string message;
    // The step that broke verification, crashed or hung.
    std::string pass;
};

class DiffFuzzer {
public:
    explicit DiffFuzzer(const DiffFuzzerOptions& options = {});
    ~DiffFuzzer();

    /// Runs every enabled tier on `fn_name` and compares the answers.
    DiffResult run_test(const Module& mod, std::string_view fn_name,
                        const std::vector<RuntimeValue>& args, uint64_t seed);

    /// Tier 0: reference interpreter.
    TierResult run_tier0_interp(const Module& mod, std::string_view fn_name,
                                const std::vector<RuntimeValue>& args);

    /// Tier 4/5: the bytecode tier (FastInterpreter).
    TierResult run_fast_interp(const Module& mod, std::string_view fn_name,
                               const std::vector<RuntimeValue>& args);

    /// Tier 1: native JIT, module as given.
    TierResult run_tier1_jit_unopt(const Module& mod, std::string_view fn_name,
                                   const std::vector<RuntimeValue>& args);

    /// Tier 2: native JIT after the configured optimization pipeline.
    TierResult run_tier2_jit_opt(const Module& mod, std::string_view fn_name,
                                 const std::vector<RuntimeValue>& args);

    /// Runs the configured pipeline on a copy of `mod`, verifying after every
    /// step. `after_step`, when set, is called after each verified step with
    /// the current module and may stop the pipeline by returning false.
    OptimizeOutcome optimize(const Module& mod,
                             const std::function<bool(std::string_view, const Module&)>& after_step = {});

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

    TierResult run_jit(const Module& mod, std::string_view fn_name,
                       const std::vector<RuntimeValue>& args, std::string_view label);
    std::string bisect_first_bad_step(const Module& mod, std::string_view fn_name,
                                      const std::vector<RuntimeValue>& args,
                                      const TierResult& expected, char runner);
};

/// True when two tier results are the same answer.
bool tier_results_match(const TierResult& a, const TierResult& b);

} // namespace brass::fuzz
