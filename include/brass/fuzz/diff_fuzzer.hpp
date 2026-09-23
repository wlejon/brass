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
    TierResult tier6_baseline;    // x64 baseline JIT, original module
    // The baseline tier rejected the program at compile time (an opcode it
    // does not compile); its result is not compared, and its fault_message
    // names the rejected operation.
    bool baseline_rejected = false;
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
    // The x64 baseline JIT (the tiering layer's Tier 1) on the original
    // module. A program it rejects at compile time is skipped, not failed.
    bool tier6_baseline = false;
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

    /// Tier 6: the x64 baseline JIT, module as given. `rejected` is set when
    /// the baseline compiler rejects a function (codegen::UnsupportedOperation).
    TierResult run_baseline(const Module& mod, std::string_view fn_name,
                            const std::vector<RuntimeValue>& args, bool& rejected);

    /// Runs the configured pipeline on a copy of `mod`, verifying after every
    /// step. `after_step`, when set, is called after each verified step with
    /// the current module and may stop the pipeline by returning false.
    OptimizeOutcome optimize(const Module& mod,
                             const std::function<bool(std::string_view, const Module&)>& after_step = {});

    /// Protected execution runner that catches hardware faults / SEH exceptions.
    static bool run_protected(const std::function<void()>& action, std::string& fault_msg);

    /// What run_with_watchdog does with an action that outlives its limit. A
    /// thread cannot be stopped safely in-process (TerminateThread can leave
    /// heap and lock state corrupt, and a still-running action writes into
    /// state its caller has moved on from), so there is no "kill and go on".
    enum class WatchdogPolicy {
        // Report "[FAIL] Seed <current seed> [Watchdog timeout]" on stdout and
        // end the process (exit code 124): a chunked campaign reports the
        // chunk as crashed with that class. For actions that touch state the
        // caller reuses: every fuzz tier.
        ExitOnTimeout,
        // Return false and leave the action running on a detached thread.
        // Only for actions that own everything they touch.
        AbandonOnTimeout,
    };

    /// Watchdog runner that executes an action with a strict time limit.
    /// Returns true if it finished in time. `what` names the action in the
    /// ExitOnTimeout report.
    static bool run_with_watchdog(const std::function<void()>& action, uint32_t timeout_ms,
                                  WatchdogPolicy policy = WatchdogPolicy::AbandonOnTimeout,
                                  std::string_view what = {});

    /// The seed ExitOnTimeout reports; run_test sets it.
    static void set_watchdog_seed(uint64_t seed) noexcept;

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

    // `schedule` runs the pre- and post-RA instruction schedulers, as
    // brass-opt's JIT and AOT paths do by default; the optimized tier uses
    // them so the unoptimized tier checks scheduled code against unscheduled.
    TierResult run_jit(const Module& mod, std::string_view fn_name,
                       const std::vector<RuntimeValue>& args, std::string_view label, bool schedule);
    std::string bisect_first_bad_step(const Module& mod, std::string_view fn_name,
                                      const std::vector<RuntimeValue>& args,
                                      const TierResult& expected, char runner);
};

/// True when two tier results are the same answer.
bool tier_results_match(const TierResult& a, const TierResult& b);

} // namespace brass::fuzz
