#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/pgo/instrument.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>

#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>

namespace brass::fuzz {

static void fuzz_deopt_exit() {}

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool values_match(const RuntimeValue& v1, const RuntimeValue& v2) {
    if (v1.is_f64()) {
        double d1 = v1.as_f64();
        double d2 = v2.as_f64();
        // NaN payloads are not part of the answer.
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

std::string_view failure_kind(const TierResult& r) {
    switch (r.status) {
        case ExecutionStatus::Success: return "mismatch";
        case ExecutionStatus::Timeout: return "timeout";
        case ExecutionStatus::CompilationFailure: return "compile";
        case ExecutionStatus::VerificationFailure: return "verify";
        default: break;
    }
    const std::string& m = r.fault_message;
    if (m.find("Integer Overflow") != std::string::npos) return "int-overflow-trap";
    if (m.find("Integer Divide by Zero") != std::string::npos) return "div-zero-trap";
    if (m.find("Access Violation") != std::string::npos) return "access-violation";
    if (m.find("Illegal Instruction") != std::string::npos) return "illegal-instruction";
    if (m.find("bad_alloc") != std::string::npos) return "bad-alloc";
    if (m.find("Exception") != std::string::npos) return "exception";
    return "fault";
}

std::string describe(const TierResult& r) {
    std::ostringstream ss;
    if (r.status == ExecutionStatus::Success) {
        ss << r.value;
    } else {
        ss << status_name(r.status) << ": " << r.fault_message;
    }
    return ss.str();
}

} // namespace

bool tier_results_match(const TierResult& a, const TierResult& b) {
    if (a.status != b.status) return false;
    if (a.status != ExecutionStatus::Success) return true;
    return values_match(a.value, b.value);
}

DiffFuzzer::DiffFuzzer(const DiffFuzzerOptions& options) : options_(options) {}
DiffFuzzer::~DiffFuzzer() = default;

TierResult DiffFuzzer::run_tier0_interp(const Module& mod, std::string_view fn_name,
                                        const std::vector<RuntimeValue>& args) {
    TierResult res;
    const auto t0 = std::chrono::steady_clock::now();

    auto res_box = std::make_shared<TierResult>();
    std::string fn_name_str(fn_name);

    // The interpreter's own brass_gc_alloc is used as-is: any size clamp or
    // ignored pointer mask here would make it disagree with the JIT's GC.
    auto worker_task = [res_box, &mod, fn_name_str, args]() {
        std::string fault;
        bool prot_ok = run_protected([&]() {
            Interpreter interp;
            interp.set_max_instructions(500'000);
            interp.register_external_function("brass_pgo_inc", [](Interpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_void();
            });
            interp.register_external_function("fuzz_deopt_exit", [](Interpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_void();
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

    if (!run_with_watchdog(worker_task, options_.timeout_ms, WatchdogPolicy::ExitOnTimeout, "Interpreter")) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in Interpreter";
    } else {
        res = std::move(*res_box);
    }
    res.duration_ms = elapsed_ms(t0);
    return res;
}

TierResult DiffFuzzer::run_fast_interp(const Module& mod, std::string_view fn_name,
                                       const std::vector<RuntimeValue>& args) {
    TierResult res;
    const auto t0 = std::chrono::steady_clock::now();

    auto res_box = std::make_shared<TierResult>();
    std::string fn_name_str(fn_name);

    auto worker_task = [res_box, &mod, fn_name_str, args]() {
        std::string fault;
        bool prot_ok = run_protected([&]() {
            FastInterpreter interp;
            // Bytecode runs more instructions than MIR (parallel copies,
            // address arithmetic), so its budget is a multiple of the
            // reference interpreter's.
            interp.set_max_instructions(16'000'000);
            interp.register_external_function("brass_pgo_inc", [](FastInterpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_void();
            });
            interp.register_external_function("fuzz_deopt_exit", [](FastInterpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_void();
            });
            const Function* fn = mod.get_function(fn_name_str);
            if (!fn) throw InterpreterException("Function @" + fn_name_str + " not found");
            interp.set_module(&mod);
            res_box->value = interp.run(*fn, args);
            res_box->status = ExecutionStatus::Success;
        }, fault);
        if (!prot_ok) {
            if (fault.find("Maximum instruction execution count exceeded") != std::string::npos) {
                res_box->status = ExecutionStatus::Timeout;
                res_box->fault_message = "Instruction execution limit exceeded in FastInterpreter";
            } else {
                res_box->status = ExecutionStatus::CrashOrFault;
                res_box->fault_message = std::move(fault);
            }
        }
    };

    if (!run_with_watchdog(worker_task, options_.timeout_ms, WatchdogPolicy::ExitOnTimeout, "FastInterpreter")) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in FastInterpreter";
    } else {
        res = std::move(*res_box);
    }
    res.duration_ms = elapsed_ms(t0);
    return res;
}

TierResult DiffFuzzer::run_jit(const Module& mod, std::string_view fn_name,
                               const std::vector<RuntimeValue>& args, std::string_view label, bool schedule) {
    TierResult res;
    const auto t0 = std::chrono::steady_clock::now();

    auto jit = std::make_shared<codegen::JitExecutionEngine>(Target::host());
    jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    jit->register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    jit->register_external_symbol("fuzz_deopt_exit", reinterpret_cast<void*>(&fuzz_deopt_exit));

    bool compiled = false;
    std::string compile_fault;
    codegen::SchedOptions sched;
    sched.enable_pre_ra = schedule;
    sched.enable_post_ra = schedule;
    const bool compile_ok = run_protected([&]() { compiled = jit->compile_and_load(mod, 0, sched); }, compile_fault);
    if (!compile_ok || !compiled) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = "JIT " + std::string(label) + " compilation failed" +
                            (compile_ok ? std::string() : ": " + compile_fault);
        res.duration_ms = elapsed_ms(t0);
        return res;
    }

    auto res_box = std::make_shared<TierResult>();
    std::string fn_name_str(fn_name);

    auto worker_task = [jit, res_box, fn_name_str, args]() {
        gc::Heap heap;
        gc::HeapScope heap_scope(heap);

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

    if (!run_with_watchdog(worker_task, options_.timeout_ms, WatchdogPolicy::ExitOnTimeout,
                           "JIT " + std::string(label))) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in JIT " + std::string(label);
    } else {
        res = std::move(*res_box);
    }
    res.duration_ms = elapsed_ms(t0);
    return res;
}

TierResult DiffFuzzer::run_tier1_jit_unopt(const Module& mod, std::string_view fn_name,
                                          const std::vector<RuntimeValue>& args) {
    return run_jit(mod, fn_name, args, "unoptimized", false);
}

OptimizeOutcome DiffFuzzer::optimize(const Module& mod,
                                     const std::function<bool(std::string_view, const Module&)>& after_step) {
    struct State {
        std::mutex mtx;
        std::string current;
        std::unique_ptr<Module> module;
        OptimizeOutcome outcome;
    };
    auto state = std::make_shared<State>();
    state->module = clone_module(mod);
    if (!state->module) {
        OptimizeOutcome out;
        out.status = ExecutionStatus::CompilationFailure;
        out.message = "Failed to clone module for optimization";
        return out;
    }
    const FuzzPipeline pipeline = options_.pipeline;
    const std::vector<std::string> skip = options_.skip_passes;

    auto task = [state, pipeline, skip, after_step]() {
        Module& m = *state->module;
        PassPipelineHooks hooks;
        hooks.before_pass = [&](std::string_view name) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->current = std::string(name);
        };
        hooks.after_pass = [&](std::string_view name) {
            DiagnosticReporter diag;
            if (!verify_module(m, &diag)) {
                state->outcome.status = ExecutionStatus::VerificationFailure;
                state->outcome.pass = std::string(name);
                state->outcome.message = "Verification failed after " + std::string(name) + ": " + diag.format_all();
                return false;
            }
            return !after_step || after_step(name, m);
        };
        std::string fault;
        if (!run_protected([&]() { run_fuzz_pipeline(m, pipeline, hooks, skip); }, fault)) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->outcome.status = ExecutionStatus::CrashOrFault;
            state->outcome.pass = state->current;
            state->outcome.message = "Fault in pass " + state->current + ": " + fault;
        }
    };

    OptimizeOutcome out;
    if (!run_with_watchdog(task, options_.pipeline_timeout_ms, WatchdogPolicy::ExitOnTimeout,
                           "optimization pipeline")) {
        // The worker may still own the module; leave the state to it.
        std::lock_guard<std::mutex> lock(state->mtx);
        out.status = ExecutionStatus::Timeout;
        out.pass = state->current;
        out.message = "Pipeline timeout in pass " + state->current;
        return out;
    }
    out = std::move(state->outcome);
    if (out.status == ExecutionStatus::Success) out.module = std::move(state->module);
    return out;
}

TierResult DiffFuzzer::run_tier2_jit_opt(const Module& mod, std::string_view fn_name,
                                        const std::vector<RuntimeValue>& args) {
    OptimizeOutcome opt = optimize(mod);
    if (!opt.module) {
        TierResult res;
        res.status = opt.status;
        res.fault_message = opt.message;
        return res;
    }
    return run_jit(*opt.module, fn_name, args, "optimized", true);
}

std::string DiffFuzzer::bisect_first_bad_step(const Module& mod, std::string_view fn_name,
                                              const std::vector<RuntimeValue>& args,
                                              const TierResult& expected, char runner) {
    // runner: 'i' interpreter, 'j' JIT, 'f' FastInterpreter.
    std::string bad;
    optimize(mod, [&](std::string_view step, const Module& cur) {
        TierResult r = runner == 'j' ? run_jit(cur, fn_name, args, "bisect", true)
                     : runner == 'f' ? run_fast_interp(cur, fn_name, args)
                                     : run_tier0_interp(cur, fn_name, args);
        if (tier_results_match(expected, r)) return true;
        bad = std::string(step);
        return false;
    });
    // The last step's module is the one that failed, so finding no step
    // means the failure did not happen again: the answer is nondeterministic.
    return bad.empty() ? std::string("nondeterministic") : bad;
}

DiffResult DiffFuzzer::run_test(const Module& mod, std::string_view fn_name,
                                const std::vector<RuntimeValue>& args, uint64_t seed) {
    set_watchdog_seed(seed);
    DiffResult result;
    auto fail = [&](std::string cls, std::string reason, std::string pass = {}) {
        result.passed = false;
        result.failure_class = std::move(cls);
        result.failing_pass = std::move(pass);
        std::ostringstream ss;
        ss << reason << "\nCLASS: " << result.failure_class
           << "\nPIPELINE: " << pipeline_name(options_.pipeline);
        if (!options_.skip_passes.empty()) {
            ss << "\nSKIP:";
            for (const std::string& s : options_.skip_passes) ss << ' ' << s;
        }
        ss << "\nARGS:";
        for (const RuntimeValue& a : args) ss << ' ' << static_cast<int64_t>(a.raw_bits());
        result.mismatch_reason = std::move(reason);
        result.reproducer_note = ss.str();
        if (options_.save_reproducers) {
            result.reproducer_path = save_reproducer(mod, seed, result.reproducer_note, options_.reproducer_dir);
        }
        return result;
    };

    DiagnosticReporter ver_diag;
    if (!verify_module(mod, &ver_diag)) {
        return fail("generator:invalid", "Initial module verification failed: " + ver_diag.format_all());
    }

    // The interpreter on the original program is the reference answer.
    result.tier0_interp = run_tier0_interp(mod, fn_name, args);
    if (result.tier0_interp.status != ExecutionStatus::Success) {
        return fail("interp:" + std::string(failure_kind(result.tier0_interp)),
                    "Tier 0 (Interpreter) failed: " + result.tier0_interp.fault_message);
    }
    const TierResult& expected = result.tier0_interp;

    OptimizeOutcome opt;
    if (options_.tier2_jit_opt || options_.tier3_interp_opt || options_.tier5_fast_interp_opt) {
        opt = optimize(mod);
        if (!opt.module) {
            std::string cls = opt.status == ExecutionStatus::VerificationFailure ? "verify"
                            : opt.status == ExecutionStatus::Timeout ? "pass-timeout" : "pass-fault";
            result.tier2_jit_opt.status = opt.status;
            result.tier2_jit_opt.fault_message = opt.message;
            return fail(cls + "@" + opt.pass, opt.message, opt.pass);
        }
    }

    // Optimizer bugs first: they are platform-independent and cheapest to read.
    if (options_.tier3_interp_opt) {
        result.tier3_interp_opt = run_tier0_interp(*opt.module, fn_name, args);
        if (!tier_results_match(expected, result.tier3_interp_opt)) {
            std::string pass = options_.bisect ? bisect_first_bad_step(mod, fn_name, args, expected, 'i') : "?";
            return fail("interp-opt:" + std::string(failure_kind(result.tier3_interp_opt)) + "@" + pass,
                        "Tier 3 (Interpreter on optimized) = " + describe(result.tier3_interp_opt) +
                        ", Tier 0 (Interpreter) = " + describe(expected) + "; first differing step: " + pass,
                        pass);
        }
    }

    // The bytecode tier: its compiler and interpreter share no code with
    // the reference interpreter, so a disagreement is a bytecode bug.
    if (options_.tier4_fast_interp) {
        result.tier4_fast = run_fast_interp(mod, fn_name, args);
        if (!tier_results_match(expected, result.tier4_fast)) {
            return fail("fast:" + std::string(failure_kind(result.tier4_fast)),
                        "Tier 4 (FastInterpreter) = " + describe(result.tier4_fast) +
                        ", Tier 0 (Interpreter) = " + describe(expected));
        }
    }
    if (options_.tier5_fast_interp_opt && opt.module) {
        result.tier5_fast_opt = run_fast_interp(*opt.module, fn_name, args);
        if (!tier_results_match(expected, result.tier5_fast_opt)) {
            std::string pass = options_.bisect ? bisect_first_bad_step(mod, fn_name, args, expected, 'f') : "?";
            return fail("fast-opt:" + std::string(failure_kind(result.tier5_fast_opt)) + "@" + pass,
                        "Tier 5 (FastInterpreter on optimized) = " + describe(result.tier5_fast_opt) +
                        ", Tier 0 (Interpreter) = " + describe(expected) + "; first differing step: " + pass,
                        pass);
        }
    }

    if (options_.tier1_jit_unopt) {
        result.tier1_jit_unopt = run_tier1_jit_unopt(mod, fn_name, args);
        if (!tier_results_match(expected, result.tier1_jit_unopt)) {
            // Machine code whose answer changes between runs is a different
            // bug from one that is consistently wrong; keep them apart.
            const TierResult again = run_tier1_jit_unopt(mod, fn_name, args);
            const bool flaky = !tier_results_match(result.tier1_jit_unopt, again);
            return fail(std::string(flaky ? "jit-unopt-flaky:" : "jit-unopt:") +
                            std::string(failure_kind(result.tier1_jit_unopt)),
                        "Tier 1 (JIT unopt) = " + describe(result.tier1_jit_unopt) +
                        ", Tier 0 (Interpreter) = " + describe(expected));
        }
    }

    if (options_.tier6_baseline) {
        result.tier6_baseline = run_baseline(mod, fn_name, args, result.baseline_rejected);
        if (!result.baseline_rejected && !tier_results_match(expected, result.tier6_baseline)) {
            return fail("baseline:" + std::string(failure_kind(result.tier6_baseline)),
                        "Tier 6 (baseline JIT) = " + describe(result.tier6_baseline) +
                            ", Tier 0 (Interpreter) = " + describe(expected));
        }
    }

    if (options_.tier2_jit_opt) {
        result.tier2_jit_opt = run_jit(*opt.module, fn_name, args, "optimized", true);
        if (!tier_results_match(expected, result.tier2_jit_opt)) {
            // Bisecting re-runs the JIT after every step, which only names
            // the right pass when the answers are reproducible. A program
            // the unoptimized JIT sometimes gets wrong, or optimized code
            // that answers differently each run, is a backend bug.
            for (int k = 0; k < 2; ++k) {
                const TierResult unopt = run_tier1_jit_unopt(mod, fn_name, args);
                if (!tier_results_match(expected, unopt)) {
                    return fail("jit-unopt-flaky:" + std::string(failure_kind(unopt)),
                                "Tier 1 (JIT unopt) on a re-run = " + describe(unopt) +
                                    ", Tier 0 (Interpreter) = " + describe(expected));
                }
            }
            const TierResult again = run_jit(*opt.module, fn_name, args, "optimized", true);
            if (!tier_results_match(result.tier2_jit_opt, again)) {
                return fail("jit-opt-flaky:" + std::string(failure_kind(result.tier2_jit_opt)),
                            "Tier 2 (JIT opt) = " + describe(result.tier2_jit_opt) + ", then " + describe(again) +
                                ", Tier 0 (Interpreter) = " + describe(expected));
            }
            std::string pass = options_.bisect ? bisect_first_bad_step(mod, fn_name, args, expected, 'j') : "?";
            return fail("jit-opt:" + std::string(failure_kind(result.tier2_jit_opt)) + "@" + pass,
                        "Tier 2 (JIT opt) = " + describe(result.tier2_jit_opt) +
                        ", Tier 0 (Interpreter) = " + describe(expected) + "; first differing step: " + pass,
                        pass);
        }
    }

    result.passed = true;
    return result;
}

} // namespace brass::fuzz
