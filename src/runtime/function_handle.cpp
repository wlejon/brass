// FunctionHandle: one function's dispatch state (its MIR, its baseline and
// tier-2 code and the engines that own it, its deopt entries) and the native
// call into whatever code it publishes. The dispatch table and the tier-2
// installer are code_installer.cpp.
#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/runtime/deopt.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace brass::runtime {

// ============================================================================
// FunctionHandle Implementation
// ============================================================================

FunctionHandle::FunctionHandle(std::string_view name, const Function* mir_fn)
    : name_(name), mir_function_(mir_fn) {
    if (mir_fn) sig_ = std::make_shared<const Signature>(Signature{mir_fn->return_type(), mir_fn->param_types()});
}

void FunctionHandle::set_mir_function_locked(const Function* fn) {
    mir_function_.store(fn, std::memory_order_release);
    if (fn) sig_ = std::make_shared<const Signature>(Signature{fn->return_type(), fn->param_types()});
}

void FunctionHandle::set_mir_function(const Function* fn) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    set_mir_function_locked(fn);
}

void FunctionHandle::detach_mir_function() noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    set_mir_function_locked(nullptr);
    mir_detached_ = true;
}

std::shared_ptr<const FunctionHandle::Signature> FunctionHandle::signature() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return sig_;
}

bool FunctionHandle::rebind_mir_function(const Function* fn) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    const bool had_code = native_entry() != nullptr || jit_engine_ || baseline_function_;
    // Unpublish first: an interpreter matching `fn` must not reach the old
    // code between the two stores.
    set_native_entry(nullptr);
    set_tier(TierLevel::Tier0_Interpreter);
    // The old tier-2 code's deopt resumers stay registered for frames still
    // running it: each resumes in the Function the code was compiled from
    // (deopt_entries_), not in `fn`.
    if (jit_engine_) retired_engines_.push_back(std::move(jit_engine_));
    jit_engine_.reset();
    if (baseline_function_) retired_baselines_.push_back(std::move(baseline_function_));
    baseline_function_.reset();
    set_mir_function_locked(fn);
    mir_detached_ = false;
    return had_code;
}

void FunctionHandle::set_signature(Type ret, std::vector<Type> params) {
    auto sig = std::make_shared<const Signature>(Signature{ret, std::move(params)});
    std::lock_guard<std::mutex> lock(engine_mutex_);
    sig_ = std::move(sig);
}

void FunctionHandle::set_jit_engine(std::shared_ptr<codegen::JitExecutionEngine> engine) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    // The engine being replaced is retired, never dropped: code compiled
    // while it was installed (a tier-1 caller binds a callee's native entry
    // directly) and frames on some stack may still run it, and its deopt
    // resumers and stack maps stay registered for them. Nothing tracks
    // which code or frames still reach a retired engine (no epoch or
    // quiescence point exists), so none is ever freed while the handle
    // lives: retired_engines_ holds one engine per install_tier2 over
    // installed code and per invalidate_optimized, for the handle's
    // lifetime, and the table's reverse address map keeps one entry per
    // function of each (FunctionDispatchTable::register_code_address).
    // Automatic re-tiering never reinstalls (a deopted function bails
    // out), so only a host calling install_tier2 repeatedly grows these.
    if (jit_engine_ && jit_engine_ != engine) retired_engines_.push_back(std::move(jit_engine_));
    jit_engine_ = std::move(engine);
}

bool FunctionHandle::publish_optimized(std::shared_ptr<codegen::JitExecutionEngine> engine, void* entry,
                                       const Function* compiled_from, Type ret, std::vector<Type> params,
                                       bool require_no_entry) {
    auto sig = std::make_shared<const Signature>(Signature{ret, std::move(params)});
    std::lock_guard<std::mutex> lock(engine_mutex_);
    const bool stale = retired_ || mir_detached_ || mir_function() != compiled_from ||
                       (require_no_entry && native_entry() != nullptr);
    if (stale) {
        if (engine && engine != jit_engine_) retired_engines_.push_back(std::move(engine));
        return false;
    }
    if (jit_engine_ && jit_engine_ != engine) retired_engines_.push_back(std::move(jit_engine_));
    jit_engine_ = std::move(engine);
    // The baseline code stays with the handle: invalidate_optimized falls
    // back to it, and tier-1 callers may be bound to its entry.
    sig_ = std::move(sig);
    set_native_entry(entry);
    set_tier(TierLevel::Tier2_Optimized);
    return true;
}

void FunctionHandle::mark_tier2_rejected() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    tier2_rejected_ = true;
    tier2_rejected_fn_ = mir_function();
}

void FunctionHandle::mark_tier2_rejected(const Function* fn) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    tier2_rejected_ = true;
    tier2_rejected_fn_ = fn;
}

bool FunctionHandle::tier2_rejected() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return tier2_rejected_ && tier2_rejected_fn_ == mir_function();
}

std::shared_ptr<codegen::JitExecutionEngine> FunctionHandle::jit_engine() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return jit_engine_;
}

void FunctionHandle::set_baseline_function(std::shared_ptr<codegen::BaselineCompiledFunction> compiled) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    baseline_function_ = std::move(compiled);
}

std::shared_ptr<codegen::BaselineCompiledFunction> FunctionHandle::baseline_function() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return baseline_function_;
}

void FunctionHandle::retire() noexcept {
    set_native_entry(nullptr);
    mir_function_.store(nullptr, std::memory_order_release);
    std::lock_guard<std::mutex> lock(engine_mutex_);
    retired_ = true;
    for (const DeoptEntry& d : deopt_entries_) unregister_deopt_resumer(d.entry);
    deopt_entries_.clear();
    jit_engine_.reset();
    baseline_function_.reset();
}

void FunctionHandle::invalidate_optimized() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (!jit_engine_ || tier() != TierLevel::Tier2_Optimized) return;
    retired_engines_.push_back(std::move(jit_engine_));
    jit_engine_.reset();
    void* lower = baseline_function_ ? baseline_function_->entry_point() : nullptr;
    set_native_entry(lower);
    set_tier(lower ? TierLevel::Tier1_Baseline : TierLevel::Tier0_Interpreter);
}

size_t FunctionHandle::retired_engine_count() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return retired_engines_.size();
}

void FunctionHandle::add_deopt_entry(void* entry, const Function* compiled_from) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    deopt_entries_.push_back({entry, compiled_from});
}

const Function* FunctionHandle::deopt_function(void* entry) const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    // The latest registration of `entry` wins (an address can be reused
    // only after its engine is gone, which a retained entry never is).
    for (auto it = deopt_entries_.rbegin(); it != deopt_entries_.rend(); ++it) {
        if (it->entry == entry) return it->compiled_from;
    }
    return nullptr;
}

void FunctionHandle::forget_deopt_functions(const std::unordered_set<const Function*>& dying) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    for (DeoptEntry& d : deopt_entries_) {
        if (dying.count(d.compiled_from) != 0) d.compiled_from = nullptr;
    }
}

RuntimeValue FunctionHandle::call_native(const std::vector<RuntimeValue>& args) const {
    // The entry, its owner and its signature are read together: a rebind
    // (which unpublishes the entry and replaces the signature under the
    // same lock) cannot pair this call's code with another's types.
    void* addr = nullptr;
    std::shared_ptr<codegen::JitExecutionEngine> engine;
    std::shared_ptr<const Signature> sig;
    {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        addr = native_entry();
        engine = jit_engine_;
        sig = sig_;
    }
    if (!addr) {
        throw std::runtime_error("FunctionHandle::call_native: native entry is null for " + name_);
    }

    if (engine) {
        return engine->invoke(name_, args);
    }

    const Type ret_type = sig->ret;
    [[maybe_unused]] const std::vector<Type>* ptypes = sig->params.empty() ? nullptr : &sig->params;

    // Every path below enters generated code from C++ (the thunks on each
    // host, Windows ARM64 included, and the cast-based fallback): a native
    // throw's pad search stops here, and stack walks need not unwind the
    // host's stack.
    GeneratedCodeEntryScope entry;

    // Each supported host calls through an ABI-exact thunk; the baseline and
    // cast-based fallbacks below are compiled only for any other host.
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(_WIN32) || defined(__GNUC__) || defined(__clang__))
#if defined(_WIN32)
    codegen::X64Win64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    codegen::partition_x64_win64_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    codegen::X64Win64InvokeResult result;
    codegen::x64_win64_invoke_thunk(&invoke_args, &result);

    return codegen::native_return_value(ret_type, result.rax, result.xmm0);
#elif defined(__GNUC__) || defined(__clang__)
    codegen::X64SysVInvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    codegen::partition_x64_sysv_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    codegen::X64SysVInvokeResult result;
    codegen::x64_sysv_invoke_thunk(&invoke_args, &result);

    return codegen::native_return_value(ret_type, result.rax, result.xmm0);
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    codegen::AArch64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    codegen::partition_aarch64_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    codegen::AArch64InvokeResult result;
    codegen::aarch64_invoke_thunk(&invoke_args, &result);
    return codegen::aarch64_invoke_result_value(ret_type, result);
#else
    auto baseline = baseline_function();
    if (baseline) {
        return baseline->invoke(args);
    }

    // Direct invocation fallback for common signatures when engine pointer is omitted (e.g. unit tests)
    if (args.empty()) {
        if (ret_type.is_void()) {
            reinterpret_cast<void(*)()>(addr)();
            return RuntimeValue::from_void();
        } else if (ret_type.is_float()) {
            if (ret_type.kind() == TypeKind::F32) {
                float r = reinterpret_cast<float(*)()>(addr)();
                return RuntimeValue::from_f32(r);
            }
            double r = reinterpret_cast<double(*)()>(addr)();
            return RuntimeValue::from_f64(r);
        } else if (ret_type.kind() == TypeKind::I32) {
            int32_t r = reinterpret_cast<int32_t(*)()>(addr)();
            return RuntimeValue::from_i32(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)()>(addr)();
            return RuntimeValue::from_i64(r);
        }
    }

    auto get_i64 = [&](size_t idx) -> int64_t {
        if (idx >= args.size()) return 0;
        return args[idx].as_i64();
    };
    auto get_f64 = [&](size_t idx) -> double {
        if (idx >= args.size()) return 0.0;
        return args[idx].as_f64();
    };

    auto is_float_val = [](const RuntimeValue& v) {
        return v.is_f64() || v.is_f32();
    };

    if (args.size() == 1) {
        bool f0 = is_float_val(args[0]);
        if (f0) {
            double a0 = get_f64(0);
            if (ret_type.is_void()) {
                reinterpret_cast<void(*)(double)>(addr)(a0);
                return RuntimeValue::from_void();
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double)>(addr)(a0);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double)>(addr)(a0);
                return RuntimeValue::from_i64(r);
            }
        } else {
            int64_t a0 = get_i64(0);
            if (ret_type.is_void()) {
                reinterpret_cast<void(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_void();
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_i64(r);
            }
        }
    }

    if (args.size() == 2) {
        bool f0 = is_float_val(args[0]), f1 = is_float_val(args[1]);
        if (f0 && f1) {
            double a0 = get_f64(0), a1 = get_f64(1);
            if (ret_type.is_void()) {
                reinterpret_cast<void(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_void();
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (f0 && !f1) {
            double a0 = get_f64(0); int64_t a1 = get_i64(1);
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(double, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (!f0 && f1) {
            int64_t a0 = get_i64(0); double a1 = get_f64(1);
            if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t, double)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, double)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else {
            int64_t a0 = get_i64(0), a1 = get_i64(1);
            if (ret_type.is_void()) {
                reinterpret_cast<void(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_void();
            } else if (ret_type.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        }
    }

    throw std::runtime_error("FunctionHandle::call_native: unsupported direct signature for " + name_);
#endif
}

RuntimeValue FunctionHandle::call(Interpreter& interp, const std::vector<RuntimeValue>& args) {
    void* addr = native_entry();
    if (addr != nullptr) {
        return call_native(args);
    }
    record_call();
    if (mir_function_) {
        return interp.run(*mir_function_, args);
    }
    return RuntimeValue::from_void();
}

RuntimeValue FunctionHandle::call(FastInterpreter& interp, const std::vector<RuntimeValue>& args) {
    void* addr = native_entry();
    if (addr != nullptr) {
        return call_native(args);
    }
    record_call();
    if (mir_function_) {
        return interp.run(*mir_function_, args);
    }
    return RuntimeValue::from_void();
}

} // namespace brass::runtime
