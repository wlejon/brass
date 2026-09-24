#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <unordered_set>

extern "C" void brass_pgo_inc(uint32_t);

namespace brass::runtime {

namespace {

// The tier-2 JIT pipeline: feedback-driven speculative devirtualization
// (from the type feedback of the program being compiled for), the scalar and
// CFG passes, the default loop stage and write-barrier elimination.
Pipeline tier2_pipeline(const FeedbackRegistry& feedback) {
    LoopOptOptions loop_opts;
    Pipeline p;
    p.add(passes::speculative_devirtualization(feedback));
    p.add(passes::gvn());
    p.add(passes::gvn_pre());
    p.add(passes::sccp(true));
    p.add(passes::cfg_simplify());
    p.add(passes::loop_unswitch(loop_opts, false));
    p.add(passes::jump_threading(loop_opts, false));
    p.add(passes::cfg_simplify("cfg_simplify 2"));
    p.append(loop_pipeline(loop_opts));
    p.add(passes::write_barrier_elim());
    return p;
}

bool run_tier2_optimization_pipeline(Module& mod, const FeedbackRegistry& feedback, std::string& errors) {
    run_pipeline(mod, tier2_pipeline(feedback));
    DiagnosticReporter diag;
    if (verify_module(mod, &diag)) return true;
    errors = diag.format_all();
    return false;
}

// The prefix of the symbol a canonicalized func_addr links against.
constexpr std::string_view kCanonicalFnPtrPrefix = "brass.fn_ptr:";

// Makes every func_addr of `mod` that names `fn_name` or a sibling (a
// function whose handle is bound to the Function this code was compiled
// from) yield the program's one address of that function, its module
// function stub, which is what func_addr yields in Tier 0 and Tier 1: a
// pointer is then equal to every other tier's pointer to the same function
// (a speculative identity check on it holds whichever tier made it), where
// the engine's own copy of the function would not be. The stub reaches the
// handle's current code, compiling it on demand. Each such func_addr links
// against an external symbol of `jit` bound to the stub.
void canonicalize_function_addresses(Module& mod, std::string_view fn_name, const Tier2Bindings& bindings,
                                     MultiTierPipeline& pipeline, codegen::JitExecutionEngine& jit) {
    std::unordered_map<std::string, std::string> canonical;  // name -> symbol
    auto symbol_for = [&](std::string_view name) -> const std::string* {
        const std::string key(name);
        if (auto it = canonical.find(key); it != canonical.end()) return &it->second;
        const Function* def = mod.get_function(name);
        if (!def || def->block_count() == 0) return nullptr;
        const bool known = name == fn_name ? bindings.target != nullptr : bindings.siblings.count(key) != 0;
        if (!known) return nullptr;
        // No Function: the handle exists and keeps what it is bound to.
        void* stub = pipeline.function_address(name, nullptr);
        if (!stub) return nullptr;
        std::string sym = std::string(kCanonicalFnPtrPrefix) + key;
        jit.register_external_symbol(sym, stub);
        return &canonical.emplace(key, std::move(sym)).first->second;
    };
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        for (BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst || inst->opcode() != Opcode::func_addr) continue;
                if (const std::string* sym = symbol_for(inst->symbol())) {
                    inst->set_symbol(mod.string_pool().intern(*sym));
                }
            }
        }
    }
}

} // namespace

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
#elif (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__GNUC__) || defined(__clang__))
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

// ============================================================================
// FunctionDispatchTable Implementation
// ============================================================================

namespace {
// Lets ~Module skip the table when it was never built or is already gone
// (a static Module can outlive the function-local static table).
std::atomic<bool> g_dispatch_table_alive{false};

// The live owned tables, for ~Module's check. Never freed: modules destroyed
// during static destruction still consult it.
struct OwnedTables {
    std::mutex mutex;
    std::vector<FunctionDispatchTable*> tables;
};
OwnedTables& owned_tables() {
    static OwnedTables* t = new OwnedTables();
    return *t;
}
} // namespace

FunctionDispatchTable::FunctionDispatchTable()
    : tiering_(std::make_unique<TieringRegistry>(*this)),
      pipeline_(std::make_unique<MultiTierPipeline>(*this)),
      osr_(std::make_unique<OsrCoordinator>(*tiering_)) {
    auto& reg = owned_tables();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.tables.push_back(this);
}

FunctionDispatchTable::FunctionDispatchTable(DefaultTag) : is_default_(true) {}

FunctionDispatchTable& FunctionDispatchTable::instance() {
    static FunctionDispatchTable table{DefaultTag{}};
    g_dispatch_table_alive.store(true, std::memory_order_release);
    return table;
}

TieringRegistry& FunctionDispatchTable::tiering() const noexcept {
    return tiering_ ? *tiering_ : TieringRegistry::instance();
}

MultiTierPipeline& FunctionDispatchTable::pipeline() const noexcept {
    return pipeline_ ? *pipeline_ : MultiTierPipeline::instance();
}

OsrCoordinator& FunctionDispatchTable::osr() const noexcept {
    return osr_ ? *osr_ : OsrCoordinator::instance();
}

namespace {
thread_local FunctionDispatchTable* t_current_program = nullptr;
} // namespace

FunctionDispatchTable& current_program() noexcept {
    return t_current_program ? *t_current_program : FunctionDispatchTable::instance();
}

ProgramScope::ProgramScope(FunctionDispatchTable& table) noexcept : prev_(t_current_program) {
    t_current_program = &table;
}

ProgramScope::~ProgramScope() {
    t_current_program = prev_;
}

TieringRegistry& tiering_of(FunctionDispatchTable* table) noexcept {
    return table ? table->tiering() : TieringRegistry::instance();
}

FunctionDispatchTable::~FunctionDispatchTable() {
    if (is_default_) {
        g_dispatch_table_alive.store(false, std::memory_order_release);
        return;
    }
    {
        auto& reg = owned_tables();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto& v = reg.tables;
        v.erase(std::remove(v.begin(), v.end(), this), v.end());
    }
    // Stop compiling for this program while its handles are still live (an
    // in-flight tier-2 compile installs into them and publishes its stack
    // maps here; queued ones are dropped), then drop its stack maps and
    // baseline code.
    pipeline_->release_program();
    // Release this program's code and deopt resumers, and make every
    // interpreter's cached handle pointer stale before the memory goes.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, handle] : handles_) {
        if (handle) handle->retire();
    }
    handles_.clear();
    retired_.clear();
    bump_registry_generation();
}

std::string FunctionDispatchTable::handle_into(const Module& mod) const {
    const auto& fns = mod.functions();
    if (fns.empty()) return {};
    std::unordered_set<const Function*> dying(fns.begin(), fns.end());
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [name, handle] : handles_) {
        if (handle && dying.count(handle->mir_function()) != 0) return name;
    }
    return {};
}

void FunctionDispatchTable::forget_module(const Module& mod) {
    if (!is_default_) {
        throw std::logic_error("FunctionDispatchTable::forget_module: only the default program detaches "
                               "modules; an owned table must be destroyed before its modules");
    }
    const auto& fns = mod.functions();
    if (fns.empty()) return;
    std::unordered_set<const Function*> dying(fns.begin(), fns.end());
    std::lock_guard<std::mutex> lock(mutex_);
    bool changed = false;
    for (auto& [_, handle] : handles_) {
        if (!handle) continue;
        handle->forget_deopt_functions(dying);
        if (dying.count(handle->mir_function()) != 0) {
            handle->detach_mir_function();
            changed = true;
        }
    }
    if (changed) bump_registry_generation();
}

void FunctionDispatchTable::forget_deopt_functions(const Module& mod) {
    const auto& fns = mod.functions();
    if (fns.empty()) return;
    std::unordered_set<const Function*> dying(fns.begin(), fns.end());
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, handle] : handles_) {
        if (handle) handle->forget_deopt_functions(dying);
    }
}

void FunctionDispatchTable::retire_locked(std::unique_ptr<FunctionHandle> handle) {
    if (!handle) return;
    handle->retire();
    retired_.push_back(std::move(handle));
}

void forget_module(const Module& mod) noexcept {
    std::string routed;
    try {
        if (g_dispatch_table_alive.load(std::memory_order_acquire)) {
            FunctionDispatchTable::instance().forget_module(mod);
        }
        TieringRegistry::forget_module(&mod);
        MultiTierPipeline::forget_module(&mod);
        auto& reg = owned_tables();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (FunctionDispatchTable* t : reg.tables) {
            // A program may name `mod` as its active module or keep a
            // Tier-0 interpreter on it without routing a handle into it.
            t->tiering().forget(&mod);
            t->pipeline().forget(&mod);
            // Tier-2 code a handle was rebound away from may still deopt:
            // it must not resume in a dead Function.
            t->forget_deopt_functions(mod);
            if (routed.empty()) routed = t->handle_into(mod);
        }
    } catch (...) {
        // Allocation failure while building the lookup set: leaving a stale
        // pointer behind beats throwing out of a destructor.
    }
    if (!routed.empty()) {
        std::fprintf(stderr,
                     "brass: fatal: module '%s' destroyed while a live dispatch table still routes '%s' "
                     "to it; destroy the table before its modules\n",
                     std::string(mod.name()).c_str(), routed.c_str());
        std::fflush(stderr);
        std::abort();
    }
}

FunctionHandle* FunctionDispatchTable::get_or_create(std::string_view name, const Function* fn) {
    std::string key(name);
    FunctionHandle* ptr = nullptr;
    bool retired_code = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = handles_.find(key);
        if (it != handles_.end()) {
            FunctionHandle& h = *it->second;
            if (fn && h.mir_function() != fn) {
                // A handle is keyed by name, so another function of that
                // name (a second module, or a module that replaced a
                // destroyed one) takes it over. Code compiled from the
                // function it was bound to is not this one's: retire it.
                // A handle never bound (a host installed code by name
                // first) keeps its code.
                if (h.mir_function() || h.mir_detached()) {
                    retired_code = h.rebind_mir_function(fn);
                } else {
                    h.set_mir_function(fn);
                }
                bump_registry_generation();
            }
            ptr = &h;
        } else {
            auto handle = std::make_unique<FunctionHandle>(name, fn);
            ptr = handle.get();
            handles_[key] = std::move(handle);
            bump_registry_generation();
        }
    }
    // The lazy stub, the program's pointer to the function, still jumps to
    // the retired code: re-arm it so it compiles `fn` on its next call.
    if (retired_code) {
        const auto& lazy = pipeline().baseline_compiler().lazy_symbols();
        if (lazy && lazy->resolved_target(name)) lazy->define(name, nullptr);
    }
    return ptr;
}

FunctionHandle* FunctionDispatchTable::find(std::string_view name) const {
    std::string key(name);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = handles_.find(key);
    if (it != handles_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool FunctionDispatchTable::has(std::string_view name) const {
    return find(name) != nullptr;
}

void FunctionDispatchTable::register_handle(std::unique_ptr<FunctionHandle> handle) {
    if (!handle) return;
    std::string key(handle->name());
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = handles_[key];
    retire_locked(std::move(slot));
    slot = std::move(handle);
    bump_registry_generation();
}

void FunctionDispatchTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, handle] : handles_) retire_locked(std::move(handle));
    handles_.clear();
    // The reverse table names functions of the program just cleared: a
    // stale address would map back to a same-named function of the next
    // one. Lazy stubs re-register on every func_addr, tier-2 code on
    // install, so the live program's addresses come back as it runs.
    code_addresses_.clear();
    bump_registry_generation();
}

void FunctionDispatchTable::register_code_address(const void* addr, std::string_view name) {
    if (!addr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    code_addresses_[addr] = std::string(name);
}

std::string FunctionDispatchTable::function_name_at(const void* addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = code_addresses_.find(addr);
    return it != code_addresses_.end() ? it->second : std::string();
}

size_t FunctionDispatchTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return handles_.size();
}

std::vector<FunctionHandle*> FunctionDispatchTable::all_handles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FunctionHandle*> result;
    result.reserve(handles_.size());
    for (const auto& [_, handle] : handles_) {
        result.push_back(handle.get());
    }
    return result;
}

// ============================================================================
// CodeInstaller Implementation
// ============================================================================

CodeInstaller::CodeInstaller(const Target& target)
    : target_(target), table_(&FunctionDispatchTable::instance()) {}

CodeInstaller::CodeInstaller(FunctionDispatchTable& table, const Target& target)
    : target_(target), table_(&table) {}

void CodeInstaller::register_external_symbol(std::string_view name, void* address) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    external_symbols_[std::string(name)] = address;
}

static CodeInstallResult foreign_target(const FunctionHandle& handle, std::string_view fn_name,
                                        const Module& module) {
    return {false, nullptr,
            "handle '" + std::string(handle.name()) + "' is not bound to function '" + std::string(fn_name) +
                "' of the module compiled for it ('" + std::string(module.name()) + "')",
            0};
}

CodeInstallResult CodeInstaller::install_tier2(
    FunctionHandle& handle,
    const Module& module,
    std::string_view fn_name
) {
    Tier2Bindings bindings = capture_tier2_bindings(handle, module, fn_name, &module);
    if (bindings.target_foreign) return foreign_target(handle, fn_name, module);
    auto mod_copy = clone_module(module);
    if (!mod_copy) {
        return {false, nullptr, "Failed to clone module for Tier-2 compilation", 0};
    }
    return install_tier2(handle, std::move(mod_copy), fn_name, bindings);
}

CodeInstallResult CodeInstaller::install_tier2(
    FunctionHandle& handle,
    std::unique_ptr<Module> module,
    std::string_view fn_name
) {
    if (!module) {
        return {false, nullptr, "Module is null", 0};
    }
    Tier2Bindings bindings = capture_tier2_bindings(handle, *module, fn_name, nullptr);
    return install_tier2(handle, std::move(module), fn_name, bindings);
}

Tier2Bindings CodeInstaller::capture_tier2_bindings(const FunctionHandle& handle, const Module& module,
                                                     std::string_view fn_name, const Module* source) const {
    Tier2Bindings b;
    b.target = handle.mir_function();
    if (const Function* t = b.target) {
        const Module* owner = t->parent();
        b.target_foreign = source ? t != source->get_function(fn_name)
                                  : t->name() != fn_name || !owner || owner->name() != module.name();    }
    for (const Function* fn : module.functions()) {
        if (!fn || fn->name() == fn_name) continue;
        const FunctionHandle* other = table_->find(fn->name());
        if (!other) continue;
        const Function* other_fn = other->mir_function();
        if (source && other_fn != source->get_function(fn->name())) continue;
        b.siblings.emplace(std::string(fn->name()), other_fn);
    }
    return b;
}

CodeInstallResult CodeInstaller::install_tier2(
    FunctionHandle& handle,
    std::unique_ptr<Module> module,
    std::string_view fn_name,
    const Tier2Bindings& bindings
) {
    if (!module) {
        return {false, nullptr, "Module is null", 0};
    }

    // The Function the handle was bound to when the module was cloned: the
    // code is validated against it, its frames deoptimize into it, and it is
    // published only if the handle is still bound to it when compilation
    // finishes.
    const Function* bound = bindings.target;

    // Compile errors are results, never exceptions: a background worker
    // and the invocation hook in native code both call this. A rejected
    // Function is not tried again by automatic tier-up (a handle rebound
    // since stays eligible).
    auto reject = [&](std::string msg) {
        handle.mark_tier2_rejected(bound);
        return CodeInstallResult{false, nullptr, std::move(msg), 0};
    };

    Function* target_fn = module->get_function(fn_name);
    if (!target_fn) {
        return {false, nullptr, "Function '" + std::string(fn_name) + "' not found in module", 0};
    }
    // Bound to another module's Function: never compiled for it, nor held
    // against it.
    if (bindings.target_foreign) return foreign_target(handle, fn_name, *module);
    // Rebound while the task was queued: the code could never be published.
    if (handle.mir_function() != bound) {
        return {false, nullptr,
                "handle '" + std::string(handle.name()) + "' was rebound before Tier-2 compilation started", 0};
    }

    std::shared_ptr<codegen::JitExecutionEngine> jit;
    try {
    // 1. Run full Tier-2 optimization passes
    if (std::string errors; !run_tier2_optimization_pipeline(*module, table_->tiering().type_feedback(), errors)) {
        return reject("Tier-2 optimization pipeline failed or invalidated module: " + errors);
    }

    // Every guard of the optimized code must have somewhere to deoptimize
    // to in the function the lower tiers run.
    {
        std::string why;
        if (!deopt_targets_valid(*target_fn, bound, why)) {
            return reject("Tier-2 code for '" + std::string(fn_name) + "' cannot deoptimize: " + why);
        }
    }

    // 2. Machine code generation and relocation
    jit = std::make_shared<codegen::JitExecutionEngine>(target_);

    // Register essential GC and runtime bridge symbols
    jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    jit->register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    jit->register_external_symbol("brass_record_call_feedback", reinterpret_cast<void*>(&brass_record_call_feedback));
    jit->register_external_symbol("brass_record_property_feedback", reinterpret_cast<void*>(&brass_record_property_feedback));
    install_host_symbols(*jit);

    {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        for (const auto& [name, addr] : external_symbols_) {
            jit->register_external_symbol(name, addr);
        }
    }

    // Function pointers this code makes are the ones the lower tiers make.
    // Only in a program whose pipeline runs them: its stubs compile a
    // function with no native entry on demand.
    if (table_->pipeline().is_initialized()) {
        canonicalize_function_addresses(*module, fn_name, bindings, table_->pipeline(), *jit);
    }

    // Compile and link in executable memory
    if (!jit->compile_and_load(*module)) {
        return reject("JIT compilation or relocation failed");
    }
    } catch (const std::exception& e) {
        return reject("Tier-2 compilation of '" + std::string(fn_name) + "' failed: " + e.what());
    }

    // The stack maps belong to the program: its pipeline drops them when
    // the program is destroyed.
    MultiTierPipeline& program_pipeline = table_->pipeline();
    if (program_pipeline.is_initialized()) {
        program_pipeline.add_stack_maps(jit->stack_maps());
    }

    // 3. load_object has already turned the code pages read-execute (W^X)
    //    and left the data pages after them writable.

    // 4. Retrieve compiled entry point
    void* native_code_ptr = jit->get_symbol_address(fn_name);
    if (!native_code_ptr) {
        return {false, nullptr, "Compiled symbol address not found for " + std::string(fn_name), 0};
    }

    // 5. Register the deopt continuation (a failed guard finishes the call
    //    in Tier 0), then store the engine lifetime holder and atomically
    //    publish the native entry point.
    // The resumer is unregistered when the handle retires, which the table
    // does before it goes away, so capturing both raw is safe.
    FunctionDispatchTable* table = table_;
    // A frame resumes in the Function the code was compiled from (its
    // guards were validated against it), even after the handle is rebound
    // to another one; once that Function's module is destroyed the deopt is
    // a fatal error.
    auto register_resumer = [table](FunctionHandle& h, void* entry, const Function* compiled_from_fn) {
        FunctionHandle* hp = &h;
        register_deopt_resumer(entry, [hp, table, entry](const DeoptFrame& frame) -> uint64_t {
            const Function* compiled_from = hp->deopt_function(entry);
            if (!compiled_from) {
                std::fprintf(stderr, "brass: fatal deoptimization error: tier-2 code of '%s' deoptimized but the "
                                     "Function it was compiled from is gone\n",
                             std::string(hp->name()).c_str());
                std::fflush(stderr);
                std::abort();
            }
            // A MIR exception the Tier-0 continuation throws must reach the
            // native callers' landing pads (an `invoke` in tier-2 code) as a
            // native throw does; a C++ exception passes them by.
            try {
                return table->pipeline().resume_after_deopt(*hp, frame, *table, *compiled_from);
            } catch (const InterpreterThrownException& ex) {
                deopt_handler_throw_native(ex.value().raw_bits(), UINTPTR_MAX, std::current_exception());
            } catch (const BrassException& ex) {
                deopt_handler_throw_native(ex.value().raw(), UINTPTR_MAX, std::current_exception());
            }
            return 0;
        });
        h.add_deopt_entry(entry, compiled_from_fn);
    };
    // A func_addr in this code that is not canonicalized (a function no
    // handle knows, or a program whose pipeline is not initialized) yields
    // the engine's own copy of a module function: Tier 0 maps it back to the
    // function by name.
    for (const Function* fn : module->functions()) {
        if (!fn || fn->block_count() == 0) continue;
        if (void* addr = jit->get_symbol_address(fn->name())) table_->register_code_address(addr, fn->name());
    }
    // The resumer is registered before the entry is published (a guard can
    // fail on the first call). If the handle was rebound, detached or
    // retired while this compiled, nothing is published: the code is
    // unreachable and kept alive with the handle.
    register_resumer(handle, native_code_ptr, bound);
    if (!handle.publish_optimized(jit, native_code_ptr, bound, target_fn->return_type(), target_fn->param_types(),
                                  /*require_no_entry=*/false)) {
        return {false, nullptr,
                "handle '" + std::string(handle.name()) + "' was rebound, detached or retired during Tier-2 compilation",
                0};
    }
    table_->tiering().get_feedback(handle.name()).set_tier(TierLevel::Tier2_Optimized);

    // Also publish the module's other functions to their handles, each only
    // while it is bound to the Function recorded when the module was cloned.
    for (const Function* fn : module->functions()) {
        if (!fn || fn->name() == fn_name) continue;
        auto recorded = bindings.siblings.find(std::string(fn->name()));
        if (recorded == bindings.siblings.end()) continue;
        FunctionHandle* other_handle = table_->find(fn->name());
        if (other_handle && !other_handle->has_native_entry() && other_handle->mir_function() == recorded->second) {
            void* other_ptr = jit->get_symbol_address(fn->name());
            const Function* other_fn = recorded->second;
            std::string why;
            if (other_ptr && deopt_targets_valid(*fn, other_fn, why)) {
                register_resumer(*other_handle, other_ptr, other_fn);
                other_handle->publish_optimized(jit, other_ptr, other_fn, fn->return_type(), fn->param_types(),
                                                /*require_no_entry=*/true);
            }
        }
    }

    return {true, native_code_ptr, "", 0};
}

} // namespace brass::runtime
