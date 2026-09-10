#include <brass/runtime/code_installer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <stdexcept>
#include <iostream>

extern "C" void brass_pgo_inc(uint32_t);

namespace brass::runtime {

namespace {

bool run_tier2_optimization_pipeline(Module& mod) {
    // 1. Initial GVN & GVN-PRE
    GvnOptions gvn_opts;
    gvn_module(mod, gvn_opts);

    GvnPreOptions pre_opts;
    gvn_pre_module(mod, pre_opts);

    // 2. SCCP & Speculation Guard Elimination
    SccpOptions sccp_opts;
    sccp_opts.enable_guard_elim = true;
    sccp_module(mod, sccp_opts);

    // 3. CFG Simplification
    CfgSimplifyOptions cfg_opts;
    cfg_simplify_module(mod, cfg_opts);

    // 4. Loop Unswitching & SSA Jump Threading
    unswitch_loops_in_module(mod);
    jump_thread_module(mod);
    cfg_simplify_module(mod, cfg_opts);

    // 5. High-level Loop Optimizations (Vectorize, SLP, LICM, Contraction)
    LoopOptOptions loop_opts;
    loop_opts.enable_vectorize = true;
    loop_opts.enable_slp = true;
    loop_opts.enable_licm = true;
    loop_opts.enable_fp_reassociation = mod.allow_fp_reassociation();
    optimize_module_loops(mod, loop_opts);

    // 6. Write Barrier Elimination
    WriteBarrierElimination wbe(false);
    wbe.run_on_module(mod);

    // 7. Verification check
    DiagnosticReporter diag;
    return verify_module(mod, &diag);
}

} // namespace

// ============================================================================
// FunctionHandle Implementation
// ============================================================================

FunctionHandle::FunctionHandle(std::string_view name, const Function* mir_fn)
    : name_(name), mir_function_(mir_fn) {
    if (mir_fn) {
        return_type_ = mir_fn->return_type();
        param_types_ = mir_fn->param_types();
    }
}

void FunctionHandle::set_mir_function(const Function* fn) noexcept {
    mir_function_ = fn;
    if (fn) {
        return_type_ = fn->return_type();
        param_types_ = fn->param_types();
    }
}

void FunctionHandle::set_signature(Type ret, std::vector<Type> params) {
    return_type_ = ret;
    param_types_ = std::move(params);
}

void FunctionHandle::set_jit_engine(std::shared_ptr<codegen::JitExecutionEngine> engine) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    jit_engine_ = std::move(engine);
}

std::shared_ptr<codegen::JitExecutionEngine> FunctionHandle::jit_engine() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return jit_engine_;
}

RuntimeValue FunctionHandle::call_native(const std::vector<RuntimeValue>& args) const {
    void* addr = native_entry();
    if (!addr) {
        throw std::runtime_error("FunctionHandle::call_native: native entry is null for " + name_);
    }

    auto engine = jit_engine();
    if (engine) {
        return engine->invoke(name_, args);
    }

    // Direct invocation fallback for common signatures when engine pointer is omitted (e.g. unit tests)
    if (args.empty()) {
        if (return_type_.is_void()) {
            reinterpret_cast<void(*)()>(addr)();
            return RuntimeValue::from_void();
        } else if (return_type_.is_float()) {
            if (return_type_.kind() == TypeKind::F32) {
                float r = reinterpret_cast<float(*)()>(addr)();
                return RuntimeValue::from_f32(r);
            }
            double r = reinterpret_cast<double(*)()>(addr)();
            return RuntimeValue::from_f64(r);
        } else if (return_type_.kind() == TypeKind::I32) {
            int32_t r = reinterpret_cast<int32_t(*)()>(addr)();
            return RuntimeValue::from_i32(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)()>(addr)();
            return RuntimeValue::from_i64(r);
        }
    }

    if (args.size() == 1) {
        if (return_type_.is_float()) {
            double r = reinterpret_cast<double(*)(double)>(addr)(args[0].as_f64());
            return RuntimeValue::from_f64(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)(int64_t)>(addr)(args[0].as_i64());
            return RuntimeValue::from_i64(r);
        }
    }

    if (args.size() == 2) {
        if (return_type_.is_float()) {
            double r = reinterpret_cast<double(*)(double, double)>(addr)(args[0].as_f64(), args[1].as_f64());
            return RuntimeValue::from_f64(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(addr)(args[0].as_i64(), args[1].as_i64());
            return RuntimeValue::from_i64(r);
        }
    }

    throw std::runtime_error("FunctionHandle::call_native: unsupported direct signature for " + name_);
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

// ============================================================================
// FunctionDispatchTable Implementation
// ============================================================================

FunctionDispatchTable& FunctionDispatchTable::instance() {
    static FunctionDispatchTable table;
    return table;
}

FunctionHandle* FunctionDispatchTable::get_or_create(std::string_view name, const Function* fn) {
    std::string key(name);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = handles_.find(key);
    if (it != handles_.end()) {
        if (fn && !it->second->mir_function()) {
            it->second->set_mir_function(fn);
        }
        return it->second.get();
    }
    auto handle = std::make_unique<FunctionHandle>(name, fn);
    auto* ptr = handle.get();
    handles_[key] = std::move(handle);
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
    handles_[key] = std::move(handle);
}

void FunctionDispatchTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    handles_.clear();
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
    : target_(target) {}

void CodeInstaller::register_external_symbol(std::string_view name, void* address) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    external_symbols_[std::string(name)] = address;
}

CodeInstallResult CodeInstaller::install_tier2(
    FunctionHandle& handle,
    const Module& module,
    std::string_view fn_name
) {
    auto mod_copy = clone_module(module);
    if (!mod_copy) {
        return {false, nullptr, "Failed to clone module for Tier-2 compilation", 0};
    }
    return install_tier2(handle, std::move(mod_copy), fn_name);
}

CodeInstallResult CodeInstaller::install_tier2(
    FunctionHandle& handle,
    std::unique_ptr<Module> module,
    std::string_view fn_name
) {
    if (!module) {
        return {false, nullptr, "Module is null", 0};
    }

    Function* target_fn = module->get_function(fn_name);
    if (!target_fn) {
        return {false, nullptr, "Function '" + std::string(fn_name) + "' not found in module", 0};
    }

    handle.set_signature(target_fn->return_type(), target_fn->param_types());

    // 1. Run full Tier-2 optimization passes
    if (!run_tier2_optimization_pipeline(*module)) {
        return {false, nullptr, "Tier-2 optimization pipeline failed or invalidated module", 0};
    }

    // 2. Machine code generation and relocation
    auto jit = std::make_shared<codegen::JitExecutionEngine>(target_);

    // Register essential GC and runtime bridge symbols
    jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    jit->register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    il::register_bronze_runtime_symbols(jit.get());

    {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        for (const auto& [name, addr] : external_symbols_) {
            jit->register_external_symbol(name, addr);
        }
    }

    // Compile and link in executable memory
    if (!jit->compile_and_load(*module)) {
        return {false, nullptr, "JIT compilation or relocation failed", 0};
    }

    // 3. Mark memory executable (PAGE_EXECUTE_READ via OS protection)
    jit->make_executable_read_only();

    // 4. Retrieve compiled entry point
    void* native_code_ptr = jit->get_symbol_address(fn_name);
    if (!native_code_ptr) {
        return {false, nullptr, "Compiled symbol address not found for " + std::string(fn_name), 0};
    }

    // 5. Store engine lifetime holder and atomically publish native entry point
    handle.set_jit_engine(jit);
    handle.set_native_entry(native_code_ptr);
    handle.set_tier(TierLevel::Tier2_Optimized);

    // Also update any other functions in the module if their handles exist in the dispatch table
    for (const Function* fn : module->functions()) {
        if (!fn || fn->name() == fn_name) continue;
        FunctionHandle* other_handle = FunctionDispatchTable::instance().find(fn->name());
        if (other_handle && !other_handle->has_native_entry()) {
            void* other_ptr = jit->get_symbol_address(fn->name());
            if (other_ptr) {
                other_handle->set_jit_engine(jit);
                other_handle->set_native_entry(other_ptr);
                other_handle->set_tier(TierLevel::Tier2_Optimized);
            }
        }
    }

    return {true, native_code_ptr, "", 0};
}

} // namespace brass::runtime
