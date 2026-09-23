#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/pgo/instrument.hpp>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

extern "C" void brass_tier1_record_invocation_fb(void* feedback);

namespace brass::codegen {

using namespace brass::x64;

// ============================================================================
// BaselineCompiledFunction Implementation
// ============================================================================

BaselineCompiledFunction::BaselineCompiledFunction(
    std::string_view name,
    Type return_type,
    std::vector<Type> param_types,
    std::shared_ptr<JitMemoryBlock> memory,
    void* entry_point,
    size_t code_size,
    FunctionStackMap stack_map
) : name_(name),
    return_type_(return_type),
    param_types_(std::move(param_types)),
    memory_(std::move(memory)),
    entry_point_(entry_point),
    code_size_(code_size),
    stack_map_(std::move(stack_map)) {}

RuntimeValue BaselineCompiledFunction::invoke(const std::vector<RuntimeValue>& args) const {
    if (!entry_point_) {
        throw std::runtime_error("BaselineCompiledFunction::invoke: entry point is null for " + name_);
    }

    void* addr = entry_point_;

    [[maybe_unused]] const std::vector<Type>* ptypes = param_types_.empty() ? nullptr : &param_types_;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32)
    X64Win64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_x64_win64_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    X64Win64InvokeResult result;
    x64_win64_invoke_thunk(&invoke_args, &result);

    if (return_type_.is_void()) return RuntimeValue::from_void();
    if (return_type_.is_vector()) return RuntimeValue::from_v128(return_type_, result.xmm0);
    if (return_type_.is_float()) {
        if (return_type_.kind() == TypeKind::F32) {
            float f = 0.0f;
            std::memcpy(&f, result.xmm0, sizeof(float));
            return RuntimeValue::from_f32(f);
        } else {
            double d = 0.0;
            std::memcpy(&d, result.xmm0, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
    }
    if (return_type_.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(result.rax));
    if (return_type_.is_pointer()) return RuntimeValue::from_ptr(static_cast<uintptr_t>(result.rax));
    if (return_type_.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(result.rax));
    return RuntimeValue::from_i64(static_cast<int64_t>(result.rax));
#elif defined(__GNUC__) || defined(__clang__)
    X64SysVInvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_x64_sysv_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    X64SysVInvokeResult result;
    x64_sysv_invoke_thunk(&invoke_args, &result);

    if (return_type_.is_void()) return RuntimeValue::from_void();
    if (return_type_.is_vector()) return RuntimeValue::from_v128(return_type_, result.xmm0);
    if (return_type_.is_float()) {
        if (return_type_.kind() == TypeKind::F32) {
            float f = 0.0f;
            std::memcpy(&f, result.xmm0, sizeof(float));
            return RuntimeValue::from_f32(f);
        } else {
            double d = 0.0;
            std::memcpy(&d, result.xmm0, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
    }
    if (return_type_.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(result.rax));
    if (return_type_.is_pointer()) return RuntimeValue::from_ptr(static_cast<uintptr_t>(result.rax));
    if (return_type_.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(result.rax));
    return RuntimeValue::from_i64(static_cast<int64_t>(result.rax));
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__GNUC__) || defined(__clang__)
    AArch64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_aarch64_invoke_args(args, ptypes, addr, invoke_args, stack_words);

    AArch64InvokeResult result;
    aarch64_invoke_thunk(&invoke_args, &result);

    if (return_type_.is_void()) return RuntimeValue::from_void();
    if (return_type_.is_vector()) return RuntimeValue::from_v128(return_type_, result.q0);
    if (return_type_.is_float()) {
        if (return_type_.kind() == TypeKind::F32) {
            float f = 0.0f;
            std::memcpy(&f, result.q0, sizeof(float));
            return RuntimeValue::from_f32(f);
        } else {
            double d = 0.0;
            std::memcpy(&d, result.q0, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
    }
    if (return_type_.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(result.x0));
    if (return_type_.is_pointer()) return RuntimeValue::from_ptr(static_cast<uintptr_t>(result.x0));
    if (return_type_.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(result.x0));
    return RuntimeValue::from_i64(static_cast<int64_t>(result.x0));
#endif
#endif

    auto get_i64 = [&](size_t idx) -> int64_t {
        if (idx >= args.size()) return 0;
        return args[idx].as_i64();
    };

    auto get_f64 = [&](size_t idx) -> double {
        if (idx >= args.size()) return 0.0;
        return args[idx].as_f64();
    };

    // 0 arguments
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

    // 1 argument
    if (args.size() == 1) {
        bool f0 = args[0].is_f64() || args[0].is_f32();
        if (f0) {
            double a0 = get_f64(0);
            if (return_type_.is_void()) {
                reinterpret_cast<void(*)(double)>(addr)(a0);
                return RuntimeValue::from_void();
            } else if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(double)>(addr)(a0);
                return RuntimeValue::from_f64(r);
            } else if (return_type_.kind() == TypeKind::I32) {
                int32_t r = reinterpret_cast<int32_t(*)(double)>(addr)(a0);
                return RuntimeValue::from_i32(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double)>(addr)(a0);
                return RuntimeValue::from_i64(r);
            }
        } else {
            int64_t a0 = get_i64(0);
            if (return_type_.is_void()) {
                reinterpret_cast<void(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_void();
            } else if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_f64(r);
            } else if (return_type_.kind() == TypeKind::I32) {
                int32_t r = reinterpret_cast<int32_t(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_i32(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t)>(addr)(a0);
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 2 arguments
    if (args.size() == 2) {
        bool f0 = args[0].is_f64() || args[0].is_f32();
        bool f1 = args[1].is_f64() || args[1].is_f32();

        if (f0 && f1) {
            double a0 = get_f64(0), a1 = get_f64(1);
            if (return_type_.is_void()) {
                reinterpret_cast<void(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_void();
            } else if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else if (return_type_.kind() == TypeKind::I32) {
                int32_t r = reinterpret_cast<int32_t(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_i32(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (!f0 && !f1) {
            int64_t a0 = get_i64(0), a1 = get_i64(1);
            if (return_type_.is_void()) {
                reinterpret_cast<void(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_void();
            } else if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else if (return_type_.kind() == TypeKind::I32) {
                int32_t r = reinterpret_cast<int32_t(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_i32(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (f0 && !f1) {
            double a0 = get_f64(0); int64_t a1 = get_i64(1);
            if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(double, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, int64_t)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else {
            int64_t a0 = get_i64(0); double a1 = get_f64(1);
            if (return_type_.is_float()) {
                double r = reinterpret_cast<double(*)(int64_t, double)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, double)>(addr)(a0, a1);
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 3 arguments
    if (args.size() == 3) {
        int64_t a0 = get_i64(0), a1 = get_i64(1), a2 = get_i64(2);
        if (return_type_.is_void()) {
            reinterpret_cast<void(*)(int64_t, int64_t, int64_t)>(addr)(a0, a1, a2);
            return RuntimeValue::from_void();
        } else if (return_type_.is_float()) {
            double r = reinterpret_cast<double(*)(int64_t, int64_t, int64_t)>(addr)(a0, a1, a2);
            return RuntimeValue::from_f64(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t)>(addr)(a0, a1, a2);
            return RuntimeValue::from_i64(r);
        }
    }

    // 4 arguments
    if (args.size() == 4) {
        int64_t a0 = get_i64(0), a1 = get_i64(1), a2 = get_i64(2), a3 = get_i64(3);
        if (return_type_.is_void()) {
            reinterpret_cast<void(*)(int64_t, int64_t, int64_t, int64_t)>(addr)(a0, a1, a2, a3);
            return RuntimeValue::from_void();
        } else {
            int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t, int64_t)>(addr)(a0, a1, a2, a3);
            return RuntimeValue::from_i64(r);
        }
    }

    throw std::runtime_error("BaselineCompiledFunction::invoke: unsupported argument count " +
                             std::to_string(args.size()) + " for " + name_);
}

// ============================================================================
// BaselineJitCompiler Implementation
// ============================================================================

BaselineJitCompiler::BaselineJitCompiler(Target target)
    : target_(target) {
    // Register standard symbols
    symbols_["brass_gc_safepoint"] = reinterpret_cast<void*>(&brass_gc_safepoint);
    symbols_["brass_gc_alloc"] = reinterpret_cast<void*>(&brass_gc_alloc);
    symbols_["brass_gc_collect"] = reinterpret_cast<void*>(&brass_gc_collect);
    symbols_["brass_pgo_inc"] = reinterpret_cast<void*>(&brass_pgo_inc);
    symbols_["brass_record_call_feedback"] = reinterpret_cast<void*>(&brass_record_call_feedback);
    symbols_["brass_record_property_feedback"] = reinterpret_cast<void*>(&brass_record_property_feedback);
    symbols_["brass_gc_write_barrier"] = reinterpret_cast<void*>(&brass_default_gc_write_barrier);
    symbols_["brass_tier1_record_invocation_fb"] = reinterpret_cast<void*>(&brass_tier1_record_invocation_fb);
    lazy_ = std::make_shared<LazySymbolTable>([this](std::string_view name) { return resolve_lazy(name); });
}

BaselineJitCompiler::~BaselineJitCompiler() {
    // Compiled code may outlive the compiler; its stubs then resolve nothing
    // new (register_external_symbol is gone with it).
    lazy_->detach();
}

void BaselineJitCompiler::register_external_symbol(std::string_view name, void* addr) {
    {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        symbols_[std::string(name)] = addr;
        // A module function of this name owns its stub.
        if (module_owned_.count(std::string(name))) return;
    }
    // Outside symbols_mutex_: the table resolves through resolve_symbol
    // while holding its own lock.
    lazy_->define(name, addr);
}

void BaselineJitCompiler::set_symbol_resolver(BaselineSymbolResolver resolver) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    custom_resolver_ = std::move(resolver);
}

void BaselineJitCompiler::set_dispatch_table(runtime::FunctionDispatchTable* table) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    dispatch_table_ = table;
}

runtime::FunctionDispatchTable& BaselineJitCompiler::dispatch_table() const {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    return dispatch_table_ ? *dispatch_table_ : runtime::FunctionDispatchTable::instance();
}

void* BaselineJitCompiler::lazy_stub(std::string_view name) {
    return lazy_->stub_for(name);
}

void* BaselineJitCompiler::resolve_symbol(std::string_view name) const {
    BaselineSymbolResolver custom;
    {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        auto it = symbols_.find(std::string(name));
        if (it != symbols_.end()) {
            return it->second;
        }
        custom = custom_resolver_;
    }
    if (custom) {
        void* ptr = custom(name);
        if (ptr) return ptr;
    }
    auto* handle = dispatch_table().find(name);
    if (handle && handle->native_entry()) {
        return handle->native_entry();
    }
    return nullptr;
}

void* BaselineJitCompiler::resolve_symbol_in(const Function& fn, std::string_view name) const {
    if (const Module* mod = fn.parent()) {
        if (const char* data = mod->string_symbol(name)) return const_cast<char*>(data);
        // A function the module defines shadows a registered symbol of the
        // same name (a program function called `sqrt` is not libm's), as in
        // tier 2's linker. Bound directly once this module's copy is
        // compiled; until then through the lazy stub, which compile_module
        // points at the module's copy when it installs it.
        if (const Function* def = mod->get_function(name); def && def->block_count() > 0) {
            runtime::FunctionHandle* handle = dispatch_table().find(name);
            if (handle && handle->mir_function() == def && handle->native_entry()) return handle->native_entry();
            // Not compiled yet (compile_module installs it later, or the
            // tiering layer compiles it one function at a time): the stub,
            // which with shadowing on must not resolve to a registered
            // symbol of this name.
            bool claimed = false;
            {
                std::lock_guard<std::mutex> lock(symbols_mutex_);
                if (module_functions_shadow_) claimed = module_owned_.emplace(name).second;
            }
            // A stub a registered symbol already filled is re-armed.
            // Outside symbols_mutex_ (lock order: table, then symbols).
            if (claimed && lazy_->resolved_target(name)) lazy_->define(name, nullptr);
            return nullptr;
        }
    }
    return resolve_symbol(name);
}

void* BaselineJitCompiler::resolve_lazy(std::string_view name) const {
    bool owned = false;
    {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        owned = module_owned_.count(std::string(name)) != 0;
    }
    // Both take symbols_mutex_ themselves.
    if (!owned) return resolve_symbol(name);
    runtime::FunctionHandle* handle = dispatch_table().find(name);
    return handle ? handle->native_entry() : nullptr;
}

void BaselineJitCompiler::set_module_functions_shadow(bool shadow) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    module_functions_shadow_ = shadow;
}

BaselineCompiledFunction BaselineJitCompiler::compile(const Function& fn) {
    return compile(fn, target_);
}

std::vector<BaselineCompiledFunction> BaselineJitCompiler::compile_module(const Module& mod) {
    return compile_module(mod, target_);
}

std::vector<BaselineCompiledFunction> BaselineJitCompiler::compile_module(const Module& mod, Target target) {
    std::vector<BaselineCompiledFunction> results;
    results.reserve(mod.function_count());

    runtime::FunctionDispatchTable& table = dispatch_table();
    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        table.get_or_create(fn->name(), fn);
    }

    for (auto it = mod.functions().rbegin(); it != mod.functions().rend(); ++it) {
        const auto* fn = *it;
        if (!fn) continue;
        results.push_back(compile(*fn, target));
        // The module is the whole program: a callee that is not one of its
        // functions (installed by this loop) and does not resolve now has
        // nothing left to resolve it. Rejected here rather than trapping
        // when the call runs.
        for (const std::string& sym : results.back().lazy_call_symbols()) {
            if (mod.get_function(sym) || resolve_symbol_in(*fn, sym)) continue;
            throw std::runtime_error("BaselineJitCompiler: function '" + std::string(fn->name()) +
                                     "' references symbol '" + sym +
                                     "', which is neither a function of its module nor a registered symbol");
        }
        auto* handle = table.get_or_create(fn->name(), fn);
        handle->set_native_entry(results.back().entry_point());
        handle->set_tier(runtime::TierLevel::Tier1_Baseline);
        handle->set_baseline_function(std::make_shared<BaselineCompiledFunction>(results.back()));
        // Callers compiled before this function reach it through its stub,
        // which must not fall back to a registered symbol of the same name.
        lazy_->define(fn->name(), results.back().entry_point());
    }

    return results;
}

} // namespace brass::codegen
