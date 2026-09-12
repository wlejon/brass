#include <brass/codegen/baseline_jit.hpp>
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

extern "C" void brass_tier1_record_invocation(const char* fn_name);

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
    symbols_["brass_gc_write_barrier"] = reinterpret_cast<void*>(&brass_gc_write_barrier);
    symbols_["brass_tier1_record_invocation"] = reinterpret_cast<void*>(&brass_tier1_record_invocation);
}

void BaselineJitCompiler::register_external_symbol(std::string_view name, void* addr) {
    symbols_[std::string(name)] = addr;
}

void* BaselineJitCompiler::resolve_symbol(std::string_view name) const {
    std::string key(name);
    auto it = symbols_.find(key);
    if (it != symbols_.end()) {
        return it->second;
    }
    if (custom_resolver_) {
        void* ptr = custom_resolver_(name);
        if (ptr) return ptr;
    }
    auto* handle = runtime::FunctionDispatchTable::instance().find(name);
    if (handle && handle->native_entry()) {
        return handle->native_entry();
    }
    return nullptr;
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

    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        runtime::FunctionDispatchTable::instance().get_or_create(fn->name(), fn);
    }

    for (const auto* fn : mod.functions()) {
        if (!fn) continue;
        results.push_back(compile(*fn, target));
        auto* handle = runtime::FunctionDispatchTable::instance().get_or_create(fn->name(), fn);
        handle->set_native_entry(results.back().entry_point());
        handle->set_tier(runtime::TierLevel::Tier1_Baseline);
        handle->set_baseline_function(std::make_shared<BaselineCompiledFunction>(results.back()));
    }

    return results;
}

} // namespace brass::codegen
