#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

extern "C" void brass_tier1_record_invocation(const char* fn_name) {
    if (!fn_name) return;
    if (!brass::runtime::MultiTierPipeline::instance().is_initialized()) return;
    brass::runtime::MultiTierPipeline::instance().on_invocation(fn_name);
}

namespace brass::runtime {

MultiTierPipeline& MultiTierPipeline::instance() {
    static MultiTierPipeline pipeline;
    return pipeline;
}

void MultiTierPipeline::initialize(const TieringConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    initialized_ = true;

    TieringRegistry::instance().set_default_config(config_);

    // Setup symbol resolution on baseline compiler
    baseline_compiler_.set_symbol_resolver([](std::string_view name) -> void* {
        auto* handle = FunctionDispatchTable::instance().find(name);
        if (handle && handle->native_entry()) {
            return handle->native_entry();
        }
        return nullptr;
    });

    brass_set_active_stack_maps(&active_stack_maps_);
}

void MultiTierPipeline::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_baseline_cache();
    if (BackgroundCompiler::instance().is_running()) {
        BackgroundCompiler::instance().stop();
    }
    initialized_ = false;
}

void MultiTierPipeline::set_config(const TieringConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    TieringRegistry::instance().set_default_config(config_);
}

void MultiTierPipeline::register_baseline_compiled(std::shared_ptr<codegen::BaselineCompiledFunction> fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lock(mutex_);
    baseline_functions_[std::string(fn->name())] = fn;
    active_stack_maps_.add_function(fn->stack_map());
    brass_set_active_stack_maps(&active_stack_maps_);
}

std::shared_ptr<codegen::BaselineCompiledFunction> MultiTierPipeline::find_baseline_compiled(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = baseline_functions_.find(std::string(name));
    if (it != baseline_functions_.end()) {
        return it->second;
    }
    return nullptr;
}

void MultiTierPipeline::clear_baseline_cache() {
    baseline_functions_.clear();
    active_stack_maps_.clear();
}

bool MultiTierPipeline::compile_and_install_tier1(std::string_view fn_name, const Function* fn) {
    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create(fn_name, fn);
    if (!fn) {
        fn = handle->mir_function();
    }
    if (!fn) {
        const Module* active_mod = TieringRegistry::instance().active_module();
        if (active_mod) {
            fn = active_mod->get_function(fn_name);
        }
    }
    if (!fn) {
        return false;
    }

    handle->set_signature(fn->return_type(), fn->param_types());

    auto start = std::chrono::high_resolution_clock::now();
    codegen::BaselineCompiledFunction compiled = baseline_compiler_.compile(*fn, Target::host());
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    auto compiled_ptr = std::make_shared<codegen::BaselineCompiledFunction>(std::move(compiled));
    register_baseline_compiled(compiled_ptr);

    handle->set_baseline_function(compiled_ptr);
    handle->set_native_entry(compiled_ptr->entry_point());
    handle->set_tier(TierLevel::Tier1_Baseline);

    auto& fb = TieringRegistry::instance().get_feedback(fn_name);
    fb.set_tier(TierLevel::Tier1_Baseline);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.tier1_compilations++;
        stats_.total_tier1_compile_time_us += elapsed_us;
    }

    return true;
}

bool MultiTierPipeline::enqueue_tier2(
    std::string_view fn_name,
    const Module* mod,
    FunctionHandle* handle
) {
    const Module* target_mod = mod ? mod : TieringRegistry::instance().active_module();
    if (!target_mod) return false;
    if (!handle) {
        handle = FunctionDispatchTable::instance().get_or_create(fn_name);
    }
    if (!handle) return false;

    bool enqueued = BackgroundCompiler::instance().enqueue(
        fn_name,
        *target_mod,
        handle,
        CompilePriority::Normal,
        TierLevel::Tier2_Optimized
    );

    if (enqueued) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.tier2_compilations++;
    }
    return enqueued;
}

void MultiTierPipeline::on_invocation(std::string_view fn_name) {
    auto* handle = FunctionDispatchTable::instance().find(fn_name);
    if (handle) {
        handle->record_call();
    }

    auto& fb = TieringRegistry::instance().get_feedback(fn_name);
    uint64_t count = fb.record_invocation();

    std::lock_guard<std::mutex> lock(mutex_);
    if (fb.current_tier() == TierLevel::Tier0_Interpreter) {
        stats_.tier0_invocations++;
        if (count >= config_.invocation_tier1_threshold && !fb.is_bailout_set()) {
            compile_and_install_tier1(fn_name);
        }
    } else if (fb.current_tier() == TierLevel::Tier1_Baseline) {
        stats_.tier1_invocations++;
        if (count >= config_.invocation_tier2_threshold && !fb.is_bailout_set()) {
            if (config_.enable_background_compile || TieringRegistry::instance().is_background_compile_enabled()) {
                enqueue_tier2(fn_name, TieringRegistry::instance().active_module(), handle);
            }
        }
    } else if (fb.current_tier() == TierLevel::Tier2_Optimized) {
        stats_.tier2_invocations++;
    }
}

RuntimeValue MultiTierPipeline::execute(
    Module& mod,
    std::string_view entry_fn,
    const std::vector<RuntimeValue>& args
) {
    TieringRegistry::instance().set_active_module(&mod);
    for (const auto* fn : mod.functions()) {
        if (fn) {
            FunctionDispatchTable::instance().get_or_create(fn->name(), fn);
        }
    }

    if (config_.enable_background_compile) {
        BackgroundCompiler::instance().start(config_.jit_threads);
    }

    Interpreter interp;
    il::register_bronze_interpreter_symbols(&interp);

    auto* fn = mod.get_function(entry_fn);
    if (!fn) {
        throw std::runtime_error("MultiTierPipeline::execute: function '" + std::string(entry_fn) + "' not found");
    }

    auto* handle = FunctionDispatchTable::instance().get_or_create(entry_fn, fn);
    RuntimeValue result = handle->call(interp, args);

    if (config_.enable_background_compile) {
        BackgroundCompiler::instance().wait_idle();
    }

    return result;
}

MultiTierStats MultiTierPipeline::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MultiTierPipeline::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MultiTierStats{};
}

void MultiTierPipeline::dump_stats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);
    os << "=== Multi-Tier Execution Pipeline Statistics ===\n";
    os << "  Tier 0 (Interpreter) Invocations: " << stats_.tier0_invocations << "\n";
    os << "  Tier 1 (Baseline JIT) Compilations: " << stats_.tier1_compilations << "\n";
    os << "  Tier 1 (Baseline JIT) Invocations:   " << stats_.tier1_invocations << "\n";
    if (stats_.tier1_compilations > 0) {
        double avg_us = static_cast<double>(stats_.total_tier1_compile_time_us) / static_cast<double>(stats_.tier1_compilations);
        double fns_per_sec = (avg_us > 0.0) ? (1000000.0 / avg_us) : 0.0;
        os << "  Tier 1 Total Compile Time:          " << stats_.total_tier1_compile_time_us << " us ("
           << std::fixed << std::setprecision(2) << avg_us << " us/fn, ~"
           << static_cast<uint64_t>(fns_per_sec) << " fns/sec)\n";
    }
    os << "  Tier 2 (Optimized JIT) Compilations: " << stats_.tier2_compilations << "\n";
    os << "  Tier 2 (Optimized JIT) Invocations:  " << stats_.tier2_invocations << "\n";
    os << "=================================================\n";
    TieringRegistry::instance().dump_stats(os);
}

} // namespace brass::runtime
