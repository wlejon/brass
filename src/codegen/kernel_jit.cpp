#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/core/diagnostics.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace brass::codegen {

// ============================================================================
// KernelBuilder Implementation
// ============================================================================

KernelBuilder::KernelBuilder(Module& mod, Function* fn)
    : owned_builder_(std::make_unique<Builder>(mod)),
      b_(*owned_builder_) {
    if (fn) {
        b_.set_function(fn);
    }
}

Value* KernelBuilder::relu(Value* val) {
    if (!val) return nullptr;
    if (val->type().is_vector()) {
        Value* z = b_.build_vzero(val->type());
        return b_.build_vmax(val, z);
    }
    Value* zero = b_.build_fconst_f64(0.0);
    Value* cond = b_.build_sgt(val, zero);
    return b_.build_select(cond, val, zero);
}

Value* KernelBuilder::relu_bias(Value* val, Value* bias) {
    if (!val || !bias) return nullptr;
    Value* sum = val->type().is_vector() ? b_.build_vadd(val, bias) : b_.build_add(val, bias);
    return relu(sum);
}

// ============================================================================
// KernelJit Implementation
// ============================================================================

KernelJit::KernelJit(const KernelOptions& options, Target target)
    : options_(options), target_(target) {
    if (options_.parallel_workers > 0) {
        brass_set_parallel_workers(options_.parallel_workers);
    }
}

void KernelJit::set_parallel_workers(uint32_t workers) {
    options_.parallel_workers = workers;
    brass_set_parallel_workers(workers);
}

uint32_t KernelJit::get_parallel_workers() const {
    if (options_.parallel_workers > 0) {
        return options_.parallel_workers;
    }
    return brass_get_parallel_workers();
}

void KernelJit::register_external_symbol(std::string_view name, void* address) {
    external_symbols_[std::string(name)] = address;
}

void KernelJit::setup_default_symbols(codegen::JitExecutionEngine& engine) const {
    // Parallel Runtime Symbols
    engine.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    engine.register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    engine.register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    engine.register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    engine.register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    engine.register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    engine.register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));

    // Standard Math Functions (for pure-compute kernels calling elementary transcendental/math routines)
    engine.register_external_symbol("sqrt", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::sqrt)));
    engine.register_external_symbol("sqrtf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::sqrt)));
    engine.register_external_symbol("exp", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::exp)));
    engine.register_external_symbol("expf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::exp)));
    engine.register_external_symbol("log", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::log)));
    engine.register_external_symbol("logf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::log)));
    engine.register_external_symbol("fabs", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::fabs)));
    engine.register_external_symbol("fabsf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::fabs)));
    engine.register_external_symbol("sin", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::sin)));
    engine.register_external_symbol("sinf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::sin)));
    engine.register_external_symbol("cos", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::cos)));
    engine.register_external_symbol("cosf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::cos)));
    engine.register_external_symbol("tanh", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::tanh)));
    engine.register_external_symbol("tanhf", reinterpret_cast<void*>(static_cast<float(*)(float)>(&std::tanh)));

    for (const auto& [name, addr] : external_symbols_) {
        engine.register_external_symbol(name, addr);
    }
}

std::vector<KernelLoopAnalysis> KernelJit::analyze_loops(Function& fn, const ParallelLoopOptions& options) const {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis la(fn, dom);

    ParallelLoopOptions opts = options;
    if (opts.parallel_threshold == 1000 && options_.parallel_threshold != 1000) {
        opts.parallel_threshold = options_.parallel_threshold;
    }
    if (opts.parallel_workers == 0 && options_.parallel_workers != 0) {
        opts.parallel_workers = options_.parallel_workers;
    }
    if (!opts.allow_fp_reassociation && options_.enable_fp_reassociation) {
        opts.allow_fp_reassociation = true;
    }

    std::vector<KernelLoopAnalysis> results;
    for (LoopInfo* loop : la.post_order_loops()) {
        if (!loop || !loop->header()) continue;
        ParallelLoopInfo pli;
        analyze_parallel_loop(fn, *loop, dom, pli, opts);

        KernelLoopAnalysis kla;
        kla.is_parallelizable = pli.is_parallelizable();
        kla.is_doall = (pli.kind == LoopParallelKind::DOALL);
        kla.is_reduction = (pli.kind == LoopParallelKind::Reduction);
        kla.has_const_trip_count = pli.has_const_trip_count;
        kla.const_trip_count = pli.const_trip_count;
        kla.rejection_reason = pli.rejection_reason;

        for (const auto& dep : pli.dependences) {
            KernelDependenceInfo kdep;
            kdep.kind = dep.kind;
            kdep.dir = dep.dir;
            kdep.distance = dep.distance;
            kdep.has_distance = dep.has_distance;
            kdep.is_loop_carried = dep.is_loop_carried;
            kla.dependences.push_back(kdep);
        }

        results.push_back(std::move(kla));
    }
    return results;
}

bool KernelJit::auto_parallelize(Function& fn, const ParallelLoopOptions& options) const {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);

    ParallelLoopOptions opts = options;
    if (opts.parallel_threshold == 1000 && options_.parallel_threshold != 1000) {
        opts.parallel_threshold = options_.parallel_threshold;
    }
    if (opts.parallel_workers == 0 && options_.parallel_workers != 0) {
        opts.parallel_workers = options_.parallel_workers;
    }
    if (!opts.allow_fp_reassociation && options_.enable_fp_reassociation) {
        opts.allow_fp_reassociation = true;
    }
    if (options_.parallel_stats && !opts.stats) {
        opts.stats = options_.parallel_stats;
    }

    return auto_parallelize_function(fn, dom, opts);
}

void KernelJit::run_kernel_optimizations(Module& mod) const {
    // 1. Polyhedral auto-parallelization pass if configured
    if (options_.enable_parallel) {
        std::vector<Function*> fns;
        for (Function* fn : mod.functions()) {
            if (fn) fns.push_back(fn);
        }
        for (Function* fn : fns) {
            fn->rebuild_cfg_predecessors();
            DominatorTree dom(*fn);
            ParallelLoopOptions par_opts;
            par_opts.parallel_threshold = options_.parallel_threshold;
            par_opts.parallel_workers = options_.parallel_workers;
            par_opts.allow_fp_reassociation = options_.enable_fp_reassociation || fn->allow_fp_reassociation() || mod.allow_fp_reassociation();
            par_opts.stats = options_.parallel_stats;
            auto_parallelize_function(*fn, dom, par_opts);
        }
    }

    // 2. Compute kernel optimization pass (FMA fusion, Vectorization, Unrolling, DCE, GVN)
    if (options_.enable_optimizations) {
        LoopOptOptions loop_opts;
        loop_opts.enable_avx2 = options_.enable_avx2;
        loop_opts.enable_fma = options_.enable_fma;
        loop_opts.enable_fp_reassociation = options_.enable_fp_reassociation || mod.allow_fp_reassociation();
        loop_opts.enable_vectorize = options_.enable_vectorize;
        loop_opts.enable_slp = options_.enable_slp;
        loop_opts.enable_unroll = options_.enable_unroll;
        loop_opts.unroll_factor = options_.unroll_factor;
        loop_opts.enable_f64_demote = false; // numerical kernels preserve precise types
        loop_opts.enable_parallel_loops = false; // already executed if enabled

        std::vector<Function*> fns;
        for (Function* fn : mod.functions()) {
            if (fn) fns.push_back(fn);
        }
        for (Function* fn : fns) {
            optimize_function(*fn, loop_opts);
        }
    }
}

KernelFunction KernelJit::compile(Function& fn) {
    if (fn.parent()) {
        return compile(*fn.parent(), fn.name());
    }

    Module temp_mod("kernel_isolated_module");
    temp_mod.set_allow_fp_reassociation(fn.allow_fp_reassociation() || options_.enable_fp_reassociation);
    Function* cloned = clone_function(fn, temp_mod);
    return compile(temp_mod, cloned->name());
}

KernelFunction KernelJit::compile(Module& mod, std::string_view entry_name) {
    Module working_mod(mod.name());
    working_mod.set_allow_fp_reassociation(mod.allow_fp_reassociation() || options_.enable_fp_reassociation);

    for (std::string_view sym : mod.external_symbols()) {
        working_mod.add_external_symbol(sym);
    }

    for (const auto* fn : mod.functions()) {
        if (fn) {
            clone_function(*fn, working_mod);
        }
    }

    run_kernel_optimizations(working_mod);

    for (auto* fn : working_mod.functions()) {
        if (fn) {
            fn->rebuild_cfg_predecessors();
        }
    }

    DiagnosticReporter diag;
    if (!verify_module(working_mod, &diag)) {
        throw std::runtime_error("Kernel JIT verification failed: " + diag.format_all());
    }

    auto engine = std::make_shared<codegen::JitExecutionEngine>(target_);
    setup_default_symbols(*engine);

    bool ok = engine->compile_and_load(working_mod);
    if (!ok) {
        throw std::runtime_error("Kernel JIT compile_and_load failed for module " + std::string(mod.name()));
    }

    void* entry_addr = engine->get_symbol_address(entry_name);
    if (!entry_addr) {
        throw std::runtime_error("Kernel JIT could not find entry point: " + std::string(entry_name));
    }

    return KernelFunction(std::string(entry_name), entry_addr, 0, engine);
}

} // namespace brass::codegen
