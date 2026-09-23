#include "brass_c_api_internal.hpp"
#include <brass/target/ptx_target.hpp>
#include <cstring>
#include <cstdlib>

using namespace brass;

extern "C" {

/* Kernel Options */
BrassKernelOptions brass_kernel_options_create(void) {
    try {
        return new BrassKernelOptions_T();
    } catch (...) {
        return nullptr;
    }
}

void brass_kernel_options_destroy(BrassKernelOptions opts) {
    delete opts;
}

void brass_kernel_options_set_optimize(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_optimizations = (enable != 0);
    }
}

void brass_kernel_options_set_avx2(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_avx2 = (enable != 0);
    }
}

void brass_kernel_options_set_fma(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_fma = (enable != 0);
    }
}

void brass_kernel_options_set_fp_reassociation(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_fp_reassociation = (enable != 0);
    }
}

void brass_kernel_options_set_vectorize(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_vectorize = (enable != 0);
    }
}

void brass_kernel_options_set_parallel(BrassKernelOptions opts, int enable) {
    if (opts) {
        opts->opts.enable_parallel = (enable != 0);
    }
}

void brass_kernel_options_set_parallel_threshold(BrassKernelOptions opts, uint64_t threshold) {
    if (opts) {
        opts->opts.parallel_threshold = threshold;
    }
}

void brass_kernel_options_set_parallel_workers(BrassKernelOptions opts, uint32_t workers) {
    if (opts) {
        opts->opts.parallel_workers = workers;
    }
}

void brass_kernel_options_set_unroll_factor(BrassKernelOptions opts, size_t factor) {
    if (opts) {
        opts->opts.unroll_factor = factor;
    }
}

/* Kernel JIT Engine */
BrassKernelJit brass_kernel_jit_create(BrassContext ctx, const BrassKernelOptions opts) {
    try {
        auto* kj = new BrassKernelJit_T();
        kj->ctx = ctx;
        codegen::KernelOptions kopts;
        if (opts) {
            kopts = opts->opts;
        }
        kj->jit = std::make_unique<codegen::KernelJit>(kopts, Target::host());
        ctx_retain(ctx);
        return kj;
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_kernel_jit_create", e);
        return nullptr;
    }
}

void brass_kernel_jit_destroy(BrassKernelJit kj) {
    if (!kj) return;
    BrassContext ctx = kj->ctx;
    delete kj;
    ctx_release(ctx);
}

void brass_kernel_jit_set_parallel_workers(BrassKernelJit kj, uint32_t workers) {
    if (!kj || !kj->jit) return;
    kj->jit->set_parallel_workers(workers);
}

uint32_t brass_kernel_jit_get_parallel_workers(BrassKernelJit kj) {
    if (!kj || !kj->jit) return 0;
    return kj->jit->get_parallel_workers();
}

void brass_kernel_jit_set_parallel_threshold(BrassKernelJit kj, uint64_t threshold) {
    if (!kj || !kj->jit) return;
    kj->jit->set_parallel_threshold(threshold);
}

uint64_t brass_kernel_jit_get_parallel_threshold(BrassKernelJit kj) {
    if (!kj || !kj->jit) return 0;
    return kj->jit->get_parallel_threshold();
}

BrassStatus brass_kernel_jit_register_symbol(BrassKernelJit kj, const char* name, void* address) {
    if (!kj || !kj->jit || !name) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        kj->jit->register_external_symbol(name, address);
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(kj->ctx, "brass_kernel_jit_register_symbol", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassKernelFunction brass_kernel_jit_compile(BrassKernelJit kj, BrassModule mod, const char* entry_name) {
    if (!kj || !kj->jit || !mod || !mod->mod || !entry_name) return nullptr;
    try {
        codegen::KernelFunction kfn = kj->jit->compile(*mod->mod, entry_name);
        if (!kfn.is_valid()) {
            set_ctx_error(kj->ctx, "Compiled kernel function is invalid");
            return nullptr;
        }
        auto* res = new BrassKernelFunction_T();
        res->ctx = kj->ctx;
        res->kfn = std::move(kfn);
        ctx_retain(res->ctx);
        return res;
    } catch (const std::exception& e) {
        set_ctx_exception(kj->ctx, "brass_kernel_jit_compile", e);
        return nullptr;
    }
}

BrassKernelFunction brass_kernel_jit_compile_function(BrassKernelJit kj, BrassFunction fn) {
    if (!kj || !kj->jit || !fn || !fn->func) return nullptr;
    try {
        codegen::KernelFunction kfn = kj->jit->compile(*fn->func);
        if (!kfn.is_valid()) {
            set_ctx_error(kj->ctx, "Compiled kernel function is invalid");
            return nullptr;
        }
        auto* res = new BrassKernelFunction_T();
        res->ctx = kj->ctx;
        res->kfn = std::move(kfn);
        ctx_retain(res->ctx);
        return res;
    } catch (const std::exception& e) {
        set_ctx_exception(kj->ctx, "brass_kernel_jit_compile_function", e);
        return nullptr;
    }
}

/* Compiled Kernel Function */
void* brass_kernel_function_get_address(BrassKernelFunction kfn) {
    if (!kfn || !kfn->kfn.is_valid()) return nullptr;
    return kfn->kfn.entry_point();
}

const char* brass_kernel_function_get_name(BrassKernelFunction kfn) {
    if (!kfn) return nullptr;
    return kfn->kfn.name().data();
}

size_t brass_kernel_function_get_code_size(BrassKernelFunction kfn) {
    if (!kfn) return 0;
    return kfn->kfn.code_size();
}

void brass_kernel_function_destroy(BrassKernelFunction kfn) {
    if (!kfn) return;
    BrassContext ctx = kfn->ctx;
    delete kfn;
    ctx_release(ctx);
}

/* Polyhedral Loop Dependence Analysis */
BrassStatus brass_kernel_jit_analyze_loops(
    BrassKernelJit kj,
    BrassFunction fn,
    BrassLoopAnalysis* out_analyses,
    size_t max_analyses,
    size_t* out_analysis_count
) {
    if (!kj || !kj->jit || !fn || !fn->func) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        std::vector<codegen::KernelLoopAnalysis> analyses = kj->jit->analyze_loops(*fn->func);
        if (out_analysis_count) {
            *out_analysis_count = analyses.size();
        }
        if (out_analyses && max_analyses > 0) {
            size_t n = std::min(max_analyses, analyses.size());
            for (size_t i = 0; i < n; ++i) {
                out_analyses[i].is_parallelizable = analyses[i].is_parallelizable ? 1 : 0;
                out_analyses[i].is_doall = analyses[i].is_doall ? 1 : 0;
                out_analyses[i].is_reduction = analyses[i].is_reduction ? 1 : 0;
                out_analyses[i].has_const_trip_count = analyses[i].has_const_trip_count ? 1 : 0;
                out_analyses[i].const_trip_count = analyses[i].const_trip_count;
                out_analyses[i].dependence_count = analyses[i].dependences.size();
                size_t copy_len = std::min(analyses[i].rejection_reason.size(), sizeof(out_analyses[i].rejection_reason) - 1);
                std::memcpy(out_analyses[i].rejection_reason, analyses[i].rejection_reason.data(), copy_len);
                out_analyses[i].rejection_reason[copy_len] = '\0';
            }
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(kj->ctx, "brass_kernel_jit_analyze_loops", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassStatus brass_kernel_jit_auto_parallelize(
    BrassKernelJit kj,
    BrassFunction fn,
    int* out_changed
) {
    if (!kj || !kj->jit || !fn || !fn->func) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        bool changed = kj->jit->auto_parallelize(*fn->func);
        if (out_changed) {
            *out_changed = changed ? 1 : 0;
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(kj->ctx, "brass_kernel_jit_auto_parallelize", e);
        return BRASS_ERR_GENERIC;
    }
}

/* PTX CUDA Target Emitter */
BrassStatus brass_kernel_emit_ptx(
    BrassFunction fn,
    const char* sm_arch,
    char** out_ptx,
    size_t* out_len
) {
    if (!fn || !fn->func || !out_ptx) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        target::PtxOptions opts;
        if (sm_arch && *sm_arch) {
            opts.sm_arch = sm_arch;
        }
        std::string ptx = target::PtxTarget::emit_function(*fn->func, opts);
        char* buf = static_cast<char*>(std::malloc(ptx.size() + 1));
        if (!buf) return BRASS_ERR_GENERIC;
        std::memcpy(buf, ptx.c_str(), ptx.size() + 1);
        *out_ptx = buf;
        if (out_len) {
            *out_len = ptx.size();
        }
        return BRASS_OK;
    } catch (...) {
        return BRASS_ERR_GENERIC;
    }
}

void brass_free_string(char* str) {
    if (str) {
        std::free(str);
    }
}

} /* extern "C" */
