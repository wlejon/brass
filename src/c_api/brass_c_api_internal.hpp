#pragma once

#include <brass/brass_c_api.h>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/gpu/cuda_driver.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <exception>

struct BrassType_T {
    brass::Type type;
};

struct BrassValue_T {
    BrassContext ctx = nullptr;
    brass::Value* val = nullptr;
    brass::Module* mod = nullptr;
};

struct BrassBlock_T {
    BrassContext ctx = nullptr;
    brass::BasicBlock* block = nullptr;
    brass::Function* func = nullptr;
    brass::Module* mod = nullptr;
};

struct BrassFunction_T {
    BrassContext ctx = nullptr;
    brass::Function* func = nullptr;
    brass::Module* mod = nullptr;
};

// Lifetime: the context owns every function, block and value handle. Every
// other object that remembers the context (module, builder, JIT engine, kernel
// JIT, kernel function) holds a reference on it, so objects can be destroyed
// in any order. brass_context_destroy drops the caller's reference; the
// context is freed when the last dependent object is destroyed.
struct BrassContext_T {
    std::string last_error;
    std::vector<std::unique_ptr<BrassValue_T>> values;
    std::vector<std::unique_ptr<BrassBlock_T>> blocks;
    std::vector<std::unique_ptr<BrassFunction_T>> functions;
    std::vector<BrassBuilder_T*> builders;
    size_t dependents = 0;
    bool destroyed = false;

    // Every value handle is owned by exactly one module: invalidation on
    // module destroy matches on the owner, never on "unknown".
    BrassValue wrap_value(brass::Value* v, brass::Module* m) {
        if (!v || !m) return nullptr;
        auto bv = std::make_unique<BrassValue_T>();
        bv->ctx = this;
        bv->val = v;
        bv->mod = m;
        BrassValue ptr = bv.get();
        values.push_back(std::move(bv));
        return ptr;
    }

    BrassBlock wrap_block(brass::BasicBlock* bb, brass::Function* fn, brass::Module* m) {
        if (!bb || !fn || !m) return nullptr;
        auto blk = std::make_unique<BrassBlock_T>();
        blk->ctx = this;
        blk->block = bb;
        blk->func = fn;
        blk->mod = m;
        BrassBlock ptr = blk.get();
        blocks.push_back(std::move(blk));
        return ptr;
    }

    BrassFunction wrap_function(brass::Function* fn, brass::Module* m) {
        if (!fn || !m) return nullptr;
        auto f = std::make_unique<BrassFunction_T>();
        f->ctx = this;
        f->func = fn;
        f->mod = m;
        BrassFunction ptr = f.get();
        functions.push_back(std::move(f));
        return ptr;
    }

    void invalidate_module_handles(const brass::Module* m);
};

struct BrassModule_T {
    BrassContext ctx = nullptr;
    std::unique_ptr<brass::Module> mod;
};

struct BrassBuilder_T {
    BrassContext ctx = nullptr;
    brass::Builder builder;
    brass::Function* func = nullptr;
    brass::Module* mod = nullptr;
    bool is_valid = true;
};

inline void BrassContext_T::invalidate_module_handles(const brass::Module* m) {
    if (!m) return;
    for (auto& f : functions) {
        if (f && f->mod == m) {
            f->func = nullptr;
            f->mod = nullptr;
        }
    }
    for (auto& blk : blocks) {
        if (blk && blk->mod == m) {
            blk->block = nullptr;
            blk->func = nullptr;
            blk->mod = nullptr;
        }
    }
    for (auto& v : values) {
        if (v && v->mod == m) {
            v->val = nullptr;
            v->mod = nullptr;
        }
    }
    for (auto* b : builders) {
        if (b && b->mod == m) {
            // The inner builder points into the module being freed; drop
            // its function and insertion point so nothing can reach them.
            b->builder.position_at_end(nullptr);
            b->builder.set_function(nullptr);
            b->func = nullptr;
            b->mod = nullptr;
            b->is_valid = false;
        }
    }
}

inline void ctx_retain(BrassContext ctx) {
    if (ctx) ++ctx->dependents;
}

inline void ctx_release(BrassContext ctx) {
    if (!ctx) return;
    if (ctx->dependents > 0) --ctx->dependents;
    if (ctx->destroyed && ctx->dependents == 0) delete ctx;
}

// A compiled module owns its own execution engine: its code and symbol table
// are independent of every other module compiled by the same BrassJitEngine,
// and brass_compiled_module_destroy frees them.
struct BrassCompiledModule_T {
    BrassContext ctx = nullptr;
    std::string module_name;
    BrassJitEngine jit = nullptr;
    std::unique_ptr<brass::codegen::JitExecutionEngine> engine; // null once destroyed
    std::vector<std::string> function_names;
};

struct BrassJitEngine_T {
    BrassContext ctx = nullptr;
    // Host symbols registered through brass_jit_register_symbol, applied (in
    // order) to every module compiled afterwards.
    std::vector<std::pair<std::string, void*>> external_symbols;
    std::vector<std::unique_ptr<BrassCompiledModule_T>> compiled_modules;
};

struct BrassKernelOptions_T {
    brass::codegen::KernelOptions opts;
};

struct BrassKernelFunction_T {
    BrassContext ctx = nullptr;
    brass::codegen::KernelFunction kfn;
};

struct BrassKernelJit_T {
    BrassContext ctx = nullptr;
    std::unique_ptr<brass::codegen::KernelJit> jit;
};

struct BrassGpuModule_T {
    brass::gpu::CudaModule module;
};

struct BrassGpuBuffer_T {
    brass::gpu::CudaBuffer buffer;
};

inline void set_ctx_error(BrassContext ctx, const std::string& msg) {
    if (ctx) {
        ctx->last_error = msg;
    }
}

inline void set_ctx_exception(BrassContext ctx, const char* prefix, const std::exception& e) {
    if (ctx) {
        ctx->last_error = (prefix ? std::string(prefix) + ": " : "") + e.what();
    }
}

BrassType get_type_handle(brass::Type t);
