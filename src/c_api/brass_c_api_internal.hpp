#pragma once

#include <brass/brass_c_api.h>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/il_translator/il_translator.hpp>
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

struct BrassContext_T {
    std::string last_error;
    std::vector<std::unique_ptr<BrassValue_T>> values;
    std::vector<std::unique_ptr<BrassBlock_T>> blocks;
    std::vector<std::unique_ptr<BrassFunction_T>> functions;
    std::vector<BrassBuilder_T*> builders;

    BrassValue wrap_value(brass::Value* v, brass::Module* m = nullptr) {
        if (!v) return nullptr;
        auto bv = std::make_unique<BrassValue_T>();
        bv->ctx = this;
        bv->val = v;
        bv->mod = m;
        BrassValue ptr = bv.get();
        values.push_back(std::move(bv));
        return ptr;
    }

    BrassBlock wrap_block(brass::BasicBlock* bb, brass::Function* fn, brass::Module* m = nullptr) {
        if (!bb) return nullptr;
        auto blk = std::make_unique<BrassBlock_T>();
        blk->ctx = this;
        blk->block = bb;
        blk->func = fn;
        blk->mod = m ? m : (fn ? fn->parent() : nullptr);
        BrassBlock ptr = blk.get();
        blocks.push_back(std::move(blk));
        return ptr;
    }

    BrassFunction wrap_function(brass::Function* fn, brass::Module* m) {
        if (!fn) return nullptr;
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
    const brass::Module* mod = nullptr;
    bool is_valid = true;
};

inline void BrassContext_T::invalidate_module_handles(const brass::Module* m) {
    if (!m) return;
    for (auto& f : functions) {
        if (f && (f->mod == m || (f->func && f->func->parent() == m))) {
            f->func = nullptr;
            f->mod = nullptr;
        }
    }
    for (auto& blk : blocks) {
        if (blk && (blk->mod == m || (blk->func && (blk->func->parent() == m || blk->func->parent() == nullptr)))) {
            blk->block = nullptr;
            blk->func = nullptr;
            blk->mod = nullptr;
        }
    }
    for (auto& v : values) {
        if (v && (v->mod == m || !v->mod)) {
            v->val = nullptr;
            v->mod = nullptr;
        }
    }
    for (auto* b : builders) {
        if (b && (b->mod == m || !b->mod)) {
            b->func = nullptr;
            b->mod = nullptr;
            b->is_valid = false;
        }
    }
}

struct BrassCompiledModule_T {
    BrassContext ctx = nullptr;
    std::string module_name;
    BrassJitEngine jit = nullptr;
};

struct BrassJitEngine_T {
    BrassContext ctx = nullptr;
    std::unique_ptr<brass::codegen::JitExecutionEngine> engine;
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
