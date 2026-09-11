#include "brass_c_api_internal.hpp"
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace brass;

/* Static Type Singletons */
static BrassType_T s_type_void{ Type::void_type() };
static BrassType_T s_type_i32{ Type::i32() };
static BrassType_T s_type_i64{ Type::i64() };
static BrassType_T s_type_f32{ Type::f32() };
static BrassType_T s_type_f64{ Type::f64() };
static BrassType_T s_type_ptr{ Type::ptr() };
static BrassType_T s_type_dynamic{ Type::gcref() };

static BrassType_T s_type_v128_f32{ Type::f32x4() };
static BrassType_T s_type_v128_f64{ Type::f64x2() };
static BrassType_T s_type_v128_i32{ Type::i32x4() };
static BrassType_T s_type_v128_i64{ Type::i64x2() };

static BrassType_T s_type_v256_f32{ Type::f32x8() };
static BrassType_T s_type_v256_f64{ Type::f64x4() };
static BrassType_T s_type_v256_i32{ Type::i32x8() };
static BrassType_T s_type_v256_i64{ Type::i64x4() };

BrassType get_type_handle(Type t) {
    switch (t.kind()) {
        case TypeKind::Void: return &s_type_void;
        case TypeKind::I32: return &s_type_i32;
        case TypeKind::I64: return &s_type_i64;
        case TypeKind::F32: return &s_type_f32;
        case TypeKind::F64: return &s_type_f64;
        case TypeKind::Ptr: return &s_type_ptr;
        case TypeKind::GCRef: return &s_type_dynamic;
        case TypeKind::F32x4: return &s_type_v128_f32;
        case TypeKind::F64x2: return &s_type_v128_f64;
        case TypeKind::I32x4: return &s_type_v128_i32;
        case TypeKind::I64x2: return &s_type_v128_i64;
        case TypeKind::F32x8: return &s_type_v256_f32;
        case TypeKind::F64x4: return &s_type_v256_f64;
        case TypeKind::I32x8: return &s_type_v256_i32;
        case TypeKind::I64x4: return &s_type_v256_i64;
        default: return &s_type_void;
    }
}

extern "C" {

/* Options API */
BrassOptions* brass_options_create(void) {
    try {
        auto* opts = new BrassOptions();
        opts->enable_optimizations = 1;
        opts->allow_fp_reassociation = 0;
        opts->target_format = 0;
        opts->vector_width = 0;
        return opts;
    } catch (...) {
        return nullptr;
    }
}

void brass_options_destroy(BrassOptions* opts) {
    delete opts;
}

void brass_options_set_optimize(BrassOptions* opts, int enable) {
    if (opts) {
        opts->enable_optimizations = enable;
    }
}

void brass_options_set_target_format(BrassOptions* opts, int format) {
    if (opts) {
        opts->target_format = format;
    }
}

/* Types API */
BrassType brass_type_void(void) { return &s_type_void; }
BrassType brass_type_i32(void) { return &s_type_i32; }
BrassType brass_type_i64(void) { return &s_type_i64; }
BrassType brass_type_f32(void) { return &s_type_f32; }
BrassType brass_type_f64(void) { return &s_type_f64; }
BrassType brass_type_bool(void) { return &s_type_i32; }
BrassType brass_type_ptr(void) { return &s_type_ptr; }
BrassType brass_type_dynamic(void) { return &s_type_dynamic; }

BrassType brass_type_v128(uint8_t lane_kind) {
    switch (lane_kind) {
        case BRASS_LANE_F32: return &s_type_v128_f32;
        case BRASS_LANE_F64: return &s_type_v128_f64;
        case BRASS_LANE_I32: return &s_type_v128_i32;
        case BRASS_LANE_I64: return &s_type_v128_i64;
        default: return &s_type_v128_f32;
    }
}

BrassType brass_type_v256(uint8_t lane_kind) {
    switch (lane_kind) {
        case BRASS_LANE_F32: return &s_type_v256_f32;
        case BRASS_LANE_F64: return &s_type_v256_f64;
        case BRASS_LANE_I32: return &s_type_v256_i32;
        case BRASS_LANE_I64: return &s_type_v256_i64;
        default: return &s_type_v256_f32;
    }
}

/* Context & Error Management */
BrassContext brass_context_create(void) {
    try {
        return new BrassContext_T();
    } catch (...) {
        return nullptr;
    }
}

void brass_context_destroy(BrassContext ctx) {
    delete ctx;
}

const char* brass_context_get_last_error(BrassContext ctx) {
    if (!ctx) return "";
    return ctx->last_error.c_str();
}

void brass_context_set_error(BrassContext ctx, const char* msg) {
    if (!ctx) return;
    ctx->last_error = msg ? msg : "";
}

/* Module Management */
BrassModule brass_module_create(BrassContext ctx, const char* name) {
    try {
        auto* mod = new BrassModule_T();
        mod->ctx = ctx;
        mod->mod = std::make_unique<Module>(name ? name : "module");
        return mod;
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_module_create", e);
        return nullptr;
    } catch (...) {
        set_ctx_error(ctx, "Unknown exception in brass_module_create");
        return nullptr;
    }
}

void brass_module_destroy(BrassModule mod) {
    delete mod;
}

void brass_module_add_external_symbol(BrassModule mod, const char* name) {
    if (!mod || !mod->mod || !name) return;
    try {
        mod->mod->add_external_symbol(name);
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_module_add_external_symbol", e);
    }
}

BrassStatus brass_module_verify(BrassModule mod, char* err_buf, size_t err_buf_len) {
    if (!mod || !mod->mod) {
        if (err_buf && err_buf_len > 0) {
            err_buf[0] = '\0';
        }
        return BRASS_ERR_INVALID_ARGUMENT;
    }

    try {
        for (auto* fn : mod->mod->functions()) {
            if (fn) fn->rebuild_cfg_predecessors();
        }

        DiagnosticReporter diag;
        bool ok = verify_module(*mod->mod, &diag);
        if (!ok) {
            std::string err_msg;
            for (const auto& d : diag.diagnostics()) {
                if (!err_msg.empty()) err_msg += "\n";
                err_msg += d.message;
            }
            if (err_msg.empty()) err_msg = "Module verification failed";

            if (err_buf && err_buf_len > 0) {
                size_t copy_len = (err_msg.size() < err_buf_len - 1) ? err_msg.size() : (err_buf_len - 1);
                std::memcpy(err_buf, err_msg.data(), copy_len);
                err_buf[copy_len] = '\0';
            }
            set_ctx_error(mod->ctx, err_msg);
            return BRASS_ERR_VERIFICATION_FAILED;
        }

        if (err_buf && err_buf_len > 0) {
            err_buf[0] = '\0';
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_module_verify", e);
        return BRASS_ERR_GENERIC;
    } catch (...) {
        set_ctx_error(mod->ctx, "Unknown exception in brass_module_verify");
        return BRASS_ERR_GENERIC;
    }
}

BrassStatus brass_module_print_mir(BrassModule mod, char** out_str) {
    if (!mod || !mod->mod || !out_str) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        std::string mir = to_string(*mod->mod);
        char* buf = static_cast<char*>(std::malloc(mir.size() + 1));
        if (!buf) {
            set_ctx_error(mod->ctx, "Out of memory allocating printed MIR string");
            return BRASS_ERR_OUT_OF_MEMORY;
        }
        std::memcpy(buf, mir.c_str(), mir.size() + 1);
        *out_str = buf;
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_module_print_mir", e);
        return BRASS_ERR_GENERIC;
    }
}

/* Function & CFG */
BrassFunction brass_function_create(BrassModule mod, const char* name, BrassType ret_type, const BrassType* param_types, size_t param_count) {
    if (!mod || !mod->mod || !name) return nullptr;
    try {
        std::vector<Type> params;
        params.reserve(param_count);
        for (size_t i = 0; i < param_count; ++i) {
            params.push_back((param_types && param_types[i]) ? param_types[i]->type : Type::i64());
        }

        Type rt = ret_type ? ret_type->type : Type::void_type();
        Function* fn = mod->mod->create_function(name, rt, params);
        return mod->ctx ? mod->ctx->wrap_function(fn, mod->mod.get()) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_function_create", e);
        return nullptr;
    }
}

BrassBlock brass_function_append_block(BrassFunction fn, const char* name) {
    if (!fn || !fn->func || !fn->mod) return nullptr;
    try {
        Arena& arena = fn->mod->arena();
        uint32_t id = fn->func->next_block_id();
        std::string_view sym = (name && name[0]) ? fn->mod->string_pool().intern(name) : "";
        BasicBlock* bb = arena.make<BasicBlock>(id, sym);
        bb->set_parent(fn->func);
        fn->func->append_block(bb);
        return fn->ctx ? fn->ctx->wrap_block(bb, fn->func) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(fn->ctx, "brass_function_append_block", e);
        return nullptr;
    }
}

BrassValue brass_block_add_param(BrassBlock blk, BrassType type) {
    if (!blk || !blk->block || !blk->func || !blk->func->parent() || !type) return nullptr;
    try {
        Arena& arena = blk->func->parent()->arena();
        uint32_t id = blk->func->next_value_id();
        Value* val = arena.make<Value>(id, type->type, ValueKind::BlockParam);
        blk->block->add_param(val);
        return blk->ctx ? blk->ctx->wrap_value(val) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(blk->ctx, "brass_block_add_param", e);
        return nullptr;
    }
}

BrassValue brass_block_get_param(BrassBlock blk, size_t index) {
    if (!blk || !blk->block) return nullptr;
    Value* val = blk->block->param(index);
    return (val && blk->ctx) ? blk->ctx->wrap_value(val) : nullptr;
}

BrassValue brass_function_get_param(BrassFunction fn, size_t index) {
    if (!fn || !fn->func || !fn->mod) return nullptr;
    try {
        BasicBlock* entry = fn->func->entry_block();
        if (!entry) {
            entry = fn->mod->arena().make<BasicBlock>(fn->func->next_block_id(), "entry");
            entry->set_parent(fn->func);
            fn->func->append_block(entry);
        }

        if (entry->param_count() <= index && fn->func->param_count() > entry->param_count()) {
            Arena& arena = fn->mod->arena();
            while (entry->param_count() < fn->func->param_count()) {
                size_t p_idx = entry->param_count();
                Type pt = fn->func->param_type(p_idx);
                uint32_t id = fn->func->next_value_id();
                Value* v = arena.make<Value>(id, pt, ValueKind::BlockParam);
                entry->add_param(v);
            }
        }

        Value* val = entry->param(index);
        return (val && fn->ctx) ? fn->ctx->wrap_value(val) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(fn->ctx, "brass_function_get_param", e);
        return nullptr;
    }
}

/* Bronze IL Translation Bridge */
BrassStatus brass_translate_bronze_il(BrassContext ctx, const char* il_text, size_t len, const BrassOptions* opts, BrassModule* out_mod) {
    if (!il_text || !out_mod) {
        set_ctx_error(ctx, "Invalid null arguments to brass_translate_bronze_il");
        return BRASS_ERR_INVALID_ARGUMENT;
    }

    try {
        std::string_view text = (len == 0 || len == static_cast<size_t>(-1)) ? std::string_view(il_text) : std::string_view(il_text, len);
        il::TranslatorOptions trans_opts;
        if (opts) {
            trans_opts.enable_optimizations = (opts->enable_optimizations != 0);
            trans_opts.allow_fp_reassociation = (opts->allow_fp_reassociation != 0);
            if (opts->vector_width > 0) {
                trans_opts.vector_width = static_cast<uint32_t>(opts->vector_width);
            }
        }

        DiagnosticReporter diag;
        il::TranslationResult res = il::translate_bronze_il(text, trans_opts, &diag);
        if (!res.success || !res.module) {
            std::string err = res.error_message;
            if (err.empty()) {
                for (const auto& d : diag.diagnostics()) {
                    if (!err.empty()) err += "\n";
                    err += d.message;
                }
            }
            if (err.empty()) err = "Bronze IL translation syntax/lowering error";
            set_ctx_error(ctx, err);
            *out_mod = nullptr;
            return BRASS_ERR_TRANSLATION_FAILED;
        }

        auto* mod = new BrassModule_T();
        mod->ctx = ctx;
        mod->mod = std::move(res.module);
        *out_mod = mod;
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_translate_bronze_il", e);
        *out_mod = nullptr;
        return BRASS_ERR_TRANSLATION_FAILED;
    } catch (...) {
        set_ctx_error(ctx, "Unknown exception in brass_translate_bronze_il");
        *out_mod = nullptr;
        return BRASS_ERR_TRANSLATION_FAILED;
    }
}

/* JIT Execution Engine */
BrassJitEngine brass_jit_create(BrassContext ctx) {
    try {
        auto* jit = new BrassJitEngine_T();
        jit->ctx = ctx;
        jit->engine = std::make_unique<codegen::JitExecutionEngine>(Target::host());
        il::register_bronze_runtime_symbols(jit->engine.get());
        return jit;
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_jit_create", e);
        return nullptr;
    }
}

void brass_jit_destroy(BrassJitEngine jit) {
    delete jit;
}

BrassStatus brass_jit_register_symbol(BrassJitEngine jit, const char* name, void* address) {
    if (!jit || !jit->engine || !name) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        jit->engine->register_external_symbol(name, address);
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(jit->ctx, "brass_jit_register_symbol", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassCompiledModule brass_jit_compile_module(BrassJitEngine jit, BrassModule mod) {
    if (!jit || !jit->engine || !mod || !mod->mod) return nullptr;
    try {
        for (auto* fn : mod->mod->functions()) {
            if (fn) fn->rebuild_cfg_predecessors();
        }

        bool ok = jit->engine->compile_and_load(*mod->mod);
        if (!ok) {
            set_ctx_error(jit->ctx, "JIT compilation and loading failed");
            return nullptr;
        }

        auto cmod = std::make_unique<BrassCompiledModule_T>();
        cmod->ctx = jit->ctx;
        cmod->module_name = std::string(mod->mod->name());
        cmod->jit = jit;
        BrassCompiledModule ptr = cmod.get();
        jit->compiled_modules.push_back(std::move(cmod));
        return ptr;
    } catch (const std::exception& e) {
        set_ctx_exception(jit->ctx, "brass_jit_compile_module", e);
        return nullptr;
    }
}

void* brass_jit_get_function_address(BrassJitEngine jit, const char* name) {
    if (!jit || !jit->engine || !name) return nullptr;
    return jit->engine->get_symbol_address(name);
}

void brass_compiled_module_destroy(BrassCompiledModule mod) {
    (void)mod;
}

void* brass_compiled_module_get_symbol(BrassCompiledModule mod, const char* name) {
    if (!mod || !mod->jit) return nullptr;
    return brass_jit_get_function_address(mod->jit, name);
}

/* AOT Binary Compilation */
BrassStatus brass_compile_to_object(BrassModule mod, int target_format, void** out_bytes, size_t* out_size) {
    if (!mod || !mod->mod || !out_bytes || !out_size) return BRASS_ERR_INVALID_ARGUMENT;
    *out_bytes = nullptr;
    *out_size = 0;

    try {
        Target target = Target::host();
        if (target_format == BRASS_OBJECT_COFF) {
            target = Target::x64_windows();
        } else if (target_format == BRASS_OBJECT_ELF) {
            target = Target::x64_linux();
        } else if (target_format == BRASS_OBJECT_MACHO) {
            target = Target::x64_macos();
        }

        for (auto* fn : mod->mod->functions()) {
            if (fn) fn->rebuild_cfg_predecessors();
        }

        object::ModuleCompiler compiler(target);
        object::ObjectFile obj = compiler.compile(*mod->mod);

        std::vector<uint8_t> bytes;
        if (target_format == BRASS_OBJECT_COFF || (target_format == BRASS_OBJECT_AUTO && target.is_windows())) {
            object::CoffWriter writer(obj);
            bytes = writer.write();
        } else if (target_format == BRASS_OBJECT_MACHO || (target_format == BRASS_OBJECT_AUTO && target.is_macos())) {
            object::MachOWriter writer(obj);
            bytes = writer.write();
        } else {
            object::ElfWriter writer(obj);
            bytes = writer.write();
        }

        if (bytes.empty()) {
            set_ctx_error(mod->ctx, "AOT object serialization returned 0 bytes");
            return BRASS_ERR_COMPILE_FAILED;
        }

        void* buf = std::malloc(bytes.size());
        if (!buf) {
            set_ctx_error(mod->ctx, "Out of memory allocating AOT object buffer");
            return BRASS_ERR_OUT_OF_MEMORY;
        }

        std::memcpy(buf, bytes.data(), bytes.size());
        *out_bytes = buf;
        *out_size = bytes.size();
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_compile_to_object", e);
        return BRASS_ERR_COMPILE_FAILED;
    }
}

BrassStatus brass_compile_to_shared_lib(BrassModule mod, const char* output_path, const BrassOptions* opts) {
    if (!mod || !mod->mod || !output_path) return BRASS_ERR_INVALID_ARGUMENT;

    try {
        for (auto* fn : mod->mod->functions()) {
            if (fn) fn->rebuild_cfg_predecessors();
        }

        target::LinkerOptions link_opts;
        link_opts.export_all_functions = true;
        Target target = Target::host();
        if (opts) {
            if (opts->target_format == BRASS_OBJECT_COFF) {
                target = Target::x64_windows();
                link_opts.format = target::OutputFormat::WindowsPeDll;
            } else if (opts->target_format == BRASS_OBJECT_ELF) {
                target = Target::x64_linux();
                link_opts.format = target::OutputFormat::LinuxElfSo;
            } else if (opts->target_format == BRASS_OBJECT_MACHO) {
                target = Target::x64_macos();
                link_opts.format = target::OutputFormat::MacOSMachODylib;
            }
        }

        bool ok = target::AotLinker::link_to_file(*mod->mod, output_path, target, link_opts);
        if (!ok) {
            set_ctx_error(mod->ctx, "Failed to compile and link shared library to file");
            return BRASS_ERR_COMPILE_FAILED;
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(mod->ctx, "brass_compile_to_shared_lib", e);
        return BRASS_ERR_COMPILE_FAILED;
    }
}

void brass_free_buffer(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

} /* extern "C" */
