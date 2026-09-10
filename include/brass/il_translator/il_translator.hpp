#pragma once

#include <brass/mir/module.hpp>
#include <brass/core/diagnostics.hpp>
#include <memory>
#include <string_view>
#include <string>

namespace brass {

class HostGC;

struct DemoteStats;

namespace il {

struct TranslatorOptions {
    bool enable_optimizations = true;
    bool allow_fp_reassociation = false;
    bool trace_lowering = false;
    bool enable_f64_demote = true;
    bool demote_stats = false;
    bool enable_inlining = false;
    bool enable_sroa = false;
    bool enable_gvn = true;
    bool enable_sccp = true;
    bool enable_guard_elim = true;
    bool enable_cfg_simplify = true;
    bool enable_loop_unswitch = true;
    bool enable_jump_threading = true;
    bool enable_trace_layout = true;
    bool run_escape_analysis = false;
    bool run_alias_analysis = false;
    bool enable_vectorize = true;
    bool enable_slp = true;
    bool enable_loop_tile = true;
    size_t tile_size = 16;
    bool enable_pic = true;
    bool dump_ic_stats = false;
    DemoteStats* demote_stats_collector = nullptr;
};

struct TranslationResult {
    bool success = false;
    std::unique_ptr<Module> module;
    std::string error_message;
};

// Translate Bronze textual IL directly to a Brass MIR Module
TranslationResult translate_bronze_il(
    std::string_view il_text,
    const TranslatorOptions& options = {},
    DiagnosticReporter* diag = nullptr
);

// Register Bronze runtime helper symbols into a JitExecutionEngine or runtime symbol table
void register_bronze_runtime_symbols(void* jit_engine_ptr);

// Optional custom function resolver for AOT or dynamic function lookup
void set_bronze_function_resolver(void* (*resolver)(const char*));

// Control whether Bronze print statements output to stdout
void bronze_set_print_enabled(bool enabled);

// Runtime helper symbols for Bronze execution
extern "C" {
void bronze_print_f64(double v);
void bronze_print_i32(int32_t v);
void bronze_print_dynamic(int64_t v);
void bronze_print_newline();
double bronze_f64_mod(double a, double b);

int64_t bronze_name_resolve(const char* name);
int64_t bronze_env_create(int64_t parent_box, int32_t size);
int64_t bronze_env_get(int64_t env_box, int32_t depth, int32_t index);
void bronze_env_set(int64_t env_box, int32_t depth, int32_t index, int64_t val);
int64_t bronze_create_func(const char* fn_name, int32_t param_count, int64_t env_box);
int64_t bronze_create_array(int32_t size);
int64_t bronze_create_object();
int64_t bronze_prop_get(int64_t obj_box, int32_t key_index);
void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, int32_t slot_idx, int32_t imm);
int64_t bronze_elem_get(int64_t arr_box, int64_t index_box);
void bronze_elem_set(int64_t arr_box, int64_t index_box, int64_t val, int32_t ic_slot);
void bronze_method_def(int64_t obj_box, const char* name, int32_t symbol_id, int64_t closure_box);
int64_t bronze_ic_get(uint32_t site_id, int64_t obj_box, const char* name, int32_t symbol_id);
void bronze_ic_set(uint32_t site_id, int64_t obj_box, const char* name, int32_t symbol_id, int64_t val_box);
uint64_t brass_ic_get_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id);
void brass_ic_set_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id, uint64_t val_raw);

int64_t bronze_call_dynamic_0(int64_t callee_box, int64_t this_box);
int64_t bronze_call_dynamic_1(int64_t callee_box, int64_t this_box, int64_t arg0);
int64_t bronze_call_dynamic_2(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1);
int64_t bronze_call_dynamic_3(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2);
int64_t bronze_call_dynamic_4(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3);
int64_t bronze_call_dynamic_5(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);
int64_t bronze_call_dynamic_6(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5);
int64_t bronze_call_dynamic_7(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6);
int64_t bronze_call_dynamic_8(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6, int64_t arg7);
int64_t bronze_call_dynamic_n(int64_t callee_box, int64_t this_box, int32_t argc, const int64_t* argv);
}

} // namespace il
} // namespace brass
