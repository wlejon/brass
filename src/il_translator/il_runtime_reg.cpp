#include "il_runtime.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/mir/module.hpp>
#include <cmath>
#include <vector>

namespace brass::il {

using namespace brass::runtime;

void* bronze_resolve_function(const char* name);

static Shape* g_root_shape_cached = nullptr;
static Shape** get_root_shape_storage() {
    g_root_shape_cached = ShapeRegistry::global().get_root_shape();
    return &g_root_shape_cached;
}

template <typename Engine>
void register_all_runtime_symbols_generic(Engine& jit) {
    auto reg = [&](const char* name, void* ptr) { jit.register_external_symbol(name, ptr); };

    reg("brass_tlab_refill", reinterpret_cast<void*>(&brass_tlab_refill));
    reg("brass_tlab_top", reinterpret_cast<void*>(brass_tlab_top_ptr()));
    reg("brass_tlab_end", reinterpret_cast<void*>(brass_tlab_end_ptr()));
    reg("brass_tlab_top_ptr", reinterpret_cast<void*>(&brass_tlab_top_ptr));
    reg("brass_tlab_end_ptr", reinterpret_cast<void*>(&brass_tlab_end_ptr));
    reg("brass_root_shape", reinterpret_cast<void*>(get_root_shape_storage()));

    reg("__bronze_module_env", reinterpret_cast<void*>(&g_bronze_module_env));
    reg("__bronze_key_map", reinterpret_cast<void*>(g_bronze_dummy_key_map));
    reg("__bronze_template_cells", reinterpret_cast<void*>(__bronze_template_cells));
    reg("bronze_template_object", reinterpret_cast<void*>(&bronze_template_object));
    reg("bronze_register_value_cells", reinterpret_cast<void*>(&bronze_register_value_cells));
    reg("bronze_register_fn_sources", reinterpret_cast<void*>(&bronze_register_fn_sources));
    reg("bronze_print_f64", reinterpret_cast<void*>(&bronze_print_f64));
    reg("bronze_print_i32", reinterpret_cast<void*>(&bronze_print_i32));
    reg("bronze_print_dynamic", reinterpret_cast<void*>(&bronze_print_dynamic));
    reg("bronze_print_space", reinterpret_cast<void*>(&bronze_print_space));
    reg("bronze_print_newline", reinterpret_cast<void*>(&bronze_print_newline));
    reg("bronze_print_f64_err", reinterpret_cast<void*>(&bronze_print_f64_err));
    reg("bronze_print_i32_err", reinterpret_cast<void*>(&bronze_print_i32_err));
    reg("bronze_print_dynamic_err", reinterpret_cast<void*>(&bronze_print_dynamic_err));
    reg("bronze_print_space_err", reinterpret_cast<void*>(&bronze_print_space_err));
    reg("bronze_print_newline_err", reinterpret_cast<void*>(&bronze_print_newline_err));
    reg("bronze_print_spread", reinterpret_cast<void*>(&bronze_print_spread));
    reg("bronze_print_spread_err", reinterpret_cast<void*>(&bronze_print_spread_err));
    reg("bronze_immutable_assign", reinterpret_cast<void*>(&bronze_immutable_assign));
    reg("bronze_dynamic_add", reinterpret_cast<void*>(&bronze_dynamic_add));
    reg("bronze_f64_mod", reinterpret_cast<void*>(&bronze_f64_mod));

    reg("bronze_name_resolve", reinterpret_cast<void*>(&bronze_name_resolve));
    reg("bronze_resolve_name", reinterpret_cast<void*>(&bronze_resolve_name));
    reg("bronze_env_create", reinterpret_cast<void*>(&bronze_env_create));
    reg("bronze_env_get", reinterpret_cast<void*>(&bronze_env_get));
    reg("bronze_env_get_tdz", reinterpret_cast<void*>(&bronze_env_get_tdz));
    reg("bronze_env_set", reinterpret_cast<void*>(&bronze_env_set));
    reg("bronze_env_ancestor", reinterpret_cast<void*>(&bronze_env_ancestor));
    reg("bronze_create_func", reinterpret_cast<void*>(&bronze_create_func));
    reg("bronze_create_function", reinterpret_cast<void*>(&bronze_create_function));
    reg("bronze_function_singleton", reinterpret_cast<void*>(&bronze_function_singleton));
    reg("bronze_import_meta", reinterpret_cast<void*>(&bronze_import_meta));
    reg("bronze_create_array", reinterpret_cast<void*>(&bronze_create_array));
    reg("bronze_create_object", reinterpret_cast<void*>(&bronze_create_object));
    reg("bronze_prop_get", reinterpret_cast<void*>(&bronze_prop_get));
    reg("bronze_prop_set", reinterpret_cast<void*>(&bronze_prop_set));
    reg("bronze_elem_get", reinterpret_cast<void*>(&bronze_elem_get));
    reg("bronze_elem_set", reinterpret_cast<void*>(&bronze_elem_set));
    reg("bronze_method_def", reinterpret_cast<void*>(&bronze_method_def));
    reg("bronze_method_def_computed", reinterpret_cast<void*>(&bronze_method_def_computed));
    reg("bronze_ic_get", reinterpret_cast<void*>(&bronze_ic_get));
    reg("bronze_ic_set", reinterpret_cast<void*>(&bronze_ic_set));
    reg("brass_ic_get_prop", reinterpret_cast<void*>(&brass_ic_get_prop));
    reg("brass_ic_set_prop", reinterpret_cast<void*>(&brass_ic_set_prop));
    reg("brass_dynamic_object_get_prop_str", reinterpret_cast<void*>(&brass_dynamic_object_get_prop_str));
    reg("brass_dynamic_object_set_prop_str", reinterpret_cast<void*>(&brass_dynamic_object_set_prop_str));

    reg("bronze_call_dynamic_0", reinterpret_cast<void*>(&bronze_call_dynamic_0));
    reg("bronze_call_dynamic_1", reinterpret_cast<void*>(&bronze_call_dynamic_1));
    reg("bronze_call_dynamic_2", reinterpret_cast<void*>(&bronze_call_dynamic_2));
    reg("bronze_call_dynamic_3", reinterpret_cast<void*>(&bronze_call_dynamic_3));
    reg("bronze_call_dynamic_4", reinterpret_cast<void*>(&bronze_call_dynamic_4));
    reg("bronze_call_dynamic_5", reinterpret_cast<void*>(&bronze_call_dynamic_5));
    reg("bronze_call_dynamic_6", reinterpret_cast<void*>(&bronze_call_dynamic_6));
    reg("bronze_call_dynamic_7", reinterpret_cast<void*>(&bronze_call_dynamic_7));
    reg("bronze_call_dynamic_8", reinterpret_cast<void*>(&bronze_call_dynamic_8));
    reg("bronze_call_dynamic_9", reinterpret_cast<void*>(&bronze_call_dynamic_9));
    reg("bronze_call_dynamic_10", reinterpret_cast<void*>(&bronze_call_dynamic_10));
    reg("bronze_call_dynamic_11", reinterpret_cast<void*>(&bronze_call_dynamic_11));
    reg("bronze_call_dynamic_12", reinterpret_cast<void*>(&bronze_call_dynamic_12));
    reg("bronze_call_dynamic_13", reinterpret_cast<void*>(&bronze_call_dynamic_13));
    reg("bronze_call_dynamic_14", reinterpret_cast<void*>(&bronze_call_dynamic_14));
    reg("bronze_call_dynamic_15", reinterpret_cast<void*>(&bronze_call_dynamic_15));
    reg("bronze_call_dynamic_16", reinterpret_cast<void*>(&bronze_call_dynamic_16));
    reg("bronze_call_dynamic_n", reinterpret_cast<void*>(&bronze_call_dynamic_n));

    reg("bronze_concat_begin", reinterpret_cast<void*>(&bronze_concat_begin));
    reg("bronze_concat_append", reinterpret_cast<void*>(&bronze_concat_append));
    reg("bronze_concat_end", reinterpret_cast<void*>(&bronze_concat_end));
    reg("bronze_global_get_name", reinterpret_cast<void*>(&bronze_global_get_name));
    reg("bronze_global_get", reinterpret_cast<void*>(&bronze_global_get));
    reg("bronze_typeof", reinterpret_cast<void*>(&bronze_typeof));
    reg("bronze_construct_0", reinterpret_cast<void*>(&bronze_construct_0));
    reg("bronze_construct_1", reinterpret_cast<void*>(&bronze_construct_1));
    reg("bronze_construct_2", reinterpret_cast<void*>(&bronze_construct_2));
    reg("bronze_construct_3", reinterpret_cast<void*>(&bronze_construct_3));
    reg("bronze_construct_4", reinterpret_cast<void*>(&bronze_construct_4));
    reg("bronze_construct_5", reinterpret_cast<void*>(&bronze_construct_5));
    reg("bronze_construct_6", reinterpret_cast<void*>(&bronze_construct_6));
    reg("bronze_construct_7", reinterpret_cast<void*>(&bronze_construct_7));
    reg("bronze_construct_8", reinterpret_cast<void*>(&bronze_construct_8));
    reg("bronze_construct_9", reinterpret_cast<void*>(&bronze_construct_9));
    reg("bronze_construct_10", reinterpret_cast<void*>(&bronze_construct_10));
    reg("bronze_construct_11", reinterpret_cast<void*>(&bronze_construct_11));
    reg("bronze_construct_12", reinterpret_cast<void*>(&bronze_construct_12));
    reg("bronze_construct_13", reinterpret_cast<void*>(&bronze_construct_13));
    reg("bronze_construct_14", reinterpret_cast<void*>(&bronze_construct_14));
    reg("bronze_construct_15", reinterpret_cast<void*>(&bronze_construct_15));
    reg("bronze_construct_16", reinterpret_cast<void*>(&bronze_construct_16));
    reg("bronze_construct", reinterpret_cast<void*>(&bronze_construct));
    reg("bronze_class_extends", reinterpret_cast<void*>(&bronze_class_extends));
    reg("bronze_super_call", reinterpret_cast<void*>(&bronze_super_call));
    reg("bronze_super_call_0", reinterpret_cast<void*>(&bronze_super_call_0));
    reg("bronze_super_call_1", reinterpret_cast<void*>(&bronze_super_call_1));
    reg("bronze_super_call_2", reinterpret_cast<void*>(&bronze_super_call_2));
    reg("bronze_super_call_3", reinterpret_cast<void*>(&bronze_super_call_3));
    reg("bronze_super_call_4", reinterpret_cast<void*>(&bronze_super_call_4));
    reg("bronze_super_call_5", reinterpret_cast<void*>(&bronze_super_call_5));
    reg("bronze_super_call_6", reinterpret_cast<void*>(&bronze_super_call_6));
    reg("bronze_super_call_7", reinterpret_cast<void*>(&bronze_super_call_7));
    reg("bronze_super_call_8", reinterpret_cast<void*>(&bronze_super_call_8));
    reg("bronze_super_call_9", reinterpret_cast<void*>(&bronze_super_call_9));
    reg("bronze_super_call_10", reinterpret_cast<void*>(&bronze_super_call_10));
    reg("bronze_super_call_11", reinterpret_cast<void*>(&bronze_super_call_11));
    reg("bronze_super_call_12", reinterpret_cast<void*>(&bronze_super_call_12));
    reg("bronze_super_call_13", reinterpret_cast<void*>(&bronze_super_call_13));
    reg("bronze_super_call_14", reinterpret_cast<void*>(&bronze_super_call_14));
    reg("bronze_super_call_15", reinterpret_cast<void*>(&bronze_super_call_15));
    reg("bronze_super_call_16", reinterpret_cast<void*>(&bronze_super_call_16));
    reg("bronze_super_call_n", reinterpret_cast<void*>(&bronze_super_call_n));
    reg("bronze_create_generator_object", reinterpret_cast<void*>(&bronze_create_generator_object));
    reg("bronze_create_async_generator_object", reinterpret_cast<void*>(&bronze_create_async_generator_object));
    reg("bronze_dynamic_import", reinterpret_cast<void*>(&bronze_dynamic_import));
    reg("bronze_iter_value", reinterpret_cast<void*>(&bronze_iter_value));
    reg("bronze_iter_close", reinterpret_cast<void*>(&bronze_iter_close));
    reg("bronze_iter_rest", reinterpret_cast<void*>(&bronze_iter_rest));
    reg("bronze_iter_delegate", reinterpret_cast<void*>(&bronze_iter_delegate));
    reg("bronze_async_iter_open", reinterpret_cast<void*>(&bronze_async_iter_open));
    reg("bronze_async_iter_next", reinterpret_cast<void*>(&bronze_async_iter_next));
    reg("bronze_async_iter_close", reinterpret_cast<void*>(&bronze_async_iter_close));
    reg("bronze_pattern_check", reinterpret_cast<void*>(&bronze_pattern_check));
    reg("bronze_array_append", reinterpret_cast<void*>(&bronze_array_append));
    reg("bronze_array_append_hole", reinterpret_cast<void*>(&bronze_array_append_hole));
    reg("bronze_array_spread", reinterpret_cast<void*>(&bronze_array_spread));
    reg("bronze_object_spread", reinterpret_cast<void*>(&bronze_object_spread));
    reg("bronze_object_rest", reinterpret_cast<void*>(&bronze_object_rest));
    reg("bronze_dynamic_call_spread", reinterpret_cast<void*>(&bronze_dynamic_call_spread));
    reg("bronze_call_method_spread", reinterpret_cast<void*>(&bronze_call_method_spread));
    reg("bronze_construct_spread", reinterpret_cast<void*>(&bronze_construct_spread));
    reg("bronze_super_call_spread", reinterpret_cast<void*>(&bronze_super_call_spread));
    reg("bronze_arg_at", reinterpret_cast<void*>(&bronze_arg_at));
    reg("bronze_arguments_object", reinterpret_cast<void*>(&bronze_arguments_object));
    reg("bronze_rest_args", reinterpret_cast<void*>(&bronze_rest_args));
    reg("bronze_super_get", reinterpret_cast<void*>(&bronze_super_get));
    reg("bronze_super_set", reinterpret_cast<void*>(&bronze_super_set));
    reg("bronze_object_keys", reinterpret_cast<void*>(&bronze_object_keys));
    reg("bronze_for_in_keys", reinterpret_cast<void*>(&bronze_for_in_keys));
    reg("bronze_instanceof", reinterpret_cast<void*>(&bronze_instanceof));
    reg("bronze_has_property", reinterpret_cast<void*>(&bronze_has_property));
    reg("bronze_is_nullish", reinterpret_cast<void*>(&bronze_is_nullish));
    reg("bronze_strict_eq", reinterpret_cast<void*>(&bronze_strict_eq));
    reg("bronze_loose_eq", reinterpret_cast<void*>(&bronze_loose_eq));
    reg("bronze_rel_lt", reinterpret_cast<void*>(&bronze_rel_lt));
    reg("bronze_rel_gt", reinterpret_cast<void*>(&bronze_rel_gt));
    reg("bronze_rel_le", reinterpret_cast<void*>(&bronze_rel_le));
    reg("bronze_rel_ge", reinterpret_cast<void*>(&bronze_rel_ge));
    reg("bronze_define_own_attr", reinterpret_cast<void*>(&bronze_define_own_attr));
    reg("bronze_accessor_def", reinterpret_cast<void*>(&bronze_accessor_def));
    reg("bronze_accessor_def_computed", reinterpret_cast<void*>(&bronze_accessor_def_computed));
    reg("bronze_module_namespace", reinterpret_cast<void*>(&bronze_module_namespace));
    reg("bronze_pin_guard", reinterpret_cast<void*>(&bronze_pin_guard));
    reg("bronze_census_record", reinterpret_cast<void*>(&bronze_census_record));
    reg("bronze_to_int32", reinterpret_cast<void*>(&bronze_to_int32));
    reg("bronze_to_int32_f64", reinterpret_cast<void*>(&bronze_to_int32_f64));
    reg("bronze_private_new", reinterpret_cast<void*>(&bronze_private_new));
    reg("bronze_private_has", reinterpret_cast<void*>(&bronze_private_has));
    reg("bronze_private_get", reinterpret_cast<void*>(&bronze_private_get));
    reg("bronze_private_add", reinterpret_cast<void*>(&bronze_private_add));
    reg("bronze_private_set", reinterpret_cast<void*>(&bronze_private_set));
    reg("bronze_private_misuse", reinterpret_cast<void*>(&bronze_private_misuse));
    reg("bronze_main_key_constants", reinterpret_cast<void*>(&g_bronze_main_key_constants));
    reg("bronze_register_key_manifest", reinterpret_cast<void*>(&bronze_register_key_manifest));
    reg("bronze_box_str_key", reinterpret_cast<void*>(&bronze_box_str_key));
    reg("bronze_box_str", reinterpret_cast<void*>(&bronze_box_str));
    reg("bronze_unbox_str", reinterpret_cast<void*>(&bronze_unbox_str));
    reg("bronze_unbox_f64", reinterpret_cast<void*>(&bronze_unbox_f64));
    reg("bronze_box_f64", reinterpret_cast<void*>(&bronze_box_f64));
    reg("bronze_unbox_i32", reinterpret_cast<void*>(&bronze_unbox_i32));
    reg("bronze_box_i32", reinterpret_cast<void*>(&bronze_box_i32));
    reg("bronze_unbox_bool", reinterpret_cast<void*>(&bronze_unbox_bool));
    reg("bronze_box_bool", reinterpret_cast<void*>(&bronze_box_bool));
    reg("bronze_get_new_target", reinterpret_cast<void*>(&bronze_get_new_target));
    reg("bronze_exception_get", reinterpret_cast<void*>(&bronze_exception_get));
    reg("bronze_exception_set", reinterpret_cast<void*>(&bronze_exception_set));
    reg("bronze_exception_take", reinterpret_cast<void*>(&bronze_exception_take));
    reg("bronze_exception_pending", reinterpret_cast<void*>(&bronze_exception_pending));
    reg("bronze_uncaught_exception", reinterpret_cast<void*>(&bronze_uncaught_exception));
    reg("bronze_gc_frame_push", reinterpret_cast<void*>(&bronze_gc_frame_push));
    reg("bronze_gc_frame_pop", reinterpret_cast<void*>(&bronze_gc_frame_pop));
    reg("bronze_pin_violation", reinterpret_cast<void*>(&bronze_pin_violation));
    reg("bronze_pin_check_array", reinterpret_cast<void*>(&bronze_pin_check_array));
    reg("bronze_pow", reinterpret_cast<void*>(&bronze_pow));
    reg("bronze_dynamic_pow", reinterpret_cast<void*>(&bronze_dynamic_pow));
    reg("bronze_dynamic_bitand", reinterpret_cast<void*>(&bronze_dynamic_bitand));
    reg("bronze_dynamic_bitor", reinterpret_cast<void*>(&bronze_dynamic_bitor));
    reg("bronze_dynamic_bitxor", reinterpret_cast<void*>(&bronze_dynamic_bitxor));
    reg("bronze_dynamic_shl", reinterpret_cast<void*>(&bronze_dynamic_shl));
    reg("bronze_dynamic_shr", reinterpret_cast<void*>(&bronze_dynamic_shr));
    reg("bronze_dynamic_ushr", reinterpret_cast<void*>(&bronze_dynamic_ushr));
    reg("bronze_dynamic_sub", reinterpret_cast<void*>(&bronze_dynamic_sub));
    reg("bronze_dynamic_mul", reinterpret_cast<void*>(&bronze_dynamic_mul));
    reg("bronze_dynamic_div", reinterpret_cast<void*>(&bronze_dynamic_div));
    reg("bronze_dynamic_mod", reinterpret_cast<void*>(&bronze_dynamic_mod));
    reg("bronze_dynamic_neg", reinterpret_cast<void*>(&bronze_dynamic_neg));
    reg("bronze_dynamic_bitnot", reinterpret_cast<void*>(&bronze_dynamic_bitnot));
    reg("sin", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::sin)));
    reg("cos", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::cos)));
    reg("sqrt", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::sqrt)));
    reg("fabs", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::fabs)));
    reg("floor", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::floor)));
    reg("ceil", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::ceil)));
    reg("trunc", reinterpret_cast<void*>(static_cast<double(*)(double)>(&std::trunc)));

    set_coro_symbol_resolver(&bronze_resolve_function);

    reg("brass_gc_write_barrier", reinterpret_cast<void*>(&brass_gc_write_barrier));
    reg("brass_gc_card_table_base", reinterpret_cast<void*>(&brass_gc_card_table_base));
    reg("brass_gc_heap_base", reinterpret_cast<void*>(&brass_gc_heap_base));
    reg("bronze_create_async_machine", reinterpret_cast<void*>(&bronze_create_async_machine));
    reg("bronze_bigint_literal", reinterpret_cast<void*>(&bronze_bigint_literal));
    reg("bronze_async_machine", reinterpret_cast<void*>(&bronze_async_machine));
    reg("bronze_async_start", reinterpret_cast<void*>(&bronze_async_start));
    reg("bronze_async_await", reinterpret_cast<void*>(&bronze_async_await));
    reg("bronze_to_string", reinterpret_cast<void*>(&bronze_to_string));
    reg("bronze_prop_delete", reinterpret_cast<void*>(&bronze_prop_delete));
    reg("bronze_elem_delete", reinterpret_cast<void*>(&bronze_elem_delete));
    reg("bronze_iter_open", reinterpret_cast<void*>(&bronze_iter_open));
    reg("bronze_iter_step", reinterpret_cast<void*>(&bronze_iter_step));
    reg("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    reg("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    reg("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    reg("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    reg("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    reg("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    reg("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));
}

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit) {
    set_active_jit(&jit);
    register_all_runtime_symbols_generic(jit);
}

void unregister_all_runtime_symbols() {
    set_active_jit(nullptr);
    set_coro_symbol_resolver(nullptr);
}

void register_bronze_runtime_symbols(void* jit_engine_ptr) {
    if (jit_engine_ptr) {
        auto* jit = reinterpret_cast<codegen::JitExecutionEngine*>(jit_engine_ptr);
        register_all_runtime_symbols(*jit);
    } else {
        unregister_all_runtime_symbols();
    }
}

void register_bronze_baseline_symbols(void* baseline_jit_ptr) {
    if (baseline_jit_ptr) {
        auto* compiler = reinterpret_cast<codegen::BaselineJitCompiler*>(baseline_jit_ptr);
        register_all_runtime_symbols_generic(*compiler);
    }
}

void register_bronze_interpreter_symbols(void* interp_ptr) {
    if (!interp_ptr) return;
    auto* interp = reinterpret_cast<Interpreter*>(interp_ptr);
    interp->register_external_function("bronze_print_f64", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_f64(args[0].as_f64());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_i32", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_i32(args[0].as_i32());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_dynamic", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_dynamic(args[0].as_i64());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_newline", [](Interpreter&, const std::vector<RuntimeValue>&) {
        bronze_print_newline();
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_f64_mod", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        return (args.size() >= 2) ? RuntimeValue::from_f64(bronze_f64_mod(args[0].as_f64(), args[1].as_f64())) : RuntimeValue::from_f64(0.0);
    });
    interp->register_external_function("bronze_call_dynamic_0", [](Interpreter&, const std::vector<RuntimeValue>& a) {
        return RuntimeValue::from_i64(bronze_call_dynamic_0(a.size() > 0 ? a[0].as_i64() : 0, a.size() > 1 ? a[1].as_i64() : 0));
    });
    interp->register_external_function("bronze_call_dynamic_1", [](Interpreter&, const std::vector<RuntimeValue>& a) {
        return RuntimeValue::from_i64(bronze_call_dynamic_1(a.size() > 0 ? a[0].as_i64() : 0, a.size() > 1 ? a[1].as_i64() : 0, a.size() > 2 ? a[2].as_i64() : 0));
    });
    interp->register_external_function("bronze_call_dynamic_2", [](Interpreter&, const std::vector<RuntimeValue>& a) {
        return RuntimeValue::from_i64(bronze_call_dynamic_2(a.size() > 0 ? a[0].as_i64() : 0, a.size() > 1 ? a[1].as_i64() : 0, a.size() > 2 ? a[2].as_i64() : 0, a.size() > 3 ? a[3].as_i64() : 0));
    });
}

void register_all_module_external_symbols(Module* mod, const std::string& entry_symbol) {
    if (!mod) return;
    static const char* const kSymbols[] = {
        "bronze_print_f64",
        "bronze_print_i32",
        "bronze_print_dynamic",
        "bronze_print_space",
        "bronze_print_newline",
        "bronze_print_f64_err",
        "bronze_print_i32_err",
        "bronze_print_dynamic_err",
        "bronze_print_space_err",
        "bronze_print_newline_err",
        "bronze_print_spread",
        "bronze_print_spread_err",
        "bronze_immutable_assign",
        "bronze_dynamic_add",
        "bronze_f64_mod",
        "bronze_name_resolve",
        "bronze_resolve_name",
        "bronze_env_create",
        "bronze_env_get",
        "bronze_env_get_tdz",
        "bronze_env_set",
        "bronze_env_ancestor",
        "bronze_create_func",
        "bronze_create_function",
        "bronze_function_singleton",
        "bronze_import_meta",
        "bronze_create_array",
        "bronze_create_object",
        "bronze_prop_get",
        "bronze_prop_set",
        "bronze_elem_get",
        "bronze_elem_set",
        "bronze_method_def",
        "bronze_method_def_computed",
        "bronze_define_own_attr",
        "bronze_accessor_def",
        "bronze_accessor_def_computed",
        "bronze_module_namespace",
        "bronze_ic_get",
        "bronze_ic_set",
        "brass_ic_get_prop",
        "brass_ic_set_prop",
        "brass_dynamic_object_get_prop_str",
        "brass_dynamic_object_set_prop_str",
        "bronze_call_dynamic_0",
        "bronze_call_dynamic_1",
        "bronze_call_dynamic_2",
        "bronze_call_dynamic_3",
        "bronze_call_dynamic_4",
        "bronze_call_dynamic_5",
        "bronze_call_dynamic_6",
        "bronze_call_dynamic_7",
        "bronze_call_dynamic_8",
        "bronze_call_dynamic_9",
        "bronze_call_dynamic_10",
        "bronze_call_dynamic_11",
        "bronze_call_dynamic_12",
        "bronze_call_dynamic_13",
        "bronze_call_dynamic_14",
        "bronze_call_dynamic_15",
        "bronze_call_dynamic_16",
        "bronze_call_dynamic_n",
        "bronze_get_new_target",
        "bronze_create_async_machine",
        "bronze_bigint_literal",
        "bronze_async_machine",
        "bronze_async_start",
        "bronze_async_await",
        "bronze_to_string",
        "bronze_prop_delete",
        "bronze_elem_delete",
        "bronze_iter_open",
        "bronze_iter_step",
        "brass_coro_create",
        "brass_coro_resume",
        "brass_coro_is_done",
        "brass_coro_destroy",
        "brass_tlab_refill",
        "brass_tlab_top",
        "brass_tlab_end",
        "brass_tlab_top_ptr",
        "brass_tlab_end_ptr",
        "brass_root_shape",
        "__bronze_key_map",
        "__bronze_template_cells",
        "bronze_template_object",
        "bronze_register_value_cells",
        "bronze_register_fn_sources",
        "bronze_concat_begin",
        "bronze_concat_append",
        "bronze_concat_end",
        "bronze_global_get_name",
        "bronze_global_get",
        "bronze_typeof",
        "bronze_construct_0",
        "bronze_construct_1",
        "bronze_construct_2",
        "bronze_construct_3",
        "bronze_construct_4",
        "bronze_construct_5",
        "bronze_construct_6",
        "bronze_construct_7",
        "bronze_construct_8",
        "bronze_construct_9",
        "bronze_construct_10",
        "bronze_construct_11",
        "bronze_construct_12",
        "bronze_construct_13",
        "bronze_construct_14",
        "bronze_construct_15",
        "bronze_construct_16",
        "bronze_construct",
        "bronze_class_extends",
        "bronze_super_call",
        "bronze_super_call_0",
        "bronze_super_call_1",
        "bronze_super_call_2",
        "bronze_super_call_3",
        "bronze_super_call_4",
        "bronze_super_call_5",
        "bronze_super_call_6",
        "bronze_super_call_7",
        "bronze_super_call_8",
        "bronze_super_call_9",
        "bronze_super_call_10",
        "bronze_super_call_11",
        "bronze_super_call_12",
        "bronze_super_call_13",
        "bronze_super_call_14",
        "bronze_super_call_15",
        "bronze_super_call_16",
        "bronze_super_call_n",
        "bronze_create_generator_object",
        "bronze_create_async_generator_object",
        "bronze_dynamic_import",
        "bronze_iter_value",
        "bronze_iter_close",
        "bronze_iter_rest",
        "bronze_iter_delegate",
        "bronze_async_iter_open",
        "bronze_async_iter_next",
        "bronze_async_iter_close",
        "bronze_pattern_check",
        "bronze_array_append",
        "bronze_array_append_hole",
        "bronze_array_spread",
        "bronze_object_spread",
        "bronze_object_rest",
        "bronze_dynamic_call_spread",
        "bronze_call_method_spread",
        "bronze_construct_spread",
        "bronze_super_call_spread",
        "bronze_arg_at",
        "bronze_arguments_object",
        "bronze_rest_args",
        "bronze_super_get",
        "bronze_super_set",
        "bronze_object_keys",
        "bronze_for_in_keys",
        "bronze_instanceof",
        "bronze_has_property",
        "bronze_is_nullish",
        "bronze_strict_eq",
        "bronze_loose_eq",
        "bronze_rel_lt",
        "bronze_rel_gt",
        "bronze_rel_le",
        "bronze_rel_ge",
        "bronze_pin_guard",
        "bronze_census_record",
        "bronze_to_int32",
        "bronze_to_int32_f64",
        "bronze_private_new",
        "bronze_private_has",
        "bronze_private_get",
        "bronze_private_add",
        "bronze_private_set",
        "bronze_private_misuse",
        "bronze_census_register",
        "__bronze_census_out_path",
        "__bronze_census_sites",
        "bronze_register_key_manifest",
        "bronze_box_str_key",
        "bronze_box_str",
        "bronze_unbox_str",
        "bronze_unbox_f64",
        "bronze_box_f64",
        "bronze_unbox_i32",
        "bronze_box_i32",
        "bronze_unbox_bool",
        "bronze_box_bool",
        "bronze_exception_get",
        "bronze_exception_set",
        "bronze_exception_take",
        "bronze_exception_pending",
        "bronze_uncaught_exception",
        "bronze_gc_frame_push",
        "bronze_gc_frame_pop",
        "bronze_pin_violation",
        "bronze_pin_check_array",
        "bronze_pow",
        "bronze_dynamic_pow",
        "bronze_dynamic_bitand",
        "bronze_dynamic_bitor",
        "bronze_dynamic_bitxor",
        "bronze_dynamic_shl",
        "bronze_dynamic_shr",
        "bronze_dynamic_ushr",
        "bronze_dynamic_sub",
        "bronze_dynamic_mul",
        "bronze_dynamic_div",
        "bronze_dynamic_mod",
        "bronze_dynamic_neg",
        "bronze_dynamic_bitnot",
        "sin",
        "cos",
        "sqrt",
        "fabs",
        "floor",
        "ceil",
        "trunc",
    };
    for (const char* sym : kSymbols) {
        mod->add_external_symbol(sym);
    }
    const std::string key_sym = (entry_symbol.empty() || entry_symbol == "main" || entry_symbol == "bronze_main")
                                    ? "bronze_main_key_constants"
                                    : (entry_symbol + "_key_constants");
    mod->add_external_symbol(key_sym);
}

} // namespace brass::il
