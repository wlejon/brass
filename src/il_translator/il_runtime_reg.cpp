#include "il_runtime.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <cmath>
#include <vector>

namespace brass::il {

using namespace brass::runtime;

void* bronze_resolve_function(const char* name);

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit) {
    set_active_jit(&jit);
    auto reg = [&](const char* name, void* ptr) { jit.register_external_symbol(name, ptr); };

    reg("__bronze_module_env", reinterpret_cast<void*>(&g_bronze_module_env));
    reg("bronze_register_value_cells", reinterpret_cast<void*>(&bronze_register_value_cells));
    reg("bronze_print_f64", reinterpret_cast<void*>(&bronze_print_f64));
    reg("bronze_print_i32", reinterpret_cast<void*>(&bronze_print_i32));
    reg("bronze_print_dynamic", reinterpret_cast<void*>(&bronze_print_dynamic));
    reg("bronze_print_space", reinterpret_cast<void*>(&bronze_print_space));
    reg("bronze_print_newline", reinterpret_cast<void*>(&bronze_print_newline));
    reg("bronze_dynamic_add", reinterpret_cast<void*>(&bronze_dynamic_add));
    reg("bronze_f64_mod", reinterpret_cast<void*>(&bronze_f64_mod));

    reg("bronze_name_resolve", reinterpret_cast<void*>(&bronze_name_resolve));
    reg("bronze_env_create", reinterpret_cast<void*>(&bronze_env_create));
    reg("bronze_env_get", reinterpret_cast<void*>(&bronze_env_get));
    reg("bronze_env_set", reinterpret_cast<void*>(&bronze_env_set));
    reg("bronze_create_func", reinterpret_cast<void*>(&bronze_create_func));
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
    reg("bronze_super_call_n", reinterpret_cast<void*>(&bronze_super_call_n));
    reg("bronze_arg_at", reinterpret_cast<void*>(&bronze_arg_at));
    reg("bronze_arguments_object", reinterpret_cast<void*>(&bronze_arguments_object));
    reg("bronze_rest_args", reinterpret_cast<void*>(&bronze_rest_args));
    reg("bronze_super_get", reinterpret_cast<void*>(&bronze_super_get));
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
    reg("bronze_exception_get", reinterpret_cast<void*>(&bronze_exception_get));
    reg("bronze_exception_set", reinterpret_cast<void*>(&bronze_exception_set));
    reg("bronze_exception_take", reinterpret_cast<void*>(&bronze_exception_take));
    reg("bronze_exception_pending", reinterpret_cast<void*>(&bronze_exception_pending));
    reg("bronze_uncaught_exception", reinterpret_cast<void*>(&bronze_uncaught_exception));
    reg("bronze_pin_violation", reinterpret_cast<void*>(&bronze_pin_violation));
    reg("bronze_pin_check_array", reinterpret_cast<void*>(&bronze_pin_check_array));
    reg("bronze_pow", reinterpret_cast<void*>(&bronze_pow));
    reg("bronze_dynamic_pow", reinterpret_cast<void*>(&bronze_dynamic_pow));
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
    reg("bronze_async_start", reinterpret_cast<void*>(&bronze_async_start));
    reg("bronze_async_await", reinterpret_cast<void*>(&bronze_async_await));
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

} // namespace brass::il
