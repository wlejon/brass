#pragma once

#include <cstdint>
#include <string>

namespace brass {
class HostGC;

namespace codegen {
class JitExecutionEngine;
}

namespace il {

constexpr uint64_t kUndefinedTag = 0xFFF6000000000000ULL;
constexpr uint64_t kNullTag      = 0xFFF5000000000000ULL;
constexpr uint64_t kBoolTag      = 0xFFF4000000000000ULL;
constexpr uint64_t kInt32Tag     = 0xFFF3000000000000ULL;
constexpr uint64_t kPrintTag     = 0xFFFE000000000001ULL;
constexpr uint64_t kPrintErrTag  = 0xFFFE000000000002ULL;

std::string format_js_number(double v);

extern "C" {
void bronze_print_f64(double v);
void bronze_print_i32(int32_t v);
void bronze_print_dynamic(int64_t v);
void bronze_print_space();
void bronze_print_newline();
int64_t bronze_dynamic_add(int64_t a, int64_t b);
double bronze_f64_mod(double a, double b);

int64_t bronze_name_resolve(const char* name);
int64_t bronze_env_create(int64_t parent_box, int32_t size);
int64_t bronze_env_get(int64_t env_box, int32_t depth, int32_t index);
void bronze_env_set(int64_t env_box, int32_t depth, int32_t index, int64_t val);
int64_t bronze_create_func(void* code_ptr, int32_t param_count, int64_t env_box);
int64_t bronze_create_array(int32_t size);
int64_t bronze_create_object();
int64_t bronze_prop_get(int64_t obj_box, int32_t key_index, uint64_t* ic_entry);
void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, uint64_t* ic_entry, int32_t strict);
int64_t bronze_elem_get(int64_t arr_box, int64_t index_box);
void bronze_elem_set(int64_t arr_box, int64_t index_box, int64_t val, int32_t ic_slot);
void bronze_method_def(int64_t obj_box, int32_t key_index, int64_t closure_box);
void bronze_method_def_computed(int64_t obj_box, int64_t key_box, int64_t closure_box);
void bronze_define_own_attr(uint64_t obj_bits, uint32_t key_index, uint64_t val_bits, uint32_t mask);
void bronze_accessor_def(uint64_t obj_bits, uint32_t key_index, uint64_t getter_bits, uint64_t setter_bits, int32_t enumerable);
void bronze_accessor_def_computed(uint64_t obj_bits, uint64_t key_bits, uint64_t getter_bits, uint64_t setter_bits, int32_t enumerable);
uint64_t bronze_module_namespace(uint64_t src_bits);
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
void bronze_register_value_cells(uint64_t* cells, uint64_t count);

int64_t bronze_concat_begin(int64_t a, int64_t b, uint32_t remaining);
int64_t bronze_concat_append(int64_t a, int64_t b);
int64_t bronze_concat_end(int64_t a);
int64_t bronze_global_get_name(const char* name);
uint64_t bronze_global_get(uint32_t key_index, uint64_t* cache_cell);
uint64_t bronze_typeof(uint64_t bits);
int64_t bronze_construct_0(int64_t callee_box);
int64_t bronze_construct_1(int64_t callee_box, int64_t arg0);
int64_t bronze_construct_2(int64_t callee_box, int64_t arg0, int64_t arg1);
int64_t bronze_construct_3(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2);
int64_t bronze_construct_4(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3);
int64_t bronze_construct_5(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);
int64_t bronze_construct_6(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5);
int64_t bronze_construct_7(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6);
int64_t bronze_construct_8(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6, int64_t arg7);
int64_t bronze_construct(int64_t callee_box, uint32_t argc, const int64_t* argv);
void bronze_class_extends(int64_t sub_box, int64_t super_box);
int64_t bronze_super_call(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv);
int64_t bronze_super_call_0(int64_t sub_box, int64_t this_box);
int64_t bronze_super_call_1(int64_t sub_box, int64_t this_box, int64_t a0);
int64_t bronze_super_call_2(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1);
int64_t bronze_super_call_3(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2);
int64_t bronze_super_call_4(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3);
int64_t bronze_super_call_5(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4);
int64_t bronze_super_call_6(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5);
int64_t bronze_super_call_7(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6);
int64_t bronze_super_call_8(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7);
int64_t bronze_super_call_n(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv);
int64_t bronze_arg_at(uint32_t argc, const int64_t* argv, uint32_t index);
int64_t bronze_arguments_object(uint32_t argc, const int64_t* argv, int64_t callee, int32_t is_strict);
int64_t bronze_rest_args(uint32_t argc, const int64_t* argv, uint32_t first_index);
int64_t bronze_super_get(int64_t proto_box, uint32_t key_index, int64_t this_box);
int64_t bronze_object_keys(int64_t obj_box);
int64_t bronze_for_in_keys(int64_t obj_box);
int32_t bronze_instanceof(int64_t a_box, int64_t b_box);
int32_t bronze_has_property(int64_t key_box, int64_t obj_box);
int32_t bronze_is_nullish(int64_t val_box);
int32_t bronze_strict_eq(int64_t a, int64_t b);
int32_t bronze_loose_eq(int64_t a, int64_t b);
int32_t bronze_rel_lt(int64_t a, int64_t b);
int32_t bronze_rel_gt(int64_t a, int64_t b);
int32_t bronze_rel_le(int64_t a, int64_t b);
int32_t bronze_rel_ge(int64_t a, int64_t b);
void bronze_pin_guard(int64_t val_box, int32_t shape, const char* name);
void bronze_census_record(uint32_t key_id, uint32_t site_info, uint64_t value_bits);
uint64_t bronze_exception_get();
void bronze_exception_set(uint64_t bits);
uint64_t bronze_exception_take();
int32_t bronze_exception_pending();
void bronze_uncaught_exception();
uint64_t bronze_pin_violation(uint32_t key_index, uint64_t bits);
void bronze_pin_check_array(uint32_t key_index, uint64_t bits);
double bronze_pow(double base, double exponent);
uint64_t bronze_dynamic_pow(uint64_t l, uint64_t r);
void bronze_register_key_manifest(const uint8_t* data);
uint64_t bronze_box_str_key(uint32_t key_index);
double bronze_unbox_f64(uint64_t bits);
uint64_t bronze_box_f64(double v);
int32_t bronze_unbox_i32(uint64_t bits);
uint64_t bronze_box_i32(int32_t v);
int32_t bronze_unbox_bool(uint64_t bits);
uint64_t bronze_box_bool(int32_t v);
}


void register_all_runtime_symbols(codegen::JitExecutionEngine& jit);
void unregister_all_runtime_symbols();
codegen::JitExecutionEngine* get_active_jit();
void set_active_jit(codegen::JitExecutionEngine* jit);

extern uint32_t g_bronze_main_key_constants;
extern int64_t g_bronze_module_env;

} // namespace il
} // namespace brass
