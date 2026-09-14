#pragma once

#include <cstdint>
#include <string>

namespace brass {
class HostGC;
class Module;

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

void bronze_print_f64(double v);
void bronze_print_i32(int32_t v);
void bronze_print_dynamic(int64_t v);
void bronze_print_space();
void bronze_print_newline();
void bronze_print_f64_err(double v);
void bronze_print_i32_err(int32_t v);
void bronze_print_dynamic_err(int64_t v);
void bronze_print_space_err();
void bronze_print_newline_err();
void bronze_print_spread(uint64_t v);
void bronze_print_spread_err(uint64_t v);
uint64_t bronze_immutable_assign();
int64_t bronze_dynamic_add(int64_t a, int64_t b);
double bronze_f64_mod(double a, double b);

int64_t bronze_name_resolve(const char* name);
uint64_t bronze_resolve_name(uint32_t key_index, int32_t soft);
int64_t bronze_env_create(int64_t parent_box, int32_t size);
int64_t bronze_env_get(int64_t env_box, int32_t depth, int32_t index);
uint64_t bronze_env_get_tdz(uint64_t env_bits, uint32_t depth, uint32_t index, uint32_t key_index);
void bronze_env_set(int64_t env_box, int32_t depth, int32_t index, int64_t val);
uint64_t bronze_env_ancestor(uint64_t env_bits, uint32_t depth);
int64_t bronze_create_func(void* code_ptr, int32_t param_count, int64_t env_box);
uint64_t bronze_create_function(void* code, uint32_t arity, uint32_t length, uint32_t name_key, uint32_t fn_flags, uint64_t env_bits);
int64_t bronze_function_singleton(void* code, uint32_t arity, uint32_t length, uint32_t name_key, uint32_t fn_flags, uint64_t* slot_cell);
uint64_t bronze_import_meta(uint32_t url_key_index);
int32_t bronze_to_int32(uint64_t bits);
int32_t bronze_to_int32_f64(double d);
uint64_t bronze_private_new(void);
bool bronze_private_has(uint64_t tableBits, uint64_t objBits, uint32_t nameKeyIndex);
uint64_t bronze_private_get(uint64_t tableBits, uint64_t objBits, uint32_t nameKeyIndex);
void bronze_private_add(uint64_t tableBits, uint64_t objBits, uint64_t valueBits);
void bronze_private_set(uint64_t tableBits, uint64_t objBits, uint64_t valueBits, uint32_t nameKeyIndex);
uint64_t bronze_private_misuse(uint32_t nameKeyIndex, uint32_t code);
int64_t bronze_create_array(int32_t size);
int64_t bronze_create_object();
int64_t bronze_prop_get(int64_t obj_box, int32_t key_index, uint64_t* ic_entry = nullptr);
void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, uint64_t* ic_entry = nullptr, int32_t strict = 1);
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

extern "C" {
void* bronze_tls_block_addr();
uint64_t brass_ic_get_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id);
void brass_ic_set_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id, uint64_t val_raw);
}

struct BronzeClosure {
    char fn_name[64];
    void* code_ptr;
    int64_t env_box;
    uint32_t param_count;
};

void* bronze_resolve_function(const char* name);

int64_t bronze_call_dynamic_0(int64_t callee_box, int64_t this_box);
int64_t bronze_call_dynamic_1(int64_t callee_box, int64_t this_box, int64_t arg0);
int64_t bronze_call_dynamic_2(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1);
int64_t bronze_call_dynamic_3(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2);
int64_t bronze_call_dynamic_4(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3);
int64_t bronze_call_dynamic_5(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);
int64_t bronze_call_dynamic_6(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5);
int64_t bronze_call_dynamic_7(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6);
int64_t bronze_call_dynamic_8(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6, int64_t arg7);
int64_t bronze_call_dynamic_9(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8);
int64_t bronze_call_dynamic_10(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9);
int64_t bronze_call_dynamic_11(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10);
int64_t bronze_call_dynamic_12(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11);
int64_t bronze_call_dynamic_13(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12);
int64_t bronze_call_dynamic_14(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13);
int64_t bronze_call_dynamic_15(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14);
int64_t bronze_call_dynamic_16(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15);
int64_t bronze_call_dynamic_n(int64_t callee_box, int64_t this_box, int32_t argc, const int64_t* argv);
void bronze_register_value_cells(uint64_t* cells, uint64_t count);
void bronze_register_fn_sources(const char* text, uint32_t text_len, const uint64_t* entries, uint32_t count);

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
int64_t bronze_construct_9(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8);
int64_t bronze_construct_10(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9);
int64_t bronze_construct_11(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10);
int64_t bronze_construct_12(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11);
int64_t bronze_construct_13(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12);
int64_t bronze_construct_14(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13);
int64_t bronze_construct_15(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14);
int64_t bronze_construct_16(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15);
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
int64_t bronze_super_call_9(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8);
int64_t bronze_super_call_10(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9);
int64_t bronze_super_call_11(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10);
int64_t bronze_super_call_12(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11);
int64_t bronze_super_call_13(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12);
int64_t bronze_super_call_14(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13);
int64_t bronze_super_call_15(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14);
int64_t bronze_super_call_16(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15);
int64_t bronze_super_call_n(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv);
int64_t bronze_arg_at(uint32_t argc, const int64_t* argv, uint32_t index);
int64_t bronze_arguments_object(uint32_t argc, const int64_t* argv, int64_t callee, int32_t is_strict);
int64_t bronze_rest_args(uint32_t argc, const int64_t* argv, uint32_t first_index);
int64_t bronze_super_get(int64_t proto_box, uint32_t key_index, int64_t this_box);
void bronze_super_set(uint64_t proto_bits, uint32_t key_idx, uint64_t this_bits, uint64_t val_bits, int32_t strict);
uint64_t bronze_create_generator_object(uint64_t fn_val);
uint64_t bronze_create_async_generator_object(uint64_t fn_val);
uint64_t bronze_bigint_literal(uint32_t key_index);
uint64_t bronze_async_machine(uint64_t resume_bits);
uint64_t bronze_to_string(uint64_t bits);
int32_t bronze_prop_delete(uint64_t obj, uint32_t key_id, int32_t strict);
int32_t bronze_elem_delete(uint64_t obj, uint64_t idx, int32_t strict);
uint64_t bronze_dynamic_import(uint64_t spec_box, uint32_t base_key);
uint64_t bronze_iter_value(uint64_t iter_rec);
void bronze_iter_close(uint64_t iter_rec, int32_t suppress);
uint64_t bronze_iter_rest(uint64_t iter_rec);
uint64_t bronze_iter_delegate(uint64_t iter_rec, uint64_t mode, uint64_t sent);
uint64_t bronze_async_iter_open(uint64_t iter_rec);
uint64_t bronze_async_iter_next(uint64_t iter_rec);
void bronze_async_iter_close(uint64_t iter_rec, int32_t suppress);
uint64_t bronze_pattern_check(uint64_t src, uint32_t key_idx);
void bronze_array_append(uint64_t arr, uint64_t val);
void bronze_array_append_hole(uint64_t arr);
void bronze_array_spread(uint64_t arr, uint64_t val);
void bronze_object_spread(uint64_t obj, uint64_t val);
uint64_t bronze_object_rest(uint64_t src, uint64_t excluded);
uint64_t bronze_dynamic_call_spread(uint64_t callee, uint64_t this_val, uint64_t args_arr);
uint64_t bronze_call_method_spread(uint64_t this_val, uint32_t key_index, uint64_t args_arr, void* ic_entry);
uint64_t bronze_construct_spread(uint64_t callee, uint64_t args_arr);
uint64_t bronze_super_call_spread(uint64_t base, uint64_t this_val, uint64_t args_arr);
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
void* bronze_gc_frame_push(uint32_t count);
void bronze_gc_frame_pop();
uint64_t bronze_pin_violation(uint32_t key_index, uint64_t bits);
void bronze_pin_check_array(uint32_t key_index, uint64_t bits);
double bronze_pow(double base, double exponent);
uint64_t bronze_dynamic_pow(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_bitand(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_bitor(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_bitxor(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_shl(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_shr(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_ushr(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_sub(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_mul(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_div(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_mod(uint64_t l, uint64_t r);
uint64_t bronze_dynamic_neg(uint64_t bits);
uint64_t bronze_dynamic_bitnot(uint64_t bits);
void bronze_register_key_manifest(const uint8_t* data, uint32_t* key_map = nullptr);
uint64_t bronze_box_str_key(uint32_t key_index);
uint64_t bronze_box_str(const char* s);
const char* bronze_unbox_str(uint64_t bits);
double bronze_unbox_f64(uint64_t bits);
uint64_t bronze_box_f64(double v);
int32_t bronze_unbox_i32(uint64_t bits);
uint64_t bronze_box_i32(int32_t v);
int32_t bronze_unbox_bool(uint64_t bits);
uint64_t bronze_box_bool(int32_t v);
uint64_t bronze_get_new_target();

extern "C" {
uint64_t bronze_template_object(uint64_t cookedBits, uint64_t rawBits, uint64_t* cell);
extern uint64_t __bronze_template_cells[1024];
}

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit);
void unregister_all_runtime_symbols();
void register_all_module_external_symbols(Module* mod, const std::string& entry_symbol);
codegen::JitExecutionEngine* get_active_jit();
void set_active_jit(codegen::JitExecutionEngine* jit);

extern uint32_t g_bronze_main_key_constants;
extern int64_t g_bronze_module_env;
extern uint32_t g_bronze_dummy_key_map[4096];

} // namespace il
} // namespace brass

