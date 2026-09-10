#pragma once

#include <cstdint>
#include <string>

namespace brass {
class HostGC;

namespace codegen {
class JitExecutionEngine;
}

namespace il {

constexpr uint64_t kUndefinedTag = 0xFFFC000000000000ULL;
constexpr uint64_t kNullTag      = 0xFFFA000000000000ULL;
constexpr uint64_t kBoolTag      = 0xFFFB000000000000ULL;
constexpr uint64_t kInt32Tag     = 0xFFF9000000000000ULL;
constexpr uint64_t kPrintTag     = 0xFFFE000000000001ULL;
constexpr uint64_t kPrintErrTag  = 0xFFFE000000000002ULL;

std::string format_js_number(double v);

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
int64_t bronze_create_func(void* code_ptr, int32_t param_count, int64_t env_box);
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
void bronze_register_value_cells(uint64_t* cells, uint64_t count);

int64_t bronze_concat_begin(int64_t a, int64_t b, uint32_t remaining);
int64_t bronze_concat_append(int64_t a, int64_t b);
int64_t bronze_concat_end(int64_t a);
int64_t bronze_global_get_name(const char* name);
int64_t bronze_construct_0(int64_t callee_box);
int64_t bronze_construct_1(int64_t callee_box, int64_t arg0);
int64_t bronze_construct_2(int64_t callee_box, int64_t arg0, int64_t arg1);
int64_t bronze_construct_3(int64_t callee_box, int64_t arg0, int64_t arg1, int64_t arg2);
int64_t bronze_construct(int64_t callee_box, uint32_t argc, const int64_t* argv);
void bronze_class_extends(int64_t sub_box, int64_t super_box);
int64_t bronze_super_call(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv);
int64_t bronze_super_get(int64_t proto_box, uint32_t key_index, int64_t this_box);
int32_t bronze_instanceof(int64_t a_box, int64_t b_box);
int32_t bronze_has_property(int64_t key_box, int64_t obj_box);
int32_t bronze_is_nullish(int64_t val_box);
}

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit);
void unregister_all_runtime_symbols();
codegen::JitExecutionEngine* get_active_jit();

} // namespace il
} // namespace brass
