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
int64_t bronze_create_func(const char* fn_name, int32_t param_count, int64_t env_box);
int64_t bronze_create_array(int32_t size);

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

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit);

} // namespace il
} // namespace brass
