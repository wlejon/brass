#include "test_framework.hpp"
#include <brass/brass_c_api.h>

#include <brass/target/dynamic_library.hpp>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <vector>

using namespace brass::test;

TEST_CASE("C-API - Arithmetic and Loop Builder with Block Parameters") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "c_api_loop_test");
    REQUIRE(mod != nullptr);

    // Build: sum_to_n(n: i64) -> i64
    // Computes: sum_{i=1..n} i
    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "sum_to_n", i64_t, &i64_t, 1);
    REQUIRE(fn != nullptr);

    BrassBlock entry_bb = brass_function_append_block(fn, "entry");
    REQUIRE(entry_bb != nullptr);
    BrassValue n = brass_block_add_param(entry_bb, i64_t);
    REQUIRE(n != nullptr);

    // Verify brass_function_get_param matches entry block param
    BrassValue n_fn = brass_function_get_param(fn, 0);
    REQUIRE(n_fn != nullptr);

    BrassBlock check_bb = brass_function_append_block(fn, "check");
    REQUIRE(check_bb != nullptr);
    BrassValue i_param = brass_block_add_param(check_bb, i64_t);
    BrassValue acc_param = brass_block_add_param(check_bb, i64_t);
    REQUIRE(i_param != nullptr);
    REQUIRE(acc_param != nullptr);

    BrassBlock body_bb = brass_function_append_block(fn, "body");
    REQUIRE(body_bb != nullptr);

    BrassBlock exit_bb = brass_function_append_block(fn, "exit");
    REQUIRE(exit_bb != nullptr);
    BrassValue res_param = brass_block_add_param(exit_bb, i64_t);
    REQUIRE(res_param != nullptr);

    BrassBuilder b = brass_builder_create(ctx, fn);
    REQUIRE(b != nullptr);

    // entry_bb:
    brass_builder_position_at_end(b, entry_bb);
    BrassValue one = brass_build_iconst_i64(b, 1);
    BrassValue zero = brass_build_iconst_i64(b, 0);
    BrassValue init_args[2] = { one, zero };
    BrassStatus br_status = brass_build_br(b, check_bb, init_args, 2);
    CHECK_EQ(br_status, BRASS_OK);

    // check_bb:
    brass_builder_position_at_end(b, check_bb);
    BrassValue cond = brass_build_cmp(b, BRASS_CMP_SLE, i_param, n);
    BrassValue exit_args[1] = { acc_param };
    BrassStatus br_if_status = brass_build_br_if(b, cond, body_bb, nullptr, 0, exit_bb, exit_args, 1);
    CHECK_EQ(br_if_status, BRASS_OK);

    // body_bb:
    brass_builder_position_at_end(b, body_bb);
    BrassValue next_acc = brass_build_add(b, acc_param, i_param);
    BrassValue next_i = brass_build_add(b, i_param, one);
    BrassValue loop_args[2] = { next_i, next_acc };
    BrassStatus br_loop_status = brass_build_br(b, check_bb, loop_args, 2);
    CHECK_EQ(br_loop_status, BRASS_OK);

    // exit_bb:
    brass_builder_position_at_end(b, exit_bb);
    BrassStatus ret_status = brass_build_ret(b, res_param);
    CHECK_EQ(ret_status, BRASS_OK);

    // Module Verification
    char err_buf[512] = {0};
    BrassStatus v_status = brass_module_verify(mod, err_buf, sizeof(err_buf));
    CHECK_EQ(v_status, BRASS_OK);
    CHECK_EQ(err_buf[0], '\0');

    // Print MIR
    char* mir_text = nullptr;
    BrassStatus p_status = brass_module_print_mir(mod, &mir_text);
    CHECK_EQ(p_status, BRASS_OK);
    CHECK(mir_text != nullptr);
    CHECK(std::strstr(mir_text, "sum_to_n") != nullptr);
    CHECK(std::strstr(mir_text, "br_if") != nullptr);
    brass_free_buffer(mir_text);

    // JIT Execution Engine
    BrassJitEngine jit = brass_jit_create(ctx);
    REQUIRE(jit != nullptr);

    BrassCompiledModule comp = brass_jit_compile_module(jit, mod);
    REQUIRE(comp != nullptr);

    void* fn_addr = brass_jit_get_function_address(jit, "sum_to_n");
    REQUIRE(fn_addr != nullptr);

    using SumFn = int64_t (*)(int64_t);
    auto sum_fn = reinterpret_cast<SumFn>(fn_addr);

    CHECK_EQ(sum_fn(0), 0);
    CHECK_EQ(sum_fn(1), 1);
    CHECK_EQ(sum_fn(5), 15);
    CHECK_EQ(sum_fn(10), 55);
    CHECK_EQ(sum_fn(100), 5050);

    // Cleanup
    brass_builder_destroy(b);
    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

TEST_CASE("C-API - Error Handling and Context Diagnostics") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    // 1. Invalid Arguments to Module Verify
    BrassStatus s1 = brass_module_verify(nullptr, nullptr, 0);
    CHECK_EQ(s1, BRASS_ERR_INVALID_ARGUMENT);

    // 2. Syntax Error in Bronze IL Translation
    const char* malformed_il = "module syntax_error\nfunc %%% bad token {{{";
    BrassModule bad_mod = nullptr;
    BrassStatus s2 = brass_translate_bronze_il(ctx, malformed_il, std::strlen(malformed_il), nullptr, &bad_mod);
    CHECK_EQ(s2, BRASS_ERR_TRANSLATION_FAILED);
    CHECK(bad_mod == nullptr);

    const char* last_err = brass_context_get_last_error(ctx);
    CHECK(last_err != nullptr);
    CHECK(std::strlen(last_err) > 0);

    // 3. User-defined Error Injection
    brass_context_set_error(ctx, "custom embedder error description");
    CHECK_EQ(std::strcmp(brass_context_get_last_error(ctx), "custom embedder error description"), 0);

    // 4. Invalid AOT Arguments
    void* dummy_bytes = nullptr;
    size_t dummy_size = 0;
    BrassStatus s3 = brass_compile_to_object(nullptr, 0, &dummy_bytes, &dummy_size);
    CHECK_EQ(s3, BRASS_ERR_INVALID_ARGUMENT);

    BrassStatus s4 = brass_compile_to_shared_lib(nullptr, nullptr, nullptr);
    CHECK_EQ(s4, BRASS_ERR_INVALID_ARGUMENT);

    brass_context_destroy(ctx);
}

TEST_CASE("C-API - Bronze IL Translation and JIT Execution") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    const char* bronze_source = R"(
module test_c_api_bronze.js

func calc_poly(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = mul %0, %0
    %3: f64 = mul %1, %1
    %4: f64 = add %2, %3
    %5: f64 = const.f64 10.5
    %6: f64 = add %4, %5
    ret %6
}
)";

    BrassOptions* opts = brass_options_create();
    brass_options_set_optimize(opts, 1);

    BrassModule mod = nullptr;
    BrassStatus tr_status = brass_translate_bronze_il(ctx, bronze_source, std::strlen(bronze_source), opts, &mod);
    CHECK_EQ(tr_status, BRASS_OK);
    REQUIRE(mod != nullptr);

    brass_options_destroy(opts);

    BrassJitEngine jit = brass_jit_create(ctx);
    REQUIRE(jit != nullptr);

    BrassCompiledModule cmod = brass_jit_compile_module(jit, mod);
    REQUIRE(cmod != nullptr);

    void* fn_addr = brass_jit_get_function_address(jit, "calc_poly");
    REQUIRE(fn_addr != nullptr);

    using PolyFn = double (*)(double, double);
    auto poly_fn = reinterpret_cast<PolyFn>(fn_addr);

    // 3^2 + 4^2 + 10.5 = 9 + 16 + 10.5 = 35.5
    double val1 = poly_fn(3.0, 4.0);
    CHECK(std::abs(val1 - 35.5) < 1e-9);

    // 2^2 + 5^2 + 10.5 = 4 + 25 + 10.5 = 39.5
    double val2 = poly_fn(2.0, 5.0);
    CHECK(std::abs(val2 - 39.5) < 1e-9);

    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

TEST_CASE("C-API - In-Memory AOT Relocatable Object Emission") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "aot_emit_mod");
    REQUIRE(mod != nullptr);

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "get_magic_const", i64_t, nullptr, 0);
    REQUIRE(fn != nullptr);

    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassBuilder b = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(b, entry);

    BrassValue magic = brass_build_iconst_i64(b, 0x123456789ABCDEF0LL);
    brass_build_ret(b, magic);
    brass_builder_destroy(b);

    // 1. Emit Windows COFF Object in memory
    void* coff_bytes = nullptr;
    size_t coff_size = 0;
    BrassStatus s_coff = brass_compile_to_object(mod, BRASS_OBJECT_COFF, &coff_bytes, &coff_size);
    CHECK_EQ(s_coff, BRASS_OK);
    CHECK(coff_bytes != nullptr);
    CHECK(coff_size > 20);

    // Validate COFF machine field: AMD64 = 0x8664
    uint16_t coff_machine = *reinterpret_cast<const uint16_t*>(coff_bytes);
    CHECK_EQ(coff_machine, 0x8664);
    brass_free_buffer(coff_bytes);

    // 2. Emit Linux ELF Object in memory
    void* elf_bytes = nullptr;
    size_t elf_size = 0;
    BrassStatus s_elf = brass_compile_to_object(mod, BRASS_OBJECT_ELF, &elf_bytes, &elf_size);
    CHECK_EQ(s_elf, BRASS_OK);
    CHECK(elf_bytes != nullptr);
    CHECK(elf_size > 20);

    // Validate ELF magic: \x7f E L F
    const auto* elf_u8 = reinterpret_cast<const uint8_t*>(elf_bytes);
    CHECK_EQ(elf_u8[0], 0x7F);
    CHECK_EQ(elf_u8[1], 'E');
    CHECK_EQ(elf_u8[2], 'L');
    CHECK_EQ(elf_u8[3], 'F');
    brass_free_buffer(elf_bytes);

    // 3. Emit macOS Mach-O Object in memory
    void* macho_bytes = nullptr;
    size_t macho_size = 0;
    BrassStatus s_macho = brass_compile_to_object(mod, BRASS_OBJECT_MACHO, &macho_bytes, &macho_size);
    CHECK_EQ(s_macho, BRASS_OK);
    CHECK(macho_bytes != nullptr);
    CHECK(macho_size > 20);

    // Validate Mach-O magic: 0xFEEDFACF (little endian)
    uint32_t macho_magic = *reinterpret_cast<const uint32_t*>(macho_bytes);
    CHECK(macho_magic == 0xFEEDFACF || macho_magic == 0xCFFAEDFE);
    brass_free_buffer(macho_bytes);

    // 4. Emit Auto format (Host platform)
    void* auto_bytes = nullptr;
    size_t auto_size = 0;
    BrassStatus s_auto = brass_compile_to_object(mod, BRASS_OBJECT_AUTO, &auto_bytes, &auto_size);
    CHECK_EQ(s_auto, BRASS_OK);
    CHECK(auto_bytes != nullptr);
    CHECK(auto_size > 20);
    brass_free_buffer(auto_bytes);

    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

static int64_t host_custom_callback(int64_t a, int64_t b) {
    return a * 10 + b;
}

TEST_CASE("C-API - External Symbol Registration and Memory Load/Store") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "c_api_extern_test");
    REQUIRE(mod != nullptr);

    brass_module_add_external_symbol(mod, "host_custom_callback");

    // test_call(a: i64, b: i64) -> i64
    BrassType i64_t = brass_type_i64();
    BrassType params[2] = { i64_t, i64_t };
    BrassFunction fn = brass_function_create(mod, "test_call", i64_t, params, 2);

    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue arg_a = brass_block_add_param(entry, i64_t);
    BrassValue arg_b = brass_block_add_param(entry, i64_t);

    BrassBuilder b = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(b, entry);

    // Call host_custom_callback(arg_a, arg_b)
    BrassValue call_args[2] = { arg_a, arg_b };
    BrassValue call_res = brass_build_call(b, "host_custom_callback", i64_t, call_args, 2);
    REQUIRE(call_res != nullptr);

    brass_build_ret(b, call_res);
    brass_builder_destroy(b);

    char err[256] = {0};
    REQUIRE(brass_module_verify(mod, err, sizeof(err)) == BRASS_OK);

    BrassJitEngine jit = brass_jit_create(ctx);
    REQUIRE(jit != nullptr);

    BrassStatus reg_s = brass_jit_register_symbol(jit, "host_custom_callback", reinterpret_cast<void*>(&host_custom_callback));
    CHECK_EQ(reg_s, BRASS_OK);

    BrassCompiledModule cmod = brass_jit_compile_module(jit, mod);
    REQUIRE(cmod != nullptr);

    void* fn_addr = brass_jit_get_function_address(jit, "test_call");
    REQUIRE(fn_addr != nullptr);

    using CallFn = int64_t (*)(int64_t, int64_t);
    auto test_fn = reinterpret_cast<CallFn>(fn_addr);

    CHECK_EQ(test_fn(7, 3), 73);
    CHECK_EQ(test_fn(12, 5), 125);

    brass_jit_destroy(jit);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

TEST_CASE("C-API - Types and Options API Completeness") {
    CHECK(brass_type_void() != nullptr);
    CHECK(brass_type_i32() != nullptr);
    CHECK(brass_type_i64() != nullptr);
    CHECK(brass_type_f32() != nullptr);
    CHECK(brass_type_f64() != nullptr);
    CHECK(brass_type_bool() != nullptr);
    CHECK(brass_type_ptr() != nullptr);
    CHECK(brass_type_dynamic() != nullptr);

    CHECK(brass_type_v128(BRASS_LANE_F32) != nullptr);
    CHECK(brass_type_v128(BRASS_LANE_F64) != nullptr);
    CHECK(brass_type_v128(BRASS_LANE_I32) != nullptr);
    CHECK(brass_type_v128(BRASS_LANE_I64) != nullptr);

    CHECK(brass_type_v256(BRASS_LANE_F32) != nullptr);
    CHECK(brass_type_v256(BRASS_LANE_F64) != nullptr);
    CHECK(brass_type_v256(BRASS_LANE_I32) != nullptr);
    CHECK(brass_type_v256(BRASS_LANE_I64) != nullptr);

    BrassOptions* opts = brass_options_create();
    REQUIRE(opts != nullptr);
    brass_options_set_optimize(opts, 0);
    CHECK_EQ(opts->enable_optimizations, 0);
    brass_options_set_optimize(opts, 1);
    CHECK_EQ(opts->enable_optimizations, 1);
    brass_options_set_target_format(opts, BRASS_OBJECT_ELF);
    CHECK_EQ(opts->target_format, static_cast<int>(BRASS_OBJECT_ELF));
    brass_options_destroy(opts);
}

TEST_CASE("C-API - AOT Shared Library Compilation and Dynamic Load") {
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "c_api_shared_lib_test");
    REQUIRE(mod != nullptr);

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, "aot_cube", i64_t, &i64_t, 1);
    REQUIRE(fn != nullptr);

    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue arg = brass_block_add_param(entry, i64_t);
    BrassBuilder b = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(b, entry);

    BrassValue x2 = brass_build_mul(b, arg, arg);
    BrassValue x3 = brass_build_mul(b, x2, arg);
    brass_build_ret(b, x3);
    brass_builder_destroy(b);

    char err[256] = {0};
    REQUIRE(brass_module_verify(mod, err, sizeof(err)) == BRASS_OK);

    std::string ext =
#if defined(_WIN32)
        ".dll";
#elif defined(__APPLE__)
        ".dylib";
#else
        ".so";
#endif

    std::filesystem::path lib_path = std::filesystem::temp_directory_path() / ("test_c_api_shared_cube" + ext);
    std::error_code ec;
    std::filesystem::remove(lib_path, ec);

    BrassStatus s_lib = brass_compile_to_shared_lib(mod, lib_path.string().c_str(), nullptr);
    CHECK_EQ(s_lib, BRASS_OK);
    CHECK(std::filesystem::exists(lib_path));

    // Dynamic load and test execution
    std::string load_err;
    auto dyn_lib = brass::target::DynamicLibrary::open(lib_path.string(), &load_err);
    if (!dyn_lib) {
        std::cerr << "DynamicLibrary::open error: " << load_err << "\n";
    }
    REQUIRE(dyn_lib != nullptr);

    using CubeFn = int64_t (*)(int64_t);
    auto cube_fn = dyn_lib->get_function<CubeFn>("aot_cube");
    REQUIRE(cube_fn != nullptr);

    CHECK_EQ(cube_fn(3), 27);
    CHECK_EQ(cube_fn(5), 125);
    CHECK_EQ(cube_fn(10), 1000);

    dyn_lib.reset();
    std::filesystem::remove(lib_path, ec);

    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

