#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
#include "../../src/il_translator/il_runtime.hpp"
#include <brass/mir/printer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include "msvc_toolchain_helper.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace brass;
using namespace brass::il;
using namespace brass::test;

TEST_CASE("Bronze IL - Arithmetic, Math, and Bitwise Translation & JIT") {
    const char* il_source = R"(
module test_arithmetic.js

func mathOps(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = add %0, %1
    %3: f64 = sub %0, %1
    %4: f64 = mul %0, %1
    %5: f64 = div %0, %1
    %6: f64 = neg %0
    %7: f64 = add %2, %3
    %8: f64 = add %7, %4
    %9: f64 = add %8, %5
    %10: f64 = add %9, %6
    ret %10
}

func bitOps(%0: f64, %1: f64) -> f64 {
  b0:
    %2: i32 = to.int32 %0
    %3: i32 = to.int32 %1
    %4: f64 = and %2, %3
    %5: f64 = or %2, %3
    %6: f64 = xor %2, %3
    %7: f64 = shl %2, %3
    %8: f64 = shr %2, %3
    %9: f64 = ushr %2, %3
    %10: f64 = add %4, %5
    %11: f64 = add %10, %6
    %12: f64 = add %11, %7
    %13: f64 = add %12, %8
    %14: f64 = add %13, %9
    ret %14
}

func main() -> f64 {
  b0:
    %0: f64 = const.f64 10
    %1: f64 = const.f64 4
    %2: f64 = call @mathOps(%0, %1)
    %3: f64 = const.f64 42
    %4: f64 = const.f64 3
    %5: f64 = call @bitOps(%3, %4)
    %6: f64 = add %2, %5
    ret %6
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto math_fn = jit.get_function_ptr<double(*)(double, double)>("mathOps");
    REQUIRE(math_fn != nullptr);
    double math_res = math_fn(10.0, 4.0);
    CHECK(std::abs(math_res - 52.5) < 1e-9);

    auto bit_fn = jit.get_function_ptr<double(*)(double, double)>("bitOps");
    REQUIRE(bit_fn != nullptr);
    double bit_res = bit_fn(42.0, 3.0);
    CHECK(std::abs(bit_res - 432.0) < 1e-9);

    auto main_fn = jit.get_function_ptr<double(*)()>("main");
    REQUIRE(main_fn != nullptr);
    double total = main_fn();
    CHECK(std::abs(total - 484.5) < 1e-9);
}

TEST_CASE("Bronze IL - Loops, Conditionals, and Block Parameters Translation") {
    const char* il_source = R"(
module test_control_flow.js

func collatz(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 1
    %5: bool = cmp.gt %2, %4
    br %5, b2, b3(%2, %3)
  b2:
    %9: f64 = const.f64 2
    %11: f64 = mod %2, %9
    %12: f64 = const.f64 0
    %13: bool = cmp.eq %11, %12
    br %13, b4, b5
  b3(%7: f64, %8: f64):
    ret %8
  b4:
    %15: f64 = const.f64 2
    %17: f64 = div %2, %15
    jump b6(%17)
  b5:
    %18: f64 = const.f64 3
    %20: f64 = mul %18, %2
    %21: f64 = const.f64 1
    %23: f64 = add %20, %21
    jump b6(%23)
  b6(%24: f64):
    %25: f64 = const.f64 1
    %26: f64 = add %3, %25
    jump b1(%24, %26)
}

func fib(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 1
    %3: f64 = const.f64 0
    jump b1(%1, %2, %3)
  b1(%4: f64, %5: f64, %6: f64):
    %7: bool = cmp.lt %6, %0
    br %7, b2, b3(%4, %5, %6)
  b2:
    %11: f64 = add %4, %5
    %12: f64 = const.f64 1
    %13: f64 = add %6, %12
    jump b1(%5, %11, %13)
  b3(%8: f64, %9: f64, %10: f64):
    ret %8
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);
    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto collatz_fn = jit.get_function_ptr<double(*)(double)>("collatz");
    REQUIRE(collatz_fn != nullptr);
    CHECK_EQ(collatz_fn(27.0), 111.0);
    CHECK_EQ(collatz_fn(1.0), 0.0);
    CHECK_EQ(collatz_fn(6.0), 8.0);

    auto fib_fn = jit.get_function_ptr<double(*)(double)>("fib");
    REQUIRE(fib_fn != nullptr);
    CHECK_EQ(fib_fn(0.0), 0.0);
    CHECK_EQ(fib_fn(1.0), 1.0);
    CHECK_EQ(fib_fn(10.0), 55.0);
    CHECK_EQ(fib_fn(20.0), 6765.0);
}

TEST_CASE("Bronze IL - End-to-End AOT compile_to_object and Link") {
    const char* il_source = R"(
module test_aot_program.js

func compute_checksum(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = add %0, %1
    %3: f64 = mul %0, %1
    %4: f64 = add %2, %3
    ret %4
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

#if defined(_WIN32)
    char temp_path_buf[MAX_PATH];
    DWORD path_len = GetTempPathA(MAX_PATH, temp_path_buf);
    REQUIRE(path_len > 0);
    std::string base_dir = std::string(temp_path_buf) + "brass_bronze_aot_test";
    CreateDirectoryA(base_dir.c_str(), NULL);

    std::string obj_file = base_dir + "\\bronze_kernel.obj";
    std::string dll_file = base_dir + "\\bronze_kernel.dll";

    HostEngine engine;
    REQUIRE(engine.compile_to_object(*res.module, obj_file));

    std::string link_cmd = "link.exe /NOLOGO /DLL /OUT:\"" + dll_file + "\" \"" + obj_file + "\" /NOENTRY";
    int link_rc = std::system(link_cmd.c_str());
    if (link_rc == 0) {
        HMODULE hModule = LoadLibraryA(dll_file.c_str());
        REQUIRE(hModule != NULL);

        using ChecksumFn = double(*)(double, double);
        auto fn = reinterpret_cast<ChecksumFn>(GetProcAddress(hModule, "compute_checksum"));
        REQUIRE(fn != nullptr);

        double result = fn(5.0, 7.0);
        CHECK_EQ(result, 47.0);

        FreeLibrary(hModule);
    }
#endif
}

#if defined(_WIN32)
static HMODULE g_aot_active_dll = nullptr;

static void* aot_symbol_resolver(const char* name) {
    if (!name || !g_aot_active_dll) return nullptr;
    return reinterpret_cast<void*>(GetProcAddress(g_aot_active_dll, name));
}
#endif

TEST_CASE("Bronze IL - 22-Program Live Corpus JIT and AOT Execution") {
    std::vector<std::string> corpus_files = {
        "01_arithmetic",
        "02_bitwise",
        "03_collatz",
        "04_fib_iter",
        "05_fib_rec",
        "06_ackermann",
        "07_prime_count",
        "08_newton_sqrt",
        "09_closures",
        "10_loop_capture",
        "11_matrix_recurrence",
        "12_counter_closure",
        "13_nested_curry",
        "14_array_loop",
        "15_nested_acc",
        "16_param_bounds",
        "17_large_int_overflow",
        "18_gcd_iter",
        "19_vec3_acc",
        "20_mat4_mul",
        "21_quat_norm",
        "22_bbox_expand"
    };

    auto normalize = [](std::string s) {
        std::string out;
        bool in_space = false;
        for (char c : s) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!in_space && !out.empty()) {
                    out += ' ';
                    in_space = true;
                }
            } else {
                out += c;
                in_space = false;
            }
        }
        if (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    };

    struct BronzeRuntimeTable {
        void (*print_f64)(double);
        void (*print_i32)(int32_t);
        void (*print_dynamic)(int64_t);
        void (*print_newline)();
        double (*f64_mod)(double, double);
        int64_t (*name_resolve)(const char*);
        int64_t (*env_create)(int64_t, int32_t);
        int64_t (*env_get)(int64_t, int32_t, int32_t);
        void (*env_set)(int64_t, int32_t, int32_t, int64_t);
        int64_t (*create_func)(void*, int32_t, int64_t);
        int64_t (*create_array)(int32_t);
        int64_t (*create_object)();
        int64_t (*prop_get)(int64_t, int32_t, uint64_t*);
        void (*prop_set)(int64_t, int32_t, int64_t, uint64_t*, int32_t);
        int64_t (*elem_get)(int64_t, int64_t);
        void (*elem_set)(int64_t, int64_t, int64_t, int32_t);
        int64_t (*call_dynamic_0)(int64_t, int64_t);
        int64_t (*call_dynamic_1)(int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_2)(int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_3)(int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_4)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_5)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_6)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_7)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_8)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        int64_t (*call_dynamic_n)(int64_t, int64_t, int32_t, const int64_t*);
        void (*method_def)(int64_t, int32_t, int64_t);
        int64_t (*ic_get)(uint32_t, int64_t, const char*, int32_t);
        void (*ic_set)(uint32_t, int64_t, const char*, int32_t, int64_t);
        uint64_t (*brass_ic_get)(uint32_t, uint64_t, const char*, uint32_t);
        void (*brass_ic_set)(uint32_t, uint64_t, const char*, uint32_t, uint64_t);
        uint64_t (*brass_dyn_get_str)(uint64_t, const char*);
        void (*brass_dyn_set_str)(uint64_t, const char*, uint64_t);
    };

    BronzeRuntimeTable host_rt = {
        &brass::il::bronze_print_f64,
        &brass::il::bronze_print_i32,
        &brass::il::bronze_print_dynamic,
        &brass::il::bronze_print_newline,
        &brass::il::bronze_f64_mod,
        &brass::il::bronze_name_resolve,
        &brass::il::bronze_env_create,
        &brass::il::bronze_env_get,
        &brass::il::bronze_env_set,
        &brass::il::bronze_create_func,
        &brass::il::bronze_create_array,
        &brass::il::bronze_create_object,
        &brass::il::bronze_prop_get,
        &brass::il::bronze_prop_set,
        &brass::il::bronze_elem_get,
        &brass::il::bronze_elem_set,
        &brass::il::bronze_call_dynamic_0,
        &brass::il::bronze_call_dynamic_1,
        &brass::il::bronze_call_dynamic_2,
        &brass::il::bronze_call_dynamic_3,
        &brass::il::bronze_call_dynamic_4,
        &brass::il::bronze_call_dynamic_5,
        &brass::il::bronze_call_dynamic_6,
        &brass::il::bronze_call_dynamic_7,
        &brass::il::bronze_call_dynamic_8,
        &brass::il::bronze_call_dynamic_n,
        &brass::il::bronze_method_def,
        &brass::il::bronze_ic_get,
        &brass::il::bronze_ic_set,
        &::brass_ic_get_prop,
        &::brass_ic_set_prop,
        &brass::runtime::brass_dynamic_object_get_prop_str,
        &brass::runtime::brass_dynamic_object_set_prop_str
    };
    (void)host_rt;

    bool msvc_ready = MsvcToolchain::is_available();
    std::filesystem::path aot_dir;
    std::filesystem::path wrapper_cpp;
    if (msvc_ready) {
        aot_dir = MsvcToolchain::temp_dir() / "bronze_corpus_aot";
        std::error_code ec;
        std::filesystem::remove_all(aot_dir, ec);
        std::filesystem::create_directories(aot_dir, ec);
        wrapper_cpp = aot_dir / "bronze_runtime_bridge.cpp";
        std::ofstream ofs(wrapper_cpp);
        ofs << "#define NOMINMAX\n"
            << "#define WIN32_LEAN_AND_MEAN\n"
            << "#include <windows.h>\n"
            << "#include <cstdint>\n"
            << "#include <cstddef>\n"
            << "#include <cstring>\n"
            << "#include <iostream>\n\n"
            << "struct BronzeRuntimeTable {\n"
            << "    void (*print_f64)(double);\n"
            << "    void (*print_i32)(int32_t);\n"
            << "    void (*print_dynamic)(int64_t);\n"
            << "    void (*print_newline)();\n"
            << "    double (*f64_mod)(double, double);\n"
            << "    int64_t (*name_resolve)(const char*);\n"
            << "    int64_t (*env_create)(int64_t, int32_t);\n"
            << "    int64_t (*env_get)(int64_t, int32_t, int32_t);\n"
            << "    void (*env_set)(int64_t, int32_t, int32_t, int64_t);\n"
            << "    int64_t (*create_func)(void*, int32_t, int64_t);\n"
            << "    int64_t (*create_array)(int32_t);\n"
            << "    int64_t (*create_object)();\n"
            << "    int64_t (*prop_get)(int64_t, int32_t, uint64_t*);\n"
            << "    void (*prop_set)(int64_t, int32_t, int64_t, uint64_t*, int32_t);\n"
            << "    int64_t (*elem_get)(int64_t, int64_t);\n"
            << "    void (*elem_set)(int64_t, int64_t, int64_t, int32_t);\n"
            << "    int64_t (*call_dynamic_0)(int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_1)(int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_2)(int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_3)(int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_4)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_5)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_6)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_7)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_8)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
            << "    int64_t (*call_dynamic_n)(int64_t, int64_t, int32_t, const int64_t*);\n"
            << "    void (*method_def)(int64_t, const char*, int32_t, int64_t);\n"
            << "    int64_t (*ic_get)(uint32_t, int64_t, const char*, int32_t);\n"
            << "    void (*ic_set)(uint32_t, int64_t, const char*, int32_t, int64_t);\n"
            << "    uint64_t (*brass_ic_get)(uint32_t, uint64_t, const char*, uint32_t);\n"
            << "    void (*brass_ic_set)(uint32_t, uint64_t, const char*, uint32_t, uint64_t);\n"
            << "    uint64_t (*brass_dyn_get_str)(uint64_t, const char*);\n"
            << "    void (*brass_dyn_set_str)(uint64_t, const char*, uint64_t);\n"
            << "};\n\n"
            << "static BronzeRuntimeTable g_rt = {};\n\n"
            << "extern \"C\" {\n"
            << "    __declspec(dllexport) void init_bronze_runtime(const BronzeRuntimeTable* table) {\n"
            << "        if (table) g_rt = *table;\n"
            << "    }\n"
            << "    void bronze_print_f64(double v) { if (g_rt.print_f64) g_rt.print_f64(v); }\n"
            << "    void bronze_print_i32(int32_t v) { if (g_rt.print_i32) g_rt.print_i32(v); }\n"
            << "    void bronze_print_dynamic(int64_t v) { if (g_rt.print_dynamic) g_rt.print_dynamic(v); }\n"
            << "    void bronze_print_space() { std::cout << \" \"; }\n"
            << "    void bronze_print_newline() { if (g_rt.print_newline) g_rt.print_newline(); }\n"
            << "    double bronze_f64_mod(double a, double b) { return g_rt.f64_mod ? g_rt.f64_mod(a, b) : 0.0; }\n"
            << "    int64_t bronze_name_resolve(const char* name) { return g_rt.name_resolve ? g_rt.name_resolve(name) : 0; }\n"
            << "    int64_t bronze_env_create(int64_t parent, int32_t sz) { return g_rt.env_create ? g_rt.env_create(parent, sz) : 0; }\n"
            << "    int64_t bronze_env_get(int64_t env, int32_t d, int32_t idx) { return g_rt.env_get ? g_rt.env_get(env, d, idx) : 0; }\n"
            << "    void bronze_env_set(int64_t env, int32_t d, int32_t idx, int64_t v) { if (g_rt.env_set) g_rt.env_set(env, d, idx, v); }\n"
            << "    int64_t bronze_create_func(void* fn_name, int32_t pc, int64_t env) { return g_rt.create_func ? g_rt.create_func(fn_name, pc, env) : 0; }\n"
            << "    int64_t bronze_create_array(int32_t sz) { return g_rt.create_array ? g_rt.create_array(sz) : 0; }\n"
            << "    int64_t bronze_create_object() { return g_rt.create_object ? g_rt.create_object() : 0; }\n"
            << "    int64_t bronze_prop_get(int64_t o, int32_t k, uint64_t* ic = nullptr) { return g_rt.prop_get ? g_rt.prop_get(o, k, ic) : 0; }\n"
            << "    void bronze_prop_set(int64_t o, int32_t k, int64_t v, uint64_t* ic = nullptr, int32_t st = 1) { if (g_rt.prop_set) g_rt.prop_set(o, k, v, ic, st); }\n"
            << "    int64_t bronze_elem_get(int64_t a, int64_t i) { return g_rt.elem_get ? g_rt.elem_get(a, i) : 0; }\n"
            << "    void bronze_elem_set(int64_t a, int64_t i, int64_t v, int32_t s) { if (g_rt.elem_set) g_rt.elem_set(a, i, v, s); }\n"
            << "    int64_t bronze_call_dynamic_0(int64_t c, int64_t th) { return g_rt.call_dynamic_0 ? g_rt.call_dynamic_0(c, th) : 0; }\n"
            << "    int64_t bronze_call_dynamic_1(int64_t c, int64_t th, int64_t a0) { return g_rt.call_dynamic_1 ? g_rt.call_dynamic_1(c, th, a0) : 0; }\n"
            << "    int64_t bronze_call_dynamic_2(int64_t c, int64_t th, int64_t a0, int64_t a1) { return g_rt.call_dynamic_2 ? g_rt.call_dynamic_2(c, th, a0, a1) : 0; }\n"
            << "    int64_t bronze_call_dynamic_3(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2) { return g_rt.call_dynamic_3 ? g_rt.call_dynamic_3(c, th, a0, a1, a2) : 0; }\n"
            << "    int64_t bronze_call_dynamic_4(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2, int64_t a3) { return g_rt.call_dynamic_4 ? g_rt.call_dynamic_4(c, th, a0, a1, a2, a3) : 0; }\n"
            << "    int64_t bronze_call_dynamic_5(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4) { return g_rt.call_dynamic_5 ? g_rt.call_dynamic_5(c, th, a0, a1, a2, a3, a4) : 0; }\n"
            << "    int64_t bronze_call_dynamic_6(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5) { return g_rt.call_dynamic_6 ? g_rt.call_dynamic_6(c, th, a0, a1, a2, a3, a4, a5) : 0; }\n"
            << "    int64_t bronze_call_dynamic_7(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6) { return g_rt.call_dynamic_7 ? g_rt.call_dynamic_7(c, th, a0, a1, a2, a3, a4, a5, a6) : 0; }\n"
            << "    int64_t bronze_call_dynamic_8(int64_t c, int64_t th, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7) { return g_rt.call_dynamic_8 ? g_rt.call_dynamic_8(c, th, a0, a1, a2, a3, a4, a5, a6, a7) : 0; }\n"
            << "    int64_t bronze_call_dynamic_n(int64_t c, int64_t th, int32_t ac, const int64_t* av) { return g_rt.call_dynamic_n ? g_rt.call_dynamic_n(c, th, ac, av) : 0; }\n"
            << "    void bronze_method_def(int64_t o, const char* n, int32_t s, int64_t c) { if (g_rt.method_def) g_rt.method_def(o, n, s, c); }\n"
            << "    int64_t bronze_ic_get(uint32_t sid, int64_t o, const char* n, int32_t s) { return g_rt.ic_get ? g_rt.ic_get(sid, o, n, s) : 0; }\n"
            << "    void bronze_ic_set(uint32_t sid, int64_t o, const char* n, int32_t s, int64_t v) { if (g_rt.ic_set) g_rt.ic_set(sid, o, n, s, v); }\n"
            << "    uint64_t brass_ic_get_prop(uint32_t sid, uint64_t o, const char* n, uint32_t s) { return g_rt.brass_ic_get ? g_rt.brass_ic_get(sid, o, n, s) : 0; }\n"
            << "    void brass_ic_set_prop(uint32_t sid, uint64_t o, const char* n, uint32_t s, uint64_t v) { if (g_rt.brass_ic_set) g_rt.brass_ic_set(sid, o, n, s, v); }\n"
            << "    uint64_t brass_dynamic_object_get_prop_str(uint64_t o, const char* n) { return g_rt.brass_dyn_get_str ? g_rt.brass_dyn_get_str(o, n) : 0; }\n"
            << "    void brass_dynamic_object_set_prop_str(uint64_t o, const char* n, uint64_t v) { if (g_rt.brass_dyn_set_str) g_rt.brass_dyn_set_str(o, n, v); }\n"
            << "    uint64_t bronze_create_async_machine(void* f, uint32_t s, uint64_t m, uint64_t e) { (void)f; (void)s; (void)m; (void)e; return 0; }\n"
            << "    uint64_t bronze_async_start(uint64_t f, uint64_t a) { (void)f; (void)a; return 0; }\n"
            << "    uint64_t bronze_async_await(uint64_t f, uint64_t v) { (void)f; return v; }\n"
            << "    uint64_t bronze_iter_open(uint64_t g) { return g; }\n"
            << "    uint64_t bronze_iter_step(uint64_t i) { (void)i; return 0; }\n"
            << "    uintptr_t brass_coro_create(void* f, uint32_t s, uint64_t m) { (void)f; (void)s; (void)m; return 0; }\n"
            << "    uint64_t brass_coro_resume(uintptr_t f, uint64_t v) { (void)f; (void)v; return 0; }\n"
            << "    uint32_t brass_coro_is_done(uintptr_t f) { (void)f; return 1; }\n"
            << "    void brass_coro_destroy(uintptr_t f) { (void)f; }\n"
            << "    void brass_gc_write_barrier(uintptr_t o, uintptr_t v) { (void)o; (void)v; }\n"
            << "    uint64_t __bronze_module_env = 0;\n"
            << "    void* __bronze_template_cells = 0;\n"
            << "    void* bronze_main_key_constants = 0;\n"
            << "    uint32_t __bronze_key_map[1024] = {0};\n"
            << "    void bronze_register_key_manifest(const void* p, uint32_t* m) {\n"
            << "        (void)p;\n"
            << "        uint32_t* map = m ? m : __bronze_key_map;\n"
            << "        for (uint32_t i = 0; i < 1024; ++i) map[i] = i;\n"
            << "    }\n"
            << "    struct BronzeGcFrame {\n"
            << "        BronzeGcFrame* prev;\n"
            << "        uint64_t count;\n"
            << "        uint64_t slots[1];\n"
            << "    };\n"
            << "    static thread_local uint64_t g_shadow_stack[131072];\n"
            << "    static thread_local size_t g_shadow_top = 0;\n"
            << "    static thread_local BronzeGcFrame* g_frame_top = nullptr;\n"
            << "    void* bronze_gc_frame_push(uint32_t count) {\n"
            << "        size_t cur = g_shadow_top;\n"
            << "        g_shadow_top += 2 + count;\n"
            << "        BronzeGcFrame* frame = (BronzeGcFrame*)&g_shadow_stack[cur];\n"
            << "        frame->prev = g_frame_top;\n"
            << "        frame->count = count;\n"
            << "        for (uint32_t i = 0; i < count; ++i) frame->slots[i] = 0xFFF6000000000000ULL;\n"
            << "        g_frame_top = frame;\n"
            << "        return frame;\n"
            << "    }\n"
            << "    void bronze_gc_frame_pop() {\n"
            << "        if (g_frame_top) {\n"
            << "            g_shadow_top = (uint64_t*)g_frame_top - g_shadow_stack;\n"
            << "            g_frame_top = g_frame_top->prev;\n"
            << "        }\n"
            << "    }\n"
            << "    double bronze_unbox_f64(int64_t v) { double d; memcpy(&d, &v, sizeof(d)); return d; }\n"
            << "    int64_t bronze_box_f64(double v) { int64_t i; memcpy(&i, &v, sizeof(i)); return i; }\n"
            << "    int32_t bronze_to_int32_f64(double d) { return (int32_t)(uint32_t)(int64_t)d; }\n"
            << "    int32_t bronze_to_int32(uint64_t b) { return (int32_t)(uint32_t)b; }\n"
            << "    uint64_t bronze_to_string(uint64_t b) { return b; }\n"
            << "    uint64_t bronze_to_numeric(uint64_t b) { return b; }\n"
            << "    void bronze_register_value_cells(uint64_t* c, uint64_t n) { (void)c; (void)n; }\n"
            << "    int64_t bronze_concat_begin(int64_t a, int64_t b, uint32_t r) { (void)b; (void)r; return a; }\n"
            << "    int64_t bronze_concat_append(int64_t a, int64_t b) { (void)b; return a; }\n"
            << "    int64_t bronze_concat_end(int64_t a) { return a; }\n"
            << "    int64_t bronze_global_get_name(const char* n) { return g_rt.name_resolve ? g_rt.name_resolve(n) : 0; }\n"
            << "    int64_t bronze_construct_0(int64_t c) { return c; }\n"
            << "    int64_t bronze_construct_1(int64_t c, int64_t a0) { (void)a0; return c; }\n"
            << "    int64_t bronze_construct_2(int64_t c, int64_t a0, int64_t a1) { (void)a0; (void)a1; return c; }\n"
            << "    int64_t bronze_construct_3(int64_t c, int64_t a0, int64_t a1, int64_t a2) { (void)a0; (void)a1; (void)a2; return c; }\n"
            << "    int64_t bronze_construct(int64_t c, uint32_t ac, const int64_t* av) { (void)ac; (void)av; return c; }\n"
            << "    void bronze_class_extends(int64_t sub, int64_t sup) { (void)sub; (void)sup; }\n"
            << "    int64_t bronze_super_call(int64_t sub, int64_t th, uint32_t ac, const int64_t* av) { (void)sub; (void)th; (void)ac; (void)av; return 0; }\n"
            << "    int64_t bronze_super_get(int64_t p, uint32_t k, int64_t th) { (void)p; (void)k; (void)th; return 0; }\n"
            << "    int32_t bronze_instanceof(int64_t a, int64_t b) { (void)a; (void)b; return 0; }\n"
            << "    int32_t bronze_has_property(int64_t k, int64_t o) { (void)k; (void)o; return 0; }\n"
            << "    int32_t bronze_is_nullish(int64_t v) { (void)v; return 0; }\n"
            << "    uint64_t bronze_create_function(void* code, uint32_t arity, uint32_t len, uint32_t name, uint32_t flags, uint64_t env) {\n"
            << "        return (uint64_t)bronze_create_func(code, (int32_t)arity, (int64_t)env);\n"
            << "    }\n"
            << "    int64_t bronze_function_singleton(void* code, uint32_t arity, uint32_t len, uint32_t name, uint32_t flags, uint64_t* slot) {\n"
            << "        return bronze_create_func(code, (int32_t)arity, (int64_t)0xFFF1000000000000ULL);\n"
            << "    }\n"
            << "    double bronze_to_dbl(int64_t v) {\n"
            << "        uint64_t u = (uint64_t)v;\n"
            << "        if (u <= 0xFFF0000000000000ULL) {\n"
            << "            double d; memcpy(&d, &v, 8); return d;\n"
            << "        } else if ((u >> 48) == 0xFFF9 || (u >> 48) == 0xFFF3) {\n"
            << "            return (double)(int32_t)(u & 0xFFFFFFFFULL);\n"
            << "        }\n"
            << "        return 0.0;\n"
            << "    }\n"
            << "    int32_t bronze_rel_lt(int64_t a, int64_t b) { return bronze_to_dbl(a) < bronze_to_dbl(b) ? 1 : 0; }\n"
            << "    int32_t bronze_rel_gt(int64_t a, int64_t b) { return bronze_to_dbl(a) > bronze_to_dbl(b) ? 1 : 0; }\n"
            << "    int32_t bronze_rel_le(int64_t a, int64_t b) { return bronze_to_dbl(a) <= bronze_to_dbl(b) ? 1 : 0; }\n"
            << "    int32_t bronze_rel_ge(int64_t a, int64_t b) { return bronze_to_dbl(a) >= bronze_to_dbl(b) ? 1 : 0; }\n"
            << "    int64_t bronze_dynamic_add(int64_t a, int64_t b) {\n"
            << "        double res = bronze_to_dbl(a) + bronze_to_dbl(b);\n"
            << "        int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_sub(int64_t a, int64_t b) {\n"
            << "        double res = bronze_to_dbl(a) - bronze_to_dbl(b);\n"
            << "        int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_mul(int64_t a, int64_t b) {\n"
            << "        double res = bronze_to_dbl(a) * bronze_to_dbl(b);\n"
            << "        int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_div(int64_t a, int64_t b) {\n"
            << "        double res = bronze_to_dbl(a) / bronze_to_dbl(b);\n"
            << "        int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_mod(int64_t a, int64_t b) {\n"
            << "        double res = bronze_f64_mod(bronze_to_dbl(a), bronze_to_dbl(b));\n"
            << "        int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_neg(int64_t a) {\n"
            << "        if (((uint64_t)a >> 48) <= 0xFFF8) {\n"
            << "            double da; memcpy(&da, &a, 8); double res = -da; int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "        }\n"
            << "        return (int64_t)0xFFF1000000000000ULL;\n"
            << "    }\n"
            << "    int64_t bronze_dynamic_bitand(int64_t a, int64_t b) { return (int64_t)((uint64_t)(bronze_to_int32(a) & bronze_to_int32(b)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_bitor(int64_t a, int64_t b) { return (int64_t)((uint64_t)(bronze_to_int32(a) | bronze_to_int32(b)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_bitxor(int64_t a, int64_t b) { return (int64_t)((uint64_t)(bronze_to_int32(a) ^ bronze_to_int32(b)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_bitnot(int64_t a) { return (int64_t)((uint64_t)(~bronze_to_int32(a)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_shl(int64_t a, int64_t b) { return (int64_t)((uint64_t)(bronze_to_int32(a) << (bronze_to_int32(b) & 31)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_shr(int64_t a, int64_t b) { return (int64_t)((uint64_t)(bronze_to_int32(a) >> (bronze_to_int32(b) & 31)) | 0xFFF9000000000000ULL); }\n"
            << "    int64_t bronze_dynamic_ushr(int64_t a, int64_t b) { return (int64_t)((uint64_t)((uint32_t)bronze_to_int32(a) >> (bronze_to_int32(b) & 31)) | 0xFFF9000000000000ULL); }\n"
            << "    int32_t bronze_strict_eq(int64_t a, int64_t b) { return a == b; }\n"
            << "    int32_t bronze_loose_eq(int64_t a, int64_t b) { return a == b; }\n"
            << "    int64_t bronze_numeric_step(int64_t a, int32_t inc) {\n"
            << "        if (((uint64_t)a >> 48) <= 0xFFF8) {\n"
            << "            double da; memcpy(&da, &a, 8); double res = inc ? da + 1.0 : da - 1.0; int64_t r; memcpy(&r, &res, 8); return r;\n"
            << "        }\n"
            << "        return a;\n"
            << "    }\n"
            << "    uint64_t bronze_resolve_name(uint32_t k, int32_t s) { (void)k; (void)s; return 0xFFF1000000000000ULL; }\n"
            << "    uint64_t bronze_global_get(uint32_t k, uint64_t* c) { (void)k; (void)c; return 0xFFF1000000000000ULL; }\n"
            << "    int64_t bronze_bigint_literal(uint32_t k) { (void)k; return 0; }\n"
            << "    int64_t bronze_get_new_target() { return (int64_t)0xFFF1000000000000ULL; }\n"
            << "    uint64_t bronze_env_get_tdz(uint64_t e, uint32_t d, uint32_t i, uint32_t k) { return (uint64_t)bronze_env_get((int64_t)e, (int32_t)d, (int32_t)i); }\n"
            << "    uint64_t bronze_env_ancestor(uint64_t e, uint32_t d) { (void)d; return e; }\n"
            << "    void bronze_super_set(int64_t p, uint32_t k, int64_t th, int64_t v, int32_t s) { (void)p; (void)k; (void)th; (void)v; (void)s; }\n"
            << "    int64_t bronze_object_keys(int64_t o) { (void)o; return 0; }\n"
            << "    int64_t bronze_for_in_keys(int64_t o) { (void)o; return 0; }\n"
            << "    void bronze_method_def_computed(int64_t o, int64_t k, int64_t c) { (void)o; (void)k; (void)c; }\n"
            << "    void bronze_define_own_attr(uint64_t o, uint32_t k, uint64_t v, uint32_t m) { (void)o; (void)k; (void)v; (void)m; }\n"
            << "    void bronze_accessor_def(uint64_t o, uint32_t k, uint64_t g, uint64_t s, int32_t e) { (void)o; (void)k; (void)g; (void)s; (void)e; }\n"
            << "    void bronze_accessor_def_computed(uint64_t o, uint64_t k, uint64_t g, uint64_t s, int32_t e) { (void)o; (void)k; (void)g; (void)s; (void)e; }\n"
            << "    uint64_t bronze_module_namespace(uint64_t s) { return s; }\n"
            << "    uint64_t bronze_typeof(uint64_t b) { (void)b; return 0; }\n"
            << "    uint64_t bronze_exception_get() { return 0; }\n"
            << "    void bronze_exception_set(uint64_t b) { (void)b; }\n"
            << "    uint64_t bronze_exception_take() { return 0; }\n"
            << "    int32_t bronze_exception_pending() { return 0; }\n"
            << "    void bronze_uncaught_exception() { }\n"
            << "    uint64_t bronze_pin_violation(uint32_t k, uint64_t b) { (void)k; (void)b; return 0; }\n"
            << "    void bronze_pin_check_array(uint32_t k, uint64_t b) { (void)k; (void)b; }\n"
            << "}\n";
        ofs.close();

    }

    for (const auto& name : corpus_files) {
        std::string il_file = "tests/bronze_corpus/" + name + ".il";
        std::string exp_file = "tests/bronze_corpus/" + name + ".expected";

        std::ifstream ifs_il(il_file);
        if (!ifs_il.is_open()) {
            il_file = "../../tests/bronze_corpus/" + name + ".il";
            exp_file = "../../tests/bronze_corpus/" + name + ".expected";
            ifs_il.open(il_file);
        }
        if (!ifs_il.is_open()) {
            il_file = "../../../tests/bronze_corpus/" + name + ".il";
            exp_file = "../../../tests/bronze_corpus/" + name + ".expected";
            ifs_il.open(il_file);
        }
        if (!ifs_il.is_open()) {
            il_file = "../tests/bronze_corpus/" + name + ".il";
            exp_file = "../tests/bronze_corpus/" + name + ".expected";
            ifs_il.open(il_file);
        }
        if (!ifs_il.is_open()) {
            il_file = "/home/j/projects/brass/tests/bronze_corpus/" + name + ".il";
            exp_file = "/home/j/projects/brass/tests/bronze_corpus/" + name + ".expected";
            ifs_il.open(il_file);
        }
        if (!ifs_il.is_open()) {
            il_file = "D:/projects/brass/tests/bronze_corpus/" + name + ".il";
            exp_file = "D:/projects/brass/tests/bronze_corpus/" + name + ".expected";
            ifs_il.open(il_file);
        }
        REQUIRE(ifs_il.is_open());
        std::stringstream ss_il;
        ss_il << ifs_il.rdbuf();
        std::string il_content = ss_il.str();

        std::ifstream ifs_exp(exp_file);
        REQUIRE(ifs_exp.is_open());
        std::stringstream ss_exp;
        ss_exp << ifs_exp.rdbuf();
        std::string expected_output = ss_exp.str();
        std::string expected_norm = normalize(expected_output);

        // =========================================================================
        // Leg (a): JIT Execution with Optimizations OFF
        // =========================================================================
        {
            TranslatorOptions opts_unopt;
            opts_unopt.enable_optimizations = false;
            DiagnosticReporter diag_unopt;
            TranslationResult res_unopt = translate_bronze_il(il_content, opts_unopt, &diag_unopt);
            REQUIRE(res_unopt.success);
            REQUIRE(res_unopt.module != nullptr);

            codegen::JitExecutionEngine jit_unopt(Target::host());
            register_bronze_runtime_symbols(&jit_unopt);
            REQUIRE(jit_unopt.compile_and_load(*res_unopt.module));

            std::stringstream captured_unopt;
            std::streambuf* old_cout = std::cout.rdbuf(captured_unopt.rdbuf());

            auto main_fn = jit_unopt.get_function_ptr<void(*)()>("main");
            REQUIRE(main_fn != nullptr);
            main_fn();

            std::cout.rdbuf(old_cout);
            CHECK_EQ(normalize(captured_unopt.str()), expected_norm);
        }

        // =========================================================================
        // Leg (b): JIT Execution with Default Optimizations ON
        // =========================================================================
        {
            TranslatorOptions opts_opt;
            opts_opt.enable_optimizations = true;
            DiagnosticReporter diag_opt;
            TranslationResult res_opt = translate_bronze_il(il_content, opts_opt, &diag_opt);
            REQUIRE(res_opt.success);
            REQUIRE(res_opt.module != nullptr);

            codegen::JitExecutionEngine jit_opt(Target::host());
            register_bronze_runtime_symbols(&jit_opt);
            REQUIRE(jit_opt.compile_and_load(*res_opt.module));

            std::stringstream captured_opt;
            std::streambuf* old_cout = std::cout.rdbuf(captured_opt.rdbuf());

            auto main_fn = jit_opt.get_function_ptr<void(*)()>("main");
            REQUIRE(main_fn != nullptr);
            main_fn();

            std::cout.rdbuf(old_cout);
            CHECK_EQ(normalize(captured_opt.str()), expected_norm);
        }

        // =========================================================================
        // Leg (c): AOT Compile, Link, and Execute Leg
        // =========================================================================
#if defined(_WIN32)
        if (msvc_ready) {
            TranslatorOptions opts_aot;
            opts_aot.enable_optimizations = true;
            DiagnosticReporter diag_aot;
            TranslationResult res_aot = translate_bronze_il(il_content, opts_aot, &diag_aot);
            REQUIRE(res_aot.success);
            REQUIRE(res_aot.module != nullptr);

            auto obj_file = aot_dir / (name + ".obj");
            auto dll_file = aot_dir / (name + ".dll");
            auto def_file = aot_dir / (name + ".def");

            HostEngine engine;
            REQUIRE(engine.compile_to_object(*res_aot.module, obj_file.string()));
            REQUIRE(std::filesystem::exists(obj_file));

            // Export all module functions and runtime initialization entrypoint
            {
                std::ofstream ofs_def(def_file);
                ofs_def << "EXPORTS\n    init_bronze_runtime\n";
                for (const auto& fn : res_aot.module->functions()) {
                    ofs_def << "    " << fn->name() << "\n";
                }
            }

            // Link into DLL
            std::string cl_cmd = "cl.exe /nologo /LD /EHsc /MD /O2 /Fo\"" + (aot_dir / "bronze_runtime_bridge.obj").string() + "\" \"" + wrapper_cpp.string() + "\" \"" +
                                 obj_file.string() + "\" /Fe\"" + dll_file.string() + "\" /link /DEF:\"" + def_file.string() + "\"";
            int link_res = MsvcToolchain::run_msvc_cmd(cl_cmd);
            REQUIRE_EQ(link_res, 0);
            REQUIRE(std::filesystem::exists(dll_file));

            // Load and execute AOT module
            HMODULE hDll = LoadLibraryA(dll_file.string().c_str());
            REQUIRE(hDll != nullptr);

            using init_fn_t = void (*)(const BronzeRuntimeTable*);
            auto p_init_rt = reinterpret_cast<init_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "init_bronze_runtime")));
            REQUIRE(p_init_rt != nullptr);
            p_init_rt(&host_rt);

            g_aot_active_dll = hDll;
            set_bronze_function_resolver(&aot_symbol_resolver);

            std::stringstream captured_aot;
            std::streambuf* old_cout = std::cout.rdbuf(captured_aot.rdbuf());

            using main_fn_t = void (*)();
            auto aot_main = reinterpret_cast<main_fn_t>(reinterpret_cast<void*>(GetProcAddress(hDll, "main")));
            REQUIRE(aot_main != nullptr);
            aot_main();

            std::cout.rdbuf(old_cout);
            set_bronze_function_resolver(nullptr);
            g_aot_active_dll = nullptr;
            FreeLibrary(hDll);

            CHECK_EQ(normalize(captured_aot.str()), expected_norm);
        }
#endif
    }
}
