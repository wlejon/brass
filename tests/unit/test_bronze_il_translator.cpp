#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
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

static HMODULE g_aot_active_dll = nullptr;

static void* aot_symbol_resolver(const char* name) {
    if (!name || !g_aot_active_dll) return nullptr;
    return reinterpret_cast<void*>(GetProcAddress(g_aot_active_dll, name));
}

TEST_CASE("Bronze IL - 18-Program Live Corpus JIT and AOT Execution") {
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
        "18_gcd_iter"
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
        int64_t (*create_func)(const char*, int32_t, int64_t);
        int64_t (*create_array)(int32_t);
        void (*prop_set)(int64_t, int32_t, int64_t, int32_t, int32_t);
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
        &brass::il::bronze_call_dynamic_n
    };

    bool msvc_ready = MsvcToolchain::is_available();
    std::filesystem::path aot_dir;
    std::filesystem::path wrapper_cpp;
    if (msvc_ready) {
        aot_dir = MsvcToolchain::temp_dir() / "bronze_corpus_aot";
        std::filesystem::create_directories(aot_dir);
        wrapper_cpp = aot_dir / "bronze_runtime_bridge.cpp";
        std::ofstream ofs(wrapper_cpp);
        ofs << "#define NOMINMAX\n"
            << "#define WIN32_LEAN_AND_MEAN\n"
            << "#include <windows.h>\n"
            << "#include <cstdint>\n"
            << "#include <cstddef>\n\n"
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
            << "    int64_t (*create_func)(const char*, int32_t, int64_t);\n"
            << "    int64_t (*create_array)(int32_t);\n"
            << "    void (*prop_set)(int64_t, int32_t, int64_t, int32_t, int32_t);\n"
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
            << "};\n\n"
            << "static BronzeRuntimeTable g_rt = {};\n\n"
            << "extern \"C\" {\n"
            << "    __declspec(dllexport) void init_bronze_runtime(const BronzeRuntimeTable* table) {\n"
            << "        if (table) g_rt = *table;\n"
            << "    }\n"
            << "    void bronze_print_f64(double v) { if (g_rt.print_f64) g_rt.print_f64(v); }\n"
            << "    void bronze_print_i32(int32_t v) { if (g_rt.print_i32) g_rt.print_i32(v); }\n"
            << "    void bronze_print_dynamic(int64_t v) { if (g_rt.print_dynamic) g_rt.print_dynamic(v); }\n"
            << "    void bronze_print_newline() { if (g_rt.print_newline) g_rt.print_newline(); }\n"
            << "    double bronze_f64_mod(double a, double b) { return g_rt.f64_mod ? g_rt.f64_mod(a, b) : 0.0; }\n"
            << "    int64_t bronze_name_resolve(const char* name) { return g_rt.name_resolve ? g_rt.name_resolve(name) : 0; }\n"
            << "    int64_t bronze_env_create(int64_t parent, int32_t sz) { return g_rt.env_create ? g_rt.env_create(parent, sz) : 0; }\n"
            << "    int64_t bronze_env_get(int64_t env, int32_t d, int32_t idx) { return g_rt.env_get ? g_rt.env_get(env, d, idx) : 0; }\n"
            << "    void bronze_env_set(int64_t env, int32_t d, int32_t idx, int64_t v) { if (g_rt.env_set) g_rt.env_set(env, d, idx, v); }\n"
            << "    int64_t bronze_create_func(const char* fn_name, int32_t pc, int64_t env) { return g_rt.create_func ? g_rt.create_func(fn_name, pc, env) : 0; }\n"
            << "    int64_t bronze_create_array(int32_t sz) { return g_rt.create_array ? g_rt.create_array(sz) : 0; }\n"
            << "    void bronze_prop_set(int64_t o, int32_t k, int64_t v, int32_t s, int32_t imm) { if (g_rt.prop_set) g_rt.prop_set(o, k, v, s, imm); }\n"
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
            << "}\n";
    }

    for (const auto& name : corpus_files) {
        std::string il_file = "tests/bronze_corpus/" + name + ".il";
        std::string exp_file = "tests/bronze_corpus/" + name + ".expected";

        std::ifstream ifs_il(il_file);
        if (!ifs_il.is_open()) {
            il_file = "D:/projects/brass/" + il_file;
            exp_file = "D:/projects/brass/" + exp_file;
            ifs_il.open(il_file);
        }
        REQUIRE(ifs_il.is_open());
        std::stringstream ss_il;
        ss_il << ifs_il.rdbuf();
        std::string il_content = ss_il.str();

        std::ifstream ifs_exp(exp_file);
        if (!ifs_exp.is_open()) {
            exp_file = "D:/projects/brass/" + exp_file;
            ifs_exp.open(exp_file);
        }
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
            std::string cl_cmd = "cl.exe /nologo /LD /EHsc /MD /O2 \"" + wrapper_cpp.string() + "\" \"" +
                                 obj_file.string() + "\" /Fe:\"" + dll_file.string() + "\" /link /DEF:\"" + def_file.string() + "\"";
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
    }
}
