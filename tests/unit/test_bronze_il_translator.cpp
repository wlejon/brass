#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
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

TEST_CASE("Bronze IL - 11-Program Live Corpus JIT and AOT Execution") {
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
        "11_matrix_recurrence"
    };

    for (const auto& name : corpus_files) {
        std::string il_file = "tests/bronze_corpus/" + name + ".il";
        std::string exp_file = "tests/bronze_corpus/" + name + ".expected";

        std::ifstream ifs_il(il_file);
        if (!ifs_il.is_open()) {
            // Try relative to workspace root
            il_file = "D:/projects/brass/" + il_file;
            exp_file = "D:/projects/brass/" + exp_file;
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

        // 1. Translate
        DiagnosticReporter diag;
        TranslationResult res = translate_bronze_il(il_content, {}, &diag);
        REQUIRE(res.success);
        REQUIRE(res.module != nullptr);

        // 2. JIT Execute and verify
        codegen::JitExecutionEngine jit(Target::host());
        register_bronze_runtime_symbols(&jit);
        REQUIRE(jit.compile_and_load(*res.module));

        // Capture stdout
        std::stringstream captured_out;
        std::streambuf* old_cout = std::cout.rdbuf(captured_out.rdbuf());

        auto main_fn = jit.get_function_ptr<void(*)()>("main");
        REQUIRE(main_fn != nullptr);
        main_fn();

        std::cout.rdbuf(old_cout);

        // Normalize spaces
        auto normalize = [](const std::string& s) {
            std::istringstream iss(s);
            std::string word, result;
            while (iss >> word) {
                if (!result.empty()) result += " ";
                result += word;
            }
            return result;
        };

        std::string actual_norm = normalize(captured_out.str());
        std::string expected_norm = normalize(expected_output);
        CHECK_EQ(actual_norm, expected_norm);

        // 3. AOT Compile Object
        HostEngine engine;
        std::string obj_out = "tests/bronze_corpus/" + name + ".obj";
        CHECK(engine.compile_to_object(*res.module, obj_out));
        std::remove(obj_out.c_str());
    }
}
