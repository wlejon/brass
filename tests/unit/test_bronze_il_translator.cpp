#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>

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
    if (!res.success) {
        std::cerr << "Diagnostics:\n" << diag.format_all() << "\n";
    }
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto math_fn = jit.get_function_ptr<double(*)(double, double)>("mathOps");
    REQUIRE(math_fn != nullptr);
    // (10+4) + (10-4) + (10*4) + (10/4) + (-10) = 14 + 6 + 40 + 2.5 - 10 = 52.5
    double math_res = math_fn(10.0, 4.0);
    CHECK(std::abs(math_res - 52.5) < 1e-9);

    auto bit_fn = jit.get_function_ptr<double(*)(double, double)>("bitOps");
    REQUIRE(bit_fn != nullptr);
    // 42 & 3 = 2, 42 | 3 = 43, 42 ^ 3 = 41, 42 << 3 = 336, 42 >> 3 = 5, 42 >>> 3 = 5
    // Sum = 2 + 43 + 41 + 336 + 5 + 5 = 432
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
    // Collatz steps for 27: 111 steps
    CHECK_EQ(collatz_fn(27.0), 111.0);
    // Collatz steps for 1: 0 steps
    CHECK_EQ(collatz_fn(1.0), 0.0);
    // Collatz steps for 6: 8 steps (6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1)
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

    // Link into DLL using MSVC link.exe
    std::string link_cmd = "link.exe /NOLOGO /DLL /OUT:\"" + dll_file + "\" \"" + obj_file + "\" /NOENTRY";
    int link_rc = std::system(link_cmd.c_str());
    if (link_rc == 0) {
        HMODULE hModule = LoadLibraryA(dll_file.c_str());
        REQUIRE(hModule != NULL);

        using ChecksumFn = double(*)(double, double);
        auto fn = reinterpret_cast<ChecksumFn>(GetProcAddress(hModule, "compute_checksum"));
        REQUIRE(fn != nullptr);

        // (5 + 7) + (5 * 7) = 12 + 35 = 47.0
        double result = fn(5.0, 7.0);
        CHECK_EQ(result, 47.0);

        FreeLibrary(hModule);
    }
#endif
}
