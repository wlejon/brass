#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/embedding/host_gc.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace brass;
using namespace brass::il;

TEST_CASE("Differential Fuzzer - Closures and Environments under Cheney GC Stress") {
    // 1. Set up HostGC with small semispace (512 bytes) to force collection during multi-closure allocation
    HostGC host_gc(512);
    set_active_host_gc(&host_gc);

    // Test program: Mutable counter closures under deep allocation pressure
    const char* il_counter = R"(
module counter_gc.js

func __anon_fn_1(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.get %0, 0, 0
    %3: dynamic = env.get %0, 0, 1
    %4: dynamic = add %2, %3
    env.set %0, 0, 0, %4
    ret %4
}

func makeCounter(%0: dynamic, %1: dynamic, %2: dynamic) -> dynamic {
  b0:
    %3: dynamic = env.create %0, 2
    env.set %3, 0, 0, %1
    env.set %3, 0, 1, %2
    %4: dynamic = create.func @__anon_fn_1, 0, %3
    ret %4
}

func main() -> dynamic {
  b0:
    %0: dynamic = const.undefined
    %1: f64 = const.f64 10
    %2: dynamic = box.f64 %1
    %3: f64 = const.f64 5
    %4: dynamic = box.f64 %3
    %5: dynamic = call @makeCounter(%0, %2, %4)
    %6: dynamic = call.dynamic %5, %0, 0
    %7: dynamic = call.dynamic %5, %0, 0
    %8: f64 = const.f64 100
    %9: dynamic = box.f64 %8
    %10: f64 = const.f64 20
    %11: dynamic = box.f64 %10
    %12: dynamic = call @makeCounter(%0, %9, %11)
    %13: dynamic = call.dynamic %12, %0, 0
    %14: dynamic = call.dynamic %12, %0, 0
    %15: dynamic = add %6, %7
    %16: dynamic = add %15, %13
    %17: dynamic = add %16, %14
    ret %17
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_counter, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    using MainFn = int64_t(*)();
    auto main_fn = jit.get_function_ptr<MainFn>("main");
    REQUIRE(main_fn != nullptr);

    int64_t raw_res = main_fn();
    double val = 0.0;
    std::memcpy(&val, &raw_res, sizeof(double));

    // makeCounter(10, 5) -> c1() = 15, c1() = 20
    // makeCounter(100, 20) -> c2() = 120, c2() = 140
    // sum = 15 + 20 + 120 + 140 = 295
    CHECK_EQ(val, 295.0);

    // Reset active host gc
    set_active_host_gc(nullptr);
}

TEST_CASE("Differential Fuzzer - Deeply Nested Curried Closures with Cheney GC") {
    HostGC host_gc(512);
    set_active_host_gc(&host_gc);

    const char* il_curry = R"(
module curry_gc.js

func __anon_fn_2$leaf(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.get %0, 1, 0
    %3: dynamic = env.get %0, 0, 0
    %4: dynamic = add %2, %3
    %5: dynamic = add %4, %1
    ret %5
}

func __anon_fn_1(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.create %0, 1
    env.set %2, 0, 0, %1
    %3: dynamic = create.func @__anon_fn_2$leaf, 1, %2
    ret %3
}

func makeAdder3D(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.create %0, 1
    env.set %2, 0, 0, %1
    %3: dynamic = create.func @__anon_fn_1, 1, %2
    ret %3
}

func main() -> dynamic {
  b0:
    %0: dynamic = const.undefined
    %1: f64 = const.f64 100
    %2: dynamic = box.f64 %1
    %3: f64 = const.f64 200
    %4: dynamic = box.f64 %3
    %5: f64 = const.f64 300
    %6: dynamic = box.f64 %5
    %7: dynamic = call @makeAdder3D(%0, %2)
    %8: dynamic = call.dynamic %7, %0, 1, %4
    %9: dynamic = call.dynamic %8, %0, 1, %6
    ret %9
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_curry, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    using MainFn = int64_t(*)();
    auto main_fn = jit.get_function_ptr<MainFn>("main");
    REQUIRE(main_fn != nullptr);

    int64_t raw_res = main_fn();
    double val = 0.0;
    std::memcpy(&val, &raw_res, sizeof(double));

    // 100 + 200 + 300 = 600
    CHECK_EQ(val, 600.0);

    set_active_host_gc(nullptr);
}
