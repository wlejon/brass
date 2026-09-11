#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/brass_c_api.h>
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/object/object_writer.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace brass;
using namespace brass::il;
using namespace brass::test;

namespace {

// Native C-ABI implementation stubs matching bro_math_c_abi.h
extern "C" {

static double host_bro_math_lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

static double host_bro_math_clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static double host_bro_math_smoothstep(double e0, double e1, double x) {
    double d = e1 - e0;
    double t = d == 0.0 ? 0.0 : (x - e0) / d;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return t * t * (3.0 - 2.0 * t);
}

struct DummySpatialHash {
    double cell_size = 1.0;
    struct Point {
        double id;
        double x, y, z;
    };
    std::vector<Point> points;
};

static void* host_bro_SpatialHash3D_create(double cell_size) {
    auto* sh = new DummySpatialHash();
    sh->cell_size = cell_size;
    return sh;
}

static void host_bro_SpatialHash3D_destroy(void* self) {
    delete static_cast<DummySpatialHash*>(self);
}

static void* host_bro_SpatialHash3D_insert(void* self, double id, double x, double y, double z) {
    if (self) {
        auto* sh = static_cast<DummySpatialHash*>(self);
        sh->points.push_back({id, x, y, z});
    }
    return self;
}

static double host_bro_SpatialHash3D_get_size(void* self) {
    if (!self) return 0.0;
    return static_cast<double>(static_cast<DummySpatialHash*>(self)->points.size());
}

static double host_bro_SpatialHash3D_nearest(void* self, double x, double y, double z, double max_dist) {
    if (!self) return -1.0;
    auto* sh = static_cast<DummySpatialHash*>(self);
    double best_d2 = max_dist * max_dist;
    double best_id = -1.0;
    for (const auto& pt : sh->points) {
        double dx = pt.x - x;
        double dy = pt.y - y;
        double dz = pt.z - z;
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= best_d2) {
            best_d2 = d2;
            best_id = pt.id;
        }
    }
    return best_id;
}

} // extern "C"

} // namespace

TEST_CASE("Native C-ABI - Direct Scalar Math Calling from Bronze IL") {
    const char* il_source = R"(
module test_cabi_scalar.js

func test_lerp(%0: f64, %1: f64, %2: f64) -> f64 {
  b0:
    %3: f64 = call @bro_math_lerp(%0, %1, %2)
    ret %3
}

func test_clamp(%0: f64, %1: f64, %2: f64) -> f64 {
  b0:
    %3: f64 = call @bro_math_clamp(%0, %1, %2)
    ret %3
}

func test_smoothstep(%0: f64, %1: f64, %2: f64) -> f64 {
  b0:
    %3: f64 = call @bro_math_smoothstep(%0, %1, %2)
    ret %3
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    jit.register_external_symbol("bro_math_lerp", reinterpret_cast<void*>(&host_bro_math_lerp));
    jit.register_external_symbol("bro_math_clamp", reinterpret_cast<void*>(&host_bro_math_clamp));
    jit.register_external_symbol("bro_math_smoothstep", reinterpret_cast<void*>(&host_bro_math_smoothstep));

    REQUIRE(jit.compile_and_load(*res.module));

    // 1. Test lerp(10.0, 50.0, 0.25) -> 20.0
    auto fn_lerp = jit.get_function_ptr<double (*)(double, double, double)>("test_lerp");
    REQUIRE(fn_lerp != nullptr);
    CHECK(std::abs(fn_lerp(10.0, 50.0, 0.25) - 20.0) < 1e-6);
    CHECK(std::abs(fn_lerp(0.0, 100.0, 0.5) - 50.0) < 1e-6);

    // 2. Test clamp(15.0, 0.0, 10.0) -> 10.0
    auto fn_clamp = jit.get_function_ptr<double (*)(double, double, double)>("test_clamp");
    REQUIRE(fn_clamp != nullptr);
    CHECK(fn_clamp(15.0, 0.0, 10.0) == 10.0);
    CHECK(fn_clamp(-5.0, 0.0, 10.0) == 0.0);
    CHECK(fn_clamp(7.5, 0.0, 10.0) == 7.5);

    // 3. Test smoothstep(0.0, 10.0, 5.0) -> 0.5
    auto fn_smoothstep = jit.get_function_ptr<double (*)(double, double, double)>("test_smoothstep");
    REQUIRE(fn_smoothstep != nullptr);
    CHECK(std::abs(fn_smoothstep(0.0, 10.0, 5.0) - 0.5) < 1e-6);
}

TEST_CASE("Native C-ABI - Stateful Object Handle Invocation (SpatialHash3D)") {
    const char* il_source = R"(
module test_spatial_hash.js

func test_spatial_hash(%0: f64) -> f64 {
  b0:
    %1: dynamic = call @bro_SpatialHash3D_create(%0)
    %2: f64 = const.f64 42
    %3: f64 = const.f64 10
    %4: f64 = const.f64 20
    %5: f64 = const.f64 30
    %6: dynamic = call @bro_SpatialHash3D_insert(%1, %2, %3, %4, %5)
    %7: f64 = const.f64 99
    %8: f64 = const.f64 100
    %9: f64 = const.f64 200
    %10: f64 = const.f64 300
    %11: dynamic = call @bro_SpatialHash3D_insert(%1, %7, %8, %9, %10)
    %12: f64 = call @bro_SpatialHash3D_get_size(%1)
    %13: f64 = const.f64 11
    %14: f64 = const.f64 21
    %15: f64 = const.f64 31
    %16: f64 = const.f64 5
    %17: f64 = call @bro_SpatialHash3D_nearest(%1, %13, %14, %15, %16)
    call @bro_SpatialHash3D_destroy(%1)
    %18: f64 = add %12, %17
    ret %18
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    if (!res.success) {
        for (const auto& d : diag.diagnostics()) {
            std::cerr << "IL Error: " << d.to_string() << "\n";
        }
    }
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    jit.register_external_symbol("bro_SpatialHash3D_create", reinterpret_cast<void*>(&host_bro_SpatialHash3D_create));
    jit.register_external_symbol("bro_SpatialHash3D_destroy", reinterpret_cast<void*>(&host_bro_SpatialHash3D_destroy));
    jit.register_external_symbol("bro_SpatialHash3D_insert", reinterpret_cast<void*>(&host_bro_SpatialHash3D_insert));
    jit.register_external_symbol("bro_SpatialHash3D_get_size", reinterpret_cast<void*>(&host_bro_SpatialHash3D_get_size));
    jit.register_external_symbol("bro_SpatialHash3D_nearest", reinterpret_cast<void*>(&host_bro_SpatialHash3D_nearest));

    REQUIRE(jit.compile_and_load(*res.module));

    auto fn_test = jit.get_function_ptr<double (*)(double)>("test_spatial_hash");
    REQUIRE(fn_test != nullptr);

    // size is 2.0, nearest to (11, 21, 31) within 5.0 is ID 42.0. Result = 2.0 + 42.0 = 44.0!
    double val = fn_test(2.0);
    CHECK(val == 44.0);
}

TEST_CASE("Native C-ABI - AOT Object Relocation Emission") {
    const char* il_source = R"(
module test_aot_math.js

func test_aot_math(%0: f64, %1: f64, %2: f64) -> f64 {
  b0:
    %3: f64 = call @bro_math_lerp(%0, %1, %2)
    ret %3
}
)";

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, {}, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    Target target = Target::host();
    object::ObjectFile obj = object::compile_module_to_object(*res.module, target);

    bool found_reloc = false;
    for (const auto& sec : obj.sections) {
        for (const auto& reloc : sec.relocations) {
            if (reloc.symbol_name == "bro_math_lerp") {
                found_reloc = true;
                break;
            }
        }
    }
    CHECK(found_reloc);
}
