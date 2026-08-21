#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Dynamic Patching While Executing") {
    std::mt19937_64 rng(3333);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_patch_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_patch_fn";

        generate_fuzz_patching(mod, fn_name, seed);

        Interpreter interp;
        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        int32_t input_x = static_cast<int32_t>(rng() % 1000);

        // Phase 1: Unpatched execution (default bias = 5, call = stub_add)
        RuntimeValue i1 = interp.run(mod, fn_name, {RuntimeValue::from_i32(input_x)});
        RuntimeValue j1 = jit.invoke(fn_name, {RuntimeValue::from_i32(input_x)});
        CHECK_EQ(i1.as_i32(), j1.as_i32());

        // Phase 2: Patch constant bias = 42
        std::string bias_name = fn_name + "_bias";
        int32_t new_bias = static_cast<int32_t>(rng() % 500 + 10);
        interp.patch_const(bias_name, new_bias);
        jit.patch_const32(bias_name, new_bias);

        RuntimeValue i2 = interp.run(mod, fn_name, {RuntimeValue::from_i32(input_x)});
        RuntimeValue j2 = jit.invoke(fn_name, {RuntimeValue::from_i32(input_x)});
        CHECK_EQ(i2.as_i32(), j2.as_i32());

        // Phase 3: Patch call site = stub_sub
        std::string call_site_name = fn_name + "_call_site";
        std::string stub_sub_name = fn_name + "_stub_sub";
        interp.patch_call(call_site_name, stub_sub_name);
        jit.patch_call(call_site_name, stub_sub_name);

        RuntimeValue i3 = interp.run(mod, fn_name, {RuntimeValue::from_i32(input_x)});
        RuntimeValue j3 = jit.invoke(fn_name, {RuntimeValue::from_i32(input_x)});
        CHECK_EQ(i3.as_i32(), j3.as_i32());
    }
}
