#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <brass/runtime/deopt.hpp>
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Speculation Guards and Mid-Loop Deoptimization") {
    std::mt19937_64 rng(9999);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_spec_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_spec_fn";

        generate_fuzz_speculation(mod, fn_name, seed);

        Interpreter interp;
        interp.set_module(&mod);
        interp.set_deopt_handler([&](Interpreter& i, const DeoptResult& d) -> RuntimeValue {
            const Function* t_fn = mod.get_function(d.exit_stub);
            if (!t_fn) return RuntimeValue::from_i64(-1);
            uint64_t buf[2] = {
                static_cast<uint64_t>(d.state_map[0].as_i64()),
                static_cast<uint64_t>(d.state_map[1].as_i64())
            };
            return i.resume(*t_fn, d.resume_id, {
                RuntimeValue::from_i32(static_cast<int32_t>(d.resume_id)),
                RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buf))
            });
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        // Test Fast Path: tag = 1, a < 1000
        {
            int64_t a = static_cast<int64_t>(rng() % 500 + 1);
            int64_t b = static_cast<int64_t>(rng() % 100 + 1);
            int32_t tag = 1;
            std::vector<RuntimeValue> args = {
                RuntimeValue::from_i64(a),
                RuntimeValue::from_i64(b),
                RuntimeValue::from_i32(tag)
            };

            RuntimeValue i_res = interp.run(mod, fn_name, args);
            RuntimeValue j_res = jit.invoke(fn_name, args);
            CHECK_EQ(i_res.as_i64(), j_res.as_i64());
        }

        // Test Slow Path 1: tag = 0 (Deopt at Guard 0 -> resume 0)
        {
            int64_t a = static_cast<int64_t>(rng() % 500 + 1);
            int64_t b = static_cast<int64_t>(rng() % 100 + 1);
            int32_t tag = 0;
            std::vector<RuntimeValue> args = {
                RuntimeValue::from_i64(a),
                RuntimeValue::from_i64(b),
                RuntimeValue::from_i32(tag)
            };

            RuntimeValue i_res = interp.run(mod, fn_name, args);
            RuntimeValue j_res = jit.invoke(fn_name, args);
            CHECK_EQ(i_res.as_i64(), j_res.as_i64());
        }

        // Test Slow Path 2: tag = 1, a >= 1000 (Deopt at Guard 1 -> resume 1)
        {
            int64_t a = static_cast<int64_t>(rng() % 500 + 1000);
            int64_t b = static_cast<int64_t>(rng() % 100 + 1);
            int32_t tag = 1;
            std::vector<RuntimeValue> args = {
                RuntimeValue::from_i64(a),
                RuntimeValue::from_i64(b),
                RuntimeValue::from_i32(tag)
            };

            RuntimeValue i_res = interp.run(mod, fn_name, args);
            RuntimeValue j_res = jit.invoke(fn_name, args);
            CHECK_EQ(i_res.as_i64(), j_res.as_i64());
        }
    }
}
