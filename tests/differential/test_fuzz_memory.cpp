#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <random>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Multi-Level Memory Indexing and Pointer Arithmetic") {
    std::mt19937_64 rng(5555);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_mem_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_mem_fn";

        generate_fuzz_memory(mod, fn_name, seed);

        size_t count = 32 + (seed % 32);
        std::vector<int64_t> interp_buf(count);
        std::vector<int64_t> jit_buf(count);
        for (size_t i = 0; i < count; ++i) {
            int64_t v = static_cast<int64_t>(rng() % 5000);
            interp_buf[i] = v;
            jit_buf[i] = v;
        }

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(interp_buf.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(count))
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        RuntimeValue jit_res = jit.invoke(fn_name, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(jit_buf.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(count))
        });

        CHECK_EQ(interp_res.as_i64(), jit_res.as_i64());
        CHECK(interp_buf == jit_buf);
    }
}
