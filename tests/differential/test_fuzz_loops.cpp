#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Complex Nested Loops with Block Arguments") {
    std::mt19937_64 rng(4242);

    for (uint64_t seed = 0; seed < 50; ++seed) {
        std::string mod_name = "fuzz_loop_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_loop_fn";

        generate_fuzz_loops(mod, fn_name, seed + 100);

        for (int t = 0; t < 3; ++t) {
            int64_t limit = static_cast<int64_t>(rng() % 50);
            int64_t init_val = static_cast<int64_t>(rng() % 1000);

            assert_diff(mod, fn_name, {
                RuntimeValue::from_i64(limit),
                RuntimeValue::from_i64(init_val)
            });
        }
    }
}
