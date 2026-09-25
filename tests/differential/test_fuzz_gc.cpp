#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Moving GC Roots Across Calls and Safepoints") {
    std::mt19937_64 rng(7777);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_gc_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_gc_fn";

        generate_fuzz_gc(mod, fn_name, seed + 500);

        int64_t v1 = static_cast<int64_t>(rng() % 10000 + 1);
        int64_t v2 = static_cast<int64_t>(rng() % 10000 + 1);
        std::vector<RuntimeValue> args = {RuntimeValue::from_i64(v1), RuntimeValue::from_i64(v2)};

        // 1. Interpreter execution with a full (moving: every young object is
        // promoted) collection at every allocation and safepoint. Full, not
        // minor: the generated MIR has no write barriers.
        Interpreter interp(gc::HeapConfig{});
        interp.heap().set_stress(gc::StressMode::Full);
        RuntimeValue interp_res = interp.run(mod, fn_name, args);

        // 2. Native JIT execution under the same stress, through the stack maps
        gc::Heap gc;
        gc.set_stress(gc::StressMode::Full);
        gc::HeapScope bind(gc);

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        brass_set_active_stack_maps(&jit.stack_maps());

        RuntimeValue jit_res = jit.invoke(fn_name, args);

        CHECK_EQ(interp_res.as_i64(), jit_res.as_i64());
        CHECK(gc.collection_count() >= 1ULL);

        brass_set_active_stack_maps(nullptr);
    }
}
