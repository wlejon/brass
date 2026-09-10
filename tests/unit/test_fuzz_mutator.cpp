#include "test_framework.hpp"
#include <brass/fuzz/ir_mutator.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>

#include <sstream>
#include <vector>
#include <cmath>

using namespace brass;
using namespace brass::fuzz;

TEST_CASE("FuzzRng_Determinism") {
    FuzzRng rng1(0x123456789ABCDEF0ULL);
    FuzzRng rng2(0x123456789ABCDEF0ULL);

    for (int i = 0; i < 100; ++i) {
        CHECK_EQ(rng1.next_u64(), rng2.next_u64());
        CHECK_EQ(rng1.next_u32(), rng2.next_u32());
        CHECK(rng1.next_f64() == rng2.next_f64());
    }
}

TEST_CASE("FuzzBoundaryValues_Coverage") {
    FuzzRng rng(42);

    bool seen_zero = false;
    bool seen_nan = false;
    bool seen_inf = false;
    bool seen_max = false;
    bool seen_min = false;

    for (int i = 0; i < 200; ++i) {
        int64_t v_i64 = rng.boundary_i64();
        if (v_i64 == 0) seen_zero = true;
        if (v_i64 == INT64_MAX) seen_max = true;
        if (v_i64 == INT64_MIN) seen_min = true;

        double v_f64 = rng.boundary_f64();
        if (std::isnan(v_f64)) seen_nan = true;
        if (std::isinf(v_f64)) seen_inf = true;
    }

    CHECK(seen_zero);
    CHECK(seen_max);
    CHECK(seen_min);
    CHECK(seen_nan);
    CHECK(seen_inf);
}

TEST_CASE("IrGenerator_Determinism") {
    IrGeneratorOptions opts;
    opts.min_instructions = 15;
    opts.max_instructions = 30;
    opts.enable_loops = true;
    opts.enable_diamonds = true;
    opts.enable_switches = true;
    opts.enable_vectors = true;

    Module mod1("det_mod1");
    Module mod2("det_mod2");

    IrGenerator gen1(opts);
    IrGenerator gen2(opts);

    Function* fn1 = gen1.generate(mod1, "fuzz_kernel", 0xCAFEBABEDEADBEEFULL);
    Function* fn2 = gen2.generate(mod2, "fuzz_kernel", 0xCAFEBABEDEADBEEFULL);

    CHECK(fn1 != nullptr);
    CHECK(fn2 != nullptr);

    std::ostringstream ss1, ss2;
    print_function(*fn1, ss1);
    print_function(*fn2, ss2);

    CHECK_EQ(ss1.str(), ss2.str());
}

TEST_CASE("IrGenerator_VerifyAllGenerated") {
    IrGeneratorOptions opts;
    opts.min_instructions = 10;
    opts.max_instructions = 25;
    opts.enable_loops = true;
    opts.enable_diamonds = true;
    opts.enable_switches = true;
    opts.enable_vectors = true;
    opts.enable_exceptions = true;

    IrGenerator generator(opts);

    for (uint64_t seed = 100; seed < 120; ++seed) {
        Module mod("fuzz_test_" + std::to_string(seed));
        Function* fn = generator.generate(mod, "test_fn", seed);
        CHECK(fn != nullptr);

        DiagnosticReporter diag;
        bool valid = verify_module(mod, &diag);
        CHECK(valid);
    }
}

TEST_CASE("IrMutator_MutationPassesPreserveValidity") {
    IrGeneratorOptions opts;
    opts.min_instructions = 15;
    opts.max_instructions = 30;
    opts.enable_loops = true;
    opts.enable_diamonds = true;
    opts.enable_switches = false;
    opts.enable_vectors = false;

    IrGenerator generator(opts);

    for (uint64_t seed = 200; seed < 210; ++seed) {
        Module mod("mutate_test_" + std::to_string(seed));
        Function* fn = generator.generate(mod, "fuzz_fn", seed);
        CHECK(fn != nullptr);

        FuzzRng rng(seed ^ 0x55555555ULL);

        // 1. Mutate constants
        IrMutator::mutate_constant_immediates(*fn, rng);
        CHECK(verify_function(*fn));

        // 2. Swap commutative operands
        IrMutator::swap_commutative_operands(*fn, rng);
        CHECK(verify_function(*fn));

        // 3. Replace opcodes
        IrMutator::replace_opcodes(*fn, rng);
        CHECK(verify_function(*fn));

        // 4. Split basic blocks
        IrMutator::split_basic_blocks(*fn, rng);
        CHECK(verify_function(*fn));

        // 5. Inject speculative guards
        IrMutator::inject_speculative_guards(*fn, rng);
        CHECK(verify_function(*fn));

        // 6. Overall mutate_function
        IrMutator::mutate_function(*fn, rng);
        CHECK(verify_function(*fn));
    }
}
