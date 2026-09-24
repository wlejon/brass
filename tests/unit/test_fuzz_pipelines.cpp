// The differential fuzzer over a fixed seed range: structured random
// programs run on the interpreter, the unoptimized JIT, and the interpreter
// and JIT after each optimization pipeline, and all answers must agree.
// A failure prints each failure class (the step that broke verification or
// first changed the answer) with one seed to replay through
// `brass-fuzz --seed=N --iterations=1 --pipeline=...`.

#include "test_framework.hpp"
#include <brass/fuzz/program_generator.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace brass;
using namespace brass::fuzz;

namespace {

constexpr uint64_t kFirstSeed = 1;
constexpr uint32_t kPrograms = 150;

size_t run_seed_range(FuzzPipeline pipeline) {
    DiffFuzzerOptions opts;
    opts.pipeline = pipeline;
    opts.save_reproducers = false;
    // Generous: ctest runs many suites in parallel on the same machine.
    opts.timeout_ms = 3000;
    DiffFuzzer fuzzer(opts);

    std::map<std::string, std::pair<size_t, uint64_t>> classes;
    size_t failures = 0;
    for (uint64_t seed = kFirstSeed; seed < kFirstSeed + kPrograms; ++seed) {
        Module mod("fuzz_mod_" + std::to_string(seed));
        ProgramGenerator gen;
        Function* fn = gen.generate(mod, "fuzz_fn", seed);
        REQUIRE(fn != nullptr);
        const std::vector<RuntimeValue> args = {
            RuntimeValue::from_i64(static_cast<int64_t>(seed % 100)),
            RuntimeValue::from_i64(static_cast<int64_t>((seed >> 8) % 100)),
        };
        DiffResult r = fuzzer.run_test(mod, "fuzz_fn", args, seed);
        if (r.passed) continue;
        ++failures;
        auto& entry = classes[r.failure_class];
        if (entry.first++ == 0) entry.second = seed;
    }
    if (failures) {
        std::cerr << "pipeline " << pipeline_name(pipeline) << ": " << failures << " of " << kPrograms
                  << " programs failed\n";
        for (const auto& [name, info] : classes) {
            std::cerr << "  " << info.first << "  " << name << "  (e.g. seed " << info.second << ")\n";
        }
    }
    return failures;
}

} // namespace

namespace {

void check_generated_program(uint64_t seed, const ProgramGeneratorOptions& options = {}) {
    Module mod("fuzz_mod_" + std::to_string(seed));
    ProgramGenerator gen(options);
    REQUIRE(gen.generate(mod, "fuzz_fn", seed) != nullptr);
    DiagnosticReporter diag;
    const bool ok = verify_module(mod, &diag);
    if (!ok) std::cerr << "seed " << seed << ": " << diag.format_all() << "\n";
    REQUIRE(ok);

    std::ostringstream text;
    print_module(mod, text);
    DiagnosticReporter pdiag;
    auto back = parse_module(text.str(), &pdiag);
    if (!back) std::cerr << "seed " << seed << ": " << pdiag.format_all() << "\n";
    REQUIRE(back != nullptr);
    REQUIRE(verify_module(*back, &pdiag));

    std::ostringstream again;
    print_module(*back, again);
    CHECK(again.str() == text.str());
}

} // namespace

TEST_CASE("Fuzz - generated programs verify and survive the text round trip") {
    for (uint64_t seed = kFirstSeed; seed < kFirstSeed + kPrograms; ++seed) {
        check_generated_program(seed);
    }
    // Seed 2381: a block split whose result failed verification was rolled
    // back with its jump left in the middle of the block.
    check_generated_program(2381);
    ProgramGeneratorOptions small;
    small.max_statements = 10;
    check_generated_program(100336, small);
}

TEST_CASE("Fuzz - every optimization pass preserves the answer on a fixed seed range") {
    CHECK_EQ(run_seed_range(FuzzPipeline::AllPasses), size_t{0});
}

TEST_CASE("Fuzz - the production pipeline preserves the answer on a fixed seed range") {
    CHECK_EQ(run_seed_range(FuzzPipeline::Production), size_t{0});
}

// optimize_module's function pipeline (with GVN-PRE and WBE ahead of it).
// Its allocation sinking once broke seed 27 and 1 in 12 programs past it.
TEST_CASE("Fuzz - the legacy optimize_module pipeline preserves the answer on a fixed seed range") {
    CHECK_EQ(run_seed_range(FuzzPipeline::Legacy), size_t{0});
}

TEST_CASE("Fuzz - every pipeline step name is unique and skip names are checked") {
    for (FuzzPipeline p : {FuzzPipeline::AllPasses, FuzzPipeline::Production, FuzzPipeline::Legacy}) {
        const std::vector<std::string> names = fuzz_pipeline(p).names();
        CHECK(!names.empty());
        CHECK(unknown_skip_names(p, {"cfg_simplify", "no_such_pass"}) == std::vector<std::string>{"no_such_pass"});
    }
}
