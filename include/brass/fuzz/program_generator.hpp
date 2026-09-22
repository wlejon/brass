#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <cstdint>
#include <string_view>

namespace brass::fuzz {

struct ProgramGeneratorOptions {
    // Nesting depth of regions (branches and loops inside each other).
    uint32_t max_depth = 3;
    // Statements in the whole entry function, across all nesting levels.
    uint32_t max_statements = 36;
    // Upper bound on the product of the trip counts of nested loops, which
    // keeps every program well inside the interpreter's instruction cap.
    uint32_t max_iteration_product = 256;
    bool enable_memory = true;
    bool enable_gc = true;
    bool enable_calls = true;
    bool enable_f64 = true;
    bool enable_vectors = true;
    // Generate `INT_MIN / -1` and `INT_MIN % -1` (defined to wrap). Off, a
    // divisor that would be -1 is replaced, so a codegen that traps on it
    // does not hide everything else.
    bool enable_div_overflow = true;
    // Apply the SSA-preserving IrMutator transforms on top.
    bool apply_mutations = true;
};

/// Generates a module-level program: a `fuzz_fn(i64, i64) -> i64` entry that
/// returns a checksum of everything it computed, plus small helper functions
/// it calls. Programs are verifier-clean, deterministic, always terminate and
/// never fold an address into the result.
class ProgramGenerator {
public:
    explicit ProgramGenerator(const ProgramGeneratorOptions& options = {}) : options_(options) {}

    /// Adds the program's functions to `mod` and returns the entry function.
    Function* generate(Module& mod, std::string_view name, uint64_t seed);

    const ProgramGeneratorOptions& options() const noexcept { return options_; }

private:
    ProgramGeneratorOptions options_;
};

} // namespace brass::fuzz
