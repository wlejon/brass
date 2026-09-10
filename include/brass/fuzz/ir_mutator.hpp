#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/builder.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <memory>

namespace brass::fuzz {

/// Fast, deterministic pseudo-random number generator parameterized by a 64-bit seed.
/// Uses SplitMix64 / XorShift algorithm for high throughput and reproducibility.
class FuzzRng {
public:
    explicit FuzzRng(uint64_t seed = 0x853c49e6748fea9bULL) noexcept;

    void seed(uint64_t s) noexcept;
    uint64_t get_seed() const noexcept { return state_; }

    uint64_t next_u64() noexcept;
    uint32_t next_u32() noexcept;
    int64_t next_i64() noexcept;
    int32_t next_i32() noexcept;
    double next_f64() noexcept; // Uniform in [0, 1)
    float next_f32() noexcept;  // Uniform in [0, 1)

    uint64_t range_u64(uint64_t min_v, uint64_t max_v) noexcept;
    int64_t range_i64(int64_t min_v, int64_t max_v) noexcept;
    size_t pick_index(size_t size) noexcept;
    bool coin_flip(double p = 0.5) noexcept;

    // Extreme and boundary values
    int32_t boundary_i32() noexcept;
    int64_t boundary_i64() noexcept;
    float boundary_f32() noexcept;
    double boundary_f64() noexcept;

private:
    uint64_t state_;
};

struct IrGeneratorOptions {
    size_t min_blocks = 2;
    size_t max_blocks = 8;
    size_t min_instructions = 10;
    size_t max_instructions = 40;
    bool enable_vectors = true;
    bool enable_loops = true;
    bool enable_diamonds = true;
    bool enable_switches = true;
    bool enable_exceptions = true;
    bool enable_memory = true;
    Type return_type = Type::i64();
    std::vector<Type> param_types = {Type::i64(), Type::i64()};
};

/// Builds valid, SSA-conformant MIR functions covering arithmetic, vectors, control flow,
/// exception handling, and boundary values.
class IrGenerator {
public:
    explicit IrGenerator(const IrGeneratorOptions& options = {}) : options_(options) {}

    Function* generate(Module& mod, std::string_view name, uint64_t seed);

    const IrGeneratorOptions& options() const noexcept { return options_; }
    void set_options(const IrGeneratorOptions& opts) noexcept { options_ = opts; }

private:
    IrGeneratorOptions options_;
};

/// Applies SSA-preserving mutation passes to a Function.
class IrMutator {
public:
    /// Mutate constant immediate values (replace with boundary values, flip bits, delta +/-1).
    static bool mutate_constant_immediates(Function& fn, FuzzRng& rng);

    /// Swap operands of commutative operations (add, mul, and, or, xor, eq, ne, vadd, etc.).
    static bool swap_commutative_operands(Function& fn, FuzzRng& rng);

    /// Replace opcodes with type-compatible alternatives (add <-> sub, and <-> or, slt <-> sgt).
    static bool replace_opcodes(Function& fn, FuzzRng& rng);

    /// Split a basic block into two blocks connected by an unconditional jump.
    static bool split_basic_blocks(Function& fn, FuzzRng& rng);

    /// Inject speculative guards (guard %cond, exit_label).
    static bool inject_speculative_guards(Function& fn, FuzzRng& rng);

    /// Applies one or more random mutation passes, ensuring verify_function still holds.
    static bool mutate_function(Function& fn, FuzzRng& rng);
};

} // namespace brass::fuzz
