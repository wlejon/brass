#pragma once

#include <brass/codegen/lir.hpp>
#include <cstddef>
#include <cstdint>

namespace brass::codegen {

struct PeepholeStats {
    size_t redundant_moves_eliminated = 0;
    size_t load_after_store_eliminated = 0;
    size_t dead_moves_eliminated = 0;
    size_t arithmetic_simplified = 0;
    size_t branches_simplified = 0;

    constexpr size_t total_optimizations() const noexcept {
        return redundant_moves_eliminated + load_after_store_eliminated +
               dead_moves_eliminated + arithmetic_simplified + branches_simplified;
    }
};

class PeepholeOptimizer {
public:
    explicit PeepholeOptimizer(LirFunction& fn);

    PeepholeStats run();

private:
    LirFunction& fn_;
    PeepholeStats stats_;

    bool run_pass();
    bool optimize_block(LirBlock& block, size_t block_index);
    bool eliminate_redundant_moves(LirBlock& block);
    bool eliminate_load_after_store(LirBlock& block);
    bool eliminate_dead_moves(LirBlock& block);
    bool simplify_arithmetic(LirBlock& block);
    bool simplify_branches(LirBlock& block, size_t block_index);

    static bool is_protected(const LirInst& inst) noexcept;
    static bool operands_equal(const LirOperand& a, const LirOperand& b) noexcept;
    static bool uses_register(const LirInst& inst, PReg reg) noexcept;
    static bool defines_register(const LirInst& inst, PReg reg) noexcept;
    static bool touches_memory(const LirInst& inst) noexcept;
};

PeepholeStats run_lir_peephole_optimizations(LirFunction& fn);

} // namespace brass::codegen
