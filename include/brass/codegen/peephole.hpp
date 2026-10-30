#pragma once

#include <brass/codegen/lir.hpp>
#include <array>
#include <cstddef>
#include <cstdint>

namespace brass::codegen {

struct PeepholeStats {
    size_t redundant_moves_eliminated = 0;
    size_t load_after_store_eliminated = 0;
    size_t dead_moves_eliminated = 0;
    size_t arithmetic_simplified = 0;
    size_t branches_simplified = 0;
    size_t bitcasts_through_slots = 0;
    size_t reloads_reused = 0;

    constexpr size_t total_optimizations() const noexcept {
        return redundant_moves_eliminated + load_after_store_eliminated +
               dead_moves_eliminated + arithmetic_simplified + branches_simplified + bitcasts_through_slots +
               reloads_reused;
    }
};

class PeepholeOptimizer {
public:
    explicit PeepholeOptimizer(LirFunction& fn);

    PeepholeStats run();

    // What a scratch register is known to hold: a spill slot's value as one
    // reload left it (reuse_scratch_reloads).
    struct ScratchHeld {
        PReg reg;
        int32_t slot = -1;  // -1: nothing known
        LirOpcode opcode = LirOpcode::Nop;
        uint8_t size = 0;
    };

private:
    LirFunction& fn_;
    PeepholeStats stats_;

    bool run_pass();
    bool optimize_block(LirBlock& block, size_t block_index);
    bool eliminate_redundant_moves(LirBlock& block);
    bool propagate_copies(LirBlock& block);
    bool eliminate_load_after_store(LirBlock& block);
    bool eliminate_dead_moves(LirBlock& block);
    bool bitcast_through_slots(LirBlock& block);
    // A spill reload into a scratch register that already holds that slot
    // (peephole_reload.cpp, which also lists the audited implicit writers).
    // Function-wide: a block entered only by falling through keeps what the
    // block before it left.
    bool reuse_scratch_reloads();
    bool reuse_scratch_reloads_block(LirBlock& block, std::array<ScratchHeld, 5>& held, PReg r11, PReg fp, PReg sp);
    bool simplify_arithmetic(LirBlock& block);
    bool simplify_branches(LirBlock& block, size_t block_index);

    static bool is_protected(const LirInst& inst) noexcept;
    static bool operands_equal(const LirOperand& a, const LirOperand& b) noexcept;
    static bool uses_register(const LirInst& inst, PReg reg) noexcept;
    static bool defines_register(const LirInst& inst, PReg reg) noexcept;
    static bool touches_memory(const LirInst& inst) noexcept;
    static bool reg_dead_from(const LirBlock& block, size_t from, PReg reg) noexcept;
    // The registers the rewriter and emitters use behind the LIR's back
    // (spill reloads, the spilled-def scratch, fabs masks, memory-to-memory
    // moves). A value in one of them does not survive to a later
    // instruction as far as the LIR can tell, so it is never forwarded.
    bool is_reserved_scratch(PReg reg) const noexcept;
};

PeepholeStats run_lir_peephole_optimizations(LirFunction& fn);

} // namespace brass::codegen
