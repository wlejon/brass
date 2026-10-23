#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/codegen/lir.hpp>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace brass::x64 {
using LIRBasicBlock = ::brass::codegen::LirBlock;
using LirBlock = ::brass::codegen::LirBlock;
} // namespace brass::x64

namespace brass::codegen {
using LIRBasicBlock = ::brass::codegen::LirBlock;
} // namespace brass::codegen

namespace brass::verify {

enum class EquivStatus : uint8_t {
    ProvenEquivalent,
    Counterexample,
    Unknown,
    Unsupported,

    // Convenience aliases
    Equivalent = ProvenEquivalent,
    CounterexampleFound = Counterexample,
    BudgetExceeded = Unknown,
    SkippedUnsupported = Unsupported
};

inline std::ostream& operator<<(std::ostream& os, EquivStatus s) {
    switch (s) {
        case EquivStatus::ProvenEquivalent: return os << "ProvenEquivalent";
        case EquivStatus::Counterexample:   return os << "Counterexample";
        case EquivStatus::Unknown:          return os << "Unknown";
        case EquivStatus::Unsupported:      return os << "Unsupported";
    }
    return os << "Unknown";
}

struct EquivOptions {
    uint32_t loop_unroll_bound = 4;
    uint32_t max_paths = 64;
    uint32_t max_sat_conflicts = 10000;
    bool check_guards = true;
    bool check_memory = true;
};

struct EquivResult {
    EquivStatus status = EquivStatus::ProvenEquivalent;
    std::vector<RuntimeValue> counterexample;
    std::string diagnostic;

    constexpr bool is_equivalent() const noexcept {
        return status == EquivStatus::ProvenEquivalent;
    }
    constexpr bool is_counterexample() const noexcept {
        return status == EquivStatus::Counterexample;
    }
};

struct LirEquivResult {
    EquivStatus status = EquivStatus::ProvenEquivalent;
    std::string diagnostic;

    constexpr bool is_equivalent() const noexcept {
        return status == EquivStatus::ProvenEquivalent;
    }
    constexpr bool is_counterexample() const noexcept {
        return status == EquivStatus::Counterexample;
    }
};

/// Verify semantic equivalence between two MIR functions (`before` and `after`).
/// Checks return value equivalence, guard failure soundness, division/modulo trap soundness,
/// and observable memory store equivalence across all symbolic bitvector inputs.
EquivResult verify_mir_equivalence(
    const Function& before,
    const Function& after,
    const EquivOptions& opts = {}
);

/// Verify semantic equivalence across all matching functions in two MIR modules.
EquivResult verify_mir_module_equivalence(
    const Module& before,
    const Module& after,
    const EquivOptions& opts = {}
);

/// Export a self-contained SMT-LIB2 (`QF_BV`) equivalence query for two MIR functions.
std::string export_mir_equivalence_smt2(
    const Function& before,
    const Function& after,
    const EquivOptions& opts = {}
);

/// Verify semantic equivalence between two x64 LIR basic blocks (`before` and `after`),
/// modeling all 16 GPRs (including 32-bit zero-extension to 64 bits), 16 XMM registers,
/// spill slots, memory reads/writes, RFLAGS (ZF, SF, OF, CF), and branch targets.
LirEquivResult verify_x64_lir_block_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    uint16_t live_out_gpr_mask = 0xFFFF,
    uint16_t live_out_xmm_mask = 0xFFFF,
    bool check_flags = true,
    uint32_t fallthrough_block_id = UINT32_MAX
);

/// Verify semantic equivalence across all basic blocks of two x64 LIR functions.
LirEquivResult verify_x64_lir_function_equivalence(
    const codegen::LirFunction& before,
    const codegen::LirFunction& after
);

/// Clone an x64 LIR function, run PeepholeOptimizer on the clone, and verify equivalence.
LirEquivResult verify_x64_peephole(
    const codegen::LirFunction& fn
);

inline LirEquivResult verify_lir_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    bool check_flags = true
) {
    return verify_x64_lir_block_equivalence(before, after, 0xFFFF, 0xFFFF, check_flags);
}

} // namespace brass::verify

namespace brass {
using verify::EquivOptions;
using verify::EquivResult;
using verify::EquivStatus;
using verify::LirEquivResult;
using verify::export_mir_equivalence_smt2;
using verify::verify_lir_equivalence;
using verify::verify_mir_equivalence;
using verify::verify_mir_module_equivalence;
using verify::verify_x64_lir_block_equivalence;
using verify::verify_x64_lir_function_equivalence;
using verify::verify_x64_peephole;
} // namespace brass

#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
