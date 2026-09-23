#pragma once

// Condition-code helpers shared by the AArch64 selector's comparison, branch,
// select and guard lowering. LIR carries x64 condition codes; the emitter maps
// them with to_aarch64_cond (B -> LO, A -> HI, ...), which is exact after an
// integer compare and after an fcmp for every condition an ordered float
// comparison is lowered to here (see emit_fused_compare).

#include <brass/mir/instruction.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <utility>

namespace brass::aarch64 {

using LirCond = brass::x64::Condition;

// {integer condition, float condition} under which `op` is true.
constexpr std::pair<LirCond, LirCond> comparison_conditions(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq:  return {LirCond::E, LirCond::E};
        case Opcode::ne:  return {LirCond::NE, LirCond::NE};
        case Opcode::slt: return {LirCond::L, LirCond::B};
        case Opcode::ult: return {LirCond::B, LirCond::B};
        case Opcode::sle: return {LirCond::LE, LirCond::BE};
        case Opcode::ule: return {LirCond::BE, LirCond::BE};
        case Opcode::sgt: return {LirCond::G, LirCond::A};
        case Opcode::ugt: return {LirCond::A, LirCond::A};
        case Opcode::sge: return {LirCond::GE, LirCond::AE};
        case Opcode::uge: return {LirCond::AE, LirCond::AE};
        default: return {LirCond::None, LirCond::None};
    }
}

// The condition that holds for (b, a) when `cond` holds for (a, b).
constexpr LirCond swap_relational_condition(LirCond cond) noexcept {
    switch (cond) {
        case LirCond::E:   return LirCond::E;
        case LirCond::NE:  return LirCond::NE;
        case LirCond::L:   return LirCond::G;
        case LirCond::LE:  return LirCond::GE;
        case LirCond::G:   return LirCond::L;
        case LirCond::GE:  return LirCond::LE;
        case LirCond::B:   return LirCond::A;
        case LirCond::BE:  return LirCond::AE;
        case LirCond::A:   return LirCond::B;
        case LirCond::AE:  return LirCond::BE;
        default: return cond;
    }
}

// `tst x, x` stands in for `cmp x, #0` only where the condition does not
// read C: ANDS clears C, while a compare with zero sets it (no borrow). The
// unsigned conditions have to see the real compare.
constexpr bool zero_test_matches_compare(LirCond cond) noexcept {
    switch (cond) {
        case LirCond::B:
        case LirCond::BE:
        case LirCond::A:
        case LirCond::AE:
            return false;
        default:
            return true;
    }
}

} // namespace brass::aarch64
