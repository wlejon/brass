#pragma once

#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>
#include <functional>

namespace brass::detail {

// Strength-reduce an i32 constant non-power-of-two `sdiv`/`udiv`/`smod`/`umod`
// at `cur` using Granlund-Montgomery / Hacker's Delight magic multiplication
// in portable MIR (`zext_i64`/`sext_i64` + 64-bit `mul` + shifts + `trunc_i32`).
// Returns the replacement Value* placed before `cur` (nullptr if not eligible).
Value* try_lower_const_div_i32_mir(
    Module& mod,
    Function& fn,
    Instruction* cur,
    Value* lhs,
    Opcode op,
    int64_t div_imm,
    const std::function<bool()>& prove_lhs_non_neg);

} // namespace brass::detail
