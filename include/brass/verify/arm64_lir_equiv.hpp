#pragma once

#include <brass/verify/smt_equiv.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <cstdint>
#include <string>

namespace brass::aarch64 {
using LIRBasicBlock = ::brass::codegen::LirBlock;
using LirBlock = ::brass::codegen::LirBlock;
} // namespace brass::aarch64

namespace brass::verify {

/// Default live-out GPR mask for AArch64: all 31 general-purpose registers X0..X30.
inline constexpr uint32_t kArm64AllGprsMask = 0x7FFFFFFFu;

/// Bit mask including X0..X30 plus SP (code 31).
inline constexpr uint32_t kArm64AllGprsAndSpMask = 0xFFFFFFFFu;

/// Default live-out SIMD/FP mask for AArch64: all 32 vector registers V0..V31.
inline constexpr uint32_t kArm64AllVecMask = 0xFFFFFFFFu;

/// Specification of an AArch64 conditional compare (`ccmp` / `ccmn`) step:
/// If `cond` holds under the incoming NZCV state, performs `cmp(lhs, rhs)` (or `cmn(lhs, rhs)`
/// when `is_cmn` is true) at 32-bit (`!is_64`) or 64-bit (`is_64`) width; otherwise sets
/// `NZCV` directly to the 4-bit immediate `nzcv_imm` (`bit 3 = N, bit 2 = Z, bit 1 = C, bit 0 = V`).
struct Arm64CcmpStep {
    bool is_64 = true;
    bool is_cmn = false;
    codegen::LirOperand lhs;
    codegen::LirOperand rhs;
    uint8_t nzcv_imm = 0; // 4-bit #nzcv immediate (N:3, Z:2, C:1, V:0)
    aarch64::Condition cond = aarch64::Condition::EQ;
};

/// Verify semantic equivalence between two AArch64 LIR basic blocks (`before` and `after`),
/// modeling:
/// - All 31 AArch64 GPRs (`X0..X30` as 64-bit `BV64`, plus `SP` and `XZR`), enforcing the
///   ARMv8-A architectural rule that any 32-bit `W`-register write unconditionally zero-extends
///   bits `[63:32]` of `Xn`,
/// - All 32 SIMD/FP `V0..V31` 128-bit registers (including scalar `S`/`D` upper-bit zeroing),
/// - Virtual registers, stack/spill slots, local frame slots, and memory loads/stores,
/// - `NZCV` condition flags (`Negative`, `Zero`, `Carry` with ARM64 not-borrow subtraction
///   polarity, and signed `oVerflow` for `adds`/`subs`/`cmp`/`cmn`/`ands`/`tst`/`fcmp`),
/// - Conditional select (`csel` / `Cmovcc`), conditional set (`cset` / `Setcc`), and
///   conditional/unconditional branches (`b.cond` / `Jcc`, `b` / `Jmp`).
LirEquivResult verify_arm64_lir_block_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    uint32_t live_out_gpr_mask = kArm64AllGprsMask,
    uint32_t live_out_vec_mask = kArm64AllVecMask,
    bool check_flags = true,
    uint32_t fallthrough_block_id = UINT32_MAX
);

inline LirEquivResult verify_aarch64_lir_block_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    uint32_t live_out_gpr_mask = kArm64AllGprsMask,
    uint32_t live_out_vec_mask = kArm64AllVecMask,
    bool check_flags = true,
    uint32_t fallthrough_block_id = UINT32_MAX
) {
    return verify_arm64_lir_block_equivalence(
        before, after, live_out_gpr_mask, live_out_vec_mask, check_flags, fallthrough_block_id);
}

inline LirEquivResult verify_arm64_lir_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    bool check_flags = true
) {
    return verify_arm64_lir_block_equivalence(
        before, after, kArm64AllGprsMask, kArm64AllVecMask, check_flags, UINT32_MAX);
}

inline LirEquivResult verify_aarch64_lir_equivalence(
    const codegen::LirBlock& before,
    const codegen::LirBlock& after,
    bool check_flags = true
) {
    return verify_arm64_lir_block_equivalence(
        before, after, kArm64AllGprsMask, kArm64AllVecMask, check_flags, UINT32_MAX);
}

/// Verify every basic block of two AArch64 LIR functions (`before` and `after`),
/// automatically computing per-block fallthrough successors and whether `NZCV` is live
/// at the end of each block.
LirEquivResult verify_arm64_lir_function_equivalence(
    const codegen::LirFunction& before,
    const codegen::LirFunction& after
);

/// Run `PeepholeOptimizer` on a clone of an AArch64 `LirFunction` and symbolically
/// verify that every optimized basic block is bit-exact equivalent to the original.
LirEquivResult verify_arm64_peephole(const codegen::LirFunction& fn);

/// Symbolically verify that an AArch64 `ccmp` / `ccmn` sequence after an initial compare
/// is bit-exact equivalent to a boolean conjunction/disjunction of two comparisons
/// evaluated under `final_cond`.
LirEquivResult verify_arm64_ccmp_chain(
    bool first_is_64,
    bool first_is_cmn,
    const codegen::LirOperand& first_lhs,
    const codegen::LirOperand& first_rhs,
    const Arm64CcmpStep& ccmp_step,
    aarch64::Condition final_cond,
    bool expect_and_conjunction,
    aarch64::Condition first_cond,
    aarch64::Condition second_cond
);

} // namespace brass::verify

namespace brass {
using verify::Arm64CcmpStep;
using verify::kArm64AllGprsAndSpMask;
using verify::kArm64AllGprsMask;
using verify::kArm64AllVecMask;
using verify::verify_aarch64_lir_block_equivalence;
using verify::verify_aarch64_lir_equivalence;
using verify::verify_arm64_ccmp_chain;
using verify::verify_arm64_lir_block_equivalence;
using verify::verify_arm64_lir_equivalence;
using verify::verify_arm64_lir_function_equivalence;
using verify::verify_arm64_peephole;
} // namespace brass
