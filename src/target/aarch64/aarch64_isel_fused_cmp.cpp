#include <brass/target/aarch64/aarch64_isel.hpp>
#include "aarch64_isel_conditions.hpp"

namespace brass::aarch64 {

using namespace brass::codegen;

// Sets the flags for a comparison consumed by the br_if / select / guard
// after it and returns the condition under which the comparison is true.
// The comparison itself may or may not have been lowered too; this reads
// only its operands.
LirCond AArch64ISel::emit_fused_compare(const Instruction& cmp_inst, LirBlock& lir_bb) {
    const Opcode cmp_op = cmp_inst.opcode();
    const auto [gpr_c, float_c] = comparison_conditions(cmp_op);
    const Value* lhs = cmp_inst.operand(0);
    const Value* rhs = cmp_inst.operand(1);

    auto flags = [&](LirOpcode op, LirOperand a, LirOperand b) {
        auto li = std::make_unique<LirInst>(op);
        li->add_use(std::move(a));
        li->add_use(std::move(b));
        lir_bb.append_inst(std::move(li));
    };

    if (lhs->type().is_float()) {
        const uint8_t sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
        const LirOpcode ucomi = sz == 4 ? LirOpcode::Ucomiss : LirOpcode::Ucomisd;
        // After fcmp an unordered result sets C and V, so HI / HS (x64 A /
        // AE) would be true for NaN. a > b is b < a (LO: false when
        // unordered), a >= b is b <= a (LS: likewise).
        if (float_c == LirCond::A || float_c == LirCond::AE) {
            flags(ucomi, LirOperand::vreg(get_vreg(rhs), sz), LirOperand::vreg(get_vreg(lhs), sz));
            return float_c == LirCond::A ? LirCond::B : LirCond::BE;
        }
        flags(ucomi, LirOperand::vreg(get_vreg(lhs), sz), LirOperand::vreg(get_vreg(rhs), sz));
        return float_c;
    }

    uint8_t sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
    if (sz == 0) sz = 8;
    const LirOpcode cmp_op32 = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
    const LirOpcode test_op = (sz == 4) ? LirOpcode::Test32 : LirOpcode::Test;
    const ImmIntInfo rhs_imm = get_imm_int_info(rhs);
    const ImmIntInfo lhs_imm = get_imm_int_info(lhs);
    const bool rhs_zero = rhs_imm.is_imm && rhs_imm.val == 0;
    const bool lhs_zero = lhs_imm.is_imm && lhs_imm.val == 0;

    // (a & b) ==/!= 0 as one tst when the `and` was left to this consumer.
    if ((cmp_op == Opcode::eq || cmp_op == Opcode::ne) && (rhs_zero || lhs_zero)) {
        const Value* non_zero = rhs_zero ? lhs : rhs;
        const Instruction* and_inst = non_zero->is_instruction() ? non_zero->defining_instruction() : nullptr;
        if (and_inst && and_inst->opcode() == Opcode::and_ && skipped_insts_.count(and_inst)) {
            const Value* a = and_inst->operand(0);
            const Value* b = and_inst->operand(1);
            const ImmIntInfo imm_b = get_imm_int_info(b);
            const ImmIntInfo imm_a = get_imm_int_info(a);
            if (imm_b.is_imm && imm_b.fits_i32) {
                flags(test_op, LirOperand::vreg(get_vreg(a), sz), LirOperand::imm(imm_b.val, sz));
            } else if (imm_a.is_imm && imm_a.fits_i32) {
                flags(test_op, LirOperand::vreg(get_vreg(b), sz), LirOperand::imm(imm_a.val, sz));
            } else {
                flags(test_op, LirOperand::vreg(get_vreg(a), sz), LirOperand::vreg(get_vreg(b), sz));
            }
            return cmp_op == Opcode::eq ? LirCond::E : LirCond::NE;
        }
    }

    if (rhs_zero && zero_test_matches_compare(gpr_c)) {
        const VReg r = get_vreg(lhs);
        flags(test_op, LirOperand::vreg(r, sz), LirOperand::vreg(r, sz));
        return gpr_c;
    }
    if (lhs_zero && zero_test_matches_compare(gpr_c)) {
        const VReg r = get_vreg(rhs);
        flags(test_op, LirOperand::vreg(r, sz), LirOperand::vreg(r, sz));
        return swap_relational_condition(gpr_c);
    }
    if (rhs_imm.is_imm && rhs_imm.fits_i32) {
        flags(cmp_op32, LirOperand::vreg(get_vreg(lhs), sz), LirOperand::imm(rhs_imm.val, sz));
        return gpr_c;
    }
    if (lhs_imm.is_imm && lhs_imm.fits_i32) {
        flags(cmp_op32, LirOperand::vreg(get_vreg(rhs), sz), LirOperand::imm(lhs_imm.val, sz));
        return swap_relational_condition(gpr_c);
    }
    flags(cmp_op32, LirOperand::vreg(get_vreg(lhs), sz), LirOperand::vreg(get_vreg(rhs), sz));
    return gpr_c;
}

} // namespace brass::aarch64
