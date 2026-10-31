#include "brass/core/env.hpp"
#include <brass/target/aarch64/aarch64_isel.hpp>
#include "aarch64_isel_conditions.hpp"
#include <cstdlib>
#include <unordered_map>

namespace brass::aarch64 {

using namespace brass::codegen;

// COMPARE REMATERIALIZATION, the AArch64 half of x64_isel_cmp.cpp's. GVN
// shares a comparison between every block it dominates (a guard's "is this
// an object" test in the hot block and the same test in the miss path's cache
// lookup become one value); with two users analyze_function's single-use rule
// cannot fuse it into either, so the hot block computes it as a 0/1 (cset),
// keeps the register alive into the miss path and tests it again at each user
// (cbnz / tst + csel). Flags never live across blocks, so the comparison is
// worth more as one cmp per user than as one register.
//
// A comparison qualifies when EVERY use of it is the condition of a br_if,
// guard or select. Each such user then emits it itself with
// emit_fused_compare, exactly as a single-use same-block one is emitted. That
// reads the same two SSA operands, which dominate every user because the
// comparison does, so it computes the same answer. emit_fused_compare never
// folds a load (it reads registers and immediates only), so no memory is
// re-read later. Excluded, as on x64:
//   - a GC root operand (gcref / tagged) can be relocated by a moving
//     collection between the definition and a later user, which rewrites the
//     root and not a non-root copy of it; only eq / ne of a root against an
//     immediate is kept, which relocation cannot change (an object stays an
//     object, its tag bits stay);
//   - float compares (the unordered cases are not every user's to handle),
//     vector compares, and narrow ones (widened first by the unfused
//     lowering only);
//   - two constant operands (the folder's to decide).
// An `and` under a rematerialized `== 0` is not fused into it: the single-use
// rule skips an `and` only beside a comparison it skipped itself, so here the
// `and` is lowered and its register tested at each user.
//
// BRASS_NO_CMP_REMAT=1 turns it off (the A/B seam, shared with x64).
void AArch64ISel::find_remat_compares(const Function& mir_fn) {
    remat_cmps_.clear();
    static const bool off = !brass::env_value("BRASS_NO_CMP_REMAT").empty();
    if (off) return;
    std::unordered_map<const Value*, uint32_t> consumer_uses;
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            const Opcode op = inst->opcode();
            if (op != Opcode::br_if && op != Opcode::guard && op != Opcode::select) continue;
            const Value* cond = inst->operand(0);
            if (cond && cond->is_instruction()) ++consumer_uses[cond];
        }
    }
    for (const auto& [v, n] : consumer_uses) {
        const Instruction* def = v->defining_instruction();
        if (!def || skipped_insts_.count(def) || !is_comparison(def->opcode()) || narrow_compare(*def)) continue;
        if (def->operand_count() < 2) continue;
        const auto it = use_count_.find(v);
        if (it == use_count_.end() || it->second != n) continue;
        const Value* lhs = def->operand(0);
        const Value* rhs = def->operand(1);
        if (!lhs || !rhs) continue;
        if (lhs->type().is_float() || rhs->type().is_float() || lhs->type().is_vector() || rhs->type().is_vector()) {
            continue;
        }
        const bool lhs_imm = get_imm_int_info(lhs).is_imm;
        const bool rhs_imm = get_imm_int_info(rhs).is_imm;
        if (lhs_imm && rhs_imm) continue;
        const bool equality = def->opcode() == Opcode::eq || def->opcode() == Opcode::ne;
        if (lhs->type().is_gc_root() && !(equality && rhs_imm)) continue;
        if (rhs->type().is_gc_root() && !(equality && lhs_imm)) continue;
        remat_cmps_.insert(def);
        skipped_insts_.insert(def);
    }
}

bool AArch64ISel::fused_into(const Instruction* cmp, const Instruction& user) const {
    if (!cmp || !is_comparison(cmp->opcode()) || !skipped_insts_.count(cmp)) return false;
    return cmp->parent() == user.parent() || remat_cmps_.count(cmp) != 0;
}

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
