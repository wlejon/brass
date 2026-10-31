#include "brass/core/env.hpp"
#include <brass/target/x64/x64_isel.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brass::x64 {

using namespace brass::codegen;

static constexpr Condition swap_relational_condition(Condition cond) noexcept {
    switch (cond) {
        case Condition::E:   return Condition::E;
        case Condition::NE:  return Condition::NE;
        case Condition::L:   return Condition::G;
        case Condition::LE:  return Condition::GE;
        case Condition::G:   return Condition::L;
        case Condition::GE:  return Condition::LE;
        case Condition::B:   return Condition::A;
        case Condition::BE:  return Condition::AE;
        case Condition::A:   return Condition::B;
        case Condition::AE:  return Condition::BE;
        default: return cond;
    }
}

bool X64ISel::fused_float_compare(Opcode cmp, Condition& cond, bool& swap_operands) noexcept {
    switch (cmp) {
        case Opcode::slt: cond = Condition::A;  swap_operands = true;  return true;
        case Opcode::sle: cond = Condition::AE; swap_operands = true;  return true;
        case Opcode::sgt: cond = Condition::A;  swap_operands = false; return true;
        case Opcode::sge: cond = Condition::AE; swap_operands = false; return true;
        default: return false;
    }
}

bool X64ISel::comparison_fusible(const Instruction& cmp) noexcept {
    if (!is_comparison(cmp.opcode()) || cmp.operand_count() < 2 || !cmp.operand(0)) return false;
    const Type t = cmp.operand(0)->type();
    if (t.is_vector()) return false;
    // A narrow comparison widens its operands first (x64_isel_narrow.cpp),
    // which only the unfused lowering does.
    if (narrow_compare(cmp)) return false;
    if (!t.is_float()) return true;
    if (is_self_nan_test(cmp)) return true;
    Condition cond = Condition::None;
    bool swap = false;
    return fused_float_compare(cmp.opcode(), cond, swap);
}

// `ne x, x` / `eq x, x` on a float: the NaN test, which is ucomis x, x and the
// parity flag alone (unordered), where a general float eq/ne needs two flags.
// A NaN-boxing front end canonicalizes every float it boxes with one.
bool X64ISel::is_self_nan_test(const Instruction& cmp) noexcept {
    return (cmp.opcode() == Opcode::eq || cmp.opcode() == Opcode::ne) && cmp.operand_count() == 2 &&
           cmp.operand(0) && cmp.operand(0) == cmp.operand(1) && cmp.operand(0)->type().is_float();
}

void X64ISel::append_fused_float_compare(const Instruction& cmp, LirBlock& lir_bb, Condition& cond) {
    bool swap = false;
    if (is_self_nan_test(cmp)) {
        cond = cmp.opcode() == Opcode::ne ? Condition::P : Condition::NP;
    } else if (!fused_float_compare(cmp.opcode(), cond, swap)) {
        throw std::logic_error("x64 isel: floating-point " + std::string(opcode_name(cmp.opcode())) +
                               " cannot be fused into its user");
    }
    const Value* a = swap ? cmp.operand(1) : cmp.operand(0);
    const Value* b = swap ? cmp.operand(0) : cmp.operand(1);
    uint8_t sz = static_cast<uint8_t>(cmp.operand(0)->type().size_in_bytes());
    if (sz == 0) sz = 8;
    auto ucomi = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Ucomiss : LirOpcode::Ucomisd);
    ucomi->add_use(LirOperand::vreg(get_vreg(a), sz));
    ucomi->add_use(LirOperand::vreg(get_vreg(b), sz));
    lir_bb.append_inst(std::move(ucomi));
}

// COMPARE REMATERIALIZATION. GVN shares a comparison between every block it
// dominates: a guard's "is this an object" test in the hot block and the same
// test recomputed by the miss path's cache lookup become one value. With two
// users it can no longer be folded into either, so the hot block computes it
// as a 0/1 (setcc, movzx), keeps the register alive into the miss path, and
// tests it again at its own select (bronze's keyed-shape guard: setb, movzbl,
// test, cmovne where one cmovb would do). Flags never live across blocks, so
// the comparison is worth more as two cmp instructions than as one register.
//
// A comparison qualifies when EVERY use of it is the condition of a br_if or
// select, or a leaf of an and-chain (x64_isel_and_chain.cpp). Each such user
// then emits the comparison itself, exactly as a single-use same-block one
// already is emitted at its user. It reads the same two SSA operands, which
// dominate every user because the comparison does, so it computes the same
// answer, with two exceptions this excludes:
//   - a load folded into the compare would re-read memory at a later point,
//     so a rematerialized compare reads every operand from a register
//     (fused_into's callers check remat_cmps_ before folding a load);
//   - a GC root operand (gcref / tagged) can be relocated between the
//     definition and a later user by a moving collection, which rewrites
//     the root and not a non-root copy of it. Only eq / ne of a root against
//     an immediate is kept, which relocation cannot change (an object stays
//     an object, its tag bits stay).
// Integer comparisons only: a float compare's flags need parity handling
// that the users do not all share.
//
// BRASS_NO_CMP_REMAT=1 turns it off (the A/B seam).
void X64ISel::find_remat_compares(const Function& mir_fn) {
    remat_cmps_.clear();
    static const bool off = !brass::env_value("BRASS_NO_CMP_REMAT").empty();
    if (off) return;
    std::unordered_map<const Value*, uint32_t> consumer_uses;
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            const Opcode op = inst->opcode();
            if (op != Opcode::br_if && op != Opcode::select) continue;
            if (op == Opcode::br_if && and_chains_.count(inst)) continue;
            const Value* cond = inst->operand(0);
            if (cond && cond->is_instruction()) ++consumer_uses[cond];
        }
    }
    for (const auto& [br, leaves] : and_chains_) {
        for (const Value* leaf : leaves) ++consumer_uses[leaf];
    }
    for (const auto& [v, n] : consumer_uses) {
        const Instruction* def = v->defining_instruction();
        if (!def || skipped_insts_.count(def) || !is_comparison(def->opcode())) continue;
        const auto it = use_count_.find(v);
        if (it == use_count_.end() || it->second != n) continue;
        if (!comparison_fusible(*def)) continue;
        const Value* lhs = def->operand(0);
        const Value* rhs = def->operand(1);
        if (!lhs || !rhs || lhs->type().is_float() || rhs->type().is_float()) continue;
        const bool lhs_imm = get_imm_int_info(lhs).is_imm;
        const bool rhs_imm = get_imm_int_info(rhs).is_imm;
        // Two constants are the folder's to decide, and which one the users
        // would take as the immediate is not worth pinning down here.
        if (lhs_imm && rhs_imm) continue;
        const bool equality = def->opcode() == Opcode::eq || def->opcode() == Opcode::ne;
        if (lhs->type().is_gc_root() && !(equality && rhs_imm)) continue;
        if (rhs->type().is_gc_root() && !(equality && lhs_imm)) continue;
        remat_cmps_.insert(def);
        skipped_insts_.insert(def);
    }
}

bool X64ISel::fused_into(const Instruction* cmp, const Instruction& user) const {
    if (!cmp || !is_comparison(cmp->opcode()) || !skipped_insts_.count(cmp)) return false;
    return cmp->parent() == user.parent() || remat_cmps_.count(cmp) != 0;
}

void X64ISel::lower_comparison(
    const Instruction& inst,
    LirBlock& lir_bb,
    Condition cond,
    Condition float_cond
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    VReg op1 = get_vreg(op1_val);
    uint8_t dst_sz = dst.size;

    if (op0.is_xmm()) {
        uint8_t sz = op0.size;
        LirOpcode ucomi_op = (sz == 4) ? LirOpcode::Ucomiss : LirOpcode::Ucomisd;
        const bool fuse_op1_load =
            op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst);
        if (!fuse_op1_load && (float_cond == Condition::B || float_cond == Condition::BE)) {
            std::swap(op0, op1);
            std::swap(op0_val, op1_val);
            float_cond = (float_cond == Condition::B) ? Condition::A : Condition::AE;
        }

        auto ucomi = std::make_unique<LirInst>(ucomi_op);
        ucomi->add_use(LirOperand::vreg(op0, sz));
        if (fuse_op1_load) {
            ucomi->add_use(get_load_mem_operand(op1_val->defining_instruction()));
        } else {
            ucomi->add_use(LirOperand::vreg(op1, sz));
        }
        lir_bb.append_inst(std::move(ucomi));

        if (float_cond == Condition::E) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = Condition::E;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_np = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_np->condition = Condition::NP;
            setcc_np->add_def(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(setcc_np));

            auto movzx_dst = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_dst->add_def(LirOperand::vreg(dst, dst_sz));
            movzx_dst->add_use(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(movzx_dst));

            auto movzx_tmp = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_tmp->add_def(LirOperand::vreg(tmp, dst_sz));
            movzx_tmp->add_use(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(movzx_tmp));

            auto and_inst = std::make_unique<LirInst>((dst_sz == 4) ? LirOpcode::And32 : LirOpcode::And);
            and_inst->add_def(LirOperand::vreg(dst, dst_sz));
            and_inst->add_use(LirOperand::vreg(dst, dst_sz));
            and_inst->add_use(LirOperand::vreg(tmp, dst_sz));
            and_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(and_inst));
        } else if (float_cond == Condition::NE) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = Condition::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_p = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_p->condition = Condition::P;
            setcc_p->add_def(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(setcc_p));

            auto movzx_dst = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_dst->add_def(LirOperand::vreg(dst, dst_sz));
            movzx_dst->add_use(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(movzx_dst));

            auto movzx_tmp = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_tmp->add_def(LirOperand::vreg(tmp, dst_sz));
            movzx_tmp->add_use(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(movzx_tmp));

            auto or_inst = std::make_unique<LirInst>((dst_sz == 4) ? LirOpcode::Or32 : LirOpcode::Or);
            or_inst->add_def(LirOperand::vreg(dst, dst_sz));
            or_inst->add_use(LirOperand::vreg(dst, dst_sz));
            or_inst->add_use(LirOperand::vreg(tmp, dst_sz));
            or_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(or_inst));
        } else if (float_cond == Condition::B || float_cond == Condition::BE) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = float_cond;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_np = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_np->condition = Condition::NP;
            setcc_np->add_def(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(setcc_np));

            auto movzx_dst = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_dst->add_def(LirOperand::vreg(dst, dst_sz));
            movzx_dst->add_use(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(movzx_dst));

            auto movzx_tmp = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx_tmp->add_def(LirOperand::vreg(tmp, dst_sz));
            movzx_tmp->add_use(LirOperand::vreg(tmp, 1));
            lir_bb.append_inst(std::move(movzx_tmp));

            auto and_inst = std::make_unique<LirInst>((dst_sz == 4) ? LirOpcode::And32 : LirOpcode::And);
            and_inst->add_def(LirOperand::vreg(dst, dst_sz));
            and_inst->add_use(LirOperand::vreg(dst, dst_sz));
            and_inst->add_use(LirOperand::vreg(tmp, dst_sz));
            and_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(and_inst));
        } else {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = float_cond;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
            movzx->add_def(LirOperand::vreg(dst, dst_sz));
            movzx->add_use(LirOperand::vreg(dst, 1));
            movzx->mir_origin = &inst;
            lir_bb.append_inst(std::move(movzx));
        }
    } else {
        uint8_t sz = static_cast<uint8_t>(op0_val->type().size_in_bytes());
        if (sz == 0) sz = 8;
        Condition final_cond = cond;

        ImmIntInfo imm1 = get_imm_int_info(op1_val);
        ImmIntInfo imm0 = get_imm_int_info(op0_val);

        if (imm1.is_imm && imm1.val == 0) {
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op0, sz));
            test_inst->add_use(LirOperand::vreg(op0, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm0.is_imm && imm0.val == 0) {
            final_cond = swap_relational_condition(cond);
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op1, sz));
            test_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm1.is_imm && imm1.fits_i32) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::imm(imm1.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (imm0.is_imm && imm0.fits_i32) {
            final_cond = swap_relational_condition(cond);
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            cmp_inst->add_use(LirOperand::imm(imm0.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(get_load_mem_operand(op1_val->defining_instruction()));
            lir_bb.append_inst(std::move(cmp_inst));
        } else {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        }

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = final_cond;
        setcc->add_def(LirOperand::vreg(dst, 1));
        lir_bb.append_inst(std::move(setcc));

        auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
        movzx->add_def(LirOperand::vreg(dst, dst_sz));
        movzx->add_use(LirOperand::vreg(dst, 1));
        movzx->mir_origin = &inst;
        lir_bb.append_inst(std::move(movzx));
    }
}

} // namespace brass::x64
