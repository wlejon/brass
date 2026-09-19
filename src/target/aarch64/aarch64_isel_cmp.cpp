#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;
using LirCond = brass::x64::Condition;
using x64::invert;

static constexpr std::pair<LirCond, LirCond> get_comparison_conditions(Opcode op) noexcept {
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

static constexpr LirCond swap_relational_condition(LirCond cond) noexcept {
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

void AArch64ISel::lower_comparison(
    const Instruction& inst,
    LirBlock& lir_bb,
    x64::Condition cond,
    x64::Condition float_cond
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

        auto ucomi = std::make_unique<LirInst>(ucomi_op);
        ucomi->add_use(LirOperand::vreg(op0, sz));
        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            ucomi->add_use(get_load_mem_operand(op1_val->defining_instruction()));
        } else {
            ucomi->add_use(LirOperand::vreg(op1, sz));
        }
        lir_bb.append_inst(std::move(ucomi));

        if (float_cond == LirCond::E) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = LirCond::E;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_np = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_np->condition = LirCond::NP;
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
        } else if (float_cond == LirCond::NE) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = LirCond::NE;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_p = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_p->condition = LirCond::P;
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
        } else if (float_cond == LirCond::B || float_cond == LirCond::BE || float_cond == LirCond::A || float_cond == LirCond::AE) {
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = float_cond;
            setcc->add_def(LirOperand::vreg(dst, 1));
            lir_bb.append_inst(std::move(setcc));

            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 1);
            auto setcc_np = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc_np->condition = LirCond::NP;
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
        LirCond final_cond = cond;

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

void AArch64ISel::lower_branch_if(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode());

    LirCond branch_cond = LirCond::NE;

    if (is_fused_cmp) {
        Opcode cmp_op = cmp_inst->opcode();
        auto [gpr_c, float_c] = get_comparison_conditions(cmp_op);
        const Value* lhs = cmp_inst->operand(0);
        const Value* rhs = cmp_inst->operand(1);

        if (lhs->type().is_float()) {
            branch_cond = float_c;
            if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(ucomi));
            } else {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(LirOperand::vreg(get_vreg(rhs), 8));
                lir_bb.append_inst(std::move(ucomi));
            }
        } else {
            uint8_t sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
            if (sz == 0) sz = 8;
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            LirOpcode test_lir_op = (sz == 4) ? LirOpcode::Test32 : LirOpcode::Test;

            ImmIntInfo rhs_imm = get_imm_int_info(rhs);
            ImmIntInfo lhs_imm = get_imm_int_info(lhs);

            if ((cmp_op == Opcode::eq || cmp_op == Opcode::ne) && ((rhs_imm.is_imm && rhs_imm.val == 0) || (lhs_imm.is_imm && lhs_imm.val == 0))) {
                const Value* non_zero = (rhs_imm.is_imm && rhs_imm.val == 0) ? lhs : rhs;
                if (non_zero->is_instruction() && skipped_insts_.count(non_zero->defining_instruction()) && non_zero->defining_instruction()->opcode() == Opcode::and_) {
                    const Instruction* and_inst = non_zero->defining_instruction();
                    const Value* a = and_inst->operand(0);
                    const Value* b = and_inst->operand(1);
                    ImmIntInfo imm_b = get_imm_int_info(b);
                    ImmIntInfo imm_a = get_imm_int_info(a);

                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    if (imm_b.is_imm && imm_b.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::imm(imm_b.val, sz));
                    } else if (imm_a.is_imm && imm_a.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                        test_lir->add_use(LirOperand::imm(imm_a.val, sz));
                    } else {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                    }
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (cmp_op == Opcode::eq) ? LirCond::E : LirCond::NE;
                } else {
                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    VReg reg = get_vreg(non_zero);
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (rhs_imm.is_imm && rhs_imm.val == 0) ? gpr_c : swap_relational_condition(gpr_c);
                }
            } else if (rhs_imm.is_imm && rhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = gpr_c;
            } else if (lhs_imm.is_imm && lhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = swap_relational_condition(gpr_c);
            } else if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::imm(rhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                branch_cond = swap_relational_condition(gpr_c);
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                cmp_lir->add_use(LirOperand::imm(lhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(cmp_lir));
            } else {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(cmp_lir));
            }
        }
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
        branch_cond = LirCond::NE;
    }

    auto emit_target_args = [&](LirBlock& bb, const BranchTarget& target) {
        if (target.args.empty()) return;
        auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
        for (size_t i = 0; i < target.args.size(); ++i) {
            const Value* arg = target.args[i];
            const Value* param = target.block->param(i);
            if (param->type().is_v256()) {
                VRegPair arg_p = get_vreg_pair(arg);
                VRegPair param_p = get_vreg_pair(param);
                pcopy->add_def(LirOperand::vreg(param_p.lo, 16));
                pcopy->add_use(LirOperand::vreg(arg_p.lo, 16));
                pcopy->add_def(LirOperand::vreg(param_p.hi, 16));
                pcopy->add_use(LirOperand::vreg(arg_p.hi, 16));
            } else {
                VReg arg_v = get_vreg(arg);
                VReg param_v = get_vreg(param);
                uint8_t sz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, sz));
                pcopy->add_use(LirOperand::vreg(arg_v, sz));
            }
        }
        bb.append_inst(std::move(pcopy));
    };

    if (t_target.args.empty() && f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(f_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    } else if (t_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        jmp_f->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_f));
    } else if (f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(f_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));
    } else {
        auto* false_trampoline = lir_fn_->create_block("br_if_false");

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(false_trampoline->id));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));

        emit_target_args(*false_trampoline, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        false_trampoline->append_inst(std::move(jmp_f));
    }
}

} // namespace brass::aarch64
