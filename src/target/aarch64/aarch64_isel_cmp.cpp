#include <brass/target/aarch64/aarch64_isel.hpp>
#include "aarch64_isel_conditions.hpp"

namespace brass::aarch64 {

using namespace brass::codegen;
using x64::invert;

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
        if (sz == 0 || op0.size == 8 || op1.size == 8) sz = 8;
        LirCond final_cond = cond;

        ImmIntInfo imm1 = get_imm_int_info(op1_val);
        ImmIntInfo imm0 = get_imm_int_info(op0_val);

        if (imm1.is_imm && imm1.val == 0 && zero_test_matches_compare(cond)) {
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op0, sz));
            test_inst->add_use(LirOperand::vreg(op0, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm0.is_imm && imm0.val == 0 && zero_test_matches_compare(cond)) {
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
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode()) && skipped_insts_.count(cmp_inst);

    LirCond branch_cond = LirCond::NE;

    if (is_fused_cmp) {
        branch_cond = emit_fused_compare(*cmp_inst, lir_bb);
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
        link_blocks(lir_bb, *false_trampoline);
        if (auto* f_lir = lir_fn_->get_block_by_id(f_target.block->id())) link_blocks(*false_trampoline, *f_lir);

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
