#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_return(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        const auto* ret_val = inst.operand(0);
        VReg ret_vreg = get_vreg(ret_val);
        Type t = ret_val->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (t.is_v256()) {
            VRegPair ret_pair = get_vreg_pair(ret_val);
            auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_lo->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            mov_lo->add_use(LirOperand::vreg(ret_pair.lo, 16));
            lir_bb.append_inst(std::move(mov_lo));

            auto mov_hi = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_hi->add_def(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
            mov_hi->add_use(LirOperand::vreg(ret_pair.hi, 16));
            lir_bb.append_inst(std::move(mov_hi));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else if (t.is_float()) {
            LirOpcode ret_mov_op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else if (t.is_vector()) {
            auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_ret->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, 16));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else {
            auto mov_ret = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        }
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(ret_inst));
}

void AArch64ISel::lower_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& target = inst.branch_target();
    if (!target.block) return;

    if (!target.args.empty()) {
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
        lir_bb.append_inst(std::move(pcopy));
    }

    auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_inst->add_use(LirOperand::label(target.block->id()));
    jmp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(jmp_inst));
}

} // namespace brass::aarch64
