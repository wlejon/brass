#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_throw(const Instruction& inst, LirBlock& lir_bb) {
    const auto* val = inst.operand(0);
    VReg val_v = get_vreg(val);
    uint8_t sz = 8;

    lir_fn_->frame.has_calls = true;

    const Type vt = val->type();
    if (vt.is_vector()) codegen::throw_unsupported("aarch64 isel (throw)", "a vector exception value");
    if (vt.is_float()) {
        // The value's bits, from its FP register (an f32's zero-extended).
        const uint8_t fsz = vt.kind() == TypeKind::F32 ? 4 : 8;
        auto mov_arg = std::make_unique<LirInst>(LirOpcode::Movd_gx);
        mov_arg->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, fsz), FixedConstraint::aarch64_gpr(GPR::X0));
        mov_arg->add_use(LirOperand::vreg(val_v, fsz));
        lir_bb.append_inst(std::move(mov_arg));
    } else {
        auto mov_arg = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_arg->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
        mov_arg->add_use(LirOperand::vreg(val_v, sz));
        lir_bb.append_inst(std::move(mov_arg));
    }

    auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
    call_lir->callee_symbol = "brass_throw";
    call_lir->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
    call_lir->add_use(LirOperand::symbol("brass_throw"));
    call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));
}

void AArch64ISel::lower_resume(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        lower_throw(inst, lir_bb);
    } else {
        lir_fn_->frame.has_calls = true;

        auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
        call_lir->callee_symbol = "brass_rethrow";
        call_lir->add_use(LirOperand::symbol("brass_rethrow"));
        call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
        call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
        call_lir->mir_origin = &inst;
        lir_bb.append_inst(std::move(call_lir));
    }
}

void AArch64ISel::lower_landing_pad(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.type() != Type::void_type() && inst.result()) {
        VReg dst_v = get_vreg(inst.result());
        uint8_t sz = dst_v.size;
        if (inst.type().is_vector()) codegen::throw_unsupported("aarch64 isel (landing_pad)", "a vector exception value");
        if (inst.type().is_float()) {
            // The thrown bits arrive in X0; a float lives in an FP register.
            const uint8_t fsz = inst.type().kind() == TypeKind::F32 ? 4 : 8;
            auto mov_inst = std::make_unique<LirInst>(LirOpcode::Movd_xg);
            mov_inst->add_def(LirOperand::vreg(dst_v, fsz));
            mov_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, fsz), FixedConstraint::aarch64_gpr(GPR::X0));
            mov_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(mov_inst));
            return;
        }
        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_inst->add_def(LirOperand::vreg(dst_v, sz));
        mov_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
        mov_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(mov_inst));
    }
}

void AArch64ISel::lower_invoke(const Instruction& inst, LirBlock& lir_bb) {
    lir_fn_->frame.has_calls = true;

    size_t num_args = inst.operand_count();
    auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
    call_lir->callee_symbol = std::string(inst.symbol());
    call_lir->is_invoke = true;
    if (inst.unwind_target().block) {
        call_lir->unwind_block_id = inst.unwind_target().block->id();
    }

    size_t gpr_idx = 0;
    size_t fpr_idx = 0;
    size_t stack_idx = 0;

    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(i);
        VReg arg_vreg = get_vreg(arg_val);
        Type t = arg_val->type();
        uint8_t sz = (t.size_in_bytes() == 4) ? 4 : 8;
        bool is_fpr = (t.is_float() || t.is_vector());
        LirOpcode mov_op = is_fpr ? ((sz == 16) ? LirOpcode::Movups : (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd))
                                  : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);

        if (is_fpr) {
            if (fpr_idx < 8) {
                FPR freg = static_cast<FPR>(fpr_idx++);
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                mov_arg->add_def(LirOperand::preg_aarch64_fpr(freg, sz), FixedConstraint::aarch64_fpr(freg));
                mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(freg, sz), FixedConstraint::aarch64_fpr(freg));
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                int32_t disp = static_cast<int32_t>(stack_idx * 8);
                stack_idx++;
                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        } else {
            if (gpr_idx < 8) {
                GPR greg = static_cast<GPR>(gpr_idx++);
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                mov_arg->add_def(LirOperand::preg_aarch64_gpr(greg, sz), FixedConstraint::aarch64_gpr(greg));
                mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                call_lir->add_use(LirOperand::preg_aarch64_gpr(greg, sz), FixedConstraint::aarch64_gpr(greg));
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                int32_t disp = static_cast<int32_t>(stack_idx * 8);
                stack_idx++;
                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        }
    }

    size_t required_stack_space = stack_idx * 8;
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);

    call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
    call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.type() != Type::void_type() && inst.result()) {
        Type t = inst.type();
        uint8_t sz = (t.size_in_bytes() == 4) ? 4 : 8;
        VReg res_vreg = get_vreg(inst.result());
        bool is_fpr = (t.is_float() || t.is_vector());
        LirOpcode ret_mov = is_fpr ? ((sz == 16) ? LirOpcode::Movups : (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd))
                                   : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        auto mov_ret = std::make_unique<LirInst>(ret_mov);
        mov_ret->add_def(LirOperand::vreg(res_vreg, sz));
        if (is_fpr) {
            mov_ret->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
        } else {
            mov_ret->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
        }
        lir_bb.append_inst(std::move(mov_ret));
    }

    const auto& normal_target = inst.normal_target();
    if (normal_target.block) {
        if (!normal_target.args.empty()) {
            if (normal_target.args.size() == 1) {
                VReg arg_v = get_vreg(normal_target.args[0]);
                VReg param_v = get_vreg(normal_target.block->param(0));
                uint8_t sz = param_v.size;
                if (arg_v != param_v) {
                    auto mov_inst = std::make_unique<LirInst>(
                        param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd)) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                    );
                    mov_inst->add_def(LirOperand::vreg(param_v, sz));
                    mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                    lir_bb.append_inst(std::move(mov_inst));
                }
            } else {
                auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
                for (size_t i = 0; i < normal_target.args.size(); ++i) {
                    VReg arg_v = get_vreg(normal_target.args[i]);
                    VReg param_v = get_vreg(normal_target.block->param(i));
                    uint8_t sz = param_v.size;
                    pcopy->add_def(LirOperand::vreg(param_v, sz));
                    pcopy->add_use(LirOperand::vreg(arg_v, sz));
                }
                lir_bb.append_inst(std::move(pcopy));
            }
        }
        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(normal_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    }
}

} // namespace brass::aarch64
