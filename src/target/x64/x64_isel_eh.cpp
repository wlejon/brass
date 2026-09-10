#include <brass/target/x64/x64_isel.hpp>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_throw(const Instruction& inst, LirBlock& lir_bb) {
    const auto* val = inst.operand(0);
    VReg val_v = get_vreg(val);
    uint8_t sz = 8;

    lir_fn_->frame.has_calls = true;
    size_t req_stack = (cc_.kind() == CallingConvKind::Win64) ? 32 : 0;
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, req_stack);

    GPR arg0 = (cc_.kind() == CallingConvKind::Win64) ? GPR::RCX : GPR::RDI;
    auto mov_arg = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_arg->add_def(LirOperand::preg_gpr(arg0, sz), FixedConstraint::gpr(arg0));
    mov_arg->add_use(LirOperand::vreg(val_v, sz));
    lir_bb.append_inst(std::move(mov_arg));

    auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
    call_lir->callee_symbol = "brass_throw";
    call_lir->add_use(LirOperand::preg_gpr(arg0, sz), FixedConstraint::gpr(arg0));
    call_lir->add_use(LirOperand::symbol("brass_throw"));
    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));
}

void X64ISel::lower_resume(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        lower_throw(inst, lir_bb);
    } else {
        lir_fn_->frame.has_calls = true;
        size_t req_stack = (cc_.kind() == CallingConvKind::Win64) ? 32 : 0;
        lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, req_stack);

        auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
        call_lir->callee_symbol = "brass_rethrow";
        call_lir->add_use(LirOperand::symbol("brass_rethrow"));
        call_lir->mir_origin = &inst;
        lir_bb.append_inst(std::move(call_lir));
    }
}

void X64ISel::lower_landing_pad(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.type() != Type::void_type() && inst.result()) {
        VReg dst_v = get_vreg(inst.result());
        uint8_t sz = dst_v.size;
        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_inst->add_def(LirOperand::vreg(dst_v, sz));
        mov_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
        mov_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(mov_inst));
    }
}

void X64ISel::lower_invoke(const Instruction& inst, LirBlock& lir_bb) {
    lir_fn_->frame.has_calls = true;

    size_t num_args = inst.operand_count();
    size_t required_stack_space = 0;
    if (cc_.kind() == CallingConvKind::Win64) {
        required_stack_space = std::max(size_t(32), 32 + (num_args > 4 ? (num_args - 4) * 8 : 0));
    } else {
        size_t stack_args = (num_args > 6 ? (num_args - 6) : 0);
        required_stack_space = stack_args * 8;
    }
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);

    auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
    call_lir->callee_symbol = std::string(inst.symbol());
    call_lir->is_invoke = true;
    if (inst.unwind_target().block) {
        call_lir->unwind_block_id = inst.unwind_target().block->id();
    }

    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(i);
        VReg arg_vreg = get_vreg(arg_val);
        Type t = arg_val->type();
        uint8_t sz = (t.size_in_bytes() == 4) ? 4 : 8;
        LirOpcode mov_op = t.is_float() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                if (t.is_float()) {
                    XMM xreg = static_cast<XMM>(i);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, 8));
                    call_lir->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                }
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                int32_t disp = static_cast<int32_t>(32 + (i - 4) * 8);
                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        } else {
            if (t.is_float()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    auto mov_arg = std::make_unique<LirInst>(LirOpcode::Movsd);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, 8));
                    call_lir->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    lir_bb.append_inst(std::move(mov_arg));
                } else {
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    xmm_idx++;
                    auto mov_stack = std::make_unique<LirInst>(LirOpcode::Movsd);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, 8));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, 8));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            } else {
                if (gpr_idx < cc_.num_arg_gprs()) {
                    GPR greg = cc_.arg_gpr(gpr_idx++);
                    auto mov_arg = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    lir_bb.append_inst(std::move(mov_arg));
                } else {
                    size_t stack_idx = (gpr_idx - cc_.num_arg_gprs()) + (xmm_idx > cc_.num_arg_xmms() ? (xmm_idx - cc_.num_arg_xmms()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    gpr_idx++;
                    auto mov_stack = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            }
        }
    }

    call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    // Result value if any
    if (inst.type() != Type::void_type() && inst.result()) {
        Type t = inst.type();
        uint8_t sz = (t.size_in_bytes() == 4) ? 4 : 8;
        VReg res_vreg = get_vreg(inst.result());
        LirOpcode ret_mov = t.is_float() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        auto mov_ret = std::make_unique<LirInst>(ret_mov);
        mov_ret->add_def(LirOperand::vreg(res_vreg, sz));
        if (t.is_float()) {
            mov_ret->add_use(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
        } else {
            mov_ret->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
        }
        lir_bb.append_inst(std::move(mov_ret));
    }

    // Branch to normal_target
    const auto& normal_target = inst.normal_target();
    if (normal_target.block) {
        if (!normal_target.args.empty()) {
            if (normal_target.args.size() == 1) {
                VReg arg_v = get_vreg(normal_target.args[0]);
                VReg param_v = get_vreg(normal_target.block->param(0));
                uint8_t sz = param_v.size;
                if (arg_v != param_v) {
                    auto mov_inst = std::make_unique<LirInst>(
                        param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
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

} // namespace brass::x64
