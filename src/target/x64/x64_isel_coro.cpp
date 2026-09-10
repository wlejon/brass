#include <brass/target/x64/x64_isel.hpp>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_coro(const Instruction& inst, LirBlock& lir_bb) {
    lir_fn_->frame.has_calls = true;
    size_t req_stack = (cc_.kind() == CallingConvKind::Win64) ? 32 : 0;
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, req_stack);

    GPR arg0 = (cc_.kind() == CallingConvKind::Win64) ? GPR::RCX : GPR::RDI;
    GPR arg1 = (cc_.kind() == CallingConvKind::Win64) ? GPR::RDX : GPR::RSI;
    GPR arg2 = (cc_.kind() == CallingConvKind::Win64) ? GPR::R8  : GPR::RDX;

    switch (inst.opcode()) {
        case Opcode::coro_create: {
            auto mov_fn = std::make_unique<LirInst>(LirOpcode::Movabs);
            mov_fn->add_def(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            mov_fn->add_use(LirOperand::symbol(std::string(inst.symbol())));
            lir_bb.append_inst(std::move(mov_fn));

            auto mov_slots = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov_slots->add_def(LirOperand::preg_gpr(arg1, 4), FixedConstraint::gpr(arg1));
            mov_slots->add_use(LirOperand::imm(16, 4));
            lir_bb.append_inst(std::move(mov_slots));

            auto mov_mask = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_mask->add_def(LirOperand::preg_gpr(arg2, 8), FixedConstraint::gpr(arg2));
            mov_mask->add_use(LirOperand::imm(0, 8));
            lir_bb.append_inst(std::move(mov_mask));

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_create";
            call_lir->add_use(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            call_lir->add_use(LirOperand::preg_gpr(arg1, 4), FixedConstraint::gpr(arg1));
            call_lir->add_use(LirOperand::preg_gpr(arg2, 8), FixedConstraint::gpr(arg2));
            call_lir->add_use(LirOperand::symbol("brass_coro_create"));
            call_lir->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));

            if (inst.result()) {
                VReg dst = get_vreg(inst.result());
                auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_res->add_def(LirOperand::vreg(dst, 8));
                mov_res->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
                lir_bb.append_inst(std::move(mov_res));
            }
            break;
        }

        case Opcode::coro_resume: {
            VReg frame_vreg = get_vreg(inst.operand(0));
            auto mov_frame = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_frame->add_def(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            mov_frame->add_use(LirOperand::vreg(frame_vreg, 8));
            lir_bb.append_inst(std::move(mov_frame));

            if (inst.operand_count() > 1 && inst.operand(1)) {
                VReg input_vreg = get_vreg(inst.operand(1));
                auto mov_input = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_input->add_def(LirOperand::preg_gpr(arg1, 8), FixedConstraint::gpr(arg1));
                mov_input->add_use(LirOperand::vreg(input_vreg, 8));
                lir_bb.append_inst(std::move(mov_input));
            } else {
                auto mov_zero = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_zero->add_def(LirOperand::preg_gpr(arg1, 8), FixedConstraint::gpr(arg1));
                mov_zero->add_use(LirOperand::imm(0, 8));
                lir_bb.append_inst(std::move(mov_zero));
            }

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_resume";
            call_lir->add_use(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            call_lir->add_use(LirOperand::preg_gpr(arg1, 8), FixedConstraint::gpr(arg1));
            call_lir->add_use(LirOperand::symbol("brass_coro_resume"));
            call_lir->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));

            if (inst.result()) {
                VReg dst = get_vreg(inst.result());
                auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_res->add_def(LirOperand::vreg(dst, 8));
                mov_res->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
                lir_bb.append_inst(std::move(mov_res));
            }
            break;
        }

        case Opcode::coro_destroy: {
            VReg frame_vreg = get_vreg(inst.operand(0));
            auto mov_frame = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_frame->add_def(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            mov_frame->add_use(LirOperand::vreg(frame_vreg, 8));
            lir_bb.append_inst(std::move(mov_frame));

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_destroy";
            call_lir->add_use(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            call_lir->add_use(LirOperand::symbol("brass_coro_destroy"));
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));
            break;
        }

        case Opcode::coro_suspend: {
            if (inst.operand_count() > 0 && inst.operand(0)) {
                VReg val_v = get_vreg(inst.operand(0));
                auto mov_ret = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_ret->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
                mov_ret->add_use(LirOperand::vreg(val_v, 8));
                lir_bb.append_inst(std::move(mov_ret));
            }
            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            break;
        }

        default:
            break;
    }
}

} // namespace brass::x64
