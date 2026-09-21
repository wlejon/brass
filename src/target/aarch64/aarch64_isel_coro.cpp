#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/mir/module.hpp>
#include <brass/runtime/coroutine.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

struct CoroFrameDescriptor {
    uint32_t slot_count = 0;
    uint64_t pointer_mask = 0;
};

CoroFrameDescriptor analyze_coro_callee(const Function* callee) {
    CoroFrameDescriptor desc{0, 0};
    if (!callee) return desc;

    for (const auto* bb : callee->blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::store || inst->opcode() == Opcode::load) {
                if (inst->offset() >= runtime::CORO_OFFSET_SLOTS) {
                    uint32_t slot = static_cast<uint32_t>((inst->offset() - runtime::CORO_OFFSET_SLOTS) / 8);
                    desc.slot_count = std::max(desc.slot_count, slot + 1);
                    if (inst->memory_type().is_pointer_or_gcref() ||
                        (inst->opcode() == Opcode::store && inst->operand(1) && inst->operand(1)->type().is_pointer_or_gcref())) {
                        if (slot < 64) {
                            desc.pointer_mask |= (1ULL << slot);
                        }
                    }
                }
            }
        }
    }
    return desc;
}

} // namespace

void AArch64ISel::lower_coro(const Instruction& inst, LirBlock& lir_bb) {
    lir_fn_->frame.has_calls = true;

    GPR arg0 = GPR::X0;
    GPR arg1 = GPR::X1;
    GPR arg2 = GPR::X2;

    switch (inst.opcode()) {
        case Opcode::coro_create: {
            auto mov_fn = std::make_unique<LirInst>(LirOpcode::Movabs);
            mov_fn->add_def(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            mov_fn->add_use(LirOperand::symbol(std::string(inst.symbol())));
            lir_bb.append_inst(std::move(mov_fn));

            uint32_t slot_count = 16;
            uint64_t pointer_mask = 0;
            VReg slot_vreg{};
            VReg mask_vreg{};

            const Function* callee_fn = (mir_fn_ && mir_fn_->parent())
                ? mir_fn_->parent()->get_function(inst.symbol())
                : nullptr;
            if (callee_fn) {
                auto desc = analyze_coro_callee(callee_fn);
                if (desc.slot_count > 0) {
                    slot_count = desc.slot_count;
                    pointer_mask = desc.pointer_mask;
                }
            }

            if (inst.operand_count() >= 2) {
                ImmIntInfo imm0 = get_imm_int_info(inst.operand(0));
                if (imm0.is_imm) {
                    slot_count = static_cast<uint32_t>(imm0.val);
                } else {
                    slot_vreg = get_vreg(inst.operand(0));
                }
                ImmIntInfo imm1 = get_imm_int_info(inst.operand(1));
                if (imm1.is_imm) {
                    pointer_mask = static_cast<uint64_t>(imm1.val);
                } else {
                    mask_vreg = get_vreg(inst.operand(1));
                }
            } else if (inst.operand_count() == 1) {
                ImmIntInfo imm0 = get_imm_int_info(inst.operand(0));
                if (imm0.is_imm) {
                    slot_count = static_cast<uint32_t>(imm0.val);
                } else {
                    slot_vreg = get_vreg(inst.operand(0));
                }
            } else if (inst.imm_i64() != 0) {
                slot_count = static_cast<uint32_t>(inst.imm_i64());
            }

            if (slot_count < 1) slot_count = 1;

            auto mov_slots = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov_slots->add_def(LirOperand::preg_aarch64_gpr(arg1, 4), FixedConstraint::aarch64_gpr(arg1));
            if (slot_vreg.is_valid()) {
                mov_slots->add_use(LirOperand::vreg(slot_vreg, 4));
            } else {
                mov_slots->add_use(LirOperand::imm(static_cast<int64_t>(slot_count), 4));
            }
            lir_bb.append_inst(std::move(mov_slots));

            auto mov_mask = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_mask->add_def(LirOperand::preg_aarch64_gpr(arg2, 8), FixedConstraint::aarch64_gpr(arg2));
            if (mask_vreg.is_valid()) {
                mov_mask->add_use(LirOperand::vreg(mask_vreg, 8));
            } else {
                mov_mask->add_use(LirOperand::imm(static_cast<int64_t>(pointer_mask), 8));
            }
            lir_bb.append_inst(std::move(mov_mask));

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_create";
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg1, 4), FixedConstraint::aarch64_gpr(arg1));
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg2, 8), FixedConstraint::aarch64_gpr(arg2));
            call_lir->add_use(LirOperand::symbol("brass_coro_create"));
            call_lir->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
            call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
            call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));

            if (inst.result()) {
                VReg dst = get_vreg(inst.result());
                auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_res->add_def(LirOperand::vreg(dst, 8));
                mov_res->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
                lir_bb.append_inst(std::move(mov_res));
            }
            break;
        }

        case Opcode::coro_resume: {
            VReg frame_vreg = get_vreg(inst.operand(0));
            auto mov_frame = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_frame->add_def(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            mov_frame->add_use(LirOperand::vreg(frame_vreg, 8));
            lir_bb.append_inst(std::move(mov_frame));

            if (inst.operand_count() > 1 && inst.operand(1)) {
                VReg input_vreg = get_vreg(inst.operand(1));
                auto mov_input = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_input->add_def(LirOperand::preg_aarch64_gpr(arg1, 8), FixedConstraint::aarch64_gpr(arg1));
                mov_input->add_use(LirOperand::vreg(input_vreg, 8));
                lir_bb.append_inst(std::move(mov_input));
            } else {
                auto mov_zero = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_zero->add_def(LirOperand::preg_aarch64_gpr(arg1, 8), FixedConstraint::aarch64_gpr(arg1));
                mov_zero->add_use(LirOperand::imm(0, 8));
                lir_bb.append_inst(std::move(mov_zero));
            }

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_resume";
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg1, 8), FixedConstraint::aarch64_gpr(arg1));
            call_lir->add_use(LirOperand::symbol("brass_coro_resume"));
            call_lir->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
            call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
            call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));

            if (inst.result()) {
                VReg dst = get_vreg(inst.result());
                auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_res->add_def(LirOperand::vreg(dst, 8));
                mov_res->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
                lir_bb.append_inst(std::move(mov_res));
            }
            break;
        }

        case Opcode::coro_destroy: {
            VReg frame_vreg = get_vreg(inst.operand(0));
            auto mov_frame = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_frame->add_def(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            mov_frame->add_use(LirOperand::vreg(frame_vreg, 8));
            lir_bb.append_inst(std::move(mov_frame));

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_destroy";
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            call_lir->add_use(LirOperand::symbol("brass_coro_destroy"));
            call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
            call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));
            break;
        }

        case Opcode::coro_suspend: {
            if (inst.operand_count() > 0 && inst.operand(0)) {
                VReg val_v = get_vreg(inst.operand(0));
                auto mov_ret = std::make_unique<LirInst>(LirOpcode::Mov);
                mov_ret->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
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

} // namespace brass::aarch64
