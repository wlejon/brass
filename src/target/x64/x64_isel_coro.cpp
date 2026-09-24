#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/runtime/coroutine.hpp>
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
            // `coro_create @f(args...)`: the operands are the coroutine's
            // arguments. The frame shape comes from the lowered body of @f;
            // argument i is stored in frame slot i, where the body loads it.
            const Module* callee_mod = callee_module_ ? callee_module_ : (mir_fn_ ? mir_fn_->parent() : nullptr);
            const Function* callee_fn = callee_mod ? callee_mod->get_function(inst.symbol()) : nullptr;
            if (!callee_fn) {
                throw_unsupported("x64 isel (coro)", "coro_create of a function not in the module: " + std::string(inst.symbol()));
            }
            if (!is_lowered_coro_body(*callee_fn)) {
                throw_unsupported("x64 isel (coro)", "coro_create of " + std::string(inst.symbol()) +
                                  ", which has not been lowered by CoroTransformPass");
            }
            if (!inst.result()) {
                throw_unsupported("x64 isel (coro)", "coro_create without a result");
            }
            CoroFrameLayout layout = compute_coro_frame_layout(*callee_fn);
            uint32_t slot_count = std::max(layout.slot_count, coro_create_slot_count(inst));

            auto mov_fn = std::make_unique<LirInst>(LirOpcode::Movabs);
            mov_fn->add_def(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            mov_fn->add_use(LirOperand::symbol(std::string(inst.symbol())));
            lir_bb.append_inst(std::move(mov_fn));

            auto mov_slots = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov_slots->add_def(LirOperand::preg_gpr(arg1, 4), FixedConstraint::gpr(arg1));
            mov_slots->add_use(LirOperand::imm(static_cast<int64_t>(slot_count), 4));
            lir_bb.append_inst(std::move(mov_slots));

            auto mov_mask = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_mask->add_def(LirOperand::preg_gpr(arg2, 8), FixedConstraint::gpr(arg2));
            mov_mask->add_use(LirOperand::imm(static_cast<int64_t>(layout.pointer_mask), 8));
            lir_bb.append_inst(std::move(mov_mask));

            auto call_lir = std::make_unique<LirInst>(LirOpcode::Call);
            call_lir->callee_symbol = "brass_coro_create";
            call_lir->add_use(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            call_lir->add_use(LirOperand::preg_gpr(arg1, 4), FixedConstraint::gpr(arg1));
            call_lir->add_use(LirOperand::preg_gpr(arg2, 8), FixedConstraint::gpr(arg2));
            call_lir->add_use(LirOperand::symbol("brass_coro_create"));
            finish_call(*call_lir, Type::i64());
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));

            VReg dst = get_vreg(inst.result());
            auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_res->add_def(LirOperand::vreg(dst, 8));
            mov_res->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
            lir_bb.append_inst(std::move(mov_res));

            // Arguments fill consecutive slots; a vector spans
            // coro_slot_count of them (unaligned full-width store).
            uint32_t slot = 0;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                VReg arg_v = get_vreg(inst.operand(i));
                uint8_t sz = arg_v.size;
                LirOpcode op;
                if (arg_v.is_xmm()) {
                    if (sz == 16) op = LirOpcode::Movups;
                    else if (sz == 32) op = LirOpcode::Vmovups;
                    else if (sz == 4 || sz == 8) op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
                    else throw_unsupported("x64 isel (coro)", "coro_create argument of " + std::to_string(sz) + " bytes");
                } else {
                    if (sz != 4 && sz != 8) throw_unsupported("x64 isel (coro)", "coro_create argument narrower than 32 bits");
                    op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
                }
                const uint32_t slots = coro_slot_count(inst.operand(i)->type());
                if (slots * 8 < sz) throw_unsupported("x64 isel (coro)", "coro_create argument wider than its frame slots");
                auto st = std::make_unique<LirInst>(op);
                st->add_def(LirOperand::mem(dst, static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8), sz));
                slot += slots;
                st->add_use(LirOperand::vreg(arg_v, sz));
                st->mir_origin = &inst;
                lir_bb.append_inst(std::move(st));
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
            // The entry for generated callers: a throw from the body is
            // raised again natively, so this function's pads see it (JIT and
            // AOT link the same symbol).
            call_lir->callee_symbol = "brass_coro_resume_from_generated";
            call_lir->add_use(LirOperand::preg_gpr(arg0, 8), FixedConstraint::gpr(arg0));
            call_lir->add_use(LirOperand::preg_gpr(arg1, 8), FixedConstraint::gpr(arg1));
            call_lir->add_use(LirOperand::symbol("brass_coro_resume_from_generated"));
            finish_call(*call_lir, Type::i64());
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
            finish_call(*call_lir, Type::void_type());
            call_lir->mir_origin = &inst;
            lir_bb.append_inst(std::move(call_lir));
            break;
        }

        case Opcode::coro_suspend:
            // Only an unlowered body still has one; returning here would lose
            // the state and every value live across the suspend.
            throw_unsupported("x64 isel (coro)",
                              "coro_suspend (the coroutine body must be lowered by CoroTransformPass first)");

        default:
            codegen::throw_unsupported("x64 isel (coro)", opcode_name(inst.opcode()));
    }
}

} // namespace brass::x64
