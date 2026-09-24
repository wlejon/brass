#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/runtime/coroutine.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_coro(const Instruction& inst, LirBlock& lir_bb) {
    lir_fn_->frame.has_calls = true;

    GPR arg0 = GPR::X0;
    GPR arg1 = GPR::X1;
    GPR arg2 = GPR::X2;

    switch (inst.opcode()) {
        case Opcode::coro_create: {
            // `coro_create @f(args...)`: the operands are the coroutine's
            // arguments. The frame shape comes from the lowered body of @f;
            // argument i is stored in frame slot i, where the body loads it.
            const Module* callee_mod = callee_module_ ? callee_module_ : (mir_fn_ ? mir_fn_->parent() : nullptr);
            const Function* callee_fn = callee_mod ? callee_mod->get_function(inst.symbol()) : nullptr;
            if (!callee_fn) {
                throw_unsupported("aarch64 isel (coro)", "coro_create of a function not in the module: " + std::string(inst.symbol()));
            }
            if (!is_lowered_coro_body(*callee_fn)) {
                throw_unsupported("aarch64 isel (coro)", "coro_create of " + std::string(inst.symbol()) +
                                  ", which has not been lowered by CoroTransformPass");
            }
            if (!inst.result()) {
                throw_unsupported("aarch64 isel (coro)", "coro_create without a result");
            }
            CoroFrameLayout layout = compute_coro_frame_layout(*callee_fn);
            uint32_t slot_count = std::max(layout.slot_count, coro_create_slot_count(inst));

            auto mov_fn = std::make_unique<LirInst>(LirOpcode::Movabs);
            mov_fn->add_def(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            mov_fn->add_use(LirOperand::symbol(std::string(inst.symbol())));
            lir_bb.append_inst(std::move(mov_fn));

            auto mov_slots = std::make_unique<LirInst>(LirOpcode::Mov32);
            mov_slots->add_def(LirOperand::preg_aarch64_gpr(arg1, 4), FixedConstraint::aarch64_gpr(arg1));
            mov_slots->add_use(LirOperand::imm(static_cast<int64_t>(slot_count), 4));
            lir_bb.append_inst(std::move(mov_slots));

            auto mov_mask = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_mask->add_def(LirOperand::preg_aarch64_gpr(arg2, 8), FixedConstraint::aarch64_gpr(arg2));
            mov_mask->add_use(LirOperand::imm(static_cast<int64_t>(layout.pointer_mask), 8));
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

            VReg dst = get_vreg(inst.result());
            auto mov_res = std::make_unique<LirInst>(LirOpcode::Mov);
            mov_res->add_def(LirOperand::vreg(dst, 8));
            mov_res->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
            lir_bb.append_inst(std::move(mov_res));

            // Arguments fill consecutive slots; a vector spans
            // coro_slot_count of them (a v256 is stored as its two halves).
            uint32_t slot = 0;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                const Type arg_t = inst.operand(i)->type();
                const int32_t off = static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8);
                slot += coro_slot_count(arg_t);
                if (arg_t.is_v256()) {
                    VRegPair pair = get_vreg_pair(inst.operand(i));
                    for (int half = 0; half < 2; ++half) {
                        auto st = std::make_unique<LirInst>(LirOpcode::Movups);
                        st->add_def(LirOperand::mem(dst, off + half * 16, 16));
                        st->add_use(LirOperand::vreg(half ? pair.hi : pair.lo, 16));
                        st->mir_origin = &inst;
                        lir_bb.append_inst(std::move(st));
                    }
                    continue;
                }
                VReg arg_v = get_vreg(inst.operand(i));
                uint8_t sz = arg_v.size;
                LirOpcode op;
                if (arg_v.is_xmm()) {
                    if (sz == 16) op = LirOpcode::Movups;
                    else if (sz == 4 || sz == 8) op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
                    else throw_unsupported("aarch64 isel (coro)", "coro_create argument of " + std::to_string(sz) + " bytes");
                } else {
                    if (sz != 4 && sz != 8) throw_unsupported("aarch64 isel (coro)", "coro_create argument narrower than 32 bits");
                    op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
                }
                auto st = std::make_unique<LirInst>(op);
                st->add_def(LirOperand::mem(dst, off, sz));
                st->add_use(LirOperand::vreg(arg_v, sz));
                st->mir_origin = &inst;
                lir_bb.append_inst(std::move(st));
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
            // The entry for generated callers: a throw from the body is
            // raised again natively, so this function's pads see it (JIT and
            // AOT link the same symbol).
            call_lir->callee_symbol = "brass_coro_resume_from_generated";
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg0, 8), FixedConstraint::aarch64_gpr(arg0));
            call_lir->add_use(LirOperand::preg_aarch64_gpr(arg1, 8), FixedConstraint::aarch64_gpr(arg1));
            call_lir->add_use(LirOperand::symbol("brass_coro_resume_from_generated"));
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

        case Opcode::coro_suspend:
            // Only an unlowered body still has one; returning here would lose
            // the state and every value live across the suspend.
            throw_unsupported("aarch64 isel (coro)",
                              "coro_suspend (the coroutine body must be lowered by CoroTransformPass first)");

        default:
            codegen::throw_unsupported("aarch64 isel (coro)", opcode_name(inst.opcode()));
    }
}

} // namespace brass::aarch64
