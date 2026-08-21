#include <brass/target/x64/x64_isel.hpp>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

static constexpr std::pair<Condition, Condition> get_comparison_conditions(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq:  return {Condition::E, Condition::E};
        case Opcode::ne:  return {Condition::NE, Condition::NE};
        case Opcode::slt: return {Condition::L, Condition::B};
        case Opcode::ult: return {Condition::B, Condition::B};
        case Opcode::sle: return {Condition::LE, Condition::BE};
        case Opcode::ule: return {Condition::BE, Condition::BE};
        case Opcode::sgt: return {Condition::G, Condition::A};
        case Opcode::ugt: return {Condition::A, Condition::A};
        case Opcode::sge: return {Condition::GE, Condition::AE};
        case Opcode::uge: return {Condition::AE, Condition::AE};
        default: return {Condition::E, Condition::E};
    }
}

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

void X64ISel::lower_call(const Instruction& inst, LirBlock& lir_bb) {
    size_t num_args = inst.operand_count();
    size_t start_arg = 0;
    VReg callee_vreg;

    if (inst.opcode() == Opcode::call_indirect) {
        callee_vreg = get_vreg(inst.operand(0));
        start_arg = 1;
        num_args = inst.operand_count() - 1;
    }

    size_t required_stack_space = 0;
    if (cc_.kind() == CallingConvKind::Win64) {
        required_stack_space = std::max(size_t(32), 32 + (num_args > 4 ? (num_args - 4) * 8 : 0));
    } else {
        size_t stack_args = (num_args > 6 ? (num_args - 6) : 0);
        required_stack_space = stack_args * 8;
    }
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);

    auto call_lir = std::make_unique<LirInst>(
        inst.opcode() == Opcode::call_indirect ? LirOpcode::CallIndirect : LirOpcode::Call
    );

    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(start_arg + i);
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

    if (inst.opcode() == Opcode::call_indirect) {
        call_lir->add_use(LirOperand::vreg(callee_vreg, 8));
    } else if (inst.opcode() == Opcode::patchable_call) {
        call_lir->is_patchable = true;
        call_lir->patch_symbol = std::string(inst.symbol());
        std::string callee_name = inst.extra_symbol().empty() ? std::string(inst.symbol()) : std::string(inst.extra_symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->add_use(LirOperand::symbol(callee_name));
    } else {
        call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    }

    call_lir->clobbered_gprs = cc_.caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.caller_saved_xmm_mask();

    Type ret_t = inst.type();
    if (!ret_t.is_void()) {
        uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
        if (ret_t.is_float()) {
            call_lir->add_def(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
        } else {
            call_lir->add_def(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
        }
    }

    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.produces_value()) {
        VReg dst = get_vreg(inst.result());
        if (ret_t.is_float()) {
            auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movsd);
            mov_ret->add_def(LirOperand::vreg(dst, 8));
            mov_ret->add_use(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
            lir_bb.append_inst(std::move(mov_ret));
        } else {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
            mov_ret->add_use(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
            lir_bb.append_inst(std::move(mov_ret));
        }
    }
}

void X64ISel::lower_safepoint(const Instruction& inst, LirBlock& lir_bb) {
    if (cc_.kind() == CallingConvKind::Win64) {
        lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, size_t(32));
    }

    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = inst.resume_id();
    sp_inst->clobbered_gprs = cc_.caller_saved_gpr_mask();
    sp_inst->clobbered_xmms = cc_.caller_saved_xmm_mask();

    for (const auto& entry : val_to_vreg_) {
        if (entry.second.is_gcref) {
            sp_inst->live_gcrefs.push_back(entry.second);
        }
    }
    sp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sp_inst));
}

void X64ISel::lower_guard(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode());

    Condition deopt_cond = Condition::E;

    if (is_fused_cmp) {
        Opcode cmp_op = cmp_inst->opcode();
        auto [gpr_c, float_c] = get_comparison_conditions(cmp_op);
        const Value* lhs = cmp_inst->operand(0);
        const Value* rhs = cmp_inst->operand(1);

        if (lhs->type().is_float()) {
            deopt_cond = invert(float_c);
            auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
            ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
            ucomi->add_use(LirOperand::vreg(get_vreg(rhs), 8));
            lir_bb.append_inst(std::move(ucomi));
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
                    deopt_cond = (cmp_op == Opcode::eq) ? Condition::NE : Condition::E;
                } else {
                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    VReg reg = get_vreg(non_zero);
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    lir_bb.append_inst(std::move(test_lir));
                    deopt_cond = invert((rhs_imm.is_imm && rhs_imm.val == 0) ? gpr_c : swap_relational_condition(gpr_c));
                }
            } else if (rhs_imm.is_imm && rhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                deopt_cond = invert(gpr_c);
            } else if (lhs_imm.is_imm && lhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                deopt_cond = invert(swap_relational_condition(gpr_c));
            } else if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                deopt_cond = invert(gpr_c);
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::imm(rhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                deopt_cond = invert(swap_relational_condition(gpr_c));
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                cmp_lir->add_use(LirOperand::imm(lhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else {
                deopt_cond = invert(gpr_c);
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
        deopt_cond = Condition::E;
    }

    auto* deopt_block = lir_fn_->create_block("guard_deopt");

    auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc_inst->condition = deopt_cond;
    jcc_inst->add_use(LirOperand::label(deopt_block->id));
    lir_bb.append_inst(std::move(jcc_inst));

    // Lower guard deopt exit payload into deopt_block
    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = inst.resume_id();
    exit_inst->exit_symbol = std::string(inst.symbol());

    for (const auto* val : inst.state_map()) {
        if (val) {
            VReg vr = get_vreg(val);
            exit_inst->add_use(LirOperand::vreg(vr, vr.size));
        }
    }
    deopt_block->append_inst(std::move(exit_inst));
}

} // namespace brass::x64
