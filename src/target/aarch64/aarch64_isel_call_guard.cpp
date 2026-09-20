#include <brass/target/aarch64/aarch64_isel.hpp>
#include <algorithm>

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
        default: return {LirCond::E, LirCond::E};
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

void AArch64ISel::lower_call(const Instruction& inst, LirBlock& lir_bb) {
    size_t num_args = inst.operand_count();
    size_t start_arg = 0;
    VReg callee_vreg;

    if (inst.opcode() == Opcode::call_indirect) {
        callee_vreg = get_vreg(inst.operand(0));
        start_arg = 1;
        num_args = inst.operand_count() - 1;
    }

    auto call_lir = std::make_unique<LirInst>(
        inst.opcode() == Opcode::call_indirect ? LirOpcode::CallIndirect : LirOpcode::Call
    );

    size_t gpr_idx = 0;
    size_t fpr_idx = 0;
    size_t stack_bytes = 0;
    bool is_apple = (cc_.kind() == CallingConvKind::AppleAAPCS64);

    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(start_arg + i);
        Type t = arg_val->type();

        if (t.is_v256()) {
            VRegPair pair = get_vreg_pair(arg_val);
            if (fpr_idx + 1 < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                FPR r_hi = static_cast<FPR>(fpr_idx++);
                auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_lo->add_def(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                mov_lo->add_use(LirOperand::vreg(pair.lo, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                lir_bb.append_inst(std::move(mov_lo));

                auto mov_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_hi->add_def(LirOperand::preg_aarch64_fpr(r_hi, 16), FixedConstraint::aarch64_fpr(r_hi));
                mov_hi->add_use(LirOperand::vreg(pair.hi, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_hi, 16), FixedConstraint::aarch64_fpr(r_hi));
                lir_bb.append_inst(std::move(mov_hi));
            } else if (fpr_idx < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_lo->add_def(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                mov_lo->add_use(LirOperand::vreg(pair.lo, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                lir_bb.append_inst(std::move(mov_lo));

                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += 16;

                auto mov_stack = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, 16));
                mov_stack->add_use(LirOperand::vreg(pair.hi, 16));
                lir_bb.append_inst(std::move(mov_stack));
            } else {
                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += 32;

                auto mov_stack_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack_lo->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, 16));
                mov_stack_lo->add_use(LirOperand::vreg(pair.lo, 16));
                lir_bb.append_inst(std::move(mov_stack_lo));

                auto mov_stack_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack_hi->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp + 16, 16));
                mov_stack_hi->add_use(LirOperand::vreg(pair.hi, 16));
                lir_bb.append_inst(std::move(mov_stack_hi));
            }
            continue;
        }

        VReg arg_vreg = get_vreg(arg_val);
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;
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
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

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
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        }
    }

    size_t required_stack_space = (stack_bytes + 15) & ~size_t(15);
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);
    lir_fn_->frame.has_calls = true;

    if (inst.opcode() == Opcode::call_indirect) {
        call_lir->add_use(LirOperand::vreg(callee_vreg, 8));
    } else if (inst.opcode() == Opcode::patchable_call) {
        call_lir->is_patchable = true;
        call_lir->patch_symbol = std::string(inst.symbol());
        std::string callee_name = inst.extra_symbol().empty() ? std::string(inst.symbol()) : std::string(inst.extra_symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->add_use(LirOperand::symbol(callee_name));
    } else {
        call_lir->callee_symbol = std::string(inst.symbol());
        call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    }

    call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();

    Type ret_t = inst.type();
    if (!ret_t.is_void()) {
        if (ret_t.is_v256()) {
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
        } else if (ret_t.is_float() || ret_t.is_vector()) {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, ret_sz), FixedConstraint::aarch64_fpr(FPR::V0));
        } else {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            call_lir->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, ret_sz), FixedConstraint::aarch64_gpr(GPR::X0));
        }
    }

    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.produces_value()) {
        if (ret_t.is_v256()) {
            VRegPair dst_pair = get_vreg_pair(inst.result());
            auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_lo->add_def(LirOperand::vreg(dst_pair.lo, 16));
            mov_lo->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            lir_bb.append_inst(std::move(mov_lo));

            auto mov_hi = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_hi->add_def(LirOperand::vreg(dst_pair.hi, 16));
            mov_hi->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
            lir_bb.append_inst(std::move(mov_hi));
        } else {
            VReg dst = get_vreg(inst.result());
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            if (ret_t.is_float()) {
                auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
                mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
                mov_ret->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, ret_sz), FixedConstraint::aarch64_fpr(FPR::V0));
                lir_bb.append_inst(std::move(mov_ret));
            } else if (ret_t.is_vector()) {
                auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movaps);
                mov_ret->add_def(LirOperand::vreg(dst, 16));
                mov_ret->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
                lir_bb.append_inst(std::move(mov_ret));
            } else {
                auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
                mov_ret->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, ret_sz), FixedConstraint::aarch64_gpr(GPR::X0));
                lir_bb.append_inst(std::move(mov_ret));
            }
        }
    }
}

void AArch64ISel::lower_safepoint(const Instruction& inst, LirBlock& lir_bb) {
    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = inst.resume_id();
    sp_inst->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    sp_inst->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();

    for (const auto& entry : val_to_vreg_) {
        if (entry.second.is_gcref) {
            sp_inst->live_gcrefs.push_back(entry.second);
        }
    }
    sp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sp_inst));
}

void AArch64ISel::lower_write_barrier(const Instruction& inst, LirBlock& lir_bb) {
    const Value* obj_val = inst.operand(0);
    const Value* val_val = inst.operand(1);
    if (!obj_val || !val_val) return;

    VReg obj_vreg = get_vreg(obj_val);
    VReg val_vreg = get_vreg(val_val);

    auto mov_obj = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_obj->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
    mov_obj->add_use(LirOperand::vreg(obj_vreg, 8));
    lir_bb.append_inst(std::move(mov_obj));

    auto mov_val = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_val->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8), FixedConstraint::aarch64_gpr(GPR::X1));
    mov_val->add_use(LirOperand::vreg(val_vreg, 8));
    lir_bb.append_inst(std::move(mov_val));

    auto call_wb = std::make_unique<LirInst>(LirOpcode::Call);
    call_wb->callee_symbol = "brass_gc_write_barrier";
    call_wb->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
    call_wb->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8), FixedConstraint::aarch64_gpr(GPR::X1));
    call_wb->add_use(LirOperand::symbol("brass_gc_write_barrier"));
    call_wb->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_wb->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
    call_wb->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_wb));
}

void AArch64ISel::lower_guard(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode());

    LirCond deopt_cond = LirCond::E;

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
                    deopt_cond = (cmp_op == Opcode::eq) ? LirCond::NE : LirCond::E;
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
        deopt_cond = LirCond::E;
    }

    auto* deopt_block = lir_fn_->create_block("guard_deopt");

    auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc_inst->condition = deopt_cond;
    jcc_inst->add_use(LirOperand::label(deopt_block->id));
    lir_bb.append_inst(std::move(jcc_inst));

    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = inst.resume_id();
    if (inst.symbol() != "@exit_stub" && inst.symbol() != "exit_stub") {
        exit_inst->exit_symbol = std::string(inst.symbol());
    }

    for (const auto* val : inst.state_map()) {
        if (val) {
            VReg vr = get_vreg(val);
            exit_inst->add_use(LirOperand::vreg(vr, vr.size));
        }
    }
    deopt_block->append_inst(std::move(exit_inst));
}

} // namespace brass::aarch64
