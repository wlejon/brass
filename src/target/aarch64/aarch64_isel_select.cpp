#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;
using LirCond = brass::x64::Condition;

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
        default: return {LirCond::None, LirCond::None};
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

// See x64_isel_mem.cpp: the pinned register sits outside the allocator's
// pool, so these are plain copies. `mov xd, sp` is `add xd, sp, #0`.
void AArch64ISel::lower_pinned_tls_read(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_aarch64_gpr(kPinnedTlsGpr, 8), FixedConstraint::aarch64_gpr(kPinnedTlsGpr));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void AArch64ISel::lower_pinned_tls_write(const Instruction& inst, LirBlock& lir_bb) {
    VReg src = get_vreg(inst.operand(0));
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::preg_aarch64_gpr(kPinnedTlsGpr, 8), FixedConstraint::aarch64_gpr(kPinnedTlsGpr));
    mov->add_use(LirOperand::vreg(src, 8));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
    lir_fn_->forced_saved_gprs |= reg_mask(kPinnedTlsGpr);
}

void AArch64ISel::lower_read_sp(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_aarch64_gpr(GPR::SP, 8), FixedConstraint::aarch64_gpr(GPR::SP));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void AArch64ISel::lower_select(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const Value* true_val = inst.operand(1);
    const Value* false_val = inst.operand(2);

    VReg dst = get_vreg(inst.result());
    bool is_float = dst.is_xmm();
    uint8_t sz = dst.size;
    if (sz == 0) sz = 8;

    VReg target_reg = dst;
    VReg g_false{};
    VReg g_true{};

    if (is_float) {
        g_false = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        g_true = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        target_reg = g_false;

        VReg f_vreg = get_vreg(false_val);
        auto mq_f = std::make_unique<LirInst>(LirOpcode::Movq_gx);
        mq_f->add_def(LirOperand::vreg(g_false, 8));
        mq_f->add_use(LirOperand::vreg(f_vreg, 8));
        lir_bb.append_inst(std::move(mq_f));

        VReg t_vreg = get_vreg(true_val);
        auto mq_t = std::make_unique<LirInst>(LirOpcode::Movq_gx);
        mq_t->add_def(LirOperand::vreg(g_true, 8));
        mq_t->add_use(LirOperand::vreg(t_vreg, 8));
        lir_bb.append_inst(std::move(mq_t));
    } else {
        ImmIntInfo false_imm = get_imm_int_info(false_val);
        if (false_imm.is_imm) {
            if (false_imm.val == 0) {
                auto xor_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                xor_inst->add_def(LirOperand::vreg(dst, sz));
                xor_inst->add_use(LirOperand::vreg(dst, sz));
                xor_inst->add_use(LirOperand::vreg(dst, sz));
                lir_bb.append_inst(std::move(xor_inst));
            } else if (sz == 4 || (false_imm.val >= INT32_MIN && false_imm.val <= INT32_MAX)) {
                auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_inst->add_def(LirOperand::vreg(dst, sz));
                mov_inst->add_use(LirOperand::imm(false_imm.val, sz));
                lir_bb.append_inst(std::move(mov_inst));
            } else {
                auto movabs_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
                movabs_inst->add_def(LirOperand::vreg(dst, 8));
                movabs_inst->add_use(LirOperand::imm(false_imm.val, 8));
                lir_bb.append_inst(std::move(movabs_inst));
            }
        } else {
            VReg f_vreg = get_vreg(false_val);
            if (dst != f_vreg) {
                auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_inst->add_def(LirOperand::vreg(dst, sz));
                mov_inst->add_use(LirOperand::vreg(f_vreg, sz));
                lir_bb.append_inst(std::move(mov_inst));
            }
        }
    }

    VReg src_vreg{};
    if (is_float) {
        src_vreg = g_true;
    } else {
        ImmIntInfo true_imm = get_imm_int_info(true_val);
        if (true_imm.is_imm) {
            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
            if (true_imm.val == 0) {
                auto xor_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                xor_inst->add_def(LirOperand::vreg(tmp, sz));
                xor_inst->add_use(LirOperand::vreg(tmp, sz));
                xor_inst->add_use(LirOperand::vreg(tmp, sz));
                lir_bb.append_inst(std::move(xor_inst));
            } else if (sz == 4 || (true_imm.val >= INT32_MIN && true_imm.val <= INT32_MAX)) {
                auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_inst->add_def(LirOperand::vreg(tmp, sz));
                mov_inst->add_use(LirOperand::imm(true_imm.val, sz));
                lir_bb.append_inst(std::move(mov_inst));
            } else {
                auto movabs_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
                movabs_inst->add_def(LirOperand::vreg(tmp, 8));
                movabs_inst->add_use(LirOperand::imm(true_imm.val, 8));
                lir_bb.append_inst(std::move(movabs_inst));
            }
            src_vreg = tmp;
        } else {
            src_vreg = get_vreg(true_val);
        }
    }

    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode()) && skipped_insts_.count(cmp_inst);

    LirCond select_cond = LirCond::NE;

    if (is_fused_cmp) {
        Opcode cmp_op = cmp_inst->opcode();
        auto [gpr_c, float_c] = get_comparison_conditions(cmp_op);
        const Value* lhs = cmp_inst->operand(0);
        const Value* rhs = cmp_inst->operand(1);

        if (lhs->type().is_float()) {
            select_cond = float_c;
            if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(ucomi));
            } else {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(LirOperand::vreg(get_vreg(rhs), 8));
                lir_bb.append_inst(std::move(ucomi));
            }
        } else {
            uint8_t cmp_sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
            if (cmp_sz == 0) cmp_sz = 8;
            LirOpcode cmp_lir_op = (cmp_sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            LirOpcode test_lir_op = (cmp_sz == 4) ? LirOpcode::Test32 : LirOpcode::Test;

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
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), cmp_sz));
                        test_lir->add_use(LirOperand::imm(imm_b.val, cmp_sz));
                    } else if (imm_a.is_imm && imm_a.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), cmp_sz));
                        test_lir->add_use(LirOperand::imm(imm_a.val, cmp_sz));
                    } else {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), cmp_sz));
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), cmp_sz));
                    }
                    lir_bb.append_inst(std::move(test_lir));
                    select_cond = (cmp_op == Opcode::eq) ? LirCond::E : LirCond::NE;
                } else {
                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    VReg reg = get_vreg(non_zero);
                    test_lir->add_use(LirOperand::vreg(reg, cmp_sz));
                    test_lir->add_use(LirOperand::vreg(reg, cmp_sz));
                    lir_bb.append_inst(std::move(test_lir));
                    select_cond = (rhs_imm.is_imm && rhs_imm.val == 0) ? gpr_c : swap_relational_condition(gpr_c);
                }
            } else if (rhs_imm.is_imm && rhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), cmp_sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), cmp_sz));
                lir_bb.append_inst(std::move(test_lir));
                select_cond = gpr_c;
            } else if (lhs_imm.is_imm && lhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), cmp_sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), cmp_sz));
                lir_bb.append_inst(std::move(test_lir));
                select_cond = swap_relational_condition(gpr_c);
            } else if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                select_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), cmp_sz));
                cmp_lir->add_use(LirOperand::imm(rhs_imm.val, cmp_sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                select_cond = swap_relational_condition(gpr_c);
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), cmp_sz));
                cmp_lir->add_use(LirOperand::imm(lhs_imm.val, cmp_sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                select_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), cmp_sz));
                cmp_lir->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(cmp_lir));
            } else {
                select_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), cmp_sz));
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), cmp_sz));
                lir_bb.append_inst(std::move(cmp_lir));
            }
        }
    } else {
        VReg c_reg = get_vreg(cond_val);
        uint8_t c_sz = c_reg.size;
        if (c_sz == 0) c_sz = 4;
        auto test_inst = std::make_unique<LirInst>(c_sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(c_reg, c_sz));
        test_inst->add_use(LirOperand::vreg(c_reg, c_sz));
        lir_bb.append_inst(std::move(test_inst));
        select_cond = LirCond::NE;
    }

    uint8_t op_sz = is_float ? 8 : sz;
    auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
    cmov->condition = select_cond;
    cmov->add_def(LirOperand::vreg(target_reg, op_sz));
    cmov->add_use(LirOperand::vreg(target_reg, op_sz));
    cmov->add_use(LirOperand::vreg(src_vreg, op_sz));
    cmov->mir_origin = &inst;
    lir_bb.append_inst(std::move(cmov));

    if (is_float) {
        auto mq_res = std::make_unique<LirInst>(LirOpcode::Movq_xg);
        mq_res->add_def(LirOperand::vreg(dst, 8));
        mq_res->add_use(LirOperand::vreg(g_false, 8));
        lir_bb.append_inst(std::move(mq_res));
    }
}

} // namespace brass::aarch64
