#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/div_magic.hpp>

namespace brass::x64 {

using namespace brass::codegen;

// sdiv / smod i64 by a constant d >= 3 that is not a power of two
// (div_magic.hpp): a one-operand imul gives the high half of M * n in RDX,
// about a quarter of an idiv's latency, and the dependent work after it is
// adds and shifts. Its sign correction reads n's own sign bit, which does not
// wait for the multiply. The divisor's constant is then read as an immediate
// (or, past 32 bits, from its register) by the remainder's multiply only.
void X64ISel::lower_signed_div_magic(const Instruction& inst, LirBlock& lir_bb, bool is_mod, const ImmIntInfo& divisor) {
    const SignedDivMagic mg = signed_div_magic64(divisor.val);
    VReg dst = get_vreg(inst.result());
    VReg n = get_vreg(inst.operand(0));

    auto mov_m = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_m->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
    mov_m->add_use(LirOperand::imm(mg.multiplier, 8));
    lir_bb.append_inst(std::move(mov_m));

    // imul n: RDX:RAX = RAX * n, signed.
    auto mulh = std::make_unique<LirInst>(LirOpcode::Smulh);
    mulh->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
    mulh->add_def(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
    mulh->add_use(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
    mulh->add_use(LirOperand::vreg(n, 8));
    lir_bb.append_inst(std::move(mulh));

    VReg q = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto mov_t = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_t->add_def(LirOperand::vreg(q, 8));
    mov_t->add_use(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
    lir_bb.append_inst(std::move(mov_t));

    if (mg.multiplier < 0) {
        auto add_n = std::make_unique<LirInst>(LirOpcode::Add);
        add_n->add_def(LirOperand::vreg(q, 8));
        add_n->add_use(LirOperand::vreg(q, 8));
        add_n->add_use(LirOperand::vreg(n, 8));
        lir_bb.append_inst(std::move(add_n));
    }
    if (mg.shift > 0) {
        auto sar = std::make_unique<LirInst>(LirOpcode::Sar);
        sar->add_def(LirOperand::vreg(q, 8));
        sar->add_use(LirOperand::vreg(q, 8));
        sar->add_use(LirOperand::imm(mg.shift, 1));
        lir_bb.append_inst(std::move(sar));
    }

    VReg sign = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto mov_s = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_s->add_def(LirOperand::vreg(sign, 8));
    mov_s->add_use(LirOperand::vreg(n, 8));
    lir_bb.append_inst(std::move(mov_s));
    auto shr_s = std::make_unique<LirInst>(LirOpcode::Shr);
    shr_s->add_def(LirOperand::vreg(sign, 8));
    shr_s->add_use(LirOperand::vreg(sign, 8));
    shr_s->add_use(LirOperand::imm(63, 1));
    lir_bb.append_inst(std::move(shr_s));

    auto add_s = std::make_unique<LirInst>(LirOpcode::Add);
    add_s->add_def(LirOperand::vreg(is_mod ? q : dst, 8));
    if (!is_mod) {
        auto mov_d = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_d->add_def(LirOperand::vreg(dst, 8));
        mov_d->add_use(LirOperand::vreg(q, 8));
        lir_bb.append_inst(std::move(mov_d));
        add_s->add_use(LirOperand::vreg(dst, 8));
        add_s->add_use(LirOperand::vreg(sign, 8));
        add_s->mir_origin = &inst;
        lir_bb.append_inst(std::move(add_s));
        return;
    }
    add_s->add_use(LirOperand::vreg(q, 8));
    add_s->add_use(LirOperand::vreg(sign, 8));
    lir_bb.append_inst(std::move(add_s));

    // r = n - q * d
    VReg prod = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto mul_d = std::make_unique<LirInst>(LirOpcode::Imul);
    mul_d->add_def(LirOperand::vreg(prod, 8));
    if (divisor.fits_i32) {
        mul_d->add_use(LirOperand::vreg(q, 8));
        mul_d->add_use(LirOperand::imm(divisor.val, 8));
    } else {
        auto mov_p = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_p->add_def(LirOperand::vreg(prod, 8));
        mov_p->add_use(LirOperand::vreg(q, 8));
        lir_bb.append_inst(std::move(mov_p));
        mul_d->add_use(LirOperand::vreg(prod, 8));
        mul_d->add_use(LirOperand::vreg(get_vreg(inst.operand(1)), 8));
    }
    lir_bb.append_inst(std::move(mul_d));

    auto mov_r = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_r->add_def(LirOperand::vreg(dst, 8));
    mov_r->add_use(LirOperand::vreg(n, 8));
    lir_bb.append_inst(std::move(mov_r));
    auto sub_r = std::make_unique<LirInst>(LirOpcode::Sub);
    sub_r->add_def(LirOperand::vreg(dst, 8));
    sub_r->add_use(LirOperand::vreg(dst, 8));
    sub_r->add_use(LirOperand::vreg(prod, 8));
    sub_r->mir_origin = &inst;
    lir_bb.append_inst(std::move(sub_r));
}

bool X64ISel::divisor_folds(const ImmIntInfo& divisor, bool is_mod) noexcept {
    if (!divisor.is_imm || divisor.val <= 0) return false;
    const uint64_t c = static_cast<uint64_t>(divisor.val);
    if ((c & (c - 1)) != 0) return false;
    // A remainder masks with C - 1 (unsigned) or ~(C - 1) (signed), which
    // an and-immediate sign-extends from 32 bits: only 2^0 .. 2^31 fit.
    return !is_mod || c <= (uint64_t{1} << 31);
}

bool X64ISel::divisor_unread(const Instruction& inst) const {
    const Opcode op = inst.opcode();
    const bool is_mod = op == Opcode::umod || op == Opcode::smod;
    const ImmIntInfo dv = get_imm_int_info(inst.operand(1));
    // lower_div_mod's first choice: the magic-number lowering reads the
    // divisor only for a remainder, and then as an immediate when it fits.
    const bool magic = (op == Opcode::sdiv || op == Opcode::smod) && inst.type() == Type::i64() && dv.is_imm &&
                       takes_signed_div_magic(dv.val) && signed_div_magic_enabled();
    if (magic) return !is_mod || dv.fits_i32;
    return divisor_folds(dv, is_mod);
}

bool X64ISel::mul_imm_folds(const ImmIntInfo& factor) noexcept {
    if (!factor.is_imm) return false;
    if (factor.fits_i32) return true;
    const uint64_t c = static_cast<uint64_t>(factor.val);
    return factor.val > 0 && (c & (c - 1)) == 0;
}

void X64ISel::lower_div_mod(
    const Instruction& inst,
    LirBlock& lir_bb,
    bool is_signed,
    bool is_mod
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    uint8_t sz = dst.size;

    ImmIntInfo imm1 = get_imm_int_info(op1_val);
    if (is_signed && sz == 8 && imm1.is_imm && takes_signed_div_magic(imm1.val) && signed_div_magic_enabled()) {
        lower_signed_div_magic(inst, lir_bb, is_mod, imm1);
        return;
    }
    if (divisor_folds(imm1, is_mod)) {
        uint64_t C = static_cast<uint64_t>(imm1.val);
        {
            int k = 0;
            while ((1ULL << k) < C) k++;
            if (!is_signed) {
                if (!is_mod) {
                    if (k == 0) {
                        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_inst->add_def(LirOperand::vreg(dst, sz));
                        mov_inst->add_use(LirOperand::vreg(op0, sz));
                        mov_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(mov_inst));
                    } else {
                        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_inst->add_def(LirOperand::vreg(dst, sz));
                        mov_inst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_inst));

                        auto shr_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_inst->add_def(LirOperand::vreg(dst, sz));
                        shr_inst->add_use(LirOperand::vreg(dst, sz));
                        shr_inst->add_use(LirOperand::imm(k, 1));
                        shr_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(shr_inst));
                    }
                    return;
                } else {
                    if (k == 0) {
                        auto zero_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                        zero_inst->add_def(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(zero_inst));
                        return;
                    } else {
                        uint64_t mask = C - 1;
                        if (mask <= INT32_MAX) {
                            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                            mov_inst->add_def(LirOperand::vreg(dst, sz));
                            mov_inst->add_use(LirOperand::vreg(op0, sz));
                            lir_bb.append_inst(std::move(mov_inst));

                            auto and_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::And32 : LirOpcode::And);
                            and_inst->add_def(LirOperand::vreg(dst, sz));
                            and_inst->add_use(LirOperand::vreg(dst, sz));
                            and_inst->add_use(LirOperand::imm(static_cast<int32_t>(mask), sz));
                            and_inst->mir_origin = &inst;
                            lir_bb.append_inst(std::move(and_inst));
                            return;
                        }
                    }
                }
            } else {
                // Signed division / mod by constant power-of-2 C = 2^k
                if (!is_mod) {
                    if (k == 0) {
                        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_inst->add_def(LirOperand::vreg(dst, sz));
                        mov_inst->add_use(LirOperand::vreg(op0, sz));
                        mov_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(mov_inst));
                        return;
                    }

                    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
                    if (k == 1) {
                        auto mov_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_tmp->add_def(LirOperand::vreg(tmp, sz));
                        mov_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_tmp));

                        auto shr_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_tmp->add_def(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::imm(sz == 4 ? 31 : 63, 1));
                        lir_bb.append_inst(std::move(shr_tmp));

                        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_dst->add_def(LirOperand::vreg(dst, sz));
                        mov_dst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_dst));

                        auto add_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Add32 : LirOpcode::Add);
                        add_dst->add_def(LirOperand::vreg(dst, sz));
                        add_dst->add_use(LirOperand::vreg(dst, sz));
                        add_dst->add_use(LirOperand::vreg(tmp, sz));
                        lir_bb.append_inst(std::move(add_dst));

                        auto sar_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sar32 : LirOpcode::Sar);
                        sar_dst->add_def(LirOperand::vreg(dst, sz));
                        sar_dst->add_use(LirOperand::vreg(dst, sz));
                        sar_dst->add_use(LirOperand::imm(1, 1));
                        sar_dst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(sar_dst));
                    } else {
                        auto mov_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_tmp->add_def(LirOperand::vreg(tmp, sz));
                        mov_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_tmp));

                        auto sar_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sar32 : LirOpcode::Sar);
                        sar_tmp->add_def(LirOperand::vreg(tmp, sz));
                        sar_tmp->add_use(LirOperand::vreg(tmp, sz));
                        sar_tmp->add_use(LirOperand::imm(sz == 4 ? 31 : 63, 1));
                        lir_bb.append_inst(std::move(sar_tmp));

                        auto shr_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_tmp->add_def(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::imm((sz == 4 ? 32 : 64) - k, 1));
                        lir_bb.append_inst(std::move(shr_tmp));

                        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_dst->add_def(LirOperand::vreg(dst, sz));
                        mov_dst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_dst));

                        auto add_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Add32 : LirOpcode::Add);
                        add_dst->add_def(LirOperand::vreg(dst, sz));
                        add_dst->add_use(LirOperand::vreg(dst, sz));
                        add_dst->add_use(LirOperand::vreg(tmp, sz));
                        lir_bb.append_inst(std::move(add_dst));

                        auto sar_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sar32 : LirOpcode::Sar);
                        sar_dst->add_def(LirOperand::vreg(dst, sz));
                        sar_dst->add_use(LirOperand::vreg(dst, sz));
                        sar_dst->add_use(LirOperand::imm(k, 1));
                        sar_dst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(sar_dst));
                    }
                    return;
                } else {
                    if (k == 0) {
                        auto zero_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                        zero_inst->add_def(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->add_use(LirOperand::vreg(dst, sz));
                        zero_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(zero_inst));
                        return;
                    }

                    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
                    if (k == 1) {
                        auto mov_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_tmp->add_def(LirOperand::vreg(tmp, sz));
                        mov_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_tmp));

                        auto shr_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_tmp->add_def(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::imm(sz == 4 ? 31 : 63, 1));
                        lir_bb.append_inst(std::move(shr_tmp));

                        auto add_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Add32 : LirOpcode::Add);
                        add_tmp->add_def(LirOperand::vreg(tmp, sz));
                        add_tmp->add_use(LirOperand::vreg(tmp, sz));
                        add_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(add_tmp));

                        auto and_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::And32 : LirOpcode::And);
                        and_tmp->add_def(LirOperand::vreg(tmp, sz));
                        and_tmp->add_use(LirOperand::vreg(tmp, sz));
                        and_tmp->add_use(LirOperand::imm(-2, sz));
                        lir_bb.append_inst(std::move(and_tmp));

                        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_dst->add_def(LirOperand::vreg(dst, sz));
                        mov_dst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_dst));

                        auto sub_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
                        sub_dst->add_def(LirOperand::vreg(dst, sz));
                        sub_dst->add_use(LirOperand::vreg(dst, sz));
                        sub_dst->add_use(LirOperand::vreg(tmp, sz));
                        sub_dst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(sub_dst));
                    } else {
                        auto mov_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_tmp->add_def(LirOperand::vreg(tmp, sz));
                        mov_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_tmp));

                        auto sar_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sar32 : LirOpcode::Sar);
                        sar_tmp->add_def(LirOperand::vreg(tmp, sz));
                        sar_tmp->add_use(LirOperand::vreg(tmp, sz));
                        sar_tmp->add_use(LirOperand::imm(sz == 4 ? 31 : 63, 1));
                        lir_bb.append_inst(std::move(sar_tmp));

                        auto shr_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shr32 : LirOpcode::Shr);
                        shr_tmp->add_def(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::vreg(tmp, sz));
                        shr_tmp->add_use(LirOperand::imm((sz == 4 ? 32 : 64) - k, 1));
                        lir_bb.append_inst(std::move(shr_tmp));

                        auto add_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Add32 : LirOpcode::Add);
                        add_tmp->add_def(LirOperand::vreg(tmp, sz));
                        add_tmp->add_use(LirOperand::vreg(tmp, sz));
                        add_tmp->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(add_tmp));

                        int32_t mask = static_cast<int32_t>(~(C - 1));
                        auto and_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::And32 : LirOpcode::And);
                        and_tmp->add_def(LirOperand::vreg(tmp, sz));
                        and_tmp->add_use(LirOperand::vreg(tmp, sz));
                        and_tmp->add_use(LirOperand::imm(mask, sz));
                        lir_bb.append_inst(std::move(and_tmp));

                        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                        mov_dst->add_def(LirOperand::vreg(dst, sz));
                        mov_dst->add_use(LirOperand::vreg(op0, sz));
                        lir_bb.append_inst(std::move(mov_dst));

                        auto sub_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
                        sub_dst->add_def(LirOperand::vreg(dst, sz));
                        sub_dst->add_use(LirOperand::vreg(dst, sz));
                        sub_dst->add_use(LirOperand::vreg(tmp, sz));
                        sub_dst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(sub_dst));
                    }
                    return;
                }
            }
        }
    }

    VReg op1 = get_vreg(op1_val);

    // 1. Move op0 into RAX
    auto mov_rax = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_rax->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    mov_rax->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_rax));

    // 2. Sign-extend RAX into RDX:RAX via cdq/cqo, or zero RDX
    if (is_signed) {
        auto extend_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Cdq : LirOpcode::Cqo);
        extend_inst->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        extend_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
        lir_bb.append_inst(std::move(extend_inst));
    } else {
        auto zero_rdx = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
        zero_rdx->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        zero_rdx->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
        lir_bb.append_inst(std::move(zero_rdx));
    }

    // 3. Emit idiv or div
    LirOpcode div_op;
    if (is_signed) {
        div_op = sz == 4 ? LirOpcode::Idiv32 : LirOpcode::Idiv;
    } else {
        div_op = sz == 4 ? LirOpcode::Div32 : LirOpcode::Div;
    }

    auto div_inst = std::make_unique<LirInst>(div_op);
    div_inst->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    div_inst->add_def(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
    div_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
    div_inst->add_use(LirOperand::preg_gpr(GPR::RDX, sz), FixedConstraint::gpr(GPR::RDX));
    div_inst->add_use(LirOperand::vreg(op1, sz));
    div_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(div_inst));

    // 4. Result is in RAX (div) or RDX (mod)
    GPR res_reg = is_mod ? GPR::RDX : GPR::RAX;
    auto mov_res = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_res->add_def(LirOperand::vreg(dst, sz));
    mov_res->add_use(LirOperand::preg_gpr(res_reg, sz), FixedConstraint::gpr(res_reg));
    lir_bb.append_inst(std::move(mov_res));
}

void X64ISel::lower_shift(
    const Instruction& inst,
    LirBlock& lir_bb,
    LirOpcode op32,
    LirOpcode op64
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    uint8_t sz = dst.size;

    ImmIntInfo imm1 = get_imm_int_info(op1_val);
    if (imm1.is_imm) {
        uint8_t shift_amt = static_cast<uint8_t>(imm1.val & (sz == 4 ? 31 : 63));
        if (shift_amt == 0) {
            auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_dst->add_def(LirOperand::vreg(dst, sz));
            mov_dst->add_use(LirOperand::vreg(op0, sz));
            mov_dst->mir_origin = &inst;
            lir_bb.append_inst(std::move(mov_dst));
            return;
        }

        auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_dst->add_def(LirOperand::vreg(dst, sz));
        mov_dst->add_use(LirOperand::vreg(op0, sz));
        lir_bb.append_inst(std::move(mov_dst));

        auto shift_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
        shift_inst->add_def(LirOperand::vreg(dst, sz));
        shift_inst->add_use(LirOperand::vreg(dst, sz));
        shift_inst->add_use(LirOperand::imm(shift_amt, 1));
        shift_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(shift_inst));
        return;
    }

    VReg op1 = get_vreg(op1_val);
    // 1. Copy op0 to dst
    auto mov_dst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    mov_dst->add_def(LirOperand::vreg(dst, sz));
    mov_dst->add_use(LirOperand::vreg(op0, sz));
    lir_bb.append_inst(std::move(mov_dst));

    // 2. x86 requires shift amount in CL (RCX)
    auto mov_cl = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_cl->add_def(LirOperand::preg_gpr(GPR::RCX, 8), FixedConstraint::gpr(GPR::RCX));
    mov_cl->add_use(LirOperand::vreg(op1, op1.size));
    lir_bb.append_inst(std::move(mov_cl));

    // 3. Shift dst by CL
    auto shift_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
    shift_inst->add_def(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::vreg(dst, sz));
    shift_inst->add_use(LirOperand::preg_gpr(GPR::RCX, 1), FixedConstraint::gpr(GPR::RCX));
    shift_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(shift_inst));
}

} // namespace brass::x64
