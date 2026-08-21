#include <brass/target/x64/x64_isel.hpp>
#include <cstring>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_instruction(const Instruction& inst, LirBlock& lir_bb) {
    switch (inst.opcode()) {
        case Opcode::iconst_i32: {
            VReg dst = get_vreg(inst.result());
            int32_t val = inst.imm_i32();
            if (val == 0) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Xor32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(dst, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::imm(val, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            }
            break;
        }
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64: {
            VReg dst = get_vreg(inst.result());
            int64_t val = inst.imm_i64();
            uint8_t sz = dst.size;
            if (val == 0) {
                auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                lir_inst->add_def(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else if (sz == 4) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::imm(val, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                if (val >= INT32_MIN && val <= INT32_MAX) {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov);
                    lir_inst->add_def(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::imm(val, 8));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                } else {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
                    lir_inst->add_def(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::imm(val, 8));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                }
            }
            break;
        }
        case Opcode::fconst_f64: {
            VReg dst = get_vreg(inst.result());
            double val = inst.imm_f64();
            if (val == 0.0) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Xorpd);
                lir_inst->add_def(LirOperand::vreg(dst, 8));
                lir_inst->add_use(LirOperand::vreg(dst, 8));
                lir_inst->add_use(LirOperand::vreg(dst, 8));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                uint64_t raw_bits = 0;
                std::memcpy(&raw_bits, &val, sizeof(double));
                VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 8);

                auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
                mabs->add_def(LirOperand::vreg(tmp, 8));
                mabs->add_use(LirOperand::imm(static_cast<int64_t>(raw_bits), 8));
                lir_bb.append_inst(std::move(mabs));

                auto mq = std::make_unique<LirInst>(LirOpcode::Movq_gx);
                mq->add_def(LirOperand::vreg(dst, 8));
                mq->add_use(LirOperand::vreg(tmp, 8));
                mq->mir_origin = &inst;
                lir_bb.append_inst(std::move(mq));
            }
            break;
        }
        case Opcode::sext_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movsxd);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::zext_i64:
        case Opcode::trunc_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::fptosi_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvttsd2si32);
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::fptosi_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvttsd2si);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::sitofp_f64_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvtsi2sd32);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::sitofp_f64_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvtsi2sd);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::bitcast_i64_f64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movq_gx);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::bitcast_f64_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movq_xg);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::add:
            lower_binary_alu(inst, lir_bb, LirOpcode::Add32, LirOpcode::Add, LirOpcode::Addsd);
            break;
        case Opcode::sub:
            lower_binary_alu(inst, lir_bb, LirOpcode::Sub32, LirOpcode::Sub, LirOpcode::Subsd);
            break;
        case Opcode::mul:
            lower_binary_alu(inst, lir_bb, LirOpcode::Imul32, LirOpcode::Imul, LirOpcode::Mulsd);
            break;
        case Opcode::sdiv:
            if (inst.type().is_float()) {
                lower_binary_alu(inst, lir_bb, LirOpcode::Nop, LirOpcode::Nop, LirOpcode::Divsd);
            } else {
                lower_div_mod(inst, lir_bb, true, false);
            }
            break;
        case Opcode::udiv:
            lower_div_mod(inst, lir_bb, false, false);
            break;
        case Opcode::smod:
            lower_div_mod(inst, lir_bb, true, true);
            break;
        case Opcode::umod:
            lower_div_mod(inst, lir_bb, false, true);
            break;
        case Opcode::neg: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            if (dst.is_xmm()) {
                uint64_t sign_mask = 0x8000000000000000ULL;
                VReg tmp_gpr = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                VReg tmp_xmm = lir_fn_->allocate_vreg(RegClass::XMM, 8);
                auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
                mabs->add_def(LirOperand::vreg(tmp_gpr, 8));
                mabs->add_use(LirOperand::imm(static_cast<int64_t>(sign_mask), 8));
                lir_bb.append_inst(std::move(mabs));

                auto mq = std::make_unique<LirInst>(LirOpcode::Movq_gx);
                mq->add_def(LirOperand::vreg(tmp_xmm, 8));
                mq->add_use(LirOperand::vreg(tmp_gpr, 8));
                lir_bb.append_inst(std::move(mq));

                auto mov_inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                mov_inst->add_def(LirOperand::vreg(dst, 8));
                mov_inst->add_use(LirOperand::vreg(src, 8));
                lir_bb.append_inst(std::move(mov_inst));

                auto xor_inst = std::make_unique<LirInst>(LirOpcode::Xorpd);
                xor_inst->add_def(LirOperand::vreg(dst, 8));
                xor_inst->add_use(LirOperand::vreg(dst, 8));
                xor_inst->add_use(LirOperand::vreg(tmp_xmm, 8));
                xor_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(xor_inst));
            } else {
                auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_inst->add_def(LirOperand::vreg(dst, sz));
                mov_inst->add_use(LirOperand::vreg(src, sz));
                lir_bb.append_inst(std::move(mov_inst));

                auto neg_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Neg32 : LirOpcode::Neg);
                neg_inst->add_def(LirOperand::vreg(dst, sz));
                neg_inst->add_use(LirOperand::vreg(dst, sz));
                neg_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(neg_inst));
            }
            break;
        }
        case Opcode::and_:
            lower_binary_alu(inst, lir_bb, LirOpcode::And32, LirOpcode::And, LirOpcode::Nop);
            break;
        case Opcode::or_:
            lower_binary_alu(inst, lir_bb, LirOpcode::Or32, LirOpcode::Or, LirOpcode::Nop);
            break;
        case Opcode::xor_:
            lower_binary_alu(inst, lir_bb, LirOpcode::Xor32, LirOpcode::Xor, LirOpcode::Xorpd);
            break;
        case Opcode::not_: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_inst->add_def(LirOperand::vreg(dst, sz));
            mov_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(mov_inst));

            auto not_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Not32 : LirOpcode::Not);
            not_inst->add_def(LirOperand::vreg(dst, sz));
            not_inst->add_use(LirOperand::vreg(dst, sz));
            not_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(not_inst));
            break;
        }
        case Opcode::shl:
            lower_shift(inst, lir_bb, LirOpcode::Shl32, LirOpcode::Shl);
            break;
        case Opcode::lshr:
            lower_shift(inst, lir_bb, LirOpcode::Shr32, LirOpcode::Shr);
            break;
        case Opcode::ashr:
            lower_shift(inst, lir_bb, LirOpcode::Sar32, LirOpcode::Sar);
            break;
        case Opcode::clz: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;

            // 1. bsr dst, src
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Bsr32 : LirOpcode::Bsr);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(lir_inst));

            // 2. dst = (63 or 31) - dst
            VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
            auto mov_tmp = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_tmp->add_def(LirOperand::vreg(tmp, sz));
            mov_tmp->add_use(LirOperand::imm(sz == 4 ? 31 : 63, sz));
            lir_bb.append_inst(std::move(mov_tmp));

            auto sub_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
            sub_inst->add_def(LirOperand::vreg(tmp, sz));
            sub_inst->add_use(LirOperand::vreg(tmp, sz));
            sub_inst->add_use(LirOperand::vreg(dst, sz));
            lir_bb.append_inst(std::move(sub_inst));

            auto mov_back = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_back->add_def(LirOperand::vreg(dst, sz));
            mov_back->add_use(LirOperand::vreg(tmp, sz));
            lir_bb.append_inst(std::move(mov_back));

            // 3. test src, src
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(src, sz));
            test_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(test_inst));

            // 4. if zero, dst = 64 (or 32)
            VReg zero_val = lir_fn_->allocate_vreg(RegClass::GPR, sz);
            auto mov_zero = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_zero->add_def(LirOperand::vreg(zero_val, sz));
            mov_zero->add_use(LirOperand::imm(sz == 4 ? 32 : 64, sz));
            lir_bb.append_inst(std::move(mov_zero));

            auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
            cmov->condition = Condition::E;
            cmov->add_def(LirOperand::vreg(dst, sz));
            cmov->add_use(LirOperand::vreg(dst, sz));
            cmov->add_use(LirOperand::vreg(zero_val, sz));
            cmov->mir_origin = &inst;
            lir_bb.append_inst(std::move(cmov));
            break;
        }
        case Opcode::ctz: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;

            // 1. bsf dst, src
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Bsf32 : LirOpcode::Bsf);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(lir_inst));

            // 2. test src, src
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(src, sz));
            test_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(test_inst));

            // 3. if zero, dst = 64 (or 32)
            VReg zero_val = lir_fn_->allocate_vreg(RegClass::GPR, sz);
            auto mov_zero = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_zero->add_def(LirOperand::vreg(zero_val, sz));
            mov_zero->add_use(LirOperand::imm(sz == 4 ? 32 : 64, sz));
            lir_bb.append_inst(std::move(mov_zero));

            auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
            cmov->condition = Condition::E;
            cmov->add_def(LirOperand::vreg(dst, sz));
            cmov->add_use(LirOperand::vreg(dst, sz));
            cmov->add_use(LirOperand::vreg(zero_val, sz));
            cmov->mir_origin = &inst;
            lir_bb.append_inst(std::move(cmov));
            break;
        }
        case Opcode::popcnt: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Popcnt32 : LirOpcode::Popcnt);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::eq:
            lower_comparison(inst, lir_bb, Condition::E, Condition::E);
            break;
        case Opcode::ne:
            lower_comparison(inst, lir_bb, Condition::NE, Condition::NE);
            break;
        case Opcode::slt:
            lower_comparison(inst, lir_bb, Condition::L, Condition::B);
            break;
        case Opcode::ult:
            lower_comparison(inst, lir_bb, Condition::B, Condition::B);
            break;
        case Opcode::sle:
            lower_comparison(inst, lir_bb, Condition::LE, Condition::BE);
            break;
        case Opcode::ule:
            lower_comparison(inst, lir_bb, Condition::BE, Condition::BE);
            break;
        case Opcode::sgt:
            lower_comparison(inst, lir_bb, Condition::G, Condition::A);
            break;
        case Opcode::ugt:
            lower_comparison(inst, lir_bb, Condition::A, Condition::A);
            break;
        case Opcode::sge:
            lower_comparison(inst, lir_bb, Condition::GE, Condition::AE);
            break;
        case Opcode::uge:
            lower_comparison(inst, lir_bb, Condition::AE, Condition::AE);
            break;
        case Opcode::load:
            lower_load(inst, lir_bb);
            break;
        case Opcode::store:
            lower_store(inst, lir_bb);
            break;
        case Opcode::load_indexed:
            lower_load_indexed(inst, lir_bb);
            break;
        case Opcode::store_indexed:
            lower_store_indexed(inst, lir_bb);
            break;
        case Opcode::call:
        case Opcode::call_indirect:
        case Opcode::patchable_call:
            lower_call(inst, lir_bb);
            break;
        case Opcode::safepoint:
            lower_safepoint(inst, lir_bb);
            break;
        case Opcode::guard:
            lower_guard(inst, lir_bb);
            break;
        case Opcode::resume_point:
            break;
        case Opcode::br:
            lower_branch(inst, lir_bb);
            break;
        case Opcode::br_if:
            lower_branch_if(inst, lir_bb);
            break;
        case Opcode::ret:
            lower_return(inst, lir_bb);
            break;
        case Opcode::unreachable: {
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Nop);
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
    }
}

void X64ISel::lower_binary_alu(
    const Instruction& inst,
    LirBlock& lir_bb,
    LirOpcode op32,
    LirOpcode op64,
    LirOpcode op_f64
) {
    VReg dst = get_vreg(inst.result());
    VReg src0 = get_vreg(inst.operand(0));
    VReg src1 = get_vreg(inst.operand(1));

    if (dst.is_xmm()) {
        auto mov_inst = std::make_unique<LirInst>(LirOpcode::Movsd);
        mov_inst->add_def(LirOperand::vreg(dst, 8));
        mov_inst->add_use(LirOperand::vreg(src0, 8));
        lir_bb.append_inst(std::move(mov_inst));

        auto alu_inst = std::make_unique<LirInst>(op_f64);
        alu_inst->add_def(LirOperand::vreg(dst, 8));
        alu_inst->add_use(LirOperand::vreg(dst, 8));
        alu_inst->add_use(LirOperand::vreg(src1, 8));
        alu_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(alu_inst));
    } else {
        uint8_t sz = dst.size;
        auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        mov_inst->add_def(LirOperand::vreg(dst, sz));
        mov_inst->add_use(LirOperand::vreg(src0, sz));
        lir_bb.append_inst(std::move(mov_inst));

        auto alu_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
        alu_inst->add_def(LirOperand::vreg(dst, sz));
        alu_inst->add_use(LirOperand::vreg(dst, sz));
        alu_inst->add_use(LirOperand::vreg(src1, sz));
        alu_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(alu_inst));
    }
}

void X64ISel::lower_div_mod(
    const Instruction& inst,
    LirBlock& lir_bb,
    bool is_signed,
    bool is_mod
) {
    VReg dst = get_vreg(inst.result());
    VReg op0 = get_vreg(inst.operand(0));
    VReg op1 = get_vreg(inst.operand(1));
    uint8_t sz = dst.size;

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
    VReg op0 = get_vreg(inst.operand(0));
    VReg op1 = get_vreg(inst.operand(1));
    uint8_t sz = dst.size;

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

void X64ISel::lower_comparison(
    const Instruction& inst,
    LirBlock& lir_bb,
    Condition cond,
    Condition float_cond
) {
    VReg dst = get_vreg(inst.result());
    VReg op0 = get_vreg(inst.operand(0));
    VReg op1 = get_vreg(inst.operand(1));
    uint8_t dst_sz = dst.size;

    if (op0.is_xmm()) {
        auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
        ucomi->add_use(LirOperand::vreg(op0, 8));
        ucomi->add_use(LirOperand::vreg(op1, 8));
        lir_bb.append_inst(std::move(ucomi));

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = float_cond;
        setcc->add_def(LirOperand::vreg(dst, 1));
        lir_bb.append_inst(std::move(setcc));

        auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
        movzx->add_def(LirOperand::vreg(dst, dst_sz));
        movzx->add_use(LirOperand::vreg(dst, 1));
        movzx->mir_origin = &inst;
        lir_bb.append_inst(std::move(movzx));
    } else {
        uint8_t sz = op0.size;
        auto cmp_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Cmp32 : LirOpcode::Cmp);
        cmp_inst->add_use(LirOperand::vreg(op0, sz));
        cmp_inst->add_use(LirOperand::vreg(op1, sz));
        lir_bb.append_inst(std::move(cmp_inst));

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = cond;
        setcc->add_def(LirOperand::vreg(dst, 1));
        lir_bb.append_inst(std::move(setcc));

        auto movzx = std::make_unique<LirInst>(LirOpcode::Movzx8);
        movzx->add_def(LirOperand::vreg(dst, dst_sz));
        movzx->add_use(LirOperand::vreg(dst, 1));
        movzx->mir_origin = &inst;
        lir_bb.append_inst(std::move(movzx));
    }
}

} // namespace brass::x64
