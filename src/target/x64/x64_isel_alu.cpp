#include <brass/target/x64/x64_isel.hpp>
#include <cstring>

namespace brass::x64 {

using namespace brass::codegen;

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

void X64ISel::lower_instruction(const Instruction& inst, LirBlock& lir_bb) {
    if (skipped_insts_.count(&inst)) {
        return;
    }

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
        case Opcode::iconst_i64: {
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
        case Opcode::patchable_const_i32: {
            VReg dst = get_vreg(inst.result());
            int32_t val = inst.imm_i32();
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
            lir_inst->is_patchable = true;
            lir_inst->patch_symbol = std::string(inst.symbol());
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::imm(val, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        case Opcode::patchable_const_i64: {
            VReg dst = get_vreg(inst.result());
            int64_t val = inst.imm_i64();
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
            lir_inst->is_patchable = true;
            lir_inst->patch_symbol = std::string(inst.symbol());
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::imm(val, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
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

                auto mq = std::make_unique<LirInst>(LirOpcode::Movq_xg);
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

                auto mq = std::make_unique<LirInst>(LirOpcode::Movq_xg);
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
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg src0 = get_vreg(op0_val);
    VReg src1 = get_vreg(op1_val);

    if (dst.is_xmm()) {
        auto emit_float_alu = [&](VReg first_src, LirOperand second_src) {
            if (dst != first_src) {
                auto mov_inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                mov_inst->add_def(LirOperand::vreg(dst, 8));
                mov_inst->add_use(LirOperand::vreg(first_src, 8));
                lir_bb.append_inst(std::move(mov_inst));
            }

            auto alu_inst = std::make_unique<LirInst>(op_f64);
            alu_inst->add_def(LirOperand::vreg(dst, 8));
            alu_inst->add_use(LirOperand::vreg(dst, 8));
            alu_inst->add_use(second_src);
            alu_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(alu_inst));
        };

        bool is_comm = (inst.opcode() == Opcode::add || inst.opcode() == Opcode::mul || inst.opcode() == Opcode::xor_);
        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_float_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (is_comm && op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_float_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (is_comm && dst == src1) {
            emit_float_alu(src1, LirOperand::vreg(src0, 8));
        } else {
            emit_float_alu(src0, LirOperand::vreg(src1, 8));
        }
        return;
    }

    uint8_t sz = dst.size;
    ImmIntInfo imm0 = get_imm_int_info(op0_val);
    ImmIntInfo imm1 = get_imm_int_info(op1_val);

    auto emit_mov_alu = [&](VReg first_src, LirOperand second_src) {
        if (dst != first_src) {
            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_inst->add_def(LirOperand::vreg(dst, sz));
            mov_inst->add_use(LirOperand::vreg(first_src, sz));
            lir_bb.append_inst(std::move(mov_inst));
        }

        auto alu_inst = std::make_unique<LirInst>(sz == 4 ? op32 : op64);
        alu_inst->add_def(LirOperand::vreg(dst, sz));
        alu_inst->add_use(LirOperand::vreg(dst, sz));
        alu_inst->add_use(second_src);
        alu_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(alu_inst));
    };

    if (inst.opcode() == Opcode::mul) {
        const Value* reg_val = imm1.is_imm ? op0_val : (imm0.is_imm ? op1_val : nullptr);
        ImmIntInfo imm_info = imm1.is_imm ? imm1 : imm0;

        if (reg_val != nullptr) {
            int64_t C = imm_info.val;
            VReg r_vreg = get_vreg(reg_val);

            if (C == 0) {
                auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                lir_inst->add_def(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
                return;
            }
            if (C == 1) {
                if (dst != r_vreg) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(r_vreg, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            if (C == 2 || C == 3 || C == 5 || C == 9) {
                Scale sc = (C == 2) ? Scale::One : ((C == 3) ? Scale::Two : ((C == 5) ? Scale::Four : Scale::Eight));
                auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                lea_inst->add_def(LirOperand::vreg(dst, sz));
                lea_inst->add_use(LirOperand::mem(r_vreg, r_vreg, sc, 0, sz));
                lea_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lea_inst));
                return;
            }
            if (C == 4 || C == 8) {
                Scale sc = (C == 4) ? Scale::Four : Scale::Eight;
                auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                lea_inst->add_def(LirOperand::vreg(dst, sz));
                LirMem m;
                m.index_vreg = r_vreg;
                m.scale = sc;
                lea_inst->add_use(LirOperand::mem_custom(m, sz));
                lea_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lea_inst));
                return;
            }
            if (C > 0 && (static_cast<uint64_t>(C) & static_cast<uint64_t>(C - 1)) == 0) {
                int k = 0;
                while ((1ULL << k) < static_cast<uint64_t>(C)) k++;
                if (dst != r_vreg) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(r_vreg, sz));
                    lir_bb.append_inst(std::move(mov_inst));
                }

                auto shl_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Shl32 : LirOpcode::Shl);
                shl_inst->add_def(LirOperand::vreg(dst, sz));
                shl_inst->add_use(LirOperand::vreg(dst, sz));
                shl_inst->add_use(LirOperand::imm(k, 1));
                shl_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(shl_inst));
                return;
            }
            if (imm_info.fits_i32) {
                auto imul_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Imul32 : LirOpcode::Imul);
                imul_inst->add_def(LirOperand::vreg(dst, sz));
                imul_inst->add_use(LirOperand::vreg(r_vreg, sz));
                imul_inst->add_use(LirOperand::imm(C, sz));
                imul_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(imul_inst));
                return;
            }
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (dst == src1) {
            emit_mov_alu(src1, LirOperand::vreg(src0, sz));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    if (inst.opcode() == Opcode::add) {
        const Value* reg_val = (imm1.is_imm && imm1.fits_i32) ? op0_val : ((imm0.is_imm && imm0.fits_i32) ? op1_val : nullptr);
        ImmIntInfo imm_info = (imm1.is_imm && imm1.fits_i32) ? imm1 : imm0;

        if (reg_val != nullptr) {
            int32_t C = static_cast<int32_t>(imm_info.val);

            // Check if reg_val is a fused instruction (e.g. mul, shl, add)
            if (reg_val->is_instruction() && skipped_insts_.count(reg_val->defining_instruction())) {
                const Instruction* def = reg_val->defining_instruction();
                if (def->opcode() == Opcode::mul) {
                    ImmIntInfo m0 = get_imm_int_info(def->operand(0));
                    ImmIntInfo m1 = get_imm_int_info(def->operand(1));
                    const Value* m_val = m1.is_imm ? def->operand(0) : def->operand(1);
                    int64_t mult = m1.is_imm ? m1.val : m0.val;
                    VReg base_vreg = get_vreg(m_val);

                    if (mult == 2 || mult == 3 || mult == 5 || mult == 9) {
                        Scale sc = (mult == 2) ? Scale::One : ((mult == 3) ? Scale::Two : ((mult == 5) ? Scale::Four : Scale::Eight));
                        auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                        lea_inst->add_def(LirOperand::vreg(dst, sz));
                        lea_inst->add_use(LirOperand::mem(base_vreg, base_vreg, sc, C, sz));
                        lea_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(lea_inst));
                        return;
                    }
                    if (mult == 4 || mult == 8) {
                        Scale sc = (mult == 4) ? Scale::Four : Scale::Eight;
                        auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                        lea_inst->add_def(LirOperand::vreg(dst, sz));
                        LirMem m;
                        m.index_vreg = base_vreg;
                        m.scale = sc;
                        m.disp = C;
                        lea_inst->add_use(LirOperand::mem_custom(m, sz));
                        lea_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(lea_inst));
                        return;
                    }
                } else if (def->opcode() == Opcode::shl) {
                    ImmIntInfo s1 = get_imm_int_info(def->operand(1));
                    if (s1.is_imm && (s1.val == 1 || s1.val == 2 || s1.val == 3)) {
                        Scale sc = (s1.val == 1) ? Scale::Two : ((s1.val == 2) ? Scale::Four : Scale::Eight);
                        VReg base_vreg = get_vreg(def->operand(0));
                        auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                        lea_inst->add_def(LirOperand::vreg(dst, sz));
                        LirMem m;
                        m.index_vreg = base_vreg;
                        m.scale = sc;
                        m.disp = C;
                        lea_inst->add_use(LirOperand::mem_custom(m, sz));
                        lea_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(lea_inst));
                        return;
                    }
                } else if (def->opcode() == Opcode::add) {
                    ImmIntInfo a0 = get_imm_int_info(def->operand(0));
                    ImmIntInfo a1 = get_imm_int_info(def->operand(1));
                    if (!a0.is_imm && !a1.is_imm) {
                        VReg base_vreg = get_vreg(def->operand(0));
                        VReg idx_vreg = get_vreg(def->operand(1));
                        auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                        lea_inst->add_def(LirOperand::vreg(dst, sz));
                        lea_inst->add_use(LirOperand::mem(base_vreg, idx_vreg, Scale::One, C, sz));
                        lea_inst->mir_origin = &inst;
                        lir_bb.append_inst(std::move(lea_inst));
                        return;
                    }
                }
            }

            VReg r_vreg = get_vreg(reg_val);
            if (C == 0) {
                if (dst != r_vreg) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(r_vreg, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            if (dst == r_vreg) {
                auto add_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Add32 : LirOpcode::Add);
                add_inst->add_def(LirOperand::vreg(dst, sz));
                add_inst->add_use(LirOperand::vreg(dst, sz));
                add_inst->add_use(LirOperand::imm(C, sz));
                add_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(add_inst));
                return;
            }
            auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
            lea_inst->add_def(LirOperand::vreg(dst, sz));
            lea_inst->add_use(LirOperand::mem(r_vreg, C, sz));
            lea_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lea_inst));
            return;
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
            emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
        } else if (dst != src0 && dst != src1) {
            auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
            lea_inst->add_def(LirOperand::vreg(dst, sz));
            lea_inst->add_use(LirOperand::mem(src0, src1, Scale::One, 0, sz));
            lea_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lea_inst));
        } else if (dst == src1) {
            emit_mov_alu(src1, LirOperand::vreg(src0, sz));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    if (inst.opcode() == Opcode::sub) {
        if (imm1.is_imm && imm1.fits_i32) {
            int32_t C = static_cast<int32_t>(imm1.val);
            if (C == 0) {
                if (dst != src0) {
                    auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_inst->add_def(LirOperand::vreg(dst, sz));
                    mov_inst->add_use(LirOperand::vreg(src0, sz));
                    mov_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mov_inst));
                }
                return;
            }
            if (dst == src0) {
                auto sub_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Sub32 : LirOpcode::Sub);
                sub_inst->add_def(LirOperand::vreg(dst, sz));
                sub_inst->add_use(LirOperand::vreg(dst, sz));
                sub_inst->add_use(LirOperand::imm(C, sz));
                sub_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(sub_inst));
                return;
            }
            if (C != INT32_MIN) {
                auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
                lea_inst->add_def(LirOperand::vreg(dst, sz));
                lea_inst->add_use(LirOperand::mem(src0, -C, sz));
                lea_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lea_inst));
                return;
            }
        }

        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
        } else if (imm1.is_imm && imm1.fits_i32) {
            emit_mov_alu(src0, LirOperand::imm(imm1.val, sz));
        } else {
            emit_mov_alu(src0, LirOperand::vreg(src1, sz));
        }
        return;
    }

    // and_, or_, xor_
    bool is_comm = (inst.opcode() == Opcode::and_ || inst.opcode() == Opcode::or_ || inst.opcode() == Opcode::xor_);
    const Value* reg_val = (imm1.is_imm && imm1.fits_i32) ? op0_val : ((is_comm && imm0.is_imm && imm0.fits_i32) ? op1_val : nullptr);
    ImmIntInfo imm_info = (imm1.is_imm && imm1.fits_i32) ? imm1 : imm0;

    if (reg_val != nullptr) {
        emit_mov_alu(get_vreg(reg_val), LirOperand::imm(imm_info.val, sz));
        return;
    }

    if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
        emit_mov_alu(src0, get_load_mem_operand(op1_val->defining_instruction()));
    } else if (is_comm && op0_val && op0_val->is_instruction() && can_fuse_load(op0_val->defining_instruction(), &inst)) {
        emit_mov_alu(src1, get_load_mem_operand(op0_val->defining_instruction()));
    } else if (is_comm && dst == src1) {
        emit_mov_alu(src1, LirOperand::vreg(src0, sz));
    } else {
        emit_mov_alu(src0, LirOperand::vreg(src1, sz));
    }
}

void X64ISel::lower_comparison(
    const Instruction& inst,
    LirBlock& lir_bb,
    Condition cond,
    Condition float_cond
) {
    VReg dst = get_vreg(inst.result());
    const Value* op0_val = inst.operand(0);
    const Value* op1_val = inst.operand(1);
    VReg op0 = get_vreg(op0_val);
    VReg op1 = get_vreg(op1_val);
    uint8_t dst_sz = dst.size;

    if (op0_val->type().is_float()) {
        if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
            ucomi->add_use(LirOperand::vreg(op0, 8));
            ucomi->add_use(get_load_mem_operand(op1_val->defining_instruction()));
            lir_bb.append_inst(std::move(ucomi));
        } else {
            auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
            ucomi->add_use(LirOperand::vreg(op0, 8));
            ucomi->add_use(LirOperand::vreg(op1, 8));
            lir_bb.append_inst(std::move(ucomi));
        }

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
        uint8_t sz = static_cast<uint8_t>(op0_val->type().size_in_bytes());
        if (sz == 0) sz = 8;
        Condition final_cond = cond;

        ImmIntInfo imm1 = get_imm_int_info(op1_val);
        ImmIntInfo imm0 = get_imm_int_info(op0_val);

        if (imm1.is_imm && imm1.val == 0) {
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op0, sz));
            test_inst->add_use(LirOperand::vreg(op0, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm0.is_imm && imm0.val == 0) {
            final_cond = swap_relational_condition(cond);
            auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
            test_inst->add_use(LirOperand::vreg(op1, sz));
            test_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(test_inst));
        } else if (imm1.is_imm && imm1.fits_i32) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::imm(imm1.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (imm0.is_imm && imm0.fits_i32) {
            final_cond = swap_relational_condition(cond);
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            cmp_inst->add_use(LirOperand::imm(imm0.val, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        } else if (op1_val && op1_val->is_instruction() && can_fuse_load(op1_val->defining_instruction(), &inst)) {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(get_load_mem_operand(op1_val->defining_instruction()));
            lir_bb.append_inst(std::move(cmp_inst));
        } else {
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            auto cmp_inst = std::make_unique<LirInst>(cmp_lir_op);
            cmp_inst->add_use(LirOperand::vreg(op0, sz));
            cmp_inst->add_use(LirOperand::vreg(op1, sz));
            lir_bb.append_inst(std::move(cmp_inst));
        }

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = final_cond;
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
