#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_fp_instruction(const Instruction& inst, LirBlock& lir_bb) {
    auto emit_unary_fp = [&](LirOpcode op, uint8_t dst_sz, uint8_t src_sz) {
        VReg dst = get_vreg(inst.result());
        const Value* src_val = inst.operand(0);
        VReg src = get_vreg(src_val);

        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(LirOperand::vreg(dst, dst_sz));
        lir_inst->add_use(LirOperand::vreg(src, src_sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    };

    switch (inst.opcode()) {
        case Opcode::sqrt_f32:
            emit_unary_fp(LirOpcode::Sqrtss, 4, 4);
            break;
        case Opcode::sqrt_f64:
            emit_unary_fp(LirOpcode::Sqrtsd, 8, 8);
            break;
        case Opcode::floor_f32:
            emit_unary_fp(LirOpcode::Floor32, 4, 4);
            break;
        case Opcode::floor_f64:
            emit_unary_fp(LirOpcode::Floor64, 8, 8);
            break;
        case Opcode::ceil_f32:
            emit_unary_fp(LirOpcode::Ceil32, 4, 4);
            break;
        case Opcode::ceil_f64:
            emit_unary_fp(LirOpcode::Ceil64, 8, 8);
            break;
        case Opcode::round_f32:
            emit_unary_fp(LirOpcode::Round32, 4, 4);
            break;
        case Opcode::round_f64:
            emit_unary_fp(LirOpcode::Round64, 8, 8);
            break;
        case Opcode::fabs_f32:
            emit_unary_fp(LirOpcode::Fabs32, 4, 4);
            break;
        case Opcode::fabs_f64:
            emit_unary_fp(LirOpcode::Fabs64, 8, 8);
            break;
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:
            lower_binary_alu(inst, lir_bb, LirOpcode::Nop, LirOpcode::Nop, LirOpcode::Minsd, LirOpcode::Minss);
            break;
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:
            lower_binary_alu(inst, lir_bb, LirOpcode::Nop, LirOpcode::Nop, LirOpcode::Maxsd, LirOpcode::Maxss);
            break;
        case Opcode::sitofp_f32_i32:
            emit_unary_fp(LirOpcode::Cvtsi2ss32, 4, 4);
            break;
        case Opcode::sitofp_f32_i64:
            emit_unary_fp(LirOpcode::Cvtsi2ss, 4, 8);
            break;
        case Opcode::sitofp_f64_i32:
            emit_unary_fp(LirOpcode::Cvtsi2sd32, 8, 4);
            break;
        case Opcode::sitofp_f64_i64:
            emit_unary_fp(LirOpcode::Cvtsi2sd, 8, 8);
            break;
        case Opcode::fptosi_i32_f32:
            emit_unary_fp(LirOpcode::Cvttss2si32, 4, 4);
            break;
        case Opcode::fptosi_i64_f32:
            emit_unary_fp(LirOpcode::Cvttss2si, 8, 4);
            break;
        case Opcode::fptosi_i32:
            emit_unary_fp(LirOpcode::Cvttsd2si32, 4, 8);
            break;
        case Opcode::fptosi_i64:
            emit_unary_fp(LirOpcode::Cvttsd2si, 8, 8);
            break;
        case Opcode::fptrunc_f32_f64:
            emit_unary_fp(LirOpcode::Cvtsd2ss, 4, 8);
            break;
        case Opcode::fpext_f64_f32:
            emit_unary_fp(LirOpcode::Cvtss2sd, 8, 4);
            break;
        case Opcode::fpext_f32_f16: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            VReg zext = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            auto zx = std::make_unique<LirInst>(LirOpcode::Movzx16);
            zx->add_def(LirOperand::vreg(zext, 4));
            zx->add_use(LirOperand::vreg(src, 2));
            lir_bb.append_inst(std::move(zx));

            VReg tmp_fpr = lir_fn_->allocate_vreg(RegClass::XMM, 4);
            auto xg = std::make_unique<LirInst>(LirOpcode::Movd_xg);
            xg->add_def(LirOperand::vreg(tmp_fpr, 4));
            xg->add_use(LirOperand::vreg(zext, 4));
            lir_bb.append_inst(std::move(xg));

            auto cvt = std::make_unique<LirInst>(LirOpcode::Cvth2ss);
            cvt->add_def(LirOperand::vreg(dst, 4));
            cvt->add_use(LirOperand::vreg(tmp_fpr, 4));
            cvt->mir_origin = &inst;
            lir_bb.append_inst(std::move(cvt));
            break;
        }
        case Opcode::fptrunc_f16_f32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            VReg tmp_fpr = lir_fn_->allocate_vreg(RegClass::XMM, 4);
            auto cvt = std::make_unique<LirInst>(LirOpcode::Cvtss2h);
            cvt->add_def(LirOperand::vreg(tmp_fpr, 4));
            cvt->add_use(LirOperand::vreg(src, 4));
            lir_bb.append_inst(std::move(cvt));

            VReg raw = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            auto gx = std::make_unique<LirInst>(LirOpcode::Movd_gx);
            gx->add_def(LirOperand::vreg(raw, 4));
            gx->add_use(LirOperand::vreg(tmp_fpr, 4));
            lir_bb.append_inst(std::move(gx));

            auto zx = std::make_unique<LirInst>(LirOpcode::Movzx16);
            zx->add_def(LirOperand::vreg(dst, 4));
            zx->add_use(LirOperand::vreg(raw, 2));
            zx->mir_origin = &inst;
            lir_bb.append_inst(std::move(zx));
            break;
        }
        case Opcode::fpext_f32_bf16: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            VReg shifted = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            auto zx = std::make_unique<LirInst>(LirOpcode::Movzx16);
            zx->add_def(LirOperand::vreg(shifted, 4));
            zx->add_use(LirOperand::vreg(src, 2));
            lir_bb.append_inst(std::move(zx));

            auto shl = std::make_unique<LirInst>(LirOpcode::Shl32);
            shl->add_def(LirOperand::vreg(shifted, 4));
            shl->add_use(LirOperand::vreg(shifted, 4));
            shl->add_use(LirOperand::imm(16, 4));
            lir_bb.append_inst(std::move(shl));

            auto xg = std::make_unique<LirInst>(LirOpcode::Movd_xg);
            xg->add_def(LirOperand::vreg(dst, 4));
            xg->add_use(LirOperand::vreg(shifted, 4));
            xg->mir_origin = &inst;
            lir_bb.append_inst(std::move(xg));
            break;
        }
        case Opcode::fptrunc_bf16_f32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto emit_bin_imm = [&](LirOpcode lop, VReg r, int64_t imm) {
                auto i = std::make_unique<LirInst>(lop);
                i->add_def(LirOperand::vreg(r, 4));
                i->add_use(LirOperand::vreg(r, 4));
                i->add_use(LirOperand::imm(imm, 4));
                lir_bb.append_inst(std::move(i));
            };
            auto emit_mov32 = [&](VReg r, VReg s) {
                auto i = std::make_unique<LirInst>(LirOpcode::Mov32);
                i->add_def(LirOperand::vreg(r, 4));
                i->add_use(LirOperand::vreg(s, 4));
                lir_bb.append_inst(std::move(i));
            };

            VReg u = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            auto gx = std::make_unique<LirInst>(LirOpcode::Movd_gx);
            gx->add_def(LirOperand::vreg(u, 4));
            gx->add_use(LirOperand::vreg(src, 4));
            lir_bb.append_inst(std::move(gx));

            VReg hi = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            emit_mov32(hi, u);
            emit_bin_imm(LirOpcode::Shr32, hi, 16);

            VReg lsb = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            emit_mov32(lsb, hi);
            emit_bin_imm(LirOpcode::And32, lsb, 1);

            VReg rnd = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            emit_mov32(rnd, u);
            emit_bin_imm(LirOpcode::Add32, rnd, 0x7FFF);
            auto add_lsb = std::make_unique<LirInst>(LirOpcode::Add32);
            add_lsb->add_def(LirOperand::vreg(rnd, 4));
            add_lsb->add_use(LirOperand::vreg(rnd, 4));
            add_lsb->add_use(LirOperand::vreg(lsb, 4));
            lir_bb.append_inst(std::move(add_lsb));
            emit_bin_imm(LirOpcode::Shr32, rnd, 16);

            VReg quiet = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            emit_mov32(quiet, hi);
            emit_bin_imm(LirOpcode::Or32, quiet, 0x40);

            VReg abs_u = lir_fn_->allocate_vreg(RegClass::GPR, 4);
            emit_mov32(abs_u, u);
            emit_bin_imm(LirOpcode::And32, abs_u, 0x7FFFFFFF);

            auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp32);
            cmp->add_use(LirOperand::vreg(abs_u, 4));
            cmp->add_use(LirOperand::imm(0x7F800000, 4));
            lir_bb.append_inst(std::move(cmp));

            auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
            cmov->condition = x64::Condition::A;
            cmov->add_def(LirOperand::vreg(rnd, 4));
            cmov->add_use(LirOperand::vreg(rnd, 4));
            cmov->add_use(LirOperand::vreg(quiet, 4));
            lir_bb.append_inst(std::move(cmov));

            auto zx = std::make_unique<LirInst>(LirOpcode::Movzx16);
            zx->add_def(LirOperand::vreg(dst, 4));
            zx->add_use(LirOperand::vreg(rnd, 2));
            zx->mir_origin = &inst;
            lir_bb.append_inst(std::move(zx));
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
        // The 32-bit pair: fmov w, s / fmov s, w.
        case Opcode::bitcast_i32_f32:
        case Opcode::bitcast_f32_i32: {
            const bool to_gpr = inst.opcode() == Opcode::bitcast_i32_f32;
            auto lir_inst = std::make_unique<LirInst>(to_gpr ? LirOpcode::Movd_gx : LirOpcode::Movd_xg);
            lir_inst->add_def(LirOperand::vreg(get_vreg(inst.result()), 4));
            lir_inst->add_use(LirOperand::vreg(get_vreg(inst.operand(0)), 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        default:
            codegen::throw_unsupported("aarch64 isel (fp)", opcode_name(inst.opcode()));
    }
}

} // namespace brass::aarch64
