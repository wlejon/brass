#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64ISel::lower_fp_instruction(const Instruction& inst, LirBlock& lir_bb) {
    auto emit_unary_fp = [&](LirOpcode op, uint8_t dst_sz, uint8_t src_sz) {
        VReg dst = get_vreg(inst.result());
        const Value* src_val = inst.operand(0);
        VReg src = get_vreg(src_val);

        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(LirOperand::vreg(dst, dst_sz));
        if (src_val && src_val->is_instruction() && can_fuse_load(src_val->defining_instruction(), &inst)) {
            lir_inst->add_use(get_load_mem_operand(src_val->defining_instruction()));
        } else {
            lir_inst->add_use(LirOperand::vreg(src, src_sz));
        }
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
        default:
            break;
    }
}

} // namespace brass::aarch64
