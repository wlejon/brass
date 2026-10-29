// The VEX vector compares, blends, shifts, conversions, rounding and sign
// masks (lir.hpp, "VEX vector ops at either width"): each one is a row of
// its encoding and operand shape, emitted by X64Encoder::vex_rr.
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/unsupported_operation.hpp>

namespace brass::codegen {

namespace {

enum class VexShape : uint8_t {
    RVM,       // dst, src1 (vvvv), src2 (rm)
    RVM_I,     // + imm
    RM,        // dst, src (rm)
    RM_I,      // + imm
    RVM_IS4,   // dst, src1 (vvvv), src2 (rm), src3 in imm[7:4]
};

struct VexForm {
    uint8_t opcode;
    uint8_t mmmmm;   // 1 = 0F, 2 = 0F38, 3 = 0F3A
    uint8_t pp;      // 0 = none, 1 = 66, 2 = F3, 3 = F2
    bool w;
    VexShape shape;
};

bool vex_form(LirOpcode op, VexForm& f) {
    using S = VexShape;
    switch (op) {
        case LirOpcode::Vcmpps:     f = {0xC2, 1, 0, false, S::RVM_I}; return true;
        case LirOpcode::Vcmppd:     f = {0xC2, 1, 1, false, S::RVM_I}; return true;
        case LirOpcode::Vpcmpeqd:   f = {0x76, 1, 1, false, S::RVM}; return true;
        case LirOpcode::Vpcmpgtd:   f = {0x66, 1, 1, false, S::RVM}; return true;
        case LirOpcode::Vpcmpeqq:   f = {0x29, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpcmpgtq:   f = {0x37, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpminsd:    f = {0x39, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpmaxsd:    f = {0x3D, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpsllvd:    f = {0x47, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpsllvq:    f = {0x47, 2, 1, true,  S::RVM}; return true;
        case LirOpcode::Vpsrlvd:    f = {0x45, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vpsrlvq:    f = {0x45, 2, 1, true,  S::RVM}; return true;
        case LirOpcode::Vpsravd:    f = {0x46, 2, 1, false, S::RVM}; return true;
        case LirOpcode::Vshufps:    f = {0xC6, 1, 0, false, S::RVM_I}; return true;
        case LirOpcode::Vblendvps:  f = {0x4A, 3, 1, false, S::RVM_IS4}; return true;
        case LirOpcode::Vblendvpd:  f = {0x4B, 3, 1, false, S::RVM_IS4}; return true;
        case LirOpcode::Vcvtdq2ps:  f = {0x5B, 1, 0, false, S::RM}; return true;
        case LirOpcode::Vcvttps2dq: f = {0x5B, 1, 2, false, S::RM}; return true;
        case LirOpcode::Vcvtdq2pd:  f = {0xE6, 1, 2, false, S::RM}; return true;
        case LirOpcode::Vcvttpd2dq: f = {0xE6, 1, 1, false, S::RM}; return true;
        case LirOpcode::Vcvtps2pd:  f = {0x5A, 1, 0, false, S::RM}; return true;
        case LirOpcode::Vcvtpd2ps:  f = {0x5A, 1, 1, false, S::RM}; return true;
        case LirOpcode::Vpmovsxdq:  f = {0x25, 2, 1, false, S::RM}; return true;
        case LirOpcode::Vroundps:   f = {0x08, 3, 1, false, S::RM_I}; return true;
        case LirOpcode::Vroundpd:   f = {0x09, 3, 1, false, S::RM_I}; return true;
        case LirOpcode::Vmovmskps:  f = {0x50, 1, 0, false, S::RM}; return true;
        case LirOpcode::Vmovmskpd:  f = {0x50, 1, 1, false, S::RM}; return true;
        default: return false;
    }
}

uint8_t reg_number(const LirOperand& op, const LirInst& inst) {
    if (!op.is_preg()) {
        throw_unsupported("x64 emit (vector)", std::string(to_string(inst.opcode)) +
                                                   " with an operand not in a register");
    }
    return static_cast<uint8_t>(op.preg_val.code & 0x0F);
}

} // namespace

void EmitContext::emit_vex_lanes_instruction(const LirInst& inst) {
    VexForm f{};
    if (!vex_form(inst.opcode, f)) {
        throw_unsupported("x64 emit (vector)", to_string(inst.opcode));
    }
    // VEX.256 when any operand is a ymm: the conversions between widths
    // (vcvtdq2pd ymm, xmm; vcvttpd2dq xmm, ymm) are L=1 forms too.
    bool l = false;
    for (const auto& d : inst.defs) l |= d.size == 32;
    for (const auto& u : inst.uses) l |= u.size == 32;

    const uint8_t dst = reg_number(inst.defs[0], inst);
    auto imm_of = [&](size_t i) { return static_cast<int>(inst.uses[i].imm_int & 0xFF); };
    switch (f.shape) {
        case VexShape::RVM:
            enc_.vex_rr(f.opcode, f.mmmmm, f.pp, f.w, l, dst,
                        reg_number(inst.uses[0], inst), reg_number(inst.uses[1], inst));
            break;
        case VexShape::RVM_I:
            enc_.vex_rr(f.opcode, f.mmmmm, f.pp, f.w, l, dst,
                        reg_number(inst.uses[0], inst), reg_number(inst.uses[1], inst), imm_of(2));
            break;
        case VexShape::RM:
            enc_.vex_rr(f.opcode, f.mmmmm, f.pp, f.w, l, dst, 0, reg_number(inst.uses[0], inst));
            break;
        case VexShape::RM_I:
            enc_.vex_rr(f.opcode, f.mmmmm, f.pp, f.w, l, dst, 0, reg_number(inst.uses[0], inst), imm_of(1));
            break;
        case VexShape::RVM_IS4:
            enc_.vex_rr(f.opcode, f.mmmmm, f.pp, f.w, l, dst,
                        reg_number(inst.uses[0], inst), reg_number(inst.uses[1], inst),
                        reg_number(inst.uses[2], inst) << 4);
            break;
    }
}

} // namespace brass::codegen
