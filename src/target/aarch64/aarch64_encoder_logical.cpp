#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <stdexcept>

namespace brass::aarch64 {

// =============================================================================
// Logical (Register): AND, BIC, ORR, ORN, EOR, EON, TST, MVN
// =============================================================================

void AArch64Encoder::and_(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x8A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::and32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x0A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::bic(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x8A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::bic32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x0A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orr(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xAA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orr32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x2A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orn(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xAA200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::orn32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x2A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eor(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xCA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eor32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x4A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eon(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0xCA200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::eon32(GPR dst, GPR src1, GPR src2) {
    buffer_.emit_inst(0x4A200000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::tst(GPR src1, GPR src2) {
    buffer_.emit_inst(0xEA000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | 31u);
}

void AArch64Encoder::tst32(GPR src1, GPR src2) {
    buffer_.emit_inst(0x6A000000u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | 31u);
}

void AArch64Encoder::mvn(GPR dst, GPR src) {
    orn(dst, GPR::XZR, src);
}

void AArch64Encoder::mvn32(GPR dst, GPR src) {
    orn32(dst, GPR::XZR, src);
}

// =============================================================================
// Logical (Immediate / Bitmask)
// =============================================================================

void AArch64Encoder::emit_logical_imm(uint32_t opc, bool is_64, GPR dst, GPR src, uint64_t imm) {
    uint32_t n = 0, immr = 0, imms = 0;
    if (!encode_logical_immediate(imm, is_64, n, immr, imms)) {
        throw std::invalid_argument("AArch64Encoder: immediate is not a valid ARMv8 logical bitmask");
    }
    uint32_t sf = is_64 ? 1u : 0u;
    uint32_t inst = (sf << 31) | (opc << 29) | (0x24u << 23) | (n << 22) |
                    (immr << 16) | (imms << 10) | (reg_code(src) << 5) | reg_code(dst);
    buffer_.emit_inst(inst);
}

void AArch64Encoder::and_imm(GPR dst, GPR src, uint64_t imm) {
    emit_logical_imm(0, true, dst, src, imm);
}

void AArch64Encoder::and32_imm(GPR dst, GPR src, uint32_t imm) {
    emit_logical_imm(0, false, dst, src, imm);
}

void AArch64Encoder::orr_imm(GPR dst, GPR src, uint64_t imm) {
    emit_logical_imm(1, true, dst, src, imm);
}

void AArch64Encoder::orr32_imm(GPR dst, GPR src, uint32_t imm) {
    emit_logical_imm(1, false, dst, src, imm);
}

void AArch64Encoder::eor_imm(GPR dst, GPR src, uint64_t imm) {
    emit_logical_imm(2, true, dst, src, imm);
}

void AArch64Encoder::eor32_imm(GPR dst, GPR src, uint32_t imm) {
    emit_logical_imm(2, false, dst, src, imm);
}

void AArch64Encoder::ands_imm(GPR dst, GPR src, uint64_t imm) {
    emit_logical_imm(3, true, dst, src, imm);
}

void AArch64Encoder::ands32_imm(GPR dst, GPR src, uint32_t imm) {
    emit_logical_imm(3, false, dst, src, imm);
}

void AArch64Encoder::tst_imm(GPR src, uint64_t imm) {
    ands_imm(GPR::XZR, src, imm);
}

void AArch64Encoder::tst32_imm(GPR src, uint32_t imm) {
    ands32_imm(GPR::XZR, src, imm);
}

} // namespace brass::aarch64
