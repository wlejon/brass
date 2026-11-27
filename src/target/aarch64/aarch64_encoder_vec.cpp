#include <brass/target/aarch64/aarch64_encoder.hpp>

namespace brass::aarch64 {

// 128-bit SIMD Vector
// 128-bit SIMD Vector (Float)
void AArch64Encoder::vec_fadd_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsub_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA0D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmul_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E20DC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fdiv_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E20FC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmla_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20CC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmin_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA0F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmax_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fneg_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EA0F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsqrt_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EA1F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fadd_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsub_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE0D400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmul_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E60DC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fdiv_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E60FC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmla_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60CC00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmin_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE0F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fmax_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60F400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fneg_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EE0F800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fsqrt_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EE1F800u | (reg_code(src) << 5) | reg_code(dst));
}

// 128-bit SIMD Vector (Integer & Bitwise)
void AArch64Encoder::vec_add_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E208400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_add_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E608400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sub_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E208400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sub_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E608400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_mul_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E209C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_mul_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E609C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_smin_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E206C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_smin_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E606C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_smax_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E206400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_smax_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E606400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umin_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E206C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umin_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E606C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_neg_16b(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E20B800u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_neg_8h(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E60B800u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_dup_16b(FPR dst, GPR src) {
    buffer_.emit_inst(0x4E010C00u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_dup_8h(FPR dst, GPR src) {
    buffer_.emit_inst(0x4E020C00u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_ins_b(FPR dst, GPR src, uint8_t lane) {
    uint32_t imm5 = ((static_cast<uint32_t>(lane) & 15u) << 1) | 1u;
    buffer_.emit_inst(0x4E001C00u | (imm5 << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_ins_h(FPR dst, GPR src, uint8_t lane) {
    uint32_t imm5 = ((static_cast<uint32_t>(lane) & 7u) << 2) | 2u;
    buffer_.emit_inst(0x4E001C00u | (imm5 << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umov_b(GPR dst, FPR src, uint8_t lane) {
    uint32_t imm5 = ((static_cast<uint32_t>(lane) & 15u) << 1) | 1u;
    buffer_.emit_inst(0x0E003C00u | (imm5 << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umov_h(GPR dst, FPR src, uint8_t lane) {
    uint32_t imm5 = ((static_cast<uint32_t>(lane) & 7u) << 2) | 2u;
    buffer_.emit_inst(0x0E003C00u | (imm5 << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_add_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sub_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_mul_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA09C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_smin_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA06C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_smax_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA06400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_add_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sub_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE08400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_and(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E201C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_orr(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA01C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_eor(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E201C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::cnt_8b(FPR dst, FPR src) {
    buffer_.emit_inst(0x0E205800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::uaddlv_h(FPR dst, FPR src) {
    buffer_.emit_inst(0x2E303800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmgt_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA0E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmgt_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE0E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmgt_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA03400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmgt_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE03400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_bsl(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E601C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_not(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E205800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_neg_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EA0B800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_neg_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EE0B800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_dup_4s(FPR dst, GPR src) {
    buffer_.emit_inst(0x4E040C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_dup_2d(FPR dst, GPR src) {
    buffer_.emit_inst(0x4E080C00u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintn_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E218800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintn_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E618800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintm_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E219800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintm_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E619800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintp_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA18800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintp_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EE18800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintz_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA19800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_frintz_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EE19800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtzs_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA1B800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtzs_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EE1B800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_scvtf_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E21D800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_scvtf_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E61D800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ucvtf_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E21D800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ucvtf_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E61D800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtl_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x0E617800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtl2_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E617800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtn_2s(FPR dst, FPR src) {
    buffer_.emit_inst(0x0E616800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcvtn2_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E616800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sxtl_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x0F20A400u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sxtl2_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4F20A400u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_xtn_2s(FPR dst, FPR src) {
    buffer_.emit_inst(0x0EA12800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_xtn2_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA12800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sqxtn_2s(FPR dst, FPR src) {
    buffer_.emit_inst(0x0EA14800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sqxtn2_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA14800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ushl_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA04400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ushl_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE04400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sshl_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA04400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sshl_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE04400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_shl_imm_4s(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 32u + (static_cast<uint32_t>(shift) & 31u);
    buffer_.emit_inst(0x4F005400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_shl_imm_2d(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 64u + (static_cast<uint32_t>(shift) & 63u);
    buffer_.emit_inst(0x4F005400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ushr_imm_4s(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 64u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x6F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ushr_imm_2d(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 128u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x6F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sshr_imm_4s(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 64u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x4F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_sshr_imm_2d(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 128u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x4F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmeq_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA08C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmeq_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE08C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmge_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EA03C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmge_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4EE03C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmhi_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA03400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmhi_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE03400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmhs_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA03C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmhs_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE03C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmeq_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E20E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmeq_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E60E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmge_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E20E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_fcmge_2d(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E60E400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmlt_zero_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EA0A800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_cmlt_zero_2d(FPR dst, FPR src) {
    buffer_.emit_inst(0x4EE0A800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_bif(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EE01C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_umin_4s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6EA06C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_umaxv_4s(FPR dst, FPR src) {
    buffer_.emit_inst(0x6EB0A800u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_dup_d1(FPR dst, FPR src) {
    buffer_.emit_inst(0x5E180400u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ins_d1(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E180400u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::vec_movi_16b(FPR dst, uint8_t imm8) {
    uint32_t abc = (static_cast<uint32_t>(imm8) >> 5) & 7u;
    uint32_t defgh = static_cast<uint32_t>(imm8) & 0x1Fu;
    buffer_.emit_inst(0x4F00E400u | (abc << 16) | (defgh << 5) | reg_code(dst));
}

void AArch64Encoder::vec_movi_8h(FPR dst, uint8_t imm8, uint8_t lsl_bits) {
    uint32_t cmode = 0x8u | ((static_cast<uint32_t>(lsl_bits) / 8u) << 1);
    uint32_t abc = (static_cast<uint32_t>(imm8) >> 5) & 7u;
    uint32_t defgh = static_cast<uint32_t>(imm8) & 0x1Fu;
    buffer_.emit_inst(0x4F000400u | (abc << 16) | (cmode << 12) | (defgh << 5) | reg_code(dst));
}

void AArch64Encoder::vec_movi_4s(FPR dst, uint8_t imm8, uint8_t lsl_bits) {
    uint32_t cmode = (static_cast<uint32_t>(lsl_bits) / 8u) << 1;
    uint32_t abc = (static_cast<uint32_t>(imm8) >> 5) & 7u;
    uint32_t defgh = static_cast<uint32_t>(imm8) & 0x1Fu;
    buffer_.emit_inst(0x4F000400u | (abc << 16) | (cmode << 12) | (defgh << 5) | reg_code(dst));
}

void AArch64Encoder::vec_ushl_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E204400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_ushl_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E604400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sshl_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E204400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sshl_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E604400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_shl_imm_16b(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 8u + (static_cast<uint32_t>(shift) & 7u);
    buffer_.emit_inst(0x4F005400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_shl_imm_8h(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 16u + (static_cast<uint32_t>(shift) & 15u);
    buffer_.emit_inst(0x4F005400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_ushr_imm_16b(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 16u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x6F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_ushr_imm_8h(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 32u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x6F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sshr_imm_16b(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 16u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x4F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_sshr_imm_8h(FPR dst, FPR src, uint8_t shift) {
    uint32_t immh_immb = 32u - static_cast<uint32_t>(shift);
    buffer_.emit_inst(0x4F000400u | (immh_immb << 16) | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmeq_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E208C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmeq_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x6E608C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmgt_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E203400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmgt_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E603400u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmge_16b(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E203C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmge_8h(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x4E603C00u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmlt_zero_16b(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E20A800u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_cmlt_zero_8h(FPR dst, FPR src) {
    buffer_.emit_inst(0x4E60A800u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umaxv_16b(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E30A800u | (reg_code(src) << 5) | reg_code(dst));
}
void AArch64Encoder::vec_umaxv_8h(FPR dst, FPR src) {
    buffer_.emit_inst(0x6E70A800u | (reg_code(src) << 5) | reg_code(dst));
}

} // namespace brass::aarch64
