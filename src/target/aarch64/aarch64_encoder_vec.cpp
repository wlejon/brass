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

} // namespace brass::aarch64
