#include <brass/target/x64/x64_encoder.hpp>

namespace brass::x64 {

// =========================================================================
// LOW-LEVEL VEX PREFIX EMISSION
// =========================================================================

void X64Encoder::emit_vex2(bool r, uint8_t vvvv, bool l, uint8_t pp) {
    uint8_t inv_r = r ? 0 : 1;
    uint8_t inv_vvvv = (~vvvv) & 0x0F;
    uint8_t l_bit = l ? 1 : 0;
    buffer_.emit8(0xC5);
    buffer_.emit8(static_cast<uint8_t>((inv_r << 7) | (inv_vvvv << 3) | (l_bit << 2) | (pp & 0x03)));
}

void X64Encoder::emit_vex3(bool w, bool r, bool x, bool b, uint8_t mmmmm, uint8_t vvvv, bool l, uint8_t pp) {
    uint8_t inv_r = r ? 0 : 1;
    uint8_t inv_x = x ? 0 : 1;
    uint8_t inv_b = b ? 0 : 1;
    uint8_t inv_vvvv = (~vvvv) & 0x0F;
    uint8_t l_bit = l ? 1 : 0;
    buffer_.emit8(0xC4);
    buffer_.emit8(static_cast<uint8_t>((inv_r << 7) | (inv_x << 6) | (inv_b << 5) | (mmmmm & 0x1F)));
    buffer_.emit8(static_cast<uint8_t>((w ? 0x80 : 0x00) | (inv_vvvv << 3) | (l_bit << 2) | (pp & 0x03)));
}

void X64Encoder::emit_vex(bool w, uint8_t mmmmm, uint8_t pp, bool l, bool r, bool x, bool b, uint8_t vvvv) {
    if (mmmmm == 1 && !w && !x && !b) {
        emit_vex2(r, vvvv, l, pp);
    } else {
        emit_vex3(w, r, x, b, mmmmm, vvvv, l, pp);
    }
}

// =========================================================================
// 256-BIT AVX / AVX2 MOVES
// =========================================================================

void X64Encoder::vmovaps(XMM dst, XMM src) {
    emit_vex(false, 1, 0, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x28);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vmovaps(XMM dst, const MemAddress& src) {
    emit_vex(false, 1, 0, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x28);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::vmovaps(const MemAddress& dst, XMM src) {
    emit_vex(false, 1, 0, true, is_extended(src), is_extended(dst.index), is_extended(dst.base), 0);
    buffer_.emit8(0x29);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::vmovups(XMM dst, XMM src) {
    emit_vex(false, 1, 0, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x10);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vmovups(XMM dst, const MemAddress& src) {
    emit_vex(false, 1, 0, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x10);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::vmovups(const MemAddress& dst, XMM src) {
    emit_vex(false, 1, 0, true, is_extended(src), is_extended(dst.index), is_extended(dst.base), 0);
    buffer_.emit8(0x11);
    emit_mem_operand(reg_code(src), dst);
}

// =========================================================================
// 256-BIT THREE-OPERAND VECTOR OPS
// =========================================================================

#define DEFINE_VEX_OP(name, opcode, mmmmm_val, pp_val, w_val)                                \
    void X64Encoder::name(XMM dst, XMM src1, XMM src2) {                                     \
        emit_vex(w_val, mmmmm_val, pp_val, true, is_extended(dst), false,                   \
                 is_extended(src2), reg_id(src1));                                          \
        buffer_.emit8(opcode);                                                               \
        emit_modrm(3, reg_code(dst), reg_code(src2));                                        \
    }                                                                                        \
    void X64Encoder::name(XMM dst, XMM src1, const MemAddress& src2) {                       \
        emit_vex(w_val, mmmmm_val, pp_val, true, is_extended(dst),                           \
                 is_extended(src2.index), is_extended(src2.base), reg_id(src1));             \
        buffer_.emit8(opcode);                                                               \
        emit_mem_operand(reg_code(dst), src2);                                               \
    }

// Float arithmetic (f32x8)
DEFINE_VEX_OP(vaddps, 0x58, 1, 0, false)
DEFINE_VEX_OP(vsubps, 0x5C, 1, 0, false)
DEFINE_VEX_OP(vmulps, 0x59, 1, 0, false)
DEFINE_VEX_OP(vdivps, 0x5E, 1, 0, false)
DEFINE_VEX_OP(vminps, 0x5D, 1, 0, false)
DEFINE_VEX_OP(vmaxps, 0x5F, 1, 0, false)

// Double arithmetic (f64x4)
DEFINE_VEX_OP(vaddpd, 0x58, 1, 1, false)
DEFINE_VEX_OP(vsubpd, 0x5C, 1, 1, false)
DEFINE_VEX_OP(vmulpd, 0x59, 1, 1, false)
DEFINE_VEX_OP(vdivpd, 0x5E, 1, 1, false)
DEFINE_VEX_OP(vminpd, 0x5D, 1, 1, false)
DEFINE_VEX_OP(vmaxpd, 0x5F, 1, 1, false)

// Integer arithmetic (i32x8 & i64x4)
DEFINE_VEX_OP(vpaddd, 0xFE, 1, 1, false)
DEFINE_VEX_OP(vpsubd, 0xFA, 1, 1, false)
DEFINE_VEX_OP(vpmulld, 0x40, 2, 1, false)
DEFINE_VEX_OP(vpaddq, 0xD4, 1, 1, false)
DEFINE_VEX_OP(vpsubq, 0xFB, 1, 1, false)

// Bitwise operations
DEFINE_VEX_OP(vandps, 0x54, 1, 0, false)
DEFINE_VEX_OP(vorps,  0x56, 1, 0, false)
DEFINE_VEX_OP(vxorps, 0x57, 1, 0, false)
DEFINE_VEX_OP(vandpd, 0x54, 1, 1, false)
DEFINE_VEX_OP(vorpd,  0x56, 1, 1, false)
DEFINE_VEX_OP(vxorpd, 0x57, 1, 1, false)
DEFINE_VEX_OP(vpand,  0xDB, 1, 1, false)
DEFINE_VEX_OP(vpor,   0xEB, 1, 1, false)
DEFINE_VEX_OP(vpxor,  0xEF, 1, 1, false)

#undef DEFINE_VEX_OP

// =========================================================================
// BROADCAST INTO 256-BIT REGISTERS
// =========================================================================

void X64Encoder::vbroadcastss(XMM dst, XMM src) {
    emit_vex(false, 2, 1, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x18);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vbroadcastss(XMM dst, const MemAddress& src) {
    emit_vex(false, 2, 1, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x18);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::vbroadcastsd(XMM dst, XMM src) {
    emit_vex(false, 2, 1, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x19);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vbroadcastsd(XMM dst, const MemAddress& src) {
    emit_vex(false, 2, 1, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x19);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::vpbroadcastd(XMM dst, XMM src) {
    emit_vex(false, 2, 1, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x58);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vpbroadcastd(XMM dst, const MemAddress& src) {
    emit_vex(false, 2, 1, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x58);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::vpbroadcastq(XMM dst, XMM src) {
    emit_vex(false, 2, 1, true, is_extended(dst), false, is_extended(src), 0);
    buffer_.emit8(0x59);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::vpbroadcastq(XMM dst, const MemAddress& src) {
    emit_vex(false, 2, 1, true, is_extended(dst), is_extended(src.index), is_extended(src.base), 0);
    buffer_.emit8(0x59);
    emit_mem_operand(reg_code(dst), src);
}

// =========================================================================
// FMA3 INSTRUCTION SET
// =========================================================================

#define DEFINE_FMA_VEC(name, opcode, w_val)                                                  \
    void X64Encoder::name(XMM dst, XMM src2, XMM src3, bool is_256) {                         \
        emit_vex(w_val, 2, 1, is_256, is_extended(dst), false,                              \
                 is_extended(src3), reg_id(src2));                                          \
        buffer_.emit8(opcode);                                                               \
        emit_modrm(3, reg_code(dst), reg_code(src3));                                        \
    }                                                                                        \
    void X64Encoder::name(XMM dst, XMM src2, const MemAddress& src3, bool is_256) {          \
        emit_vex(w_val, 2, 1, is_256, is_extended(dst),                                      \
                 is_extended(src3.index), is_extended(src3.base), reg_id(src2));             \
        buffer_.emit8(opcode);                                                               \
        emit_mem_operand(reg_code(dst), src3);                                               \
    }

DEFINE_FMA_VEC(vfmadd213ps, 0xA8, false)
DEFINE_FMA_VEC(vfmadd231ps, 0xB8, false)
DEFINE_FMA_VEC(vfmadd213pd, 0xA8, true)
DEFINE_FMA_VEC(vfmadd231pd, 0xB8, true)

#undef DEFINE_FMA_VEC

#define DEFINE_FMA_SCALAR(name, opcode, w_val)                                               \
    void X64Encoder::name(XMM dst, XMM src2, XMM src3) {                                     \
        emit_vex(w_val, 2, 1, false, is_extended(dst), false,                                \
                 is_extended(src3), reg_id(src2));                                          \
        buffer_.emit8(opcode);                                                               \
        emit_modrm(3, reg_code(dst), reg_code(src3));                                        \
    }                                                                                        \
    void X64Encoder::name(XMM dst, XMM src2, const MemAddress& src3) {                        \
        emit_vex(w_val, 2, 1, false, is_extended(dst),                                       \
                 is_extended(src3.index), is_extended(src3.base), reg_id(src2));             \
        buffer_.emit8(opcode);                                                               \
        emit_mem_operand(reg_code(dst), src3);                                               \
    }

DEFINE_FMA_SCALAR(vfmadd213ss, 0xA9, false)
DEFINE_FMA_SCALAR(vfmadd231ss, 0xB9, false)
DEFINE_FMA_SCALAR(vfmadd213sd, 0xA9, true)
DEFINE_FMA_SCALAR(vfmadd231sd, 0xB9, true)

#undef DEFINE_FMA_SCALAR

} // namespace brass::x64
