#include <brass/target/x64/x64_encoder.hpp>

namespace brass::x64 {

// =========================================================================
// 128-BIT SIMD VECTOR INSTRUCTIONS (SSE / SSE2 / SSE4.1)
// =========================================================================

void X64Encoder::movaps(XMM dst, XMM src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x28);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movaps(XMM dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x28);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movaps(const MemAddress& dst, XMM src) {
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x29);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::movups(XMM dst, XMM src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movd(XMM dst, GPR src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x6E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movd(GPR dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x7E);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::movd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x6E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movd(const MemAddress& dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x7E);
    emit_mem_operand(reg_code(src), dst);
}

// -------------------------------------------------------------------------
// Float vectors (f32x4)
// -------------------------------------------------------------------------

#define DEFINE_SSE_OP(name, opcode_byte)                                     \
    void X64Encoder::name(XMM dst, XMM src) {                                \
        emit_rex(false, is_extended(dst), false, is_extended(src));           \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_modrm(3, reg_code(dst), reg_code(src));                          \
    }                                                                         \
    void X64Encoder::name(XMM dst, const MemAddress& src) {                  \
        emit_rex(false, is_extended(dst), is_extended(src.index),             \
                 is_extended(src.base));                                      \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_mem_operand(reg_code(dst), src);                                 \
    }

DEFINE_SSE_OP(addps, 0x58)
DEFINE_SSE_OP(subps, 0x5C)
DEFINE_SSE_OP(mulps, 0x59)
DEFINE_SSE_OP(divps, 0x5E)
DEFINE_SSE_OP(minps, 0x5D)
DEFINE_SSE_OP(maxps, 0x5F)
DEFINE_SSE_OP(sqrtps, 0x51)

#undef DEFINE_SSE_OP

// -------------------------------------------------------------------------
// Double vectors (f64x2)
// -------------------------------------------------------------------------

#define DEFINE_SSE2_OP(name, opcode_byte)                                    \
    void X64Encoder::name(XMM dst, XMM src) {                                \
        buffer_.emit8(0x66);                                                  \
        emit_rex(false, is_extended(dst), false, is_extended(src));           \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_modrm(3, reg_code(dst), reg_code(src));                          \
    }                                                                         \
    void X64Encoder::name(XMM dst, const MemAddress& src) {                  \
        buffer_.emit8(0x66);                                                  \
        emit_rex(false, is_extended(dst), is_extended(src.index),             \
                 is_extended(src.base));                                      \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_mem_operand(reg_code(dst), src);                                 \
    }

DEFINE_SSE2_OP(addpd, 0x58)
DEFINE_SSE2_OP(subpd, 0x5C)
DEFINE_SSE2_OP(mulpd, 0x59)
DEFINE_SSE2_OP(divpd, 0x5E)
DEFINE_SSE2_OP(minpd, 0x5D)
DEFINE_SSE2_OP(maxpd, 0x5F)
DEFINE_SSE2_OP(sqrtpd, 0x51)

// -------------------------------------------------------------------------
// Integer vectors (i32x4 & i64x2)
// -------------------------------------------------------------------------

DEFINE_SSE2_OP(paddd, 0xFE)
DEFINE_SSE2_OP(psubd, 0xFA)
DEFINE_SSE2_OP(paddq, 0xD4)
DEFINE_SSE2_OP(psubq, 0xFB)

// -------------------------------------------------------------------------
// Bitwise vectors
// -------------------------------------------------------------------------

DEFINE_SSE2_OP(pand, 0xDB)
DEFINE_SSE2_OP(por, 0xEB)
DEFINE_SSE2_OP(pxor, 0xEF)
DEFINE_SSE2_OP(pandn, 0xDF)

#undef DEFINE_SSE2_OP

// -------------------------------------------------------------------------
// SSE4.1 Integer vector ops (pmulld, pminsd, pmaxsd)
// -------------------------------------------------------------------------

#define DEFINE_SSE41_OP(name, opcode_byte)                                   \
    void X64Encoder::name(XMM dst, XMM src) {                                \
        buffer_.emit8(0x66);                                                  \
        emit_rex(false, is_extended(dst), false, is_extended(src));           \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(0x38);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_modrm(3, reg_code(dst), reg_code(src));                          \
    }                                                                         \
    void X64Encoder::name(XMM dst, const MemAddress& src) {                  \
        buffer_.emit8(0x66);                                                  \
        emit_rex(false, is_extended(dst), is_extended(src.index),             \
                 is_extended(src.base));                                      \
        buffer_.emit8(0x0F);                                                  \
        buffer_.emit8(0x38);                                                  \
        buffer_.emit8(opcode_byte);                                           \
        emit_mem_operand(reg_code(dst), src);                                 \
    }

DEFINE_SSE41_OP(pmulld, 0x40)
DEFINE_SSE41_OP(pminsd, 0x39)
DEFINE_SSE41_OP(pmaxsd, 0x3D)

#undef DEFINE_SSE41_OP

// -------------------------------------------------------------------------
// Shifts / compares
// -------------------------------------------------------------------------

void X64Encoder::pcmpeqd(XMM dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x76);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::pcmpeqd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x76);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::pslld(XMM dst, uint8_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x72);
    emit_modrm(3, 6, reg_code(dst));
    buffer_.emit8(imm);
}

void X64Encoder::psllq(XMM dst, uint8_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x73);
    emit_modrm(3, 6, reg_code(dst));
    buffer_.emit8(imm);
}

// -------------------------------------------------------------------------
// Shuffles & broadcast
// -------------------------------------------------------------------------

void X64Encoder::shufps(XMM dst, XMM src, uint8_t imm) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xC6);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(imm);
}

void X64Encoder::shufpd(XMM dst, XMM src, uint8_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xC6);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(imm);
}

void X64Encoder::pshufd(XMM dst, XMM src, uint8_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x70);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(imm);
}

void X64Encoder::movddup(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x12);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

// -------------------------------------------------------------------------
// Lane insert & extract
// -------------------------------------------------------------------------

void X64Encoder::pinsrd(XMM dst, GPR src, uint8_t lane) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x22);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(lane);
}

void X64Encoder::pextrd(GPR dst, XMM src, uint8_t lane) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x16);
    emit_modrm(3, reg_code(src), reg_code(dst));
    buffer_.emit8(lane);
}

void X64Encoder::pinsrq(XMM dst, GPR src, uint8_t lane) {
    buffer_.emit8(0x66);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x22);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(lane);
}

void X64Encoder::pextrq(GPR dst, XMM src, uint8_t lane) {
    buffer_.emit8(0x66);
    emit_rex(true, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x16);
    emit_modrm(3, reg_code(src), reg_code(dst));
    buffer_.emit8(lane);
}

void X64Encoder::insertps(XMM dst, XMM src, uint8_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x21);
    emit_modrm(3, reg_code(dst), reg_code(src));
    buffer_.emit8(imm);
}

void X64Encoder::extractps(GPR dst, XMM src, uint8_t lane) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x3A);
    buffer_.emit8(0x17);
    emit_modrm(3, reg_code(src), reg_code(dst));
    buffer_.emit8(lane);
}

} // namespace brass::x64
