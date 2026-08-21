#include <brass/target/x64/x64_encoder.hpp>

namespace brass::x64 {

// =========================================================================
// SSE & FLOATING POINT (Single & Double)
// =========================================================================

void X64Encoder::movss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movss(const MemAddress& dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x11);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::movsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movsd(const MemAddress& dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x11);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::movups(XMM dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x10);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movups(const MemAddress& dst, XMM src) {
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x11);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::movq(XMM dst, GPR src) {
    buffer_.emit8(0x66);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x6E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movq(GPR dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(true, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x7E);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::movq(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x7E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movq(const MemAddress& dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xD6);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::movq(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x7E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

// Double FP Arithmetic
void X64Encoder::addsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x58);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::addsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x58);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::subsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::subsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mulsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x59);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::mulsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x59);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::divsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::divsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::sqrtsd(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x51);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::sqrtsd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x51);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::ucomisd(XMM dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::ucomisd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::xorpd(XMM dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x57);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::xorpd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x57);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::andpd(XMM dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x54);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::andpd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x54);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::orpd(XMM dst, XMM src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x56);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::orpd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x56);
    emit_mem_operand(reg_code(dst), src);
}

// Single FP Arithmetic
void X64Encoder::addss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x58);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::addss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x58);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::subss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::subss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mulss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x59);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::mulss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x59);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::divss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::divss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::sqrtss(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x51);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::sqrtss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x51);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::ucomiss(XMM dst, XMM src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2E);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::ucomiss(XMM dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2E);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::xorps(XMM dst, XMM src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x57);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::xorps(XMM dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x57);
    emit_mem_operand(reg_code(dst), src);
}

// FP Conversions
void X64Encoder::cvtsi2sd(XMM dst, GPR src) {
    buffer_.emit8(0xF2);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtsi2sd32(XMM dst, GPR src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtsi2sd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvtsi2sd32(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvttsd2si(GPR dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvttsd2si32(GPR dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvttsd2si(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvttsd2si32(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvtsi2ss(XMM dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtsi2ss32(XMM dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtsi2ss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvtsi2ss32(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvttss2si(GPR dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvttss2si32(GPR dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvttss2si(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvttss2si32(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x2C);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvtsd2ss(XMM dst, XMM src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtsd2ss(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF2);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cvtss2sd(XMM dst, XMM src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5A);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cvtss2sd(XMM dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0x5A);
    emit_mem_operand(reg_code(dst), src);
}

// =========================================================================
// BIT OPERATIONS
// =========================================================================

void X64Encoder::popcnt(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB8);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::popcnt32(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB8);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::popcnt(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB8);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::popcnt32(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB8);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::lzcnt(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::lzcnt32(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::lzcnt(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::lzcnt32(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::tzcnt(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::tzcnt32(GPR dst, GPR src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::tzcnt(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::tzcnt32(GPR dst, const MemAddress& src) {
    buffer_.emit8(0xF3);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::bsf(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::bsf32(GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::bsf(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::bsf32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBC);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::bsr(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::bsr32(GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::bsr(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::bsr32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBD);
    emit_mem_operand(reg_code(dst), src);
}

} // namespace brass::x64
