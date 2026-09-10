#pragma once

#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <brass/target/x64/code_buffer.hpp>
#include <cstdint>
#include <cstddef>
#include <string>

namespace brass::x64 {

class X64Encoder {
public:
    explicit X64Encoder(CodeBuffer& buffer) noexcept : buffer_(buffer) {}

    CodeBuffer& buffer() noexcept { return buffer_; }
    const CodeBuffer& buffer() const noexcept { return buffer_; }

    // =========================================================================
    // ALU: ADD, SUB, AND, OR, XOR, CMP, TEST, NOT, NEG
    // =========================================================================

    // ADD (64-bit & 32-bit)
    void add(GPR dst, GPR src);
    void add32(GPR dst, GPR src);
    void add(GPR dst, int32_t imm);
    void add32(GPR dst, int32_t imm);
    void add(GPR dst, const MemAddress& src);
    void add32(GPR dst, const MemAddress& src);
    void add(const MemAddress& dst, GPR src);
    void add32(const MemAddress& dst, GPR src);
    void add(const MemAddress& dst, int32_t imm);
    void add32(const MemAddress& dst, int32_t imm);

    // SUB (64-bit & 32-bit)
    void sub(GPR dst, GPR src);
    void sub32(GPR dst, GPR src);
    void sub(GPR dst, int32_t imm);
    void sub32(GPR dst, int32_t imm);
    void sub(GPR dst, const MemAddress& src);
    void sub32(GPR dst, const MemAddress& src);
    void sub(const MemAddress& dst, GPR src);
    void sub32(const MemAddress& dst, GPR src);
    void sub(const MemAddress& dst, int32_t imm);
    void sub32(const MemAddress& dst, int32_t imm);

    // AND (64-bit & 32-bit)
    void and_(GPR dst, GPR src);
    void and32(GPR dst, GPR src);
    void and_(GPR dst, int32_t imm);
    void and32(GPR dst, int32_t imm);
    void and_(GPR dst, const MemAddress& src);
    void and32(GPR dst, const MemAddress& src);
    void and_(const MemAddress& dst, GPR src);
    void and32(const MemAddress& dst, GPR src);
    void and_(const MemAddress& dst, int32_t imm);
    void and32(const MemAddress& dst, int32_t imm);

    // OR (64-bit & 32-bit)
    void or_(GPR dst, GPR src);
    void or32(GPR dst, GPR src);
    void or_(GPR dst, int32_t imm);
    void or32(GPR dst, int32_t imm);
    void or_(GPR dst, const MemAddress& src);
    void or32(GPR dst, const MemAddress& src);
    void or_(const MemAddress& dst, GPR src);
    void or32(const MemAddress& dst, GPR src);
    void or_(const MemAddress& dst, int32_t imm);
    void or32(const MemAddress& dst, int32_t imm);

    // XOR (64-bit & 32-bit)
    void xor_(GPR dst, GPR src);
    void xor32(GPR dst, GPR src);
    void xor_(GPR dst, int32_t imm);
    void xor32(GPR dst, int32_t imm);
    void xor_(GPR dst, const MemAddress& src);
    void xor32(GPR dst, const MemAddress& src);
    void xor_(const MemAddress& dst, GPR src);
    void xor32(const MemAddress& dst, GPR src);
    void xor_(const MemAddress& dst, int32_t imm);
    void xor32(const MemAddress& dst, int32_t imm);

    // CMP (64-bit & 32-bit)
    void cmp(GPR dst, GPR src);
    void cmp32(GPR dst, GPR src);
    void cmp(GPR dst, int32_t imm);
    void cmp32(GPR dst, int32_t imm);
    void cmp(GPR dst, const MemAddress& src);
    void cmp32(GPR dst, const MemAddress& src);
    void cmp(const MemAddress& dst, GPR src);
    void cmp32(const MemAddress& dst, GPR src);
    void cmp(const MemAddress& dst, int32_t imm);
    void cmp32(const MemAddress& dst, int32_t imm);

    // TEST (64-bit, 32-bit, 8-bit)
    void test(GPR reg1, GPR reg2);
    void test32(GPR reg1, GPR reg2);
    void test(GPR reg, int32_t imm);
    void test32(GPR reg, int32_t imm);
    void test(const MemAddress& mem, GPR reg);
    void test32(const MemAddress& mem, GPR reg);
    void test(const MemAddress& mem, int32_t imm);
    void test32(const MemAddress& mem, int32_t imm);
    void test8(GPR reg1, GPR reg2);
    void test8(GPR reg, uint8_t imm);

    // NOT & NEG (64-bit & 32-bit)
    void not_(GPR dst);
    void not32(GPR dst);
    void not_(const MemAddress& dst);
    void not32(const MemAddress& dst);
    void neg(GPR dst);
    void neg32(GPR dst);
    void neg(const MemAddress& dst);
    void neg32(const MemAddress& dst);

    // Multiply & Divide (MUL, IMUL, DIV, IDIV)
    void mul(GPR src);
    void mul32(GPR src);
    void mul(const MemAddress& src);
    void mul32(const MemAddress& src);

    void div(GPR src);
    void div32(GPR src);
    void div(const MemAddress& src);
    void div32(const MemAddress& src);

    void idiv(GPR src);
    void idiv32(GPR src);
    void idiv(const MemAddress& src);
    void idiv32(const MemAddress& src);

    void imul(GPR src);
    void imul32(GPR src);
    void imul(GPR dst, GPR src);
    void imul32(GPR dst, GPR src);
    void imul(GPR dst, const MemAddress& src);
    void imul32(GPR dst, const MemAddress& src);
    void imul(GPR dst, GPR src, int32_t imm);
    void imul32(GPR dst, GPR src, int32_t imm);
    void imul(GPR dst, int32_t imm);
    void imul32(GPR dst, int32_t imm);

    // Shifts & Rotates (SHL, SHR, SAR, ROL, ROR)
    void shl(GPR dst, uint8_t imm);
    void shl(GPR dst); // by CL
    void shl32(GPR dst, uint8_t imm);
    void shl32(GPR dst);
    void shl(const MemAddress& dst, uint8_t imm);
    void shl_cl(const MemAddress& dst);
    void shl32(const MemAddress& dst, uint8_t imm);
    void shl32_cl(const MemAddress& dst);

    void shr(GPR dst, uint8_t imm);
    void shr(GPR dst); // by CL
    void shr32(GPR dst, uint8_t imm);
    void shr32(GPR dst);
    void shr(const MemAddress& dst, uint8_t imm);
    void shr_cl(const MemAddress& dst);
    void shr32(const MemAddress& dst, uint8_t imm);
    void shr32_cl(const MemAddress& dst);

    void sar(GPR dst, uint8_t imm);
    void sar(GPR dst); // by CL
    void sar32(GPR dst, uint8_t imm);
    void sar32(GPR dst);
    void sar(const MemAddress& dst, uint8_t imm);
    void sar_cl(const MemAddress& dst);
    void sar32(const MemAddress& dst, uint8_t imm);
    void sar32_cl(const MemAddress& dst);

    void rol(GPR dst, uint8_t imm);
    void rol(GPR dst);
    void rol32(GPR dst, uint8_t imm);
    void rol32(GPR dst);

    void ror(GPR dst, uint8_t imm);
    void ror(GPR dst);
    void ror32(GPR dst, uint8_t imm);
    void ror32(GPR dst);

    // =========================================================================
    // MOVES & EXTENSIONS
    // =========================================================================

    // MOV (64-bit & 32-bit)
    void mov(GPR dst, GPR src);
    void mov32(GPR dst, GPR src);
    void mov(GPR dst, int64_t imm);
    void mov32(GPR dst, uint32_t imm);
    void movabs(GPR dst, uint64_t imm);
    void movabs(GPR dst, const std::string& symbol);
    void mov64(GPR dst, uint64_t imm);
    void mov(GPR dst, const MemAddress& src);
    void mov32(GPR dst, const MemAddress& src);
    void mov(const MemAddress& dst, GPR src);
    void mov32(const MemAddress& dst, GPR src);
    void mov(const MemAddress& dst, int32_t imm);
    void mov32(const MemAddress& dst, int32_t imm);

    // XCHG (64-bit & 32-bit)
    void xchg(GPR dst, GPR src);
    void xchg32(GPR dst, GPR src);

    // 8-bit & 16-bit Moves
    void mov8(GPR dst, GPR src);
    void mov8(GPR dst, uint8_t imm);
    void mov8(GPR dst, const MemAddress& src);
    void mov8(const MemAddress& dst, GPR src);
    void mov8(const MemAddress& dst, uint8_t imm);

    void mov16(GPR dst, GPR src);
    void mov16(GPR dst, uint16_t imm);
    void mov16(GPR dst, const MemAddress& src);
    void mov16(const MemAddress& dst, GPR src);
    void mov16(const MemAddress& dst, uint16_t imm);

    // Sign & Zero Extensions
    void movsxd(GPR dst, GPR src);
    void movsxd(GPR dst, const MemAddress& src);

    void movzx8(GPR dst, GPR src);
    void movzx8(GPR dst, const MemAddress& src);
    void movzx16(GPR dst, GPR src);
    void movzx16(GPR dst, const MemAddress& src);

    void movsx8(GPR dst, GPR src);
    void movsx8(GPR dst, const MemAddress& src);
    void movsx8_32(GPR dst, GPR src);
    void movsx8_32(GPR dst, const MemAddress& src);

    void movsx16(GPR dst, GPR src);
    void movsx16(GPR dst, const MemAddress& src);
    void movsx16_32(GPR dst, GPR src);
    void movsx16_32(GPR dst, const MemAddress& src);

    // =========================================================================
    // SSE & FLOATING POINT (Single & Double)
    // =========================================================================

    // Scalar FP Moves
    void movss(XMM dst, XMM src);
    void movss(XMM dst, const MemAddress& src);
    void movss(const MemAddress& dst, XMM src);

    void movsd(XMM dst, XMM src);
    void movsd(XMM dst, const MemAddress& src);
    void movsd(const MemAddress& dst, XMM src);

    void movups(XMM dst, const MemAddress& src);
    void movups(const MemAddress& dst, XMM src);

    void movq(XMM dst, GPR src);
    void movq(GPR dst, XMM src);
    void movq(XMM dst, const MemAddress& src);
    void movq(const MemAddress& dst, XMM src);
    void movq(XMM dst, XMM src);

    // Double-Precision Arithmetic
    void addsd(XMM dst, XMM src);
    void addsd(XMM dst, const MemAddress& src);
    void subsd(XMM dst, XMM src);
    void subsd(XMM dst, const MemAddress& src);
    void mulsd(XMM dst, XMM src);
    void mulsd(XMM dst, const MemAddress& src);
    void divsd(XMM dst, XMM src);
    void divsd(XMM dst, const MemAddress& src);
    void sqrtsd(XMM dst, XMM src);
    void sqrtsd(XMM dst, const MemAddress& src);
    void ucomisd(XMM dst, XMM src);
    void ucomisd(XMM dst, const MemAddress& src);
    void xorpd(XMM dst, XMM src);
    void xorpd(XMM dst, const MemAddress& src);
    void andpd(XMM dst, XMM src);
    void andpd(XMM dst, const MemAddress& src);
    void orpd(XMM dst, XMM src);
    void orpd(XMM dst, const MemAddress& src);

    // Single-Precision Arithmetic
    void addss(XMM dst, XMM src);
    void addss(XMM dst, const MemAddress& src);
    void subss(XMM dst, XMM src);
    void subss(XMM dst, const MemAddress& src);
    void mulss(XMM dst, XMM src);
    void mulss(XMM dst, const MemAddress& src);
    void divss(XMM dst, XMM src);
    void divss(XMM dst, const MemAddress& src);
    void sqrtss(XMM dst, XMM src);
    void sqrtss(XMM dst, const MemAddress& src);
    void ucomiss(XMM dst, XMM src);
    void ucomiss(XMM dst, const MemAddress& src);
    void xorps(XMM dst, XMM src);
    void xorps(XMM dst, const MemAddress& src);

    // FP Conversions
    void cvtsi2sd(XMM dst, GPR src);
    void cvtsi2sd32(XMM dst, GPR src);
    void cvtsi2sd(XMM dst, const MemAddress& src);
    void cvtsi2sd32(XMM dst, const MemAddress& src);

    void cvttsd2si(GPR dst, XMM src);
    void cvttsd2si32(GPR dst, XMM src);
    void cvttsd2si(GPR dst, const MemAddress& src);
    void cvttsd2si32(GPR dst, const MemAddress& src);

    void cvtsi2ss(XMM dst, GPR src);
    void cvtsi2ss32(XMM dst, GPR src);
    void cvtsi2ss(XMM dst, const MemAddress& src);
    void cvtsi2ss32(XMM dst, const MemAddress& src);

    void cvttss2si(GPR dst, XMM src);
    void cvttss2si32(GPR dst, XMM src);
    void cvttss2si(GPR dst, const MemAddress& src);
    void cvttss2si32(GPR dst, const MemAddress& src);

    void cvtsd2ss(XMM dst, XMM src);
    void cvtsd2ss(XMM dst, const MemAddress& src);
    void cvtss2sd(XMM dst, XMM src);
    void cvtss2sd(XMM dst, const MemAddress& src);

    // =========================================================================
    // 128-BIT SIMD VECTOR INSTRUCTIONS (SSE / SSE2 / SSE4.1)
    // =========================================================================

    void movaps(XMM dst, XMM src);
    void movaps(XMM dst, const MemAddress& src);
    void movaps(const MemAddress& dst, XMM src);

    void movups(XMM dst, XMM src);

    void movd(XMM dst, GPR src);
    void movd(GPR dst, XMM src);
    void movd(XMM dst, const MemAddress& src);
    void movd(const MemAddress& dst, XMM src);

    // Float vectors (f32x4)
    void addps(XMM dst, XMM src);
    void addps(XMM dst, const MemAddress& src);
    void subps(XMM dst, XMM src);
    void subps(XMM dst, const MemAddress& src);
    void mulps(XMM dst, XMM src);
    void mulps(XMM dst, const MemAddress& src);
    void divps(XMM dst, XMM src);
    void divps(XMM dst, const MemAddress& src);
    void minps(XMM dst, XMM src);
    void minps(XMM dst, const MemAddress& src);
    void maxps(XMM dst, XMM src);
    void maxps(XMM dst, const MemAddress& src);
    void sqrtps(XMM dst, XMM src);
    void sqrtps(XMM dst, const MemAddress& src);

    // Double vectors (f64x2)
    void addpd(XMM dst, XMM src);
    void addpd(XMM dst, const MemAddress& src);
    void subpd(XMM dst, XMM src);
    void subpd(XMM dst, const MemAddress& src);
    void mulpd(XMM dst, XMM src);
    void mulpd(XMM dst, const MemAddress& src);
    void divpd(XMM dst, XMM src);
    void divpd(XMM dst, const MemAddress& src);
    void minpd(XMM dst, XMM src);
    void minpd(XMM dst, const MemAddress& src);
    void maxpd(XMM dst, XMM src);
    void maxpd(XMM dst, const MemAddress& src);
    void sqrtpd(XMM dst, XMM src);
    void sqrtpd(XMM dst, const MemAddress& src);

    // Integer vectors (i32x4)
    void paddd(XMM dst, XMM src);
    void paddd(XMM dst, const MemAddress& src);
    void psubd(XMM dst, XMM src);
    void psubd(XMM dst, const MemAddress& src);
    void pmulld(XMM dst, XMM src);
    void pmulld(XMM dst, const MemAddress& src);
    void pminsd(XMM dst, XMM src);
    void pminsd(XMM dst, const MemAddress& src);
    void pmaxsd(XMM dst, XMM src);
    void pmaxsd(XMM dst, const MemAddress& src);

    // Integer vectors (i64x2)
    void paddq(XMM dst, XMM src);
    void paddq(XMM dst, const MemAddress& src);
    void psubq(XMM dst, XMM src);
    void psubq(XMM dst, const MemAddress& src);

    // Bitwise vectors
    void pand(XMM dst, XMM src);
    void pand(XMM dst, const MemAddress& src);
    void por(XMM dst, XMM src);
    void por(XMM dst, const MemAddress& src);
    void pxor(XMM dst, XMM src);
    void pxor(XMM dst, const MemAddress& src);
    void pandn(XMM dst, XMM src);
    void pandn(XMM dst, const MemAddress& src);

    // Shifts / compares
    void pcmpeqd(XMM dst, XMM src);
    void pcmpeqd(XMM dst, const MemAddress& src);
    void pslld(XMM dst, uint8_t imm);
    void psllq(XMM dst, uint8_t imm);

    // Shuffles & broadcast
    void shufps(XMM dst, XMM src, uint8_t imm);
    void shufpd(XMM dst, XMM src, uint8_t imm);
    void pshufd(XMM dst, XMM src, uint8_t imm);
    void movddup(XMM dst, XMM src);

    // Lane insert & extract
    void pinsrd(XMM dst, GPR src, uint8_t lane);
    void pextrd(GPR dst, XMM src, uint8_t lane);
    void pinsrq(XMM dst, GPR src, uint8_t lane);
    void pextrq(GPR dst, XMM src, uint8_t lane);
    void insertps(XMM dst, XMM src, uint8_t imm);
    void extractps(GPR dst, XMM src, uint8_t lane);

    // =========================================================================
    // AVX / AVX2 (256-BIT) & FMA3 INSTRUCTIONS
    // =========================================================================

    // Low-level VEX Prefix Emission
    void emit_vex2(bool r, uint8_t vvvv, bool l, uint8_t pp);
    void emit_vex3(bool w, bool r, bool x, bool b, uint8_t mmmmm, uint8_t vvvv, bool l, uint8_t pp);
    void emit_vex(bool w, uint8_t mmmmm, uint8_t pp, bool l, bool r, bool x, bool b, uint8_t vvvv);

    // 256-bit AVX/AVX2 Moves
    void vmovaps(XMM dst, XMM src);
    void vmovaps(XMM dst, const MemAddress& src);
    void vmovaps(const MemAddress& dst, XMM src);
    void vmovups(XMM dst, XMM src);
    void vmovups(XMM dst, const MemAddress& src);
    void vmovups(const MemAddress& dst, XMM src);

    // 256-bit Float Arithmetic (f32x8)
    void vaddps(XMM dst, XMM src1, XMM src2);
    void vaddps(XMM dst, XMM src1, const MemAddress& src2);
    void vsubps(XMM dst, XMM src1, XMM src2);
    void vsubps(XMM dst, XMM src1, const MemAddress& src2);
    void vmulps(XMM dst, XMM src1, XMM src2);
    void vmulps(XMM dst, XMM src1, const MemAddress& src2);
    void vdivps(XMM dst, XMM src1, XMM src2);
    void vdivps(XMM dst, XMM src1, const MemAddress& src2);
    void vminps(XMM dst, XMM src1, XMM src2);
    void vminps(XMM dst, XMM src1, const MemAddress& src2);
    void vmaxps(XMM dst, XMM src1, XMM src2);
    void vmaxps(XMM dst, XMM src1, const MemAddress& src2);

    // 256-bit Double Arithmetic (f64x4)
    void vaddpd(XMM dst, XMM src1, XMM src2);
    void vaddpd(XMM dst, XMM src1, const MemAddress& src2);
    void vsubpd(XMM dst, XMM src1, XMM src2);
    void vsubpd(XMM dst, XMM src1, const MemAddress& src2);
    void vmulpd(XMM dst, XMM src1, XMM src2);
    void vmulpd(XMM dst, XMM src1, const MemAddress& src2);
    void vdivpd(XMM dst, XMM src1, XMM src2);
    void vdivpd(XMM dst, XMM src1, const MemAddress& src2);
    void vminpd(XMM dst, XMM src1, XMM src2);
    void vminpd(XMM dst, XMM src1, const MemAddress& src2);
    void vmaxpd(XMM dst, XMM src1, XMM src2);
    void vmaxpd(XMM dst, XMM src1, const MemAddress& src2);

    // 256-bit Integer Arithmetic (i32x8 & i64x4)
    void vpaddd(XMM dst, XMM src1, XMM src2);
    void vpaddd(XMM dst, XMM src1, const MemAddress& src2);
    void vpsubd(XMM dst, XMM src1, XMM src2);
    void vpsubd(XMM dst, XMM src1, const MemAddress& src2);
    void vpmulld(XMM dst, XMM src1, XMM src2);
    void vpmulld(XMM dst, XMM src1, const MemAddress& src2);

    void vpaddq(XMM dst, XMM src1, XMM src2);
    void vpaddq(XMM dst, XMM src1, const MemAddress& src2);
    void vpsubq(XMM dst, XMM src1, XMM src2);
    void vpsubq(XMM dst, XMM src1, const MemAddress& src2);

    // 256-bit Bitwise Operations
    void vandps(XMM dst, XMM src1, XMM src2);
    void vandps(XMM dst, XMM src1, const MemAddress& src2);
    void vorps(XMM dst, XMM src1, XMM src2);
    void vorps(XMM dst, XMM src1, const MemAddress& src2);
    void vxorps(XMM dst, XMM src1, XMM src2);
    void vxorps(XMM dst, XMM src1, const MemAddress& src2);

    void vandpd(XMM dst, XMM src1, XMM src2);
    void vandpd(XMM dst, XMM src1, const MemAddress& src2);
    void vorpd(XMM dst, XMM src1, XMM src2);
    void vorpd(XMM dst, XMM src1, const MemAddress& src2);
    void vxorpd(XMM dst, XMM src1, XMM src2);
    void vxorpd(XMM dst, XMM src1, const MemAddress& src2);

    void vpand(XMM dst, XMM src1, XMM src2);
    void vpand(XMM dst, XMM src1, const MemAddress& src2);
    void vpor(XMM dst, XMM src1, XMM src2);
    void vpor(XMM dst, XMM src1, const MemAddress& src2);
    void vpxor(XMM dst, XMM src1, XMM src2);
    void vpxor(XMM dst, XMM src1, const MemAddress& src2);

    // Broadcast into 256-bit registers
    void vbroadcastss(XMM dst, XMM src);
    void vbroadcastss(XMM dst, const MemAddress& src);
    void vbroadcastsd(XMM dst, XMM src);
    void vbroadcastsd(XMM dst, const MemAddress& src);
    void vpbroadcastd(XMM dst, XMM src);
    void vpbroadcastd(XMM dst, const MemAddress& src);
    void vpbroadcastq(XMM dst, XMM src);
    void vpbroadcastq(XMM dst, const MemAddress& src);

    // FMA3: Vector FMA (both 128-bit [is_256=false] and 256-bit [is_256=true])
    void vfmadd213ps(XMM dst, XMM src2, XMM src3, bool is_256 = false);
    void vfmadd213ps(XMM dst, XMM src2, const MemAddress& src3, bool is_256 = false);
    void vfmadd231ps(XMM dst, XMM src2, XMM src3, bool is_256 = false);
    void vfmadd231ps(XMM dst, XMM src2, const MemAddress& src3, bool is_256 = false);

    void vfmadd213pd(XMM dst, XMM src2, XMM src3, bool is_256 = false);
    void vfmadd213pd(XMM dst, XMM src2, const MemAddress& src3, bool is_256 = false);
    void vfmadd231pd(XMM dst, XMM src2, XMM src3, bool is_256 = false);
    void vfmadd231pd(XMM dst, XMM src2, const MemAddress& src3, bool is_256 = false);

    // FMA3: Scalar FMA
    void vfmadd213ss(XMM dst, XMM src2, XMM src3);
    void vfmadd213ss(XMM dst, XMM src2, const MemAddress& src3);
    void vfmadd231ss(XMM dst, XMM src2, XMM src3);
    void vfmadd231ss(XMM dst, XMM src2, const MemAddress& src3);

    void vfmadd213sd(XMM dst, XMM src2, XMM src3);
    void vfmadd213sd(XMM dst, XMM src2, const MemAddress& src3);
    void vfmadd231sd(XMM dst, XMM src2, XMM src3);
    void vfmadd231sd(XMM dst, XMM src2, const MemAddress& src3);

    // =========================================================================
    // BIT OPERATIONS (POPCNT, LZCNT, TZCNT, BSF, BSR)
    // =========================================================================

    void popcnt(GPR dst, GPR src);
    void popcnt32(GPR dst, GPR src);
    void popcnt(GPR dst, const MemAddress& src);
    void popcnt32(GPR dst, const MemAddress& src);

    void lzcnt(GPR dst, GPR src);
    void lzcnt32(GPR dst, GPR src);
    void lzcnt(GPR dst, const MemAddress& src);
    void lzcnt32(GPR dst, const MemAddress& src);

    void tzcnt(GPR dst, GPR src);
    void tzcnt32(GPR dst, GPR src);
    void tzcnt(GPR dst, const MemAddress& src);
    void tzcnt32(GPR dst, const MemAddress& src);

    void bsf(GPR dst, GPR src);
    void bsf32(GPR dst, GPR src);
    void bsf(GPR dst, const MemAddress& src);
    void bsf32(GPR dst, const MemAddress& src);

    void bsr(GPR dst, GPR src);
    void bsr32(GPR dst, GPR src);
    void bsr(GPR dst, const MemAddress& src);
    void bsr32(GPR dst, const MemAddress& src);

    // =========================================================================
    // CONTROL FLOW & BRANCHES
    // =========================================================================

    void jmp(int32_t rel32_disp);
    void jmp_rel8(int8_t disp);
    void jmp(Label label);
    void jmp_short(Label label);
    void jmp_near(Label label);
    void jmp(GPR target);
    void jmp(const MemAddress& target);

    void j(Condition cond, int32_t rel32_disp);
    void j_rel8(Condition cond, int8_t disp);
    void j(Condition cond, Label label);
    void j_short(Condition cond, Label label);
    void j_near(Condition cond, Label label);

    // Convenience Conditional Jumps (near by default)
    void je(Label label)  { j(Condition::E, label); }
    void jne(Label label) { j(Condition::NE, label); }
    void jl(Label label)  { j(Condition::L, label); }
    void jle(Label label) { j(Condition::LE, label); }
    void jg(Label label)  { j(Condition::G, label); }
    void jge(Label label) { j(Condition::GE, label); }
    void jb(Label label)  { j(Condition::B, label); }
    void jbe(Label label) { j(Condition::BE, label); }
    void ja(Label label)  { j(Condition::A, label); }
    void jae(Label label) { j(Condition::AE, label); }
    void js(Label label)  { j(Condition::S, label); }
    void jns(Label label) { j(Condition::NS, label); }
    void jo(Label label)  { j(Condition::O, label); }
    void jno(Label label) { j(Condition::NO, label); }
    void jp(Label label)  { j(Condition::P, label); }
    void jnp(Label label) { j(Condition::NP, label); }

    // Convenience Conditional Jumps (short)
    void je_short(Label label)  { j_short(Condition::E, label); }
    void jne_short(Label label) { j_short(Condition::NE, label); }
    void jl_short(Label label)  { j_short(Condition::L, label); }
    void jle_short(Label label) { j_short(Condition::LE, label); }
    void jg_short(Label label)  { j_short(Condition::G, label); }
    void jge_short(Label label) { j_short(Condition::GE, label); }
    void jb_short(Label label)  { j_short(Condition::B, label); }
    void jbe_short(Label label) { j_short(Condition::BE, label); }
    void ja_short(Label label)  { j_short(Condition::A, label); }
    void jae_short(Label label) { j_short(Condition::AE, label); }
    void js_short(Label label)  { j_short(Condition::S, label); }
    void jns_short(Label label) { j_short(Condition::NS, label); }
    void jo_short(Label label)  { j_short(Condition::O, label); }
    void jno_short(Label label) { j_short(Condition::NO, label); }

    // Conditional Set (SETcc)
    void setcc(Condition cond, GPR dst8);
    void setcc(Condition cond, const MemAddress& dst8);

    void sete(GPR dst8)  { setcc(Condition::E, dst8); }
    void setne(GPR dst8) { setcc(Condition::NE, dst8); }
    void setl(GPR dst8)  { setcc(Condition::L, dst8); }
    void setle(GPR dst8) { setcc(Condition::LE, dst8); }
    void setg(GPR dst8)  { setcc(Condition::G, dst8); }
    void setge(GPR dst8) { setcc(Condition::GE, dst8); }
    void setb(GPR dst8)  { setcc(Condition::B, dst8); }
    void setbe(GPR dst8) { setcc(Condition::BE, dst8); }
    void seta(GPR dst8)  { setcc(Condition::A, dst8); }
    void setae(GPR dst8) { setcc(Condition::AE, dst8); }
    void sets(GPR dst8)  { setcc(Condition::S, dst8); }
    void setns(GPR dst8) { setcc(Condition::NS, dst8); }
    void seto(GPR dst8)  { setcc(Condition::O, dst8); }
    void setno(GPR dst8) { setcc(Condition::NO, dst8); }

    void sete(const MemAddress& dst8)  { setcc(Condition::E, dst8); }
    void setne(const MemAddress& dst8) { setcc(Condition::NE, dst8); }
    void setl(const MemAddress& dst8)  { setcc(Condition::L, dst8); }
    void setle(const MemAddress& dst8) { setcc(Condition::LE, dst8); }
    void setg(const MemAddress& dst8)  { setcc(Condition::G, dst8); }
    void setge(const MemAddress& dst8) { setcc(Condition::GE, dst8); }
    void setb(const MemAddress& dst8)  { setcc(Condition::B, dst8); }
    void setbe(const MemAddress& dst8) { setcc(Condition::BE, dst8); }
    void seta(const MemAddress& dst8)  { setcc(Condition::A, dst8); }
    void setae(const MemAddress& dst8) { setcc(Condition::AE, dst8); }

    // Conditional Move (CMOVcc)
    void cmovcc(Condition cond, GPR dst, GPR src);
    void cmovcc32(Condition cond, GPR dst, GPR src);
    void cmovcc(Condition cond, GPR dst, const MemAddress& src);
    void cmovcc32(Condition cond, GPR dst, const MemAddress& src);

    void cmove(GPR dst, GPR src)  { cmovcc(Condition::E, dst, src); }
    void cmovne(GPR dst, GPR src) { cmovcc(Condition::NE, dst, src); }
    void cmovl(GPR dst, GPR src)  { cmovcc(Condition::L, dst, src); }
    void cmovle(GPR dst, GPR src) { cmovcc(Condition::LE, dst, src); }
    void cmovg(GPR dst, GPR src)  { cmovcc(Condition::G, dst, src); }
    void cmovge(GPR dst, GPR src) { cmovcc(Condition::GE, dst, src); }
    void cmovb(GPR dst, GPR src)  { cmovcc(Condition::B, dst, src); }
    void cmovbe(GPR dst, GPR src) { cmovcc(Condition::BE, dst, src); }
    void cmova(GPR dst, GPR src)  { cmovcc(Condition::A, dst, src); }
    void cmovae(GPR dst, GPR src) { cmovcc(Condition::AE, dst, src); }
    void cmovs(GPR dst, GPR src)  { cmovcc(Condition::S, dst, src); }
    void cmovns(GPR dst, GPR src) { cmovcc(Condition::NS, dst, src); }

    // Function Calls & Returns
    void call(int32_t rel32_disp);
    void call(Label label);
    void call(const std::string& symbol);
    void call(GPR target);
    void call(const MemAddress& target);

    void ret();
    void ret(uint16_t imm);

    // =========================================================================
    // STACK & FRAME OPERATIONS
    // =========================================================================

    void push(GPR reg);
    void push(int32_t imm);
    void push(const MemAddress& mem);

    void pop(GPR reg);
    void pop(const MemAddress& mem);

    void lea(GPR dst, const MemAddress& src);
    void lea32(GPR dst, const MemAddress& src);

    void int3();
    void ud2();
    void cqo();
    void cdq();

    void nop();
    void nop(size_t count);

private:
    CodeBuffer& buffer_;

    // Low-level emission helpers
    void emit_rex(bool w, bool r, bool x, bool b);
    void emit_rex_force(bool w, bool r, bool x, bool b);
    void emit_modrm(uint8_t mod, uint8_t reg, uint8_t rm);
    void emit_sib(Scale scale, uint8_t index, uint8_t base);

    void emit_alu_op(uint8_t op_reg_rm, uint8_t op_ext, GPR dst, GPR src, bool w);
    void emit_alu_imm(uint8_t op_ext, GPR dst, int32_t imm, bool w);
    void emit_alu_mem(uint8_t op_reg_rm, GPR dst, const MemAddress& src, bool w);
    void emit_alu_mem_reg(uint8_t op_rm_reg, const MemAddress& dst, GPR src, bool w);
    void emit_alu_mem_imm(uint8_t op_ext, const MemAddress& dst, int32_t imm, bool w);

    void emit_mem_operand(uint8_t reg_code_val, const MemAddress& mem);
};

} // namespace brass::x64
